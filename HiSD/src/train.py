import argparse
from copy import deepcopy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger

import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist, squareform
from sklearn.cluster import KMeans

from dataset_loader import RLDataset
import asot
from utils import *
from metrics import ClusteringMetrics, indep_eval_metrics
from bnp_marginal import DPStickBreakingMarginal
from sklearn.metrics.cluster import normalized_mutual_info_score
import json

import os

num_eps = 1e-11

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.use_deterministic_algorithms(True, warn_only=True)  # PyTorch >=1.8



class VideoSSL(pl.LightningModule):
    def __init__(self, lr=1e-4, weight_decay=1e-4, layer_sizes=[64, 128, 40], n_clusters=20, alpha_train=0.3, alpha_eval=0.3,
                 n_ot_train=[50, 1], n_ot_eval=[50, 1], step_size=None, train_eps=0.06, eval_eps=0.01, ub_frames=False, ub_actions=True,
                 lambda_frames_train=0.05, lambda_actions_train=0.05, lambda_frames_eval=0.05, lambda_actions_eval=0.01,
                 temp=0.1, radius_gw=0.04, learn_clusters=True, n_frames=256, rho=0.1, visualize=False,
                 marginal='uniform', dp_gamma=1.0, dp_prune_eps=None, dp_gamma_prior=None, dp_decay=0.99,
                 dp_warmup=20, dp_usage='plan', dp_usage_lambda=0.01, dp_merge_cos=None, dp_merge_delta=0.1,
                 dp_train_marginal='uniform', n_gt_skills=None):
        super().__init__()
        self.lr = lr
        self.weight_decay = weight_decay
        self.n_clusters = n_clusters
        self.learn_clusters = learn_clusters
        self.layer_sizes = layer_sizes

        self.visualize = visualize

        self.alpha_train = alpha_train
        self.alpha_eval = alpha_eval
        self.n_ot_train = n_ot_train
        self.n_ot_eval = n_ot_eval
        self.step_size = step_size
        self.train_eps = train_eps
        self.eval_eps = eval_eps
        self.radius_gw = radius_gw
        self.ub_frames = ub_frames
        self.ub_actions = ub_actions
        self.lambda_frames_train = lambda_frames_train
        self.lambda_actions_train = lambda_actions_train
        self.lambda_frames_eval = lambda_frames_eval
        self.lambda_actions_eval = lambda_actions_eval
        self.exclude_cls = None # whether to exclude -1 class from evaluation

        self.temp = temp
        self.n_frames = n_frames
        self.rho = rho


        # initialize MLP
        layers = [nn.Sequential(nn.Linear(sz, sz1), nn.ReLU()) for sz, sz1 in zip(layer_sizes[:-2], layer_sizes[1:-1])]
        layers += [nn.Linear(layer_sizes[-2], layer_sizes[-1])]
        self.mlp = nn.Sequential(*layers)

        # initialize cluster centers/codebook
        d = self.layer_sizes[-1]
        self.clusters = nn.parameter.Parameter(data=F.normalize(torch.randn(self.n_clusters, d), dim=-1), requires_grad=learn_clusters)

        # Component A: learned DP stick-breaking target marginal (n_clusters acts as K_max)
        self.marginal = marginal
        self.dp_usage = dp_usage
        self.dp_usage_lambda = dp_usage_lambda
        self.dp_merge_cos = dp_merge_cos
        self.dp_merge_delta = dp_merge_delta
        self.dp_train_marginal = dp_train_marginal
        self.n_gt_skills = n_gt_skills
        if marginal == 'dp':
            self.dp = DPStickBreakingMarginal(n_clusters, gamma=dp_gamma, prune_eps=dp_prune_eps,
                                              gamma_prior=dp_gamma_prior, decay=dp_decay,
                                              warmup_updates=dp_warmup)
        else:
            self.dp = None
        self.test_pred_labels, self.test_gt_labels = [], []

        # initialize evaluation metrics
        self.mof = ClusteringMetrics(metric='mof')
        self.f1 = ClusteringMetrics(metric='f1')
        self.miou = ClusteringMetrics(metric='miou')
        self.save_hyperparameters()
        self.test_cache = []

    def save_figure_to_disk(self, fig, figure_name, global_step):
        """
        Saves the given matplotlib figure to the 'figures' folder inside the experiment folder.
        """
        # Get the base experiment directory from the logger if available.
        if hasattr(self.logger, 'log_dir'):
            base_dir = self.logger.log_dir
        else:
            base_dir = '.'
        figures_dir = os.path.join(base_dir, 'figures')
        os.makedirs(figures_dir, exist_ok=True)
        fig_path = os.path.join(figures_dir, f"{figure_name}_step_{global_step}.png")
        fig.savefig(fig_path)

    def active_clusters(self):
        if self.dp is None:
            return torch.arange(self.n_clusters, device=self.clusters.device)
        return torch.nonzero(self.dp.active).squeeze(1)

    def compute_codes(self, features):
        codes = torch.exp(features @ self.clusters.T[None, ...] / self.temp)
        if self.dp is not None:
            codes = codes * self.dp.active.float()  # pruned prototypes receive no mass
        return codes / codes.sum(dim=-1, keepdim=True)

    def segment(self, features, mask, train, relax_actions=False):
        """Run ASOT over the active prototypes (all K for uniform q) and return a (B, T, K_max) plan.

        relax_actions: solve the unbalanced-actions relaxation (KL weight dp_usage_lambda) instead;
        used only to measure data-driven usage for the q-step when the action marginal is hard.
        """
        B, T, _ = features.shape
        idx = self.active_clusters()
        k_act = len(idx)
        cost_matrix = 1. - features @ self.clusters[idx].T.unsqueeze(0)
        cost_matrix = cost_matrix + asot.temporal_prior(T, k_act, self.rho, features.device)
        # Training keeps equipartition over the *active* prototypes unless dp_train_marginal='learned':
        # feeding a learned marginal back into self-labelled representation learning invites collapse.
        use_learned = self.dp is not None and (not train or relax_actions or self.dp_train_marginal == 'learned')
        q = self.dp.q[idx] if use_learned else None
        if train:
            kw = dict(eps=self.train_eps, alpha=self.alpha_train, lambda_frames=self.lambda_frames_train,
                      lambda_actions=self.lambda_actions_train, n_iters=self.n_ot_train)
        else:
            kw = dict(eps=self.eval_eps, alpha=self.alpha_eval, lambda_frames=self.lambda_frames_eval,
                      lambda_actions=self.lambda_actions_eval, n_iters=self.n_ot_eval)
        ub_actions = self.ub_actions
        if relax_actions:
            ub_actions = True
            kw['lambda_actions'] = self.dp_usage_lambda
        plan_act, _ = asot.segment_asot(cost_matrix, mask, radius=self.radius_gw, ub_frames=self.ub_frames,
                                        ub_actions=ub_actions, step_size=self.step_size, q=q, **kw)
        plan = torch.zeros(B, T, self.n_clusters, device=features.device, dtype=plan_act.dtype)
        plan[:, :, idx] = plan_act
        return plan

    @torch.no_grad()
    def redundancy(self, features, mask):
        """Per-prototype transport-cost increase if its frames were given to the runner-up prototype.

        Small-variance (DP-means) view of the DP: a prototype is worth keeping only if it lowers the
        transport cost of its frames by more than a penalty.  Returns (sum of increases, #frames)."""
        idx = self.active_clusters()
        if len(idx) < 2:
            return None, None
        cost = 1. - features @ self.clusters[idx].T.unsqueeze(0)  # (B, T, K_act)
        cost = cost[mask]
        two = torch.topk(cost, 2, dim=1, largest=False)
        own = two.indices[:, 0]
        gain = two.values[:, 1] - two.values[:, 0]
        k_max = self.n_clusters
        sums = torch.zeros(k_max, device=cost.device).index_add_(0, idx[own], gain)
        cnts = torch.zeros(k_max, device=cost.device).index_add_(0, idx[own], torch.ones_like(gain))
        return sums, cnts

    def training_step(self, batch, batch_idx):
        features_raw, mask, gt, fname, n_subactions = batch
        with torch.no_grad():
            self.clusters.data = F.normalize(self.clusters.data, dim=-1)
        D = self.layer_sizes[-1]
        B, T, _ = features_raw.shape
        features = F.normalize(self.mlp(features_raw.reshape(-1, features_raw.shape[-1])).reshape(B, T, D), dim=-1)


        codes = self.compute_codes(features)

        with torch.no_grad():  # pseudo-labels from OT
            opt_codes = self.segment(features, mask, train=True)
            if self.dp is not None:  # q-step: closed-form stick-breaking posterior update
                if self.dp_usage == 'codes':
                    usage_src = codes
                elif self.ub_actions:
                    usage_src = opt_codes
                else:
                    # with a hard action marginal the realised usage equals q (a fixed point), so
                    # measure usage on the unbalanced relaxation of the same problem
                    usage_src = self.segment(features, mask, train=True, relax_actions=True)
                self.dp.update(DPStickBreakingMarginal.segment_usage(usage_src, mask))
                if self.dp_merge_delta is not None:
                    self.dp.accumulate_redundancy(*self.redundancy(features, mask))
                    self.dp.prune_redundant(self.dp_merge_delta)
                if self.dp_merge_cos is not None:
                    self.dp.merge_similar(self.clusters.data, self.dp_merge_cos)

        loss_ce = -((opt_codes * torch.log(codes + num_eps)) * mask[..., None]).sum(dim=2).mean()
        self.log('train_loss', loss_ce)
        if self.dp is not None:
            self.log('train_k_hat', float(self.dp.k_hat))
            self.log('train_dp_gamma', float(self.dp.gamma))
        return loss_ce

    def validation_step(self, batch, batch_idx):
        features_raw, mask, gt, fname, n_subactions = batch
        D = self.layer_sizes[-1]
        B, T, _ = features_raw.shape
        features = F.normalize(self.mlp(features_raw.reshape(-1, features_raw.shape[-1])).reshape(B, T, D), dim=-1)

        # log clustering metrics over full epoch
        segmentation = self.segment(features, mask, train=False)
        segments = segmentation.argmax(dim=2)
        self.mof.update(segments, gt, mask)
        self.f1.update(segments, gt, mask)
        self.miou.update(segments, gt, mask)

        # log clustering metrics per video
        metrics = indep_eval_metrics(segments, gt, mask, ['mof', 'f1', 'miou'])
        self.log('val_mof_per', metrics['mof'])
        self.log('val_f1_per', metrics['f1'])
        self.log('val_miou_per', metrics['miou'])

        # log validation loss
        codes = self.compute_codes(features)
        pseudo_labels = self.segment(features, mask, train=True)
        loss_ce = -((pseudo_labels * torch.log(codes + num_eps)) * mask[..., None]).sum(dim=[1, 2]).mean()
        self.log('val_loss', loss_ce)

        # plot qualitative examples of pseudo-labelling and embeddings for 5 videos evenly spaced in dataset
        spacing = max(1, int(self.trainer.num_val_batches[0] / 5))
        if batch_idx % spacing == 0 and self.visualize:
            plot_idx = int(batch_idx / spacing)
            global_step = self.trainer.global_step
            gt_cpu = gt[0].cpu().numpy()

            fdists = squareform(pdist(features[0].cpu().numpy(), 'cosine'))
            fig = plot_matrix(fdists, gt=gt_cpu, colorbar=False, title=fname[0], figsize=(5, 5),
                              xlabel='Frame index', ylabel='Frame index')
            if self.logger is not None and hasattr(self.logger, 'experiment'):
                self.logger.experiment.add_figure(f"val_pairwise_{plot_idx}", fig, global_step)
            self.save_figure_to_disk(fig, f"val_pairwise_{plot_idx}", global_step)
            plt.close()

            fig = plot_matrix(codes[0].cpu().numpy().T, gt=gt_cpu, colorbar=False, title=fname[0], figsize=(10, 5),
                             xlabel='Frame index', ylabel='Action index')
            if self.logger is not None and hasattr(self.logger, 'experiment'):
                self.logger.experiment.add_figure(f"val_P_{plot_idx}", fig, global_step)
            self.save_figure_to_disk(fig, f"val_P_{plot_idx}", global_step)
            plt.close()

            fig = plot_matrix(pseudo_labels[0].cpu().numpy().T, gt=gt_cpu, colorbar=False, title=fname[0], figsize=(10, 5),
                             xlabel='Frame index', ylabel='Action index')
            if self.logger is not None and hasattr(self.logger, 'experiment'):
                self.logger.experiment.add_figure(f"val_OT_PL_{plot_idx}", fig, global_step)
            self.save_figure_to_disk(fig, f"val_OT_PL_{plot_idx}", global_step)
            plt.close()

            fig = plot_matrix(segmentation[0].cpu().numpy().T, gt=gt_cpu, colorbar=False, title=fname[0], figsize=(10, 5),
                             xlabel='Frame index', ylabel='Action index')
            if self.logger is not None and hasattr(self.logger, 'experiment'):
                self.logger.experiment.add_figure(f"val_OT_pred_{plot_idx}", fig, global_step)
            self.save_figure_to_disk(fig, f"val_OT_pred_{plot_idx}", global_step)
            plt.close()

        return None

    def test_step(self, batch, batch_idx):
        features_raw, mask, gt, fname, n_subactions = batch
        D = self.layer_sizes[-1]
        B, T, _ = features_raw.shape
        features = F.normalize(self.mlp(features_raw.reshape(-1, features_raw.shape[-1])).reshape(B, T, D), dim=-1)

        # log clustering metrics over full epoch
        segmentation = self.segment(features, mask, train=False)
        segments = segmentation.argmax(dim=2)
        self.test_pred_labels.extend(segments[mask].tolist())
        self.test_gt_labels.extend(gt[mask].tolist())
        self.mof.update(segments, gt, mask)
        self.f1.update(segments, gt, mask)
        self.miou.update(segments, gt, mask)

        # log clustering metrics per video
        metrics = indep_eval_metrics(segments, gt, mask, ['mof', 'f1', 'miou'])
        self.log('test_mof_per', metrics['mof'])
        self.log('test_f1_per', metrics['f1'])
        self.log('test_miou_per', metrics['miou'])

        # cache videos for plotting
        self.test_cache.append([metrics['mof'], segments, gt, mask, fname])

        return None

    def on_validation_epoch_end(self):
        mof, pred_to_gt = self.mof.compute()
        f1, _ = self.f1.compute(pred_to_gt=pred_to_gt)
        miou, _ = self.miou.compute(pred_to_gt=pred_to_gt)
        self.log('val_mof_full', mof)
        self.log('val_f1_full', f1)
        self.log('val_miou_full', miou)
        self.mof.reset()
        self.f1.reset()
        self.miou.reset()

    def on_test_epoch_end(self):
        # compute global metrics
        mof, pred_to_gt = self.mof.compute()
        f1, _          = self.f1.compute(pred_to_gt=pred_to_gt)
        miou, _        = self.miou.compute(pred_to_gt=pred_to_gt)

        self.log('test_mof_full',  mof)
        self.log('test_f1_full',   f1)
        self.log('test_miou_full', miou)

        # K-free diagnostics: inferred K_hat (active prototypes), number of skills actually
        # predicted, K_hat error vs ground truth and NMI (label-permutation invariant)
        k_used = len(np.unique(self.test_pred_labels))
        k_gt = len(np.unique(self.test_gt_labels))
        k_hat = self.dp.k_hat if self.dp is not None else self.n_clusters
        self.log('test_k_hat', float(k_hat))
        self.log('test_k_used', float(k_used))
        self.log('test_k_err', float(abs(k_used - k_gt)))
        self.log('test_nmi', float(normalized_mutual_info_score(self.test_gt_labels, self.test_pred_labels)))
        if self.dp is not None:
            base_dir = getattr(self.logger, 'log_dir', None)
            if base_dir is not None:
                os.makedirs(base_dir, exist_ok=True)
                with open(os.path.join(base_dir, 'dp_marginal.json'), 'w') as f:
                    json.dump(self.dp.state_summary(), f, indent=2)
        self.test_pred_labels, self.test_gt_labels = [], []

        if self.visualize:
            for i, (mof, pred, gt, mask, fname) in enumerate(self.test_cache):
                self.test_cache[i][0] = indep_eval_metrics(pred, gt, mask, ['mof'], exclude_cls=self.exclude_cls, pred_to_gt=pred_to_gt)['mof']

                


            self.test_cache = sorted(self.test_cache, key=lambda x: x[0], reverse=True)

            saved_matching_mapping = False

            for i, (mof, pred, gt, mask, fname) in enumerate(self.test_cache):
                print("fname:", fname[0], "Mof:", self.test_cache[i][0])
                print("Predicted: ", pred)
                print("Ground truth:", gt)
                print("Mask:", mask)
                print("gt_uniq:", np.unique(self.mof.gt_labels))
                print("Predicted to GT mapping:", pred_to_gt)
                print("=" * 10)
                fig = plot_segmentation_gt(gt, pred, mask, exclude_cls=self.exclude_cls, pred_to_gt=pred_to_gt,
                                           gt_uniq=np.unique(self.mof.gt_labels), name=f'{fname[0]}')
                

                base_dir = getattr(self.logger, 'log_dir', '.')
                segments_dir = os.path.join(base_dir, 'segments')
                skills_dir   = os.path.join(base_dir, 'predicted_skills')
                mapping_dir = os.path.join(base_dir, 'mapping')
                os.makedirs(segments_dir, exist_ok=True)
                os.makedirs(skills_dir, exist_ok=True)
                os.makedirs(mapping_dir, exist_ok=True)

                if not saved_matching_mapping:
                    save_matching_mapping(pred_to_gt, out_dir=mapping_dir)
                    saved_matching_mapping = True

                skills = (
                    pred[0].cpu().numpy()
                    if hasattr(pred, 'cpu')
                    else np.array(pred)
                )
                # Save both the original format and the new segments format
                save_skill_ordering(skills, fname[0], out_dir=skills_dir)
                save_skill_segments(skills, fname[0], out_dir=skills_dir)
                


                filename = f"{i:04d}_{fname[0]}_step_{self.trainer.global_step}.png"
                fig_path = os.path.join(segments_dir, filename)
                fig.savefig(fig_path)

                plt.close()


        # if self.visualize:
        #     # 1) compute per-episode MIOU, store in slot 0
        #     for idx, (m, pred, gt, mask, fname) in enumerate(self.test_cache):
        #         val = indep_eval_metrics(
        #             pred, gt, mask,
        #             ['miou'],
        #             pred_to_gt=pred_to_gt
        #         )['miou']
        #         self.test_cache[idx][0] = val

        #     # 2) sort ALL episodes by miou descending
        #     self.test_cache.sort(key=lambda x: x[0], reverse=True)

        #     # 3) prepare output dirs
        #     base_dir     = getattr(self.logger, 'log_dir', '.')
        #     skills_dir   = os.path.join(base_dir, 'predicted_skills')
        #     segments_dir = os.path.join(base_dir, 'segments')
        #     os.makedirs(skills_dir,   exist_ok=True)
        #     os.makedirs(segments_dir, exist_ok=True)

        #     # save the overall matching once

        #     # 4) now save ALL episodes in sorted order
        #     for rank, (mof_val, pred, gt, mask, fname) in enumerate(self.test_cache):
        #         # save per-episode skill ordering
        #         skills = (
        #             pred[0].cpu().numpy().tolist()
        #             if hasattr(pred, 'cpu')
        #             else np.array(pred).tolist()
        #         )
        #         save_skill_ordering(skills, fname[0], out_dir=skills_dir)

        #         # plot segmentation
        #         fig = plot_segmentation_gt(
        #             gt, pred, mask,
        #             # pred_to_gt=pred_to_gt,
        #             # gt_uniq=np.unique(self.mof.gt_labels),
        #             name=fname[0]
        #         )
        #         # optional wandb/logger hook
        #         if self.logger is not None and hasattr(self.logger, 'experiment'):
        #             self.logger.experiment.add_figure(
        #                 f"test_segment_{rank}", fig, self.trainer.global_step
        #             )

        #         # save with zero-padded rank so files list in order
        #         filename = f"{rank:04d}_{fname[0]}_step_{self.trainer.global_step}.png"
        #         fig_path = os.path.join(segments_dir, filename)
        #         fig.savefig(fig_path)
        #         plt.close(fig)

        # reset for next epoch
        self.test_cache = []
        self.mof.reset()
        self.f1.reset()
        self.miou.reset()

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)

    def fit_clusters(self, dataloader, K):
        with torch.no_grad():
            features_full = []
            self.mlp.eval()
            for features_raw, _, _, _, _ in dataloader:
                B, T, _ = features_raw.shape
                D = self.layer_sizes[-1]
                features = F.normalize(self.mlp(features_raw.reshape(-1, features_raw.shape[-1])).reshape(B, T, D), dim=-1)
                features_full.append(features)
            features_full = torch.cat(features_full, dim=0).reshape(-1, features.shape[2]).cpu().numpy()
            kmeans = KMeans(n_clusters=K).fit(features_full) #n_init = 10
            self.mlp.train()
        self.clusters.data = torch.from_numpy(kmeans.cluster_centers_).to(self.clusters.device)
        return None


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train representation learning pipeline")

    # FUGW OT segmentation parameters
    parser.add_argument('--alpha-train', '-at', type=float, default=0.4, help='weighting of KOT term on frame features in OT')
    parser.add_argument('--alpha-eval', '-ae', type=float, default=0.7, help='weighting of KOT term on frame features in OT')
    parser.add_argument('--ub-frames', '-uf', action='store_true',
                        help='relaxes balanced assignment assumption over frames, i.e., each frame is assigned')
    parser.add_argument('--ub-actions', '-ua', action='store_true',
                        help='relaxes balanced assignment assumption over actions, i.e., each action is uniformly represented in a video')
    parser.add_argument('--lambda-frames-train', '-lft', type=float, default=0.05, help='penalty on balanced frames assumption for training')
    parser.add_argument('--lambda-actions-train', '-lat', type=float, default=0.05, help='penalty on balanced actions assumption for training')
    parser.add_argument('--lambda-frames-eval', '-lfe', type=float, default=0.05, help='penalty on balanced frames assumption for test')
    parser.add_argument('--lambda-actions-eval', '-lae', type=float, default=0.01, help='penalty on balanced actions assumption for test')
    parser.add_argument('--eps-train', '-et', type=float, default=0.07, help='entropy regularization for OT during training')
    parser.add_argument('--eps-eval', '-ee', type=float, default=0.04, help='entropy regularization for OT during val/test')
    parser.add_argument('--radius-gw', '-r', type=float, default=0.04, help='Radius parameter for GW structure loss')
    parser.add_argument('--n-ot-train', '-nt', type=int, nargs='+', default=[25, 1], help='number of outer and inner iterations for ASOT solver (train)')
    parser.add_argument('--n-ot-eval', '-no', type=int, nargs='+', default=[25, 1], help='number of outer and inner iterations for ASOT solver (eval)')
    parser.add_argument('--step-size', '-ss', type=float, default=None,
                        help='Step size/learning rate for ASOT solver. Worth setting manually if ub-frames && ub-actions')

    parser.add_argument('--dataset', '-d', type=str,  default='desktop_assembly' ,help='dataset to use for training/eval (Breakfast, YTI, FSeval, FS, desktop_assembly)')
    parser.add_argument('--feature-name',  type=str,  default='symbolic_obs' ,help='name of the features folder')
    parser.add_argument('--n-frames', '-f', type=int, default=6, help='number of frames sampled per video for train/val')
    parser.add_argument('--std-feats', '-s', action='store_true', help='standardize features per video during preprocessing')
    parser.add_argument('--save-directory', '-sd', type=str, default='runs', help='directory to save model file, plots and results')

    # representation learning params
    parser.add_argument('--n-epochs', '-ne', type=int, default=15, help='number of epochs for training')
    parser.add_argument('--batch-size', '-bs', type=int, default=8, help='batch size')
    parser.add_argument('--learning-rate', '-lr', type=float, default=1e-3, help='learning rate')
    parser.add_argument('--weight-decay', '-wd', type=float, default=1e-4, help='weight decay for optimizer')
    parser.add_argument('--k-means', '-km', action='store_false', help='do not initialize clusters with kmeans default = True')
    parser.add_argument('--layers', '-ls', default=[500, 256, 50], nargs='+', type=int, help='layer sizes for MLP (in, hidden, ..., out)')
    parser.add_argument('--rho', type=float, default=0.1, help='Factor for global structure weighting term')
    parser.add_argument('--n-clusters', '-c', type=int, default=5, help='number of actions/clusters (K_max when --marginal dp)')

    # Component A: K-free segmentation via a learned DP stick-breaking target marginal
    parser.add_argument('--marginal', type=str, default='uniform', choices=['uniform', 'dp'],
                        help='target action marginal q: uniform over K (original ASOT) or learned DP stick-breaking')
    parser.add_argument('--dp-gamma', type=float, default=1.0, help='DP concentration gamma')
    parser.add_argument('--dp-gamma-prior', type=float, nargs=2, default=None, metavar=('A0', 'B0'),
                        help='Gamma(a0, b0) hyperprior on gamma (e.g. 1 1); omitted = fixed gamma')
    parser.add_argument('--dp-prune-eps', type=float, default=None, help='prune threshold on q share (default 1/(2 K_max))')
    parser.add_argument('--dp-decay', type=float, default=0.99, help='decay of running usage counts per q-step')
    parser.add_argument('--dp-warmup', type=int, default=None, help='q-steps before pruning is allowed (overrides --dp-warmup-frac)')
    parser.add_argument('--dp-warmup-frac', type=float, default=0.5,
                        help='fraction of training steps before pruning is allowed: prototype redundancy is only '
                             'identifiable once the representation has settled')
    parser.add_argument('--dp-usage', type=str, default='plan', choices=['plan', 'codes'],
                        help='usage for the q-step: OT plan (Gamma*) or the unconstrained network codes')
    parser.add_argument('--dp-usage-lambda', type=float, default=0.01,
                        help='KL weight of the unbalanced relaxation used for the q-step when actions are balanced')
    parser.add_argument('--dp-merge-delta', type=float, default=0.1,
                        help='DP-means style pruning: drop a prototype whose frames cost < delta more (mean cosine '
                             'distance) on their runner-up prototype; negative disables')
    parser.add_argument('--dp-train-marginal', type=str, default='uniform', choices=['uniform', 'learned'],
                        help='marginal used for training pseudo-labels: equipartition over active prototypes '
                             '(collapse-safe) or the learned q; evaluation always uses the learned q')
    parser.add_argument('--dp-merge-cos', type=float, default=None,
                        help='merge active prototypes whose cosine similarity exceeds this (off by default)')

    # system/logging params
    parser.add_argument('--val-freq', '-vf', type=int, default=5, help='validation epoch frequency (epochs)')
    parser.add_argument('--visualize', '-v', action='store_true', help='generate visualizations during logging')
    parser.add_argument('--seed', type=int, default=0, help='Random seed initialization')
    parser.add_argument('--ckpt', type=str, help='path to checkpoint')
    parser.add_argument('--eval', action='store_true', help='run evaluation on test set only')

    parser.add_argument('--run', type=str, default='test_run', help='experiment run name')
    parser.add_argument('--log', action='store_true', help='whether or not to log to tensorboard')

    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    pl.seed_everything(args.seed)

    data_val = RLDataset('', args.dataset, args.n_frames, standardise=args.std_feats, random=False, feature_type=args.feature_name)
    data_train = RLDataset('', args.dataset, args.n_frames, standardise=args.std_feats, random=True, feature_type=args.feature_name)
    data_test = RLDataset('', args.dataset, None, standardise=args.std_feats, random=False, feature_type=args.feature_name)
    #Maybe combine above ^ 
    val_loader = DataLoader(data_val, batch_size=args.batch_size,shuffle=False)
    train_loader = DataLoader(data_train, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(data_test, batch_size=1, shuffle=False)


    # np.random.seed(args.seed)
    # torch.manual_seed(args.seed)
    # torch.cuda.manual_seed(args.seed)
    # torch.cuda.manual_seed_all(args.seed)

    # pl.seed_everything(args.seed)

    #Seed dataset with 0 then seed model args.seed

    if args.ckpt is not None:
        ssl = VideoSSL.load_from_checkpoint(args.ckpt)
    else:
        ssl = VideoSSL(layer_sizes=args.layers, n_clusters=args.n_clusters, alpha_train=args.alpha_train, alpha_eval=args.alpha_eval,
                       ub_frames=args.ub_frames, ub_actions=args.ub_actions, lambda_frames_train=args.lambda_frames_train, lambda_frames_eval=args.lambda_frames_eval,
                       lambda_actions_train=args.lambda_actions_train, lambda_actions_eval=args.lambda_actions_eval, step_size=args.step_size,
                       train_eps=args.eps_train, eval_eps=args.eps_eval, radius_gw=args.radius_gw, n_ot_train=args.n_ot_train, n_ot_eval=args.n_ot_eval,
                       n_frames=args.n_frames, lr=args.learning_rate, weight_decay=args.weight_decay, rho=args.rho, visualize=args.visualize,
                       marginal=args.marginal, dp_gamma=args.dp_gamma, dp_prune_eps=args.dp_prune_eps, dp_gamma_prior=args.dp_gamma_prior,
                       dp_decay=args.dp_decay, dp_usage=args.dp_usage,
                       dp_warmup=args.dp_warmup if args.dp_warmup is not None else int(args.dp_warmup_frac * args.n_epochs * len(train_loader)),
                       dp_usage_lambda=args.dp_usage_lambda, dp_merge_cos=args.dp_merge_cos,
                       dp_merge_delta=args.dp_merge_delta if args.dp_merge_delta >= 0 else None,
                       dp_train_marginal=args.dp_train_marginal, n_gt_skills=data_test.n_subactions)

    # Conditionally create the TensorBoard logger if logging is enabled.
    if args.log:
        name = f'{args.dataset}_{args.run}'
        logger = TensorBoardLogger(save_dir=args.save_directory, name=name)
    else:
        logger = None

    trainer = pl.Trainer(devices=1, check_val_every_n_epoch=args.val_freq, max_epochs=args.n_epochs, log_every_n_steps=50, logger=logger)

    if args.k_means and args.ckpt is None:
        ssl.fit_clusters(train_loader, args.n_clusters)

    if not args.eval:
        trainer.validate(ssl, val_loader)
        trainer.fit(ssl, train_loader, val_loader)

    trainer.test(ssl, dataloaders=test_loader)
