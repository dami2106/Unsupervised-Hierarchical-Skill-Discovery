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

import os

num_eps = 1e-11

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.use_deterministic_algorithms(True, warn_only=True)  # PyTorch >=1.8



class VideoSSL(pl.LightningModule):
    def __init__(self, lr=1e-4, weight_decay=1e-4, layer_sizes=[64, 128, 40], n_clusters=20, alpha_train=0.3, alpha_eval=0.3,
                 n_ot_train=[50, 1], n_ot_eval=[50, 1], step_size=None, train_eps=0.06, eval_eps=0.01, ub_frames=False, ub_actions=True,
                 lambda_frames_train=0.05, lambda_actions_train=0.05, lambda_frames_eval=0.05, lambda_actions_eval=0.01,
                 temp=0.1, radius_gw=0.04, learn_clusters=True, n_frames=256, rho=0.1, visualize=False):
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

    def training_step(self, batch, batch_idx):
        features_raw, mask, gt, fname, n_subactions = batch
        with torch.no_grad():
            self.clusters.data = F.normalize(self.clusters.data, dim=-1)
        D = self.layer_sizes[-1]
        B, T, _ = features_raw.shape
        features = F.normalize(self.mlp(features_raw.reshape(-1, features_raw.shape[-1])).reshape(B, T, D), dim=-1)


        codes = torch.exp(features @ self.clusters.T[None, ...] / self.temp)
        codes = codes / codes.sum(dim=-1, keepdim=True)



        with torch.no_grad():  # pseudo-labels from OT
            temp_prior = asot.temporal_prior(T, self.n_clusters, self.rho, features.device)
            cost_matrix = 1. - features @ self.clusters.T.unsqueeze(0)
            cost_matrix += temp_prior
            opt_codes, _ = asot.segment_asot(cost_matrix, mask, eps=self.train_eps, alpha=self.alpha_train, radius=self.radius_gw,
                                             ub_frames=self.ub_frames, ub_actions=self.ub_actions, lambda_frames=self.lambda_frames_train,
                                             lambda_actions=self.lambda_actions_train, n_iters=self.n_ot_train, step_size=self.step_size)

        loss_ce = -((opt_codes * torch.log(codes + num_eps)) * mask[..., None]).sum(dim=2).mean()
        self.log('train_loss', loss_ce)
        return loss_ce

    def validation_step(self, batch, batch_idx):
        features_raw, mask, gt, fname, n_subactions = batch
        D = self.layer_sizes[-1]
        B, T, _ = features_raw.shape
        features = F.normalize(self.mlp(features_raw.reshape(-1, features_raw.shape[-1])).reshape(B, T, D), dim=-1)

        # log clustering metrics over full epoch
        temp_prior = asot.temporal_prior(T, self.n_clusters, self.rho, features.device)
        cost_matrix = 1. - features @ self.clusters.T.unsqueeze(0)
        cost_matrix += temp_prior
        segmentation, _ = asot.segment_asot(cost_matrix, mask, eps=self.eval_eps, alpha=self.alpha_eval, radius=self.radius_gw,
                                            ub_frames=self.ub_frames, ub_actions=self.ub_actions, lambda_frames=self.lambda_frames_eval,
                                            lambda_actions=self.lambda_actions_eval, n_iters=self.n_ot_eval, step_size=self.step_size)
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
        codes = torch.exp(features @ self.clusters.T / self.temp)
        codes /= codes.sum(dim=-1, keepdim=True)
        pseudo_labels, _ = asot.segment_asot(cost_matrix, mask, eps=self.train_eps, alpha=self.alpha_train, radius=self.radius_gw,
                                             ub_frames=self.ub_frames, ub_actions=self.ub_actions, lambda_frames=self.lambda_frames_train,
                                             lambda_actions=self.lambda_actions_train, n_iters=self.n_ot_train, step_size=self.step_size)
        loss_ce = -((pseudo_labels * torch.log(codes + num_eps)) * mask[..., None]).sum(dim=[1, 2]).mean()
        self.log('val_loss', loss_ce)

        # plot qualitative examples of pseudo-labelling and embeddings for 5 videos evenly spaced in dataset
        spacing =  int(self.trainer.num_val_batches[0] / 5)
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
        temp_prior = asot.temporal_prior(T, self.n_clusters, self.rho, features.device)
        cost_matrix = 1. - features @ self.clusters.T.unsqueeze(0)
        cost_matrix += temp_prior
        segmentation, _ = asot.segment_asot(cost_matrix, mask, eps=self.eval_eps, alpha=self.alpha_eval, radius=self.radius_gw,
                                            ub_frames=self.ub_frames, ub_actions=self.ub_actions, lambda_frames=self.lambda_frames_eval,
                                            lambda_actions=self.lambda_actions_eval, n_iters=self.n_ot_eval, step_size=self.step_size)
        segments = segmentation.argmax(dim=2)
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
    parser.add_argument('--n-clusters', '-c', type=int, default=5, help='number of actions/clusters')

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
                       n_frames=args.n_frames, lr=args.learning_rate, weight_decay=args.weight_decay, rho=args.rho, visualize=args.visualize)

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
