import os
import os.path as path

import numpy as np
import torch
from torch.utils.data import Dataset





class RLDataset(Dataset):
    def __init__(self, root_dir: str, dataset, n_frames, standardise=True, split: str = None, random=True, n_videos=None, feature_type='features'):
        self.root_dir = root_dir
        self.dataset = dataset
        self.data_dir = path.join(root_dir, self.dataset)
        self.video_fnames = sorted([fname for fname in os.listdir(path.join(self.data_dir, 'groundTruth'))
                                    if len(fname.split('_')) > 1 or len(fname.split('-')) > 1])

        self.feature_type = feature_type

        if n_videos is not None:
            self.video_fnames = self.video_fnames[::int(len(self.video_fnames) / n_videos)]

        def prep(x):
            i, nm = x.rstrip().split(' ')
            return nm, int(i)
    
        action_mapping = list(map(prep, open(path.join(self.data_dir, 'mapping/mapping.txt'))))
        self.action_mapping = dict(action_mapping)
        self.n_subactions = len(set(self.action_mapping.keys()))
        self.n_frames = n_frames
        self.standardise = standardise
        self.random = random

    def __len__(self):
        return len(self.video_fnames)
    
    def __getitem__(self, idx):
        video_fname = self.video_fnames[idx]
        gt = [line.rstrip() for line in open(path.join(self.data_dir, 'groundTruth', video_fname))]
        inds, mask = self._partition_and_sample(self.n_frames, len(gt))
        gt = torch.Tensor([self.action_mapping[gt[ind]] for ind in inds]).long()
        feat_fname = path.join(self.data_dir, self.feature_type, video_fname)
        try:
            features = np.loadtxt(feat_fname + '.txt')[inds, :]
        except:
            features = np.load(feat_fname + '.npy')[inds, :]

        if self.standardise:  # normalize features
            zmask = np.ones(features.shape[0], dtype=bool)
            for rdx, row in enumerate(features):
                if np.sum(row) == 0:
                    zmask[rdx] = False
            z = features[zmask] - np.mean(features[zmask], axis=0)
            z = z / np.std(features[zmask], axis=0)
            features = np.zeros(features.shape)
            features[zmask] = z
            features = np.nan_to_num(features)
            features /= np.sqrt(features.shape[1])
        
        features = torch.from_numpy(features).float()
        return features, mask, gt, video_fname, gt.unique().shape[0]
    
    def _partition_and_sample(self, n_samples, n_frames):
        if n_samples is None:
            indices = np.arange(n_frames)
            mask = np.full(n_frames, 1, dtype=bool)
        elif n_samples < n_frames:
            if self.random:
                boundaries = np.linspace(0, n_frames-1, n_samples+1).astype(int)
                indices = np.random.randint(low=boundaries[:-1], high=boundaries[1:])
            else:
                indices = np.linspace(0, n_frames-1, n_samples).astype(int)
            mask = np.full(n_samples, 1, dtype=bool)
        else:
            indices = np.concatenate((np.arange(n_frames), np.full(n_samples - n_frames, n_frames - 1)))
            mask = np.concatenate((np.full(n_frames, 1, dtype=bool), np.zeros(n_samples - n_frames, dtype=bool)))
        return indices, mask
