"""
Fit PCA on every frame in <data_dir>/<obs_dir> and write per-episode features to
<data_dir>/pca_features/ (the layout HiSD's RLDataset reads).  Unlike
train_pca_model.py + get_pca_features.py this accepts any image size / dtype and
uses randomized SVD, so it runs on a laptop-sized machine.

    python build_pca_features.py --data_dir Traces/wsws_random --components 256
"""
import argparse
import os
from pathlib import Path

import joblib
import numpy as np
from sklearn.decomposition import PCA

parser = argparse.ArgumentParser()
parser.add_argument("--data_dir", type=str, required=True)
parser.add_argument("--components", type=int, default=650)
parser.add_argument("--out_name", type=str, default="pca_features")
parser.add_argument("--obs_dir", type=str, default="top_down_obs", help="top_down_obs or pixel_obs (local view)")
parser.add_argument("--seed", type=int, default=0)
args = parser.parse_args()

data_dir = Path(args.data_dir)
files = sorted((data_dir / args.obs_dir).glob("*.npy"))
episodes = [np.load(f) for f in files]
scale = 255. if episodes[0].dtype == np.uint8 else 1.
X = np.concatenate([e.reshape(len(e), -1) for e in episodes]).astype(np.float32) / scale
n_comp = min(args.components, X.shape[0] - 1)
print(f"{len(files)} episodes, {X.shape[0]} frames, dim {X.shape[1]} -> {n_comp} components")

pca = PCA(n_components=n_comp, svd_solver="randomized", random_state=args.seed)
pca.fit(X)
print(f"explained variance: {pca.explained_variance_ratio_.sum():.3f}")

out_dir = data_dir / args.out_name
out_dir.mkdir(parents=True, exist_ok=True)
for f, e in zip(files, episodes):
    feats = pca.transform(e.reshape(len(e), -1).astype(np.float32) / scale)
    np.save(out_dir / f.name, feats.astype(np.float32))
os.makedirs(data_dir / "pca_models", exist_ok=True)
joblib.dump({"pca": pca, "scale": scale}, data_dir / "pca_models" / f"{args.out_name}_{n_comp}.joblib")
