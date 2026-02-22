import numpy as np
import glob
from collections import Counter
import matplotlib.pyplot as plt
import os
from sklearn.decomposition import IncrementalPCA
import joblib
import random
from argparse import ArgumentParser
import glob
import numpy as np
from sklearn.decomposition import IncrementalPCA
import psutil
import os
from tqdm import tqdm
import os
import glob
import psutil
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import IncrementalPCA
import joblib
from matplotlib.backends.backend_pdf import PdfPages
from PIL import Image
import cv2
import matplotlib.animation as animation
from sklearn.decomposition import PCA
import json 

def run_pca(args):  # NEW: add sampling frequency as a parameter
    mem_info = psutil.virtual_memory()
    total_mem_mb = mem_info.total / (1024 ** 2)
    print(f"Total memory available: {total_mem_mb:.2f} MB")

    #Make directory to store PCA data and ground truth data 
    os.makedirs(args.data_dir + "/pca_features", exist_ok=True)
    os.makedirs(args.data_dir + "/pca_groundTruth", exist_ok=True)

    numpy_files = sorted(glob.glob(args.data_dir + "/observations/*"))
    process = psutil.Process(os.getpid())

    # First pass: compute total number of sampled frames
    total_samples = 0
    for f in numpy_files:
        shape = np.load(f, allow_pickle=True).shape[0]
        sampled_indices = list(range(0, shape, args.frame_freq))
        if sampled_indices[-1] != shape - 1:
            sampled_indices.append(shape - 1)
        total_samples += len(sampled_indices)
    
    print(f"Total number of sampled frames: {total_samples}")

    # Example shape for flattening
    example = np.load(numpy_files[0], allow_pickle=True)
    flat_shape = np.prod(example.shape[1:])
    final_array = np.empty((total_samples, flat_shape), dtype=np.float32)

    offset = 0
    for file_ in numpy_files:
        data = np.load(file_, allow_pickle=True).astype(np.uint8)
        print(f"Processing file: {file_} with shape: {data.shape}")

        # Sample every nth frame and ensure final frame is included
        sampled_indices = list(range(0, data.shape[0], args.frame_freq))
        if sampled_indices[-1] != data.shape[0] - 1:
            sampled_indices.append(data.shape[0] - 1)

        data = data[sampled_indices]
        data = data.astype(np.float32) / 255.0
        data = data.reshape(data.shape[0], -1)

        batch_size = data.shape[0]
        final_array[offset:offset + batch_size] = data
        offset += batch_size

        print(f"Current final_array slice: {offset}/{total_samples}")

    print("Final shape:", final_array.shape)
    mem = process.memory_info().rss / (1024 ** 2)
    print(f"Memory usage all loaded: {mem:.2f} MB")

    # Shuffle the data in place
    np.random.shuffle(final_array)


    #Check if PCA model already exists
    pca_model_path = os.path.join(args.data_dir, "pca_model.joblib")
    if os.path.exists(pca_model_path):
        print(f"PCA model already exists at {pca_model_path}. Loading...")
        pca = joblib.load(pca_model_path)
    else:
        # print("\nFitting PCA...")
        # pca = PCA(n_components=0.99)
        # pca.fit(final_array)
        # print("PCA fitted.")

        print("\nFitting Incremental PCA...")
        pca = IncrementalPCA(n_components=4000, batch_size=20000)
        pca.fit(final_array)
        print("Incremental PCA fitted.")

    print("Explained variance by top components:", pca.explained_variance_ratio_[:10])
    print("Total explained variance:", pca.explained_variance_ratio_.sum())
    print("Number of components:", pca.n_components_)

    # Save PCA model
    model_path = os.path.join(args.data_dir, "pca_model.joblib")
    joblib.dump(pca, model_path)
    print(f"PCA model saved to: {model_path}")

    # Plotting original and reconstructed images
    sample_indices = np.random.choice(final_array.shape[0], 10, replace=False)
    sampled = final_array[sample_indices]
 
    # Reconstruct images using PCA
    pca_features = pca.transform(sampled)
    reconstructed = pca.inverse_transform(pca_features)

    # Plot original and reconstructed images side by side
    fig, axes = plt.subplots(nrows=10, ncols=2, figsize=(6, 30))
    for i in range(10):
        # Original image
        axes[i, 0].imshow(sampled[i].reshape(160, 160, 3))
        axes[i, 0].axis("off")
        axes[i, 0].set_title("Original")

        # Reconstructed image
        axes[i, 1].imshow(reconstructed[i].reshape(160, 160, 3))
        axes[i, 1].axis("off")
        axes[i, 1].set_title("Reconstructed")

    plt.tight_layout()
    plt.savefig(os.path.join(args.data_dir, "reconstruction_comparison.pdf"), format="pdf", dpi=300, bbox_inches='tight')

    del final_array

    #Now use the PCA model to transform the original data and save those PCA features 
    for file_ in numpy_files:
        data = np.load(file_, allow_pickle=True).astype(np.uint8)
        ground_truth_name = file_.replace("observations", "groundTruth").replace(".npy", "")

        #Load the groud truth text file into an array split by new line 
        with open(ground_truth_name, "r") as f:
            ground_truth = f.read().splitlines()
        ground_truth = np.array(ground_truth)

        assert len(data) == len(ground_truth), f"Data length {len(data)} does not match ground truth length {len(ground_truth)}"
        print(f"Processing file: {file_} with shape: {data.shape}")

        # Sample every nth frame and ensure final frame is included
        sampled_indices = list(range(0, data.shape[0], args.frame_freq))
        if sampled_indices[-1] != data.shape[0] - 1:
            sampled_indices.append(data.shape[0] - 1)

        ground_truth = ground_truth[sampled_indices]

        data = data[sampled_indices]
        data = data.astype(np.float32) / 255.0
        data = data.reshape(data.shape[0], -1)

        # Transform the data using PCA
        pca_features = pca.transform(data)
        pca_features = pca_features.astype(np.float32)

        print(f"Current pca_features shape: {pca_features.shape}")

        # Save PCA features
        file_name = file_.split("/")[-1]
        np.save(f"{args.data_dir}/pca_features/{file_name}", pca_features)
        # Save ground truth

        with open(f"{args.data_dir}/pca_groundTruth/{file_name.replace('.npy', '')}", "w") as f:
            f.write("\n".join(ground_truth))

        print(f"PCA features saved to: {args.data_dir}/pca_features/{file_name}")
        print(f"Ground truth saved to: {args.data_dir}/pca_groundTruth/{file_name.replace('.npy', '')}")


    #Save arguments and PCA details to stats.json file 
    stats = {
        "args": vars(args),
        "pca_components": int(pca.n_components_),
        "explained_variance_ratio": pca.explained_variance_ratio_.tolist(),
        "total_explained_variance": float(pca.explained_variance_ratio_.sum())
    }
    stats_file = os.path.join(args.data_dir, "pca_stats.json")
    with open(stats_file, "w") as f:
        json.dump(stats, f, indent=4)
    print(f"Stats saved to: {stats_file}")



if __name__ == "__main__":
    parser = ArgumentParser("Run pretrained models on MineRL environment")

    parser.add_argument("--data-dir", type=str, default="Data/wooden_pickaxe")
    parser.add_argument("--frame-freq", type=int, default=6)
    # parser.add_argument("--components", type=int, default=1500)

    args = parser.parse_args()
    run_pca(args)