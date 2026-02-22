import torch
# from Clip4MC.model.clip4mc import CLIP4MC
import torchvision.transforms as T
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import gzip
import pickle
import os 
from collections import defaultdict
from tqdm import tqdm
import sys 


base = os.path.dirname(__file__)        
project_root = base

clip4mc_dir = os.path.join(project_root, "Clip4MC")
os.chdir(clip4mc_dir)
sys.path.insert(0, clip4mc_dir)

from model.clip4mc import CLIP4MC



def get_truth(ep, name_map={}):
    # --- 0) Prepare ---
    prev_counts = defaultdict(lambda: defaultdict(int))
    events = []
    walking_counter = 0
    potential_walk_frames = []
    move_keys = ['forward','back','left','right', 'jump', 'sneak', 'sprint']

    # --- 1) Collect raw events & walking candidates ---
    for i, x in enumerate(ep):
        # inventory diffs
        mine_changes = {}
        craft_changes = {}
        use_changes = {}
        for k, v in x['mine_block'].items():
            cur = v.item()
            if cur != prev_counts['mine_block'][k]:
                mine_changes[k] = cur
        for k, v in x['craft_item'].items():
            cur = v.item()
            if cur != prev_counts['craft_item'][k]:
                craft_changes[k] = cur
        for k, v in x['use_item'].items():
            cur = v.item()
            if cur != prev_counts['use_item'][k]:
                use_changes[k] = cur

        # walking detection
        no_interact = not (mine_changes or craft_changes or use_changes)
        gui_closed = not x.get('isGuiOpen', False)
        moving = any(x['action'][k][0] > 0 for k in move_keys)
        if no_interact and gui_closed and moving:
            walking_counter += 1
            potential_walk_frames.append(i)
        else:
            if walking_counter >= 20:
                for j in potential_walk_frames:
                    events.append((j, 'walked', ''))
            walking_counter = 0
            potential_walk_frames = []

        # pick one event per frame
        if mine_changes:
            action, raw = 'mined', next(iter(mine_changes))
        elif craft_changes:
            action, raw = 'crafted', next(iter(craft_changes))
        elif use_changes:
            action, raw = 'used', next(iter(use_changes))
        else:
            continue

        mapped = name_map.get(raw, raw)
        events.append((i, action, mapped))

        # update counts
        for dct, changes in [('mine_block', mine_changes),
                             ('craft_item', craft_changes),
                             ('use_item',  use_changes)]:
            for k, v in changes.items():
                prev_counts[dct][k] = v

    # trailing walk
    if walking_counter >= 20:
        for j in potential_walk_frames:
            events.append((j, 'walked', ''))

    # --- 2) Build initial contiguous segments ---
    events.sort(key=lambda e: e[0])
    segments = []
    prev_step = None
    for step, action, item in events:
        if not segments:
            segments.append([0, step, action, item])
        else:
            last = segments[-1]
            if last[2] == action and last[3] == item:
                last[1] = step
            else:
                segments.append([prev_step + 1, step, action, item])
        prev_step = step

    # --- 3) Refine craft segments to respect multi-craft GUI sessions ---
    # 3a) collect all [gui_start, gui_end] intervals
    gui_windows = []
    in_gui = False
    for i, frame in enumerate(ep):
        if frame.get('isGuiOpen', False):
            if not in_gui:
                in_gui = True
                win_start = i
        else:
            if in_gui:
                gui_windows.append((win_start, i - 1))
                in_gui = False
    if in_gui:
        gui_windows.append((win_start, len(ep) - 1))

    # 3b) for each GUI session, snap the first craft->start and last craft->end
    for win_start, win_end in gui_windows:
        craft_idxs = [
            idx for idx, (s, e, action, item) in enumerate(segments)
            if action == 'crafted' and not (e < win_start or s > win_end)
        ]
        if not craft_idxs:
            continue

        first = craft_idxs[0]
        last  = craft_idxs[-1]
        segments[first][0] = max(segments[first][0], win_start)
        segments[last][1]  = win_end

    # --- 4) Pad each segment’s end to one before the next segment’s start ---
    for i in range(len(segments) - 1):
        cur_end    = segments[i][1]
        next_start = segments[i+1][0]
        desired_end = next_start - 1

        if desired_end >= cur_end:
            # there's a gap or exact, so extend to right up to next_start-1
            segments[i][1] = desired_end
        else:
            # overlap: bump the next segment’s start forward
            segments[i+1][0] = cur_end + 1

    # --- 5) Extend the very last segment to cover the whole episode ---
    segments[-1][1] = len(ep) - 1

    # output
    for start, end, action, item in segments:
        print(f"[{start} - {end}] -> {action} {item}")

    action_list = []
    for start, end, action, item in segments:
        action_list.extend([f"{action} {item}"] * (end - start + 1))

    return action_list


if __name__ == '__main__':
    task = "iron_ingot"
    truth_dir = f'Data/{task}/groundTruth'
    features_dir = f'Data/{task}/features'
    raw_obs_dir = f'Data/{task}/raw_obs'
    os.makedirs(truth_dir, exist_ok=True)
    os.makedirs(features_dir, exist_ok=True)

    pretrained_model_path = "ViT-B-16.pt"  

    pretrained_clip = torch.jit.load(pretrained_model_path)
    model = CLIP4MC(
        frame_num=100,
        use_action=False,        
        use_brief_text=False,    
        pretrained_clip=pretrained_clip
    )
    model.eval()  

    device = torch.device("cpu")
    model = model.to(device)

    transform = T.Compose([
        T.Resize((160, 256)),
        T.ToTensor(),
    ])

    os.chdir(project_root)

    # Get all raw_obs files
    all_raw_files = sorted([
        fname for fname in os.listdir(raw_obs_dir)
        if fname.startswith("minecraft_") and fname.endswith(".pkl.gz")
    ])
    # Get existing features and groundTruth files (without extension)
    existing_features = set(
        fname.replace(".npy", "") for fname in os.listdir(features_dir) if fname.endswith(".npy")
    )
    existing_truth = set(
        fname for fname in os.listdir(truth_dir)
    )

    # If either features or groundTruth is empty, process all
    process_all = (not existing_features) or (not existing_truth)
    files_to_process = []
    for fname in all_raw_files:
        base = fname.replace(".pkl.gz", "")
        if process_all or (base not in existing_features or base not in existing_truth):
            files_to_process.append(fname)

    if not files_to_process:
        print("No new files to process.")
    else:
        for fname in tqdm(files_to_process):
            ep_idx = fname.split("_")[-1].split(".")[0]
            print(f"Processing episode {ep_idx}...")


            try:
                with gzip.open(os.path.join(raw_obs_dir, fname), 'rb') as f_gz:
                    ep_data = pickle.load(f_gz)
            except Exception as e:
                print(f"Error loading {fname}: {e}")
                continue

            embeddings = []
            with torch.no_grad():
                for obs in ep_data:
                    frame = obs['pov']
                    image = Image.fromarray(frame)
                    img_tensor = transform(image).unsqueeze(0).to(device)
                    embedding = model.get_image_embedding(img_tensor).cpu().squeeze(0)
                    embeddings.append(embedding)

            embeddings_tensor = torch.stack(embeddings).to(torch.float16)  # Or torch.float32
            embeddings_np = embeddings_tensor.numpy()
            print("Final embeddings shape:", embeddings_tensor.shape)
            np.save(os.path.join(features_dir, f"minecraft_{ep_idx}.npy"), embeddings_np)

            ground_truth = get_truth(ep_data, name_map={})
            assert len(ground_truth) == embeddings_np.shape[0], \
                f"Ground truth length {len(ground_truth)} does not match embeddings length"
            path = os.path.join(truth_dir, f"minecraft_{ep_idx}")
            with open(path, 'w') as f:
                f.write('\n'.join(ground_truth))

            del embeddings_tensor, embeddings_np, embeddings
            del ep_data, image, img_tensor, embedding  # if still referenced
            # gc.collect()




