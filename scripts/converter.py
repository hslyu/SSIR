import os
import pickle
from multiprocessing import Pool

import torch
from tqdm import tqdm

ROOT_DIR = "/home/hslyu/research/SSIR/scripts/results/"
SAVE_DIR = "/home/hslyu/research/SSIR/scripts/results_pt/"


def process_folder(idx):
    dir_path = os.path.join(ROOT_DIR, str(idx))
    master_path = os.path.join(dir_path, "master_graph.pkl")
    genetic_path = os.path.join(dir_path, "graph_genetic.pkl")

    try:
        with open(master_path, "rb") as f:
            graph = pickle.load(f)
        data_master = graph.to_torch_geometric()
    except Exception as e:
        print(f"[{idx}] master_graph.pkl error: {e}")
        return None

    if os.path.exists(genetic_path):
        try:
            with open(genetic_path, "rb") as f:
                graph_label = pickle.load(f)
            data_label = graph_label.to_torch_geometric()
        except Exception as e:
            print(f"[{idx}] graph_genetic.pkl error: {e}")
            data_label = None
    else:
        data_label = None

    # if neither data_master nore data_label are None
    if data_master is not None and data_label is not None:
        save_path = os.path.join(SAVE_DIR, f"{idx}.pt")
        torch.save((data_master, data_label), save_path)
    else:
        print(f"[{idx}] data_master or data_label is None")
    return idx


if __name__ == "__main__":
    os.makedirs(SAVE_DIR, exist_ok=True)
    total_folders = 50000

    # 프로세스 개수는 시스템에 맞게 조정
    num_processes = 32
    with Pool(processes=num_processes) as pool:
        for _ in tqdm(
            pool.imap_unordered(process_folder, range(total_folders)),
            total=total_folders,
        ):
            pass
