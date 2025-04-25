"""
run‑ratio‑connected.py
──────────────────────────────────────────────────────────────────────────────
• interleave tasks: experiment‑id가 outer, power‑level이 inner
• 각 power level마다 별도 tqdm 바를 두어 현재 진행도(n/50)와
  실시간 평균 UE ratio(connected / total)를 함께 보여줌
"""

import json
import multiprocessing as mp
import os
from collections import defaultdict

import numpy as np
from tqdm import tqdm

from ssir import basestations as bs


# ─────────────────────────────────────────────────────────────────────────────
# IO helpers
# ─────────────────────────────────────────────────────────────────────────────
def load_exp_graph(exp_id: int, env_dir: str):
    exp_dir = os.path.join(env_dir, f"exp_{exp_id:03d}")
    g = bs.IABRelayGraph()
    g.load_graph(os.path.join(exp_dir, "graph.pkl"), pkl=True)
    return g


# ─────────────────────────────────────────────────────────────────────────────
# core experiment
# ─────────────────────────────────────────────────────────────────────────────
def run_single_experiment(power_level: float, exp_id: int, env_dir: str):
    """Return (power_level, ratio_connected) for one (p, id) pair."""
    graph = load_exp_graph(exp_id, env_dir)

    # override TX‑power ratio
    for node in graph.basestations:
        node.basestation_type.config.minimum_transit_power_ratio = power_level

    graph.reset()
    graph.connect_reachable_nodes()

    N = len(graph.users)
    ratio = sum(u.has_parent() for u in graph.users) / N
    return power_level, ratio


def _worker(args):
    return run_single_experiment(*args)


# ─────────────────────────────────────────────────────────────────────────────
# main
# ─────────────────────────────────────────────────────────────────────────────
def main():
    power_levels = np.arange(0.95, 0.34, -0.05)  # 0.95→0.35
    start_id = 0
    n_per_level = 2000

    base_dir = "./results_mmf_vs_power"
    env_dir = os.path.join(base_dir, "env")
    os.makedirs(base_dir, exist_ok=True)

    # ― task list: eid outer, power inner  (※ 요청사항)
    tasks = [
        (p, eid, env_dir)
        for eid in range(start_id, start_id + n_per_level)
        for p in power_levels
    ]

    # ― tqdm bars
    total_bar = tqdm(total=len(tasks), desc="TOTAL", position=0, leave=True)
    level_bars = {
        p: tqdm(
            total=n_per_level,
            desc=f"P={p:.2f}",
            position=i + 1,
            leave=True,
            bar_format="{l_bar}{bar}|{n_fmt}/{total_fmt} [{postfix}]",
        )
        for i, p in enumerate(power_levels)
    }

    # ― result accumulator
    ratios = defaultdict(list)

    # ― run in parallel
    with mp.Pool(processes=os.cpu_count() - 1, maxtasksperchild=1) as pool:
        for p, r in pool.imap_unordered(_worker, tasks):
            ratios[p].append(r)

            # update bars
            total_bar.update()
            lb = level_bars[p]
            lb.update()
            lb.set_postfix_str(f"avg={np.mean(ratios[p]):.6f}")

    total_bar.close()
    for b in level_bars.values():
        b.close()

    # ― save aggregated means
    aggregated = {f"{p:.2f}": float(np.mean(ratios[p])) for p in power_levels}
    with open(os.path.join(base_dir, "aggregated_ratio.json"), "w") as f:
        json.dump(aggregated, f, indent=4)

    print("Aggregated JSON saved:", os.path.join(base_dir, "aggregated_ratio.json"))


if __name__ == "__main__":
    main()
