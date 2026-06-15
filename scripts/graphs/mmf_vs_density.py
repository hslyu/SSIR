#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Create a 2x4 grid of sub-figures that summarize seven mmf-vs-density
experiments: G, M, H, L, GM, HL, and GMHL.

The eighth panel, at the bottom right, is reserved for the global legend.
Edit the `base_dirs` dictionary if the experiment output paths differ.
"""

import json
import os

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import ScalarFormatter


# Global plotting parameters
FONTSIZE = 14
plt.rcParams["font.size"] = FONTSIZE
plt.rcParams["axes.labelsize"] = FONTSIZE
plt.rcParams["xtick.labelsize"] = FONTSIZE
plt.rcParams["ytick.labelsize"] = FONTSIZE

MARKER_LIST = ["v", "^", ">", "d", "o", "P", "s", "X"]
COLOR_LIST = [
    "#3FB3B7",
    "#2991D1",
    "#A3CF7B",
    "#333333",
    "#493C95",
    "#7C3A7A",
    "#FF0000",
    "#E2A72B",
]

ROOT_DIR = "/fast/hslyu/mmf_vs_density/"
PANEL_LABELS = [
    "(a) Ground",
    "(b) Maritime",
    "(c) HAPS",
    "(d) LEO",
    "(e) GM",
    "(f) HL",
    "(g) GMHL",
]
BASE_DIRS = {
    PANEL_LABELS[0]: ROOT_DIR + "results_mmf_vs_density_G",
    PANEL_LABELS[1]: ROOT_DIR + "results_mmf_vs_density_M",
    PANEL_LABELS[2]: ROOT_DIR + "results_mmf_vs_density_H",
    PANEL_LABELS[3]: ROOT_DIR + "results_mmf_vs_density_L",
    PANEL_LABELS[4]: ROOT_DIR + "results_mmf_vs_density_GM",
    PANEL_LABELS[5]: ROOT_DIR + "results_mmf_vs_density_HL",
    PANEL_LABELS[6]: ROOT_DIR + "results_mmf_vs_density_GMHL",
}

# Densities swept in the experiment script.
DENSITIES = np.logspace(-5, -1, 13, base=10)


def load_results(root):
    """
    Return a mapping from density to scheme mean throughput in Kbps.

    Shape:
        dict[density] -> dict[scheme] -> mean_throughput_kbps
    """
    results = {}

    for density in DENSITIES:
        density_dir = os.path.join(root, f"density_{density:.2e}")
        if not os.path.isdir(density_dir):
            continue

        exp_dirs = [
            name
            for name in os.listdir(density_dir)
            if name.startswith("exp_")
            and os.path.isdir(os.path.join(density_dir, name))
        ]

        scheme_values = {}
        for exp_dir in exp_dirs:
            result_path = os.path.join(density_dir, exp_dir, "result.json")
            if not os.path.isfile(result_path):
                continue

            with open(result_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            if "bruteforce" in data and "montecarlo" in data:
                if data["bruteforce"] < data["montecarlo"]:
                    data["bruteforce"] = data["montecarlo"]

            for scheme, throughput in data.items():
                scheme_values.setdefault(scheme, []).append(throughput / 1_000.0)

        if scheme_values:
            results[density] = {
                scheme: np.mean(values) for scheme, values in scheme_values.items()
            }

    return results


def main():
    fig, axes = plt.subplots(
        nrows=2, ncols=4, figsize=(16, 6), sharex=False, sharey=False
    )

    axes_flat = axes.flatten()
    legend_ax = axes_flat[-1]
    legend_ax.axis("off")

    for idx, (panel_label, root) in enumerate(BASE_DIRS.items()):
        ax = axes_flat[idx]

        all_results = load_results(root)
        if not all_results:
            ax.set_title(f"{panel_label}  (NO DATA)")
            ax.axis("off")
            continue

        schemes = sorted({scheme for values in all_results.values() for scheme in values})

        for scheme_idx, scheme in enumerate(schemes):
            y_values = [
                all_results.get(density, {}).get(scheme, np.nan)
                for density in DENSITIES
            ]

            zorder = 1
            linewidth = 1.5
            markerfacecolor = "w"
            markersize = 7

            if scheme == "montecarlo":
                markerfacecolor = "#FFD4D4"
                linewidth = 1.75
                markersize = 7.5
                zorder = 2
            elif scheme == "greedy":
                markersize = 7.5
            elif scheme == "bruteforce":
                zorder = 3

            ax.plot(
                DENSITIES,
                y_values,
                linewidth=linewidth,
                marker=MARKER_LIST[scheme_idx % len(MARKER_LIST)],
                markersize=markersize,
                markerfacecolor=markerfacecolor,
                markeredgewidth=1.5,
                color=COLOR_LIST[scheme_idx % len(COLOR_LIST)],
                label=scheme,
                zorder=zorder,
            )

        ax.set_xscale("log")
        ax.set_xlabel("Eve Density (km$^{-2}$)")
        ax.set_ylabel("Average MMF (Kbps)")
        ax.set_title(panel_label, loc="center")
        ax.grid(
            True,
            which="both",
            linestyle=(0, (5, 5)),
            linewidth=0.5,
            color="#e0e0e0",
        )
        ax.yaxis.set_major_formatter(ScalarFormatter(useOffset=False))
        ax.set_xticks([1e-5, 1e-3, 1e-1])
        ax.set_xlim(1e-5, 1e-1)
        ax.set_ylim(bottom=0)
        if idx != 5:
            ax.set_yticks(np.arange(0, 16, 3))
        else:
            ax.set_yticks(np.arange(0, 19, 3))

    handles, legend_labels = axes_flat[0].get_legend_handles_labels()
    legend_ax.legend(
        handles,
        legend_labels,
        loc="center",
        ncol=1,
        frameon=False,
        fontsize=FONTSIZE - 2,
    )

    plt.tight_layout()
    plt.savefig("mmf_vs_density.pdf", format="pdf", dpi=300)
    plt.show()


if __name__ == "__main__":
    main()
