import json
import os
from multiprocessing import Pool, cpu_count

import numpy as np
from tqdm import tqdm

# ssir-related imports
import ssir.basestations as bs


#############################################
# Helper functions for HPPP on Earth's surface
#############################################
def deg2rad(deg):
    return deg * np.pi / 180


def area_of_bbox(lat_min, lat_max, lon_min, lon_max):
    R = 6371  # Earth radius in km
    return (
        (deg2rad(lon_max) - deg2rad(lon_min))
        * R
        * R
        * (np.sin(deg2rad(lat_max)) - np.sin(deg2rad(lat_min)))
    )


def generate_hppp_on_bbox(lat_min, lat_max, lon_min, lon_max, density):
    area_km2 = area_of_bbox(lat_min, lat_max, lon_min, lon_max)
    expected_num_points = np.random.poisson(density * area_km2)

    sin_lat_min = np.sin(deg2rad(lat_min))
    sin_lat_max = np.sin(deg2rad(lat_max))
    u = np.random.uniform(sin_lat_min, sin_lat_max, expected_num_points)
    lat_samples = np.arcsin(u) * 180 / np.pi  # convert back to degrees

    lon_samples = np.random.uniform(lon_min, lon_max, expected_num_points)

    return lat_samples, lon_samples


def process_experiment(args):
    """Run one experiment 10 times and return average secrecy rate per scheme."""

    def get_eves(density, altitude):
        lat, lon = generate_hppp_on_bbox(lat_min, lat_max, lon_min, lon_max, density)
        return np.column_stack([lat, lon, np.full(len(lat), altitude)])

    exp_id, threshold_str, threshold, base_dir, num_repeat = args

    env_dir = os.path.join(base_dir, "env", f"exp_{exp_id:03d}")
    config_path = os.path.join(env_dir, "config.json")
    if not os.path.isfile(config_path):
        return {}

    with open(config_path, "r") as f:
        config = json.load(f)

    lat_min, lat_max = config["latitude_range"]
    lon_min, lon_max = config["longitude_range"]

    exp_dir = os.path.join(base_dir, f"spsc_{threshold_str}", f"exp_{exp_id:03d}")
    schemes = [
        "astar_distance",
        "astar_hop",
        "astar_spectral_efficiency",
        "genetic",
        "montecarlo",
        "greedy",
        "bruteforce",
        "bruteforce_mmf",
    ]

    # Initialize scheme -> list of results
    scheme_results = {scheme: [] for scheme in schemes}

    for _ in range(num_repeat):
        eves_maritime = get_eves(
            bs.BaseStationType.MARITIME.config.eavesdropper_density,
            bs.environmental_variables.maritime_basestations_altitude,
        )
        eves_ground = get_eves(
            bs.BaseStationType.GROUND.config.eavesdropper_density,
            bs.environmental_variables.ground_basestations_altitude,
        )
        eves_haps = get_eves(
            bs.BaseStationType.HAPS.config.eavesdropper_density,
            bs.environmental_variables.haps_basestations_altitude,
        )
        eves_leo = get_eves(
            bs.BaseStationType.LEO.config.eavesdropper_density,
            bs.environmental_variables.leo_basestations_altitude,
        )

        for scheme in schemes:
            solution_file = os.path.join(exp_dir, f"solution_{scheme}.pkl")
            if not os.path.isfile(solution_file):
                continue

            g = bs.IABRelayGraph()
            g.load_graph(solution_file, pkl=True)
            if scheme == "bruteforce":
                if threshold > 1 - 4e-5:
                    h = bs.IABRelayGraph()
                    h.load_graph(os.path.join(exp_dir, "solution_montecarlo.pkl"))
                    if g.compute_network_throughput() < h.compute_network_throughput():
                        g = h
                scheme_results["bruteforce_mmf"].append(g.compute_network_throughput())

            mm_secrecy_rate = g.compute_network_secrecy_rate(
                eves_maritime, eves_ground, eves_haps, eves_leo
            )
            scheme_results[scheme].append(mm_secrecy_rate)

    # Return average over num_repeat
    return {
        scheme: float(np.mean(vals)) if vals else None
        for scheme, vals in scheme_results.items()
    }


def evaluate_max_min_secrecy_rate():
    base_dir = "/fast/hslyu/mmf_result_1_3k"

    raw_logspace = np.concatenate(
        (np.logspace(-5, -4, 7, base=10)[:-1], np.logspace(-4, 0, 13, base=10))
    )
    raw_logspace = np.logspace(-1, 0, 4, base=10)[1:]
    thresholds_to_test = 1 - raw_logspace
    start_exp = 0
    num_experiments = 3000
    num_repeat_per_exp = 10

    output_file = os.path.join(base_dir, "avg_secrecy_rate_results.json")

    # Load existing results if available
    if os.path.isfile(output_file):
        with open(output_file, "r") as f:
            avg_secrecy_results = json.load(f)
    else:
        avg_secrecy_results = {}

    for threshold in thresholds_to_test:
        bs.environmental_variables.SPSC_probability = threshold

        threshold_str = f"{threshold:.6f}"
        avg_secrecy_results[threshold_str] = {}
        schemes = [
            "astar_distance",
            "astar_hop",
            "astar_spectral_efficiency",
            "genetic",
            "montecarlo",
            "greedy",
            "bruteforce",
            "bruteforce_mmf",
        ]
        scheme_accumulator = {scheme: [] for scheme in schemes}

        task_args = [
            (exp_id, threshold_str, threshold, base_dir, num_repeat_per_exp)
            for exp_id in range(start_exp, start_exp + num_experiments)
        ]

        with Pool(processes=cpu_count()) as pool:
            for result in tqdm(
                pool.imap_unordered(process_experiment, task_args),
                total=num_experiments,
                desc=f"Threshold={threshold_str}",
            ):
                for scheme, val in result.items():
                    if val is not None:
                        scheme_accumulator[scheme].append(val)

        avg_secrecy_results[threshold_str] = {
            scheme: float(np.mean(vals)) if vals else None
            for scheme, vals in scheme_accumulator.items()
        }

        print(
            f"[Threshold={threshold_str}] Avg secrecy rates: {avg_secrecy_results[threshold_str]}"
        )

    with open(output_file, "w") as f:
        json.dump(avg_secrecy_results, f, indent=4)


if __name__ == "__main__":
    evaluate_max_min_secrecy_rate()
