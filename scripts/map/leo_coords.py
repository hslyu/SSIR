"""
This script converts Starlink TLE data from Celestrak
(https://celestrak.org/NORAD/elements/table.php?GROUP=starlink&FORMAT=tle)
into geodetic coordinates (latitude, longitude, altitude) at a common evaluation time.

The input TLE snapshot should be taken at a specific time, e.g.:
    "Current as of 2025 Apr 21 04:05:31 UTC (Day 111)"

Each TLE entry includes an epoch indicating the orbital elements’ reference time.
The script propagates each satellite's orbit from its TLE epoch to a single evaluation time
(current UTC time by default), and computes its position in the Earth-fixed ITRS frame.

The resulting (lat, lon, alt) positions are filtered by optional geographic bounds
and returned as a list of (OBJECT_NAME, lat, lon, alt) tuples in degrees and kilometers.
"""

import csv
from multiprocessing import Pool, cpu_count
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from astropy import units as u
from astropy.coordinates import GCRS, ITRS, CartesianRepresentation, EarthLocation
from astropy.time import Time
from astropy.utils import iers
from poliastro.bodies import Earth
from poliastro.twobody import angles
from poliastro.twobody.orbit import Orbit
from tqdm import tqdm

# Enable automatic download of IERS data for accurate transforms
iers.conf.auto_download = True


def _process_row(
    args: Tuple[
        pd.Series,
        float,
        Optional[Tuple[float, float]],
        Optional[Tuple[float, float]],
        Time,
    ],
) -> Optional[Tuple[str, float, float, float]]:
    """
    Worker function to process one row of the DataFrame.
    Propagates orbit from its epoch to eval_time, then converts to geodetic coords.
    """
    row, mu_earth, lat_range, lon_range, eval_time = args

    try:
        # 1. Extract classical orbital elements
        n_rev_day = row["MEAN_MOTION"]
        n_rad_s = n_rev_day * 2.0 * np.pi / 86400.0
        a_km = (mu_earth / n_rad_s**2) ** (1.0 / 3.0)

        ecc = row["ECCENTRICITY"]
        inc = row["INCLINATION"] * u.deg
        raan = row["RA_OF_ASC_NODE"] * u.deg
        argp = row["ARG_OF_PERICENTER"] * u.deg
        M = row["MEAN_ANOMALY"] * u.deg

        # 2. Compute true anomaly from mean anomaly
        M_rad = M.to(u.rad).value
        E_rad = angles.M_to_E(M_rad, ecc)
        nu_rad = angles.E_to_nu(E_rad, ecc)

        # 3. Build initial orbit at its epoch
        epoch = Time(row["EPOCH"])
        orb = Orbit.from_classical(
            attractor=Earth,
            a=a_km * u.km,
            ecc=ecc * u.one,
            inc=inc,
            raan=raan,
            argp=argp,
            nu=nu_rad * u.rad,
            epoch=epoch,
        )

        # 4. Propagate to the common evaluation time
        dt = eval_time - epoch
        orb_eval = orb.propagate(dt)

        # 5. Get position vector at eval_time
        r_eval, _ = orb_eval.rv()
        r_vec = CartesianRepresentation(r_eval)

        # 6. Transform from GCRS to ITRS at eval_time
        gcrs = GCRS(r_vec, obstime=eval_time)
        itrs = gcrs.transform_to(ITRS(obstime=eval_time))

        # 7. Convert to geodetic lat, lon, alt
        loc = EarthLocation.from_geocentric(itrs.x, itrs.y, itrs.z)
        lat = loc.lat.deg
        lon = ((loc.lon.deg + 180.0) % 360.0) - 180.0
        alt = loc.height.to(u.km).value

        # 8. Apply optional filtering
        if lat_range and not (lat_range[0] <= lat <= lat_range[1]):
            return None
        if lon_range and not (lon_range[0] <= lon <= lon_range[1]):
            return None

        return (row["OBJECT_NAME"], lat, lon, alt)

    except Exception:
        return None


def extract_starlink_positions(
    csv_path: str,
    lat_range: Optional[Tuple[float, float]] = None,
    lon_range: Optional[Tuple[float, float]] = None,
) -> List[Tuple[str, float, float, float]]:
    """
    Parse Starlink GP CSV and convert each satellite's orbit to (lat, lon, alt)
    at a common evaluation time, with optional filtering.
    """
    # Load TLE-derived CSV
    df = pd.read_csv(csv_path)

    # Standard gravitational parameter for Earth [km^3 / s^2]
    mu_earth = Earth.k.to(u.km**3 / u.s**2).value

    # Define a single common evaluation time (UTC now)
    eval_time = Time.now()

    # Prepare arguments for multiprocessing
    args_list = [
        (row, mu_earth, lat_range, lon_range, eval_time) for _, row in df.iterrows()
    ]

    # Process in parallel with tqdm progress bar
    with Pool(processes=cpu_count()) as pool:
        results = list(
            tqdm(
                pool.imap_unordered(_process_row, args_list),
                total=len(args_list),
                desc="Processing Starlink satellites",
            )
        )

    # Filter out None results
    return [r for r in results if r is not None]


def save_positions_to_csv(
    positions: list[tuple[str, float, float, float]],
    output_path: str,
):
    """
    Save satellite positions to a CSV file with columns: name, lat, lon, alt.

    Parameters
    ----------
    positions : list of tuples
        Each tuple should be (name, lat, lon, alt).
    output_path : str
        Path to output CSV file.
    """
    with open(output_path, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["name", "lat", "lon", "alt"])
        for entry in positions:
            writer.writerow(entry)


if __name__ == "__main__":
    csv_file = "starlink_gp.csv"
    output_csv = "starlink_positions.csv"

    # Example: lat 35°–55°N, lon -120°–-75°E
    leo_coords = extract_starlink_positions(
        csv_file,
    )

    save_positions_to_csv(leo_coords, output_csv)
    print(f"Saved {len(leo_coords)} satellite positions to {output_csv}.")
