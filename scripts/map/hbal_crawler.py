import datetime as dt
import re
from concurrent.futures import ThreadPoolExecutor
from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from dateutil import parser as dtparse
from requests.adapters import HTTPAdapter
from tqdm import tqdm
from urllib3.util.retry import Retry


# ────────────────────────────────────────────────────────────────
# helpers
# ────────────────────────────────────────────────────────────────
def make_session() -> requests.Session:
    s = requests.Session()
    s.headers.update(
        {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/123.0.0.0 Safari/537.36"
        }
    )
    retry = Retry(total=3, backoff_factor=0.3, status_forcelist=[500, 502, 503, 504])
    s.mount("https://", HTTPAdapter(max_retries=retry))
    return s


# ──────────────────────────────────────────────────────────────────────────
# parse a raw string into datetime (date or datetime with time)
# ──────────────────────────────────────────────────────────────────────────
def parse_datetime_str(raw: str) -> dt.datetime | None:
    """
    Parse strings like:
      'Dec.4, 2019'
      '9/10/2020'
      'Aug. 27, 2020 - 16:43 UTC'
      'Set.10, 2020'
      'Winnemucca, Nevada, July 15, 2020'
    Return a tz-naive datetime (time=00:00 if missing).
    """
    if not raw:
        return None
    txt = raw.strip()
    # normalize month abbreviations
    txt = re.sub(r"\bSet\.", "Sep.", txt, flags=re.I)
    # unify UTC token
    txt = re.sub(r"\s+[uU][tT][cC]\b", " UTC", txt)
    # remove all dots to avoid e.g. 'Dec.4'
    txt = txt.replace(".", "")
    try:
        dtobj = dtparse.parse(txt)
        # drop tzinfo if any
        return dtobj.replace(tzinfo=None)
    except Exception:
        return None


# ──────────────────────────────────────────────────────────────────────────
# single‐track crawler
# ──────────────────────────────────────────────────────────────────────────
def crawl_track(callsign: str, sample_hours: int = 3) -> pd.DataFrame:
    sess = requests.Session()
    sess.headers.update({"User-Agent": "Mozilla/5.0"})
    url = f"https://stratocat.com.ar/globos/mapas/map_{callsign}.html"
    html = sess.get(url, timeout=15).text

    # 1) extract raw duration in seconds, if present
    m_days = re.search(r"in\s+(\d+)\s+days", html, flags=re.I)
    if m_days:
        duration_sec = int(m_days.group(1)) * 86400
        duration_known = True
    else:
        m_hm = re.search(
            r"in\s+(\d+)\s+hours(?:\s+and\s+(\d+)\s+minutes)?", html, flags=re.I
        )
        if m_hm:
            h, m = int(m_hm.group(1)), int(m_hm.group(2) or 0)
            duration_sec = h * 3600 + m * 60
            duration_known = True
        else:
            duration_sec = None
            duration_known = False

    # 2) extract Launch and Ditch marker raw strings
    names = re.findall(r"name\s*:\s*'([^']*)'", html)
    launch_raw = next((n for n in names if n.startswith("Launch Site")), None)
    ditch_raw = next((n for n in names if n.startswith("Ditching")), None)

    # parse raw to datetime if possible
    launch_dt = None
    if launch_raw:
        parts = launch_raw.split(" - ")
        if len(parts) >= 4:
            raw = parts[-2] + " " + parts[-1]
        elif len(parts) == 3:
            raw = parts[-1]
        else:
            # location+date mashup
            sub = parts[-1].split(",")
            if len(sub) >= 3:
                raw = sub[-2] + ", " + sub[-1]
            else:
                raw = None
        launch_dt = parse_datetime_str(raw) if raw else None

    ditch_dt = None
    if ditch_raw:
        parts = ditch_raw.split(" - ")
        if len(parts) >= 4:
            raw = parts[-2] + " " + parts[-1]
        elif len(parts) == 3:
            raw = parts[-1]
        else:
            sub = parts[-1].split(",")
            if len(sub) >= 3:
                raw = sub[-2] + ", " + sub[-1]
            else:
                raw = None
        ditch_dt = parse_datetime_str(raw) if raw else None

    # 3) infer missing one of (launch_dt, ditch_dt, duration_sec)
    #    need at least two known
    known = [launch_dt is not None, ditch_dt is not None, duration_known]
    if sum(known) < 2:
        raise ValueError("Need at least two of launch date, ditch date, duration.")
    if launch_dt and ditch_dt and not duration_known:
        duration_sec = int((ditch_dt - launch_dt).total_seconds())
    elif launch_dt and duration_known and not ditch_dt:
        ditch_dt = launch_dt + timedelta(seconds=duration_sec)
    elif ditch_dt and duration_known and not launch_dt:
        launch_dt = ditch_dt - timedelta(seconds=duration_sec)

    # 4) extract coordinates
    pts = np.array(
        re.findall(r"\[\s*([-+\d.]+)\s*,\s*([-+\d.]+)\s*,\s*([-+\d.]+)\s*\]", html),
        dtype=float,
    )
    if pts.shape[0] < 2:
        raise ValueError("Too few coordinate samples.")

    # 5) build original timeline
    step = duration_sec / (pts.shape[0] - 1)
    orig_times = [launch_dt + timedelta(seconds=i * step) for i in range(pts.shape[0])]

    # 6) truncate duration to whole days
    days_trunc = duration_sec // 86400
    end_trunc = launch_dt + timedelta(days=days_trunc)

    # 7) resample every sample_hours until end_trunc or last point
    grid = []
    delta = timedelta(hours=sample_hours)
    t = launch_dt
    last = min(orig_times[-1], end_trunc)
    while t <= last:
        grid.append(t)
        t += delta

    # 8) nearest‐index lookup
    idx = np.rint([(g - launch_dt).total_seconds() / step for g in grid]).astype(int)
    idx = np.clip(idx, 0, pts.shape[0] - 1)

    # 9) assemble DataFrame
    return pd.DataFrame(
        {
            "callsign": callsign.upper(),
            "timestamp": grid,
            "lat": pts[idx, 0],
            "lon": pts[idx, 1],
            "alt": pts[idx, 2] / 1000,
        }
    )


# ────────────────────────────────────────────────────────────────
# parallel crawler with tqdm
# ────────────────────────────────────────────────────────────────
def crawl_range_parallel(
    start: int = 100,
    end: int = 300,
    sample_hours: int = 3,
    max_workers: int = 10,
    out_csv: Path | str | None = None,
) -> pd.DataFrame:
    """
    Crawl map_hbal{start..end}.html in parallel, resample to sample_hours grid,
    show tqdm progress at bottom (position=1), and optionally save to CSV.
    """
    callsigns = [f"hbal{i}" for i in range(start, end + 1)]
    frames: list[pd.DataFrame] = []

    def worker(cs: str):
        try:
            return cs, crawl_track(cs, sample_hours)
        except Exception as e:
            return cs, e

    with ThreadPoolExecutor(max_workers=max_workers) as exe:
        # position=1 -> leave prints above, bar stays on line 2
        for cs, val in tqdm(
            exe.map(worker, callsigns),
            total=len(callsigns),
            desc="Crawling HBAL",
            position=1,
            leave=True,
        ):
            if isinstance(val, pd.DataFrame):
                frames.append(val)
            else:
                # use tqdm.write to avoid breaking the progress bar
                tqdm.write(f"✗ {cs} – {val}")

    if not frames:
        return pd.DataFrame()
    result = pd.concat(frames, ignore_index=True).sort_values("timestamp")
    if out_csv:
        result.to_csv(out_csv, index=False)
    return result


# ────────────────────────────────────────────────────────────────
# query helper (unchanged)
# ────────────────────────────────────────────────────────────────
def get_positions(
    df: pd.DataFrame, when: dt.datetime, tol: timedelta = timedelta(hours=1.5)
) -> pd.DataFrame:
    mask = (df["timestamp"] >= when - tol) & (df["timestamp"] <= when + tol)
    return df.loc[mask, ["callsign", "timestamp", "lat", "lon", "alt"]]


# ────────────────────────────────────────────────────────────────
# example usage
# ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    df_all = crawl_range_parallel(
        0, 800, sample_hours=3, max_workers=12, out_csv="hbal_3h.csv"
    )

    # query example: 2020‑09‑01 12:00 UTC
    qtime = dt.datetime(2020, 9, 1, 12, 0)
    print(get_positions(df_all, qtime))
