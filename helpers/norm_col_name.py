# norm_col_name.py

# Noromlize the name of columns
import re
from typing import Optional
import pandas as pd

def _unit_token(unit_raw: str) -> Optional[str]:
    """Map raw unit string to a clean token like 'w_m2', 'wh_m2', 'c', 'm_s', 'deg', 'pct', 'a', 'mm', '1m'."""
    if not unit_raw:
        return None
    u = unit_raw.lower().strip()

    # unify variants
    u = u.replace("w/m²", "w/m2").replace("w/m2", "w/m2").replace("w/m2", "w/m2")
    u = u.replace("wh/m²", "wh/m2").replace("w/m2)", "w/m2)").replace("w/m2", "w/m2")
    u = u.replace("degree centigrade", "deg c").replace("°c", "deg c")
    u = u.replace("deg c", "deg c")

    # map
    if "wh/m2" in u:
        return "wh_m2"
    if "w/m2" in u:
        return "w_m2"
    if "m/s" in u:
        return "m_s"
    if "deg c" in u or "celsius" in u:
        return "c"
    if "degree" in u:
        return "deg"
    if "%" in u:
        return "pct"
    if "amp" in u or u == "a":
        return "a"
    if "mm" in u:
        return "mm"
    if "1m" in u:
        return "1m"
    return re.sub(r"[^a-z0-9]+", "_", u).strip("_") or None


def _base_token(label: str) -> str:
    """Return canonical base name token (without unit) in snake_case."""
    s = label.lower()
    s = s.replace("\n", " ")
    s = re.sub(r"\s+", " ", s).strip()

    # normalize noisy phrases
    s = s.replace("total solar irradiance on inclined plane ", "")
    s = s.replace("total solar irradiance on horizontal plane ", "")
    s = s.replace("ghi ", "ghi ").replace(" poa", " poa")
    s = s.replace("module surface ", "module ")
    s = s.replace("temperature  ", "temperature ")  # double spaces

    # explicit mappings (order matters)
    if s.startswith("time"):
        return "time"
    if "control-ppc - active power" in s:
        return "activepower"
    if "poa1" in s:
        return "poa1"
    if "poa2" in s:
        return "poa2"
    if "ghi" in s:
        return "ghi"
    if "ambient temp" in s:
        return "ambienttemp"
    if "module temperature1" in s:
        return "moduletemp1"
    if "module temperature2" in s:
        return "moduletemp2"
    if "wind speed" in s:
        return "wind_speed"
    if "wind direction" in s:
        return "wind_dir"
    if "daily rain" in s or s.startswith("rain"):
        return "rain"
    if "relative humidity" in s or s.startswith("humidity"):
        return "humidity"
    if "soiling loss index geff" in s:
        return "soiling_loss_geff"
    if "soiling loss index" in s:      # isc
        return "soiling_loss_isc"
    if "isc test" in s:
        return "isc_test"
    if "isc ref" in s:
        return "isc_ref"
    if "temperature reference cell" in s:
        return "temp_refcell"
    if "temperature test" in s:
        return "temp_test"
    if "geff reference" in s:
        return "geff_ref"
    if "geff test" in s:
        return "geff_test"

    # fallback: compact snake
    return re.sub(r"[^a-z0-9]+", "_", s).strip("_")


def normalize_cols_keep_units(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize solar dataframe columns to short snake_case names and keep units as tokens.
    - Removes suffixes like '.1'
    - Preserves units inside parentheses and maps them to tokens (w_m2, wh_m2, c, m_s, deg, pct, a, mm, 1m)
    - Handles newlines and spacing variations
    """
    new_cols = []
    for col in df.columns:
        raw = str(col)

        # drop trailing suffix like .1 / .2
        raw = re.sub(r"\.\d+$", "", raw)

        # extract first unit in parentheses (if any)
        m = re.search(r"\(([^)]*)\)", raw)
        unit_raw = m.group(1) if m else ""

        # remove all (...) parts from label for base tokening
        label_wo_unit = re.sub(r"\([^)]*\)", "", raw).strip()

        base = _base_token(label_wo_unit)
        unit = _unit_token(unit_raw)

        # special: active power has interval (1m) not a physical unit
        if base == "activepower":
            # if we found "1m" in unit -> append only _1m
            if unit == "1m":
                new_name = f"{base}_1m"
            else:
                new_name = base
        else:
            new_name = base if unit is None else f"{base}_{unit}"

        new_cols.append(new_name)

    out = df.copy()
    out.columns = new_cols
    return out