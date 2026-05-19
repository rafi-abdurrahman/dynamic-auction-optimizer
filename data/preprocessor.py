"""
Loads and preprocesses EUROSTAT datasets for the auction simulation.

Expected raw files in data/raw/ (EUROSTAT bulk-download TSV format):
  apro_mt_pwgtm.tsv   — slaughtering in slaughterhouses
  apro_mt_pslothm.tsv — slaughtering outside slaughterhouses
  apro_ec_poulm.tsv   — poultry & egg production
  apro_mk_colm.tsv    — milk collection & dairy products

Output: unified DataFrame indexed by date (daily), with one column per
product [milk, eggs, poultry, beef] in tonnes.
"""

import re
import warnings
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).parent
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"

# apro_mt_pwgtm / apro_mt_pslothm: beef codes (B1000, B1100, B1200)
_SLAUGHTER_UNIT = "THS_T"

# apro_ec_poulm: poultry (A5130O) and eggs (A5130P)
_POULTRY_ANIMAL_CODE = "A5130O"
_EGG_ANIMAL_CODE = "A5130P"
_POULM_UNIT = "THS"

# apro_mk_colm: milk collection
_MILK_DAIRYPROD_CODE = "D1110D"  # Raw cow milk collected
_DAIRY_UNIT = "THS_T"


def _parse_eurostat_tsv(path: Path) -> pd.DataFrame:
    """Parse an EUROSTAT bulk-download TSV into a tidy long DataFrame with columns [*dims, date, value].

    Row 0 is a header with comma-separated dimension names followed by
    tab-separated monthly period columns. Observation flags like "123.4 b"
    or ": z" are stripped; unparseable values become NaN.
    """
    raw = path.read_text(encoding="utf-8")
    lines = [ln.rstrip("\n") for ln in raw.splitlines() if ln.strip()]

    header_line = lines[0]
    key_header, *time_cols = header_line.split("\t")
    # Header looks like "freq,unit,meat,geo\TIME_PERIOD" — drop the suffix
    dim_names = re.sub(r"\\.*$", "", key_header).split(",")

    def _to_period(s: str) -> Optional[pd.Period]:
        s = s.strip()
        m = re.match(r"^(\d{4})M(\d{2})$", s)
        if m:
            return pd.Period(f"{m.group(1)}-{m.group(2)}", freq="M")
        try:
            return pd.Period(s, freq="M")
        except Exception:
            return None

    periods = [_to_period(c) for c in time_cols]

    records = []
    for line in lines[1:]:
        parts = line.split("\t")
        key_str = parts[0]
        values = parts[1:]
        dims = key_str.split(",")
        if len(dims) != len(dim_names):
            continue
        for period, raw_val in zip(periods, values):
            if period is None:
                continue
            raw_val = raw_val.strip()
            num_str = re.sub(r"[a-zA-Z ]+$", "", raw_val).strip()
            try:
                val = float(num_str)
            except ValueError:
                val = np.nan
            records.append(dict(zip(dim_names, dims)) | {"date": period, "value": val})

    return pd.DataFrame(records)


def _extract_beef(country: str) -> pd.Series:
    """Return monthly beef production (tonnes) for country."""
    series_list = []
    for fname in ("apro_mt_pwgtm.tsv", "apro_mt_pslothm.tsv"):
        fpath = RAW_DIR / fname
        if not fpath.exists():
            warnings.warn(f"Missing raw file: {fpath}", UserWarning)
            continue
        df = _parse_eurostat_tsv(fpath)
        # Beef codes start with "B" (B1000, B1100, B1200)
        mask = (
            (df["geo"].str.upper() == country.upper())
            & (df["meat"].str.upper().str.startswith("B"))
            & (df["unit"].str.upper() == _SLAUGHTER_UNIT)
        )
        sub = df.loc[mask, ["date", "value"]].set_index("date")["value"]
        series_list.append(sub)

    if not series_list:
        return pd.Series(dtype=float, name="beef")

    combined = pd.concat(series_list).groupby(level=0).sum()
    combined = combined * 1_000.0  # THS_T → tonnes
    combined.name = "beef"
    return combined


def _extract_poultry(country: str) -> pd.Series:
    """Return monthly poultry-meat production (tonnes) for country."""
    fpath = RAW_DIR / "apro_ec_poulm.tsv"
    if not fpath.exists():
        warnings.warn(f"Missing raw file: {fpath}", UserWarning)
        return pd.Series(dtype=float, name="poultry")

    df = _parse_eurostat_tsv(fpath)
    mask = (
        (df["geo"].str.upper() == country.upper())
        & (df["animals"].str.upper() == _POULTRY_ANIMAL_CODE)
        & (df["unit"].str.upper() == _POULM_UNIT)
    )
    s = df.loc[mask, ["date", "value"]].set_index("date")["value"]
    s = s * 1_000.0  # THS → tonnes
    s.name = "poultry"
    return s


def _extract_eggs(country: str) -> pd.Series:
    """Return monthly egg/broiler production (tonnes) for country.

    EUROSTAT A5130P represents broiler chicken production in thousand tonnes.
    """
    fpath = RAW_DIR / "apro_ec_poulm.tsv"
    if not fpath.exists():
        warnings.warn(f"Missing raw file: {fpath}", UserWarning)
        return pd.Series(dtype=float, name="eggs")

    df = _parse_eurostat_tsv(fpath)
    mask = (
        (df["geo"].str.upper() == country.upper())
        & (df["animals"].str.upper() == _EGG_ANIMAL_CODE)
        & (df["unit"].str.upper() == _POULM_UNIT)
    )
    s = df.loc[mask, ["date", "value"]].set_index("date")["value"]
    s = s * 1_000.0  # THS → tonnes
    s.name = "eggs"
    return s


def _extract_milk(country: str) -> pd.Series:
    """Return monthly raw milk collected (tonnes) for country."""
    fpath = RAW_DIR / "apro_mk_colm.tsv"
    if not fpath.exists():
        warnings.warn(f"Missing raw file: {fpath}", UserWarning)
        return pd.Series(dtype=float, name="milk")

    df = _parse_eurostat_tsv(fpath)
    mask = (
        (df["geo"].str.upper() == country.upper())
        & (df["dairyprod"].str.upper() == _MILK_DAIRYPROD_CODE)
        & (df["unit"].str.upper() == _DAIRY_UNIT)
    )
    s = df.loc[mask, ["date", "value"]].set_index("date")["value"]
    s = s * 1_000.0  # THS_T → tonnes
    s.name = "milk"
    return s


PRODUCTS = ("milk", "eggs", "poultry", "beef")


def load_monthly(country: str = "DE") -> pd.DataFrame:
    """Load and combine all four product series at monthly resolution (PeriodIndex, tonnes)."""
    series = {
        "milk": _extract_milk(country),
        "eggs": _extract_eggs(country),
        "poultry": _extract_poultry(country),
        "beef": _extract_beef(country),
    }
    df = pd.DataFrame(series)
    df = df.dropna(how="all")
    df = df.ffill().bfill()
    df = df.sort_index()
    return df


def load_daily(
    country: str = "DE",
    auction_fraction: float = 1.0 / 20,
) -> pd.DataFrame:
    """Return a daily-resolution DataFrame of auction lot sizes (tonnes).

    Monthly production is spread evenly across calendar days; auction_fraction
    (default 1/20) sets what share of daily production constitutes one lot.
    """
    monthly = load_monthly(country)
    if monthly.empty:
        raise RuntimeError(
            "No data loaded — ensure raw TSV files are placed in data/raw/. "
            "See README for download instructions."
        )

    daily_rows = []
    for period, row in monthly.iterrows():
        month_start = period.to_timestamp(how="S")
        month_end = period.to_timestamp(how="E").normalize()
        days_in_month = (month_end - month_start).days + 1
        dates = pd.date_range(month_start, periods=days_in_month, freq="D")
        daily_vals = row.values / days_in_month
        for d in dates:
            daily_rows.append([d] + daily_vals.tolist())

    daily = pd.DataFrame(daily_rows, columns=["date"] + list(monthly.columns))
    daily = daily.set_index("date").sort_index()
    daily = daily * auction_fraction
    return daily


def get_depletion_rates(country: str = "DE") -> pd.Series:
    """Estimate average daily depletion per product (mean daily production as proxy)."""
    daily = load_daily(country, auction_fraction=1.0)
    return daily.mean()


def save_processed(country: str = "DE") -> Path:
    """Preprocess and save daily auction data to data/processed/."""
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    daily = load_daily(country)
    out_path = PROCESSED_DIR / f"auction_quantities_{country}.csv"
    daily.to_csv(out_path)
    return out_path


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Preprocess EUROSTAT data for the auction simulator."
    )
    parser.add_argument(
        "--country", default="DE", help="ISO-2 country code (default: DE)"
    )
    parser.add_argument(
        "--auction-fraction",
        type=float,
        default=1.0 / 20,
        help="Fraction of daily production per auction lot (default: 0.05)",
    )
    args = parser.parse_args()

    print(f"Loading data for country: {args.country}")
    monthly = load_monthly(args.country)
    print(f"Monthly data shape: {monthly.shape}")
    print(monthly.tail())

    out = save_processed(args.country)
    print(f"Saved processed data to: {out}")

    print("\nMean daily depletion rates (tonnes):")
    print(get_depletion_rates(args.country).round(2))
