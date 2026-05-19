from typing import Generator, Tuple, Dict, Optional
from pathlib import Path
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from data.preprocessor import load_daily, get_depletion_rates, PRODUCTS


class AuctionDataGenerator:
    """Yields intra-day auction data with noise for a given country.

    Each day's quantities are split randomly across N auction slots via
    Dirichlet sampling (so they sum to the daily total) and then perturbed
    with Gaussian noise. All quantities are normalised to [0, 1000] using
    the per-product historical maximum.
    """

    def __init__(
        self,
        country: str = "DE",
        auctions_per_day: int = 3,
        noise_std: float = 0.05,
        n_days: Optional[int] = None,
        seed: Optional[int] = None,
    ):
        self.country = country.upper()
        self.auctions_per_day = auctions_per_day
        self.noise_std = noise_std
        self.n_days = n_days
        self.seed = seed

        if seed is not None:
            np.random.seed(seed)

        # Load full daily data at auction_fraction=1.0; normalise internally
        self.daily_data = load_daily(country, auction_fraction=1.0)
        self.depletion_rates = get_depletion_rates(country)
        self.products = list(PRODUCTS)

        # Normalise to [0, 1000] using the per-product max over the full dataset.
        # The scale factor is fixed (independent of n_days) so alpha/d_max/prices
        # are all comparable across products and countries.
        _full_max = self.daily_data.max()
        _safe_max = _full_max.clip(lower=1.0)
        self.norm_scale = pd.Series(
            1000.0 / _safe_max.values,
            index=_safe_max.index,
        )
        self.normed_daily_mean = (self.daily_data.mean() * self.norm_scale).clip(lower=0.0)

        if n_days is not None:
            self.daily_data = self.daily_data.iloc[:n_days]

    def __len__(self) -> int:
        return len(self.daily_data)

    def _generate_time_slot_distribution(self) -> np.ndarray:
        """Sample a Dirichlet split across auction slots (sums to 1)."""
        alpha = np.ones(self.auctions_per_day)
        return np.random.dirichlet(alpha)

    def generate(
        self,
    ) -> Generator[Tuple[str, int, Dict[str, float]], None, None]:
        """Yield (date_str, auction_idx, quantities_dict) for each auction slot."""
        for date_idx_pd, day_row in self.daily_data.iterrows():
            auction_dist = self._generate_time_slot_distribution()
            date_str = str(date_idx_pd.date())

            for auction_idx, distribution_fraction in enumerate(auction_dist):
                quantities = {}

                for product in self.products:
                    daily_qty = day_row[product]
                    auction_qty = daily_qty * distribution_fraction
                    noise = np.random.normal(0, self.noise_std * auction_qty)
                    noisy_qty = max(0, auction_qty + noise)
                    scale = float(self.norm_scale.get(product, 1.0))
                    quantities[product] = noisy_qty * scale

                yield (date_str, auction_idx, quantities)

    def get_metadata(self) -> Dict:
        """Return configuration and data statistics."""
        return {
            "country": self.country,
            "auctions_per_day": self.auctions_per_day,
            "noise_std": self.noise_std,
            "n_days": self.n_days if self.n_days is not None else len(self.daily_data),
            "total_days": len(self.daily_data),
            "total_auctions": len(self.daily_data) * self.auctions_per_day,
            "products": self.products,
            "date_range": {
                "start": str(self.daily_data.index[0].date()),
                "end": str(self.daily_data.index[-1].date()),
            },
            "depletion_rates": self.depletion_rates.to_dict(),
        }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate intra-day auction data with noise."
    )
    parser.add_argument("--country", default="DE", help="ISO-2 country code (default: DE)")
    parser.add_argument("--auctions-per-day", type=int, default=3)
    parser.add_argument("--noise-std", type=float, default=0.05)
    parser.add_argument("--n-days", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--demo-days", type=int, default=3)

    args = parser.parse_args()

    gen = AuctionDataGenerator(
        country=args.country,
        auctions_per_day=getattr(args, 'auctions_per_day', 3),
        noise_std=getattr(args, 'noise_std', 0.05),
        n_days=getattr(args, 'n_days', None),
        seed=getattr(args, 'seed', None),
    )

    print("=" * 80)
    print("AUCTION DATA GENERATOR - METADATA")
    print("=" * 80)
    metadata = gen.get_metadata()
    for key, value in metadata.items():
        if isinstance(value, dict):
            print(f"{key}:")
            for k, v in value.items():
                if isinstance(v, float):
                    print(f"  {k}: {v:.4f}")
                else:
                    print(f"  {k}: {v}")
        else:
            print(f"{key}: {value}")

    print("\n" + "=" * 80)
    print(f"SAMPLE INTRA-DAY AUCTIONS ({args.demo_days} days × {gen.auctions_per_day} auctions/day)")
    print("=" * 80)

    demo_count = 0
    for date_str, auction_idx, quantities in gen.generate():
        if demo_count >= args.demo_days * gen.auctions_per_day:
            break

        if demo_count % gen.auctions_per_day == 0:
            print(f"\n--- Date: {date_str} ---")

        print(f"  Auction {auction_idx}:")
        for product, qty in quantities.items():
            print(f"    {product:10s}: {qty:12.2f} tonnes")

        demo_count += 1

    print("\n" + "=" * 80)
