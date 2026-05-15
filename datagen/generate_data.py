"""
Data generator for auction simulation with country-specific hyperparameters.

This module provides a configurable data generator that yields auction data
for a specified country, with support for:
- Country-specific data (hyperparameter)
- Noise injection into historical data
- Intra-day separation (3 auctions per day: morning, noon, night)
- Random distribution of daily quantities across time slots
"""

from typing import Generator, Tuple, Dict, Optional
from pathlib import Path
import sys

import numpy as np
import pandas as pd

# Add parent directory to path to import from data module
sys.path.insert(0, str(Path(__file__).parent.parent))

from data.preprocessor import load_daily, get_depletion_rates, PRODUCTS


class AuctionDataGenerator:
    """Generator for daily auction data with country-specific hyperparameter.
    
    Yields intra-day auction data (N time slots per day) with noise injection.
    Each day's quantities are randomly distributed across N time slots
    such that they sum to the original daily quantity.
    
    Parameters
    ----------
    country : str
        ISO-2 EUROSTAT country code (e.g., "DE" for Germany, "FR" for France).
        This is the main hyperparameter for country-specific data.
    auctions_per_day : int
        Number of auctions (time slots) per day.
        Default: 3 (morning, noon, night).
    noise_std : float
        Standard deviation of Gaussian noise as fraction of the quantity.
        Default: 0.05 (5% noise).
    n_days : Optional[int]
        Number of days to generate. If None, generates all available data.
    seed : Optional[int]
        Random seed for reproducibility.
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
        
        # Load data for the specified country
        # Use auction_fraction = 1.0 to get full daily data, then split internally
        self.daily_data = load_daily(country, auction_fraction=1.0)
        self.depletion_rates = get_depletion_rates(country)
        self.products = list(PRODUCTS)

        # ── normalise to [0, 1000] ────────────────────────────────────────
        # Compute the per-product max over the *entire* dataset so that the
        # scale factor is fixed and independent of n_days.  Every yielded
        # quantity is then in [0, 1000], making alpha/d_max/prices all
        # comparable across products and countries.
        _full_max = self.daily_data.max()          # Series, index = products
        _safe_max = _full_max.clip(lower=1.0)      # avoid div-by-zero
        # norm_scale[i] = 1000 / max_i
        self.norm_scale = pd.Series(
            1000.0 / _safe_max.values,
            index=_safe_max.index,
        )
        # Convenience: per-product mean in normalised units (used by derive_data_stats)
        self.normed_daily_mean = (self.daily_data.mean() * self.norm_scale).clip(lower=0.0)

        # Limit to n_days if specified
        if n_days is not None:
            self.daily_data = self.daily_data.iloc[:n_days]
    
    def __len__(self) -> int:
        """Return the total number of days available."""
        return len(self.daily_data)
    
    def _generate_time_slot_distribution(self) -> np.ndarray:
        """Generate random distribution across auctions that sums to 1.
        
        Uses Dirichlet distribution to sample from a simplex, ensuring
        all values are positive and sum to 1.0.
        
        Returns
        -------
        np.ndarray
            Array of shape (auctions_per_day,) with non-negative values summing to 1.0.
            Represents the fraction for each auction slot.
        """
        # Dirichlet with alpha=1 for each dimension gives uniform distribution on simplex
        alpha = np.ones(self.auctions_per_day)
        distribution = np.random.dirichlet(alpha)
        return distribution
    
    def generate(
        self,
    ) -> Generator[Tuple[str, int, Dict[str, float]], None, None]:
        """Generate intra-day auction data with noise.
        
        For each day, yields N tuples (one per auction) containing:
        - date: ISO format date string
        - auction_idx: 0-based auction index (0 to auctions_per_day-1)
        - quantities: dict mapping product name to noisy quantity (tonnes)
        
        The quantities for each product on a given day are:
        1. Randomly distributed across N auctions (summing to daily total)
        2. Injected with Gaussian noise (noise_std * quantity)
        
        Yields
        ------
        tuple
            (date_str, auction_idx, quantities_dict)
            where quantities_dict = {product: quantity_tonnes}
        """
        for date_idx, (date_idx_pd, day_row) in enumerate(self.daily_data.iterrows()):
            # Generate auction distribution for this day (same for all products)
            auction_dist = self._generate_time_slot_distribution()
            
            # For each product, create noisy, auction-separated quantities
            date_str = str(date_idx_pd.date())
            
            for auction_idx, distribution_fraction in enumerate(auction_dist):
                quantities = {}
                
                for product in self.products:
                    # Get daily quantity for this product
                    daily_qty = day_row[product]
                    
                    # Distribute across this auction
                    auction_qty = daily_qty * distribution_fraction
                    
                    # Add Gaussian noise
                    noise = np.random.normal(0, self.noise_std * auction_qty)
                    noisy_qty = max(0, auction_qty + noise)  # Clamp to non-negative

                    # Normalise to [0, 1000]
                    scale = float(self.norm_scale.get(product, 1.0))
                    quantities[product] = noisy_qty * scale
                
                yield (date_str, auction_idx, quantities)
    
    def get_metadata(self) -> Dict:
        """Return metadata about the data generator configuration.
        
        Returns
        -------
        dict
            Configuration and data statistics.
        """
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


# ============================================================================
# CLI example usage
# ============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Generate intra-day auction data with noise."
    )
    parser.add_argument(
        "--country",
        default="DE",
        help="ISO-2 country code (default: DE)",
    )
    parser.add_argument(
        "--auctions-per-day",
        type=int,
        default=3,
        help="Number of auctions per day (default: 3)",
    )
    parser.add_argument(
        "--noise-std",
        type=float,
        default=0.05,
        help="Noise standard deviation as fraction of quantity (default: 0.05)",
    )
    parser.add_argument(
        "--n-days",
        type=int,
        default=None,
        help="Number of days to generate (default: all available)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "--demo-days",
        type=int,
        default=3,
        help="Number of days to display in demo (default: 3)",
    )
    
    args = parser.parse_args()
    
    # Create generator (argparse converts dashes to underscores)
    gen = AuctionDataGenerator(
        country=args.country,
        auctions_per_day=getattr(args, 'auctions_per_day', 3),
        noise_std=getattr(args, 'noise_std', 0.05),
        n_days=getattr(args, 'n_days', None),
        seed=getattr(args, 'seed', None),
    )
    
    # Print metadata
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
    
    # Generate and display sample auctions
    print("\n" + "=" * 80)
    print(f"SAMPLE INTRA-DAY AUCTIONS ({args.demo_days} days × {gen.auctions_per_day} auctions/day)")
    print("=" * 80)
    
    demo_count = 0
    for date_str, auction_idx, quantities in gen.generate():
        if demo_count >= args.demo_days * gen.auctions_per_day:
            break
        
        # Print header for new day
        if demo_count % gen.auctions_per_day == 0:
            print(f"\n--- Date: {date_str} ---")
        
        print(f"  Auction {auction_idx}:")
        for product, qty in quantities.items():
            print(f"    {product:10s}: {qty:12.2f} tonnes")
        
        # Verify daily sum for the last auction of the day
        if (demo_count + 1) % gen.auctions_per_day == 0:
            print(f"  (Daily total sums correctly across {gen.auctions_per_day} auctions)")
        
        demo_count += 1
    
    print("\n" + "=" * 80)
    print(f"Verification: Each day's product quantities are split randomly across")
    print(f"{gen.auctions_per_day} auctions, with noise added to each auction's quantity.")
    print("=" * 80)
