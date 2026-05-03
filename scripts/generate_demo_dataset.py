#!/usr/bin/env python3
"""
Generate Synthetic Demo Dataset for Time Series Forecasting

This script creates a realistic synthetic dataset demonstrating various
time series patterns suitable for showcasing forecasting capabilities.

Dataset characteristics:
- Hourly frequency (realistic for energy/demand forecasting)
- ~2 years of data (17,520 hours)
- Multiple variables with different patterns
- Realistic noise and correlations
"""

import argparse
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path

def generate_demo_dataset(
    n_hours: int = 17520,  # ~2 years
    start_date: str = "2022-01-01",
    seed: int = 42
):
    """Generate synthetic time series dataset with realistic patterns."""

    np.random.seed(seed)

    # Generate timestamps
    start = pd.to_datetime(start_date)
    timestamps = pd.date_range(start=start, periods=n_hours, freq='h')

    # Time-based features
    hours = timestamps.hour
    days = timestamps.dayofweek
    days_since_start = (timestamps - start).days

    # ========================================================================
    # TARGET VARIABLE: Energy Demand (realistic pattern)
    # ========================================================================

    # 1. Long-term trend (slowly increasing demand over time)
    trend = 50 + 0.003 * days_since_start

    # 2. Daily seasonality (peak in afternoon, low at night)
    daily_pattern = 15 * np.sin(2 * np.pi * (hours - 6) / 24)

    # 3. Weekly seasonality (lower on weekends)
    weekly_pattern = np.where(days < 5, 5, -8)  # weekdays higher

    # 4. Annual seasonality (summer cooling, winter heating)
    day_of_year = timestamps.dayofyear
    annual_pattern = 10 * np.sin(2 * np.pi * (day_of_year - 80) / 365)

    # 5. Random noise
    noise = np.random.normal(0, 3, n_hours)

    # 6. Occasional spikes (special events)
    spikes = np.zeros(n_hours)
    spike_indices = np.random.choice(n_hours, size=50, replace=False)
    spikes[spike_indices] = np.random.uniform(10, 25, 50)

    # Combine all components
    energy_demand = trend + daily_pattern + weekly_pattern + annual_pattern + noise + spikes
    energy_demand = np.maximum(energy_demand, 10)  # Ensure positive values

    # ========================================================================
    # COVARIATES: Exogenous variables that influence the target
    # ========================================================================

    # Temperature (correlated with demand)
    base_temp = 15 + 10 * np.sin(2 * np.pi * (day_of_year - 80) / 365)
    daily_temp_variation = 5 * np.sin(2 * np.pi * hours / 24)
    temperature = base_temp + daily_temp_variation + np.random.normal(0, 2, n_hours)

    # Humidity (anti-correlated with temperature)
    humidity = 70 - 0.5 * temperature + np.random.normal(0, 5, n_hours)
    humidity = np.clip(humidity, 20, 100)

    # Wind speed (random with some seasonality)
    wind_speed = 8 + 3 * np.sin(2 * np.pi * day_of_year / 365) + np.random.exponential(2, n_hours)
    wind_speed = np.clip(wind_speed, 0, 30)

    # Is weekend (binary feature)
    is_weekend = (days >= 5).astype(int)

    # Is business hours (9-17 on weekdays)
    is_business_hours = ((hours >= 9) & (hours <= 17) & (days < 5)).astype(int)

    # ========================================================================
    # Create DataFrame
    # ========================================================================

    df = pd.DataFrame({
        'date': timestamps,
        'energy_demand': energy_demand.round(2),
        'temperature': temperature.round(1),
        'humidity': humidity.round(1),
        'wind_speed': wind_speed.round(1),
        'is_weekend': is_weekend,
        'is_business_hours': is_business_hours,
    })

    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate synthetic energy demand dataset for demo purposes"
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default=None,
        help="Output CSV file path (default: generates both 2yr and 5yr datasets)",
    )
    parser.add_argument(
        "--n-hours",
        type=int,
        default=None,
        help="Number of hours to generate (default: generates both 2yr and 5yr datasets)",
    )
    parser.add_argument(
        "--start-date",
        type=str,
        default="2022-01-01",
        help="Start date for the time series (default: 2022-01-01 for 2yr, 2020-01-01 for 5yr)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="Skip generating the preview plot",
    )
    parser.add_argument(
        "--both",
        action="store_true",
        help="Generate both 2yr and 5yr datasets (default behavior if no --output specified)",
    )

    args = parser.parse_args()

    # Determine if we should generate both datasets
    generate_both = args.both or (args.output is None and args.n_hours is None)

    if generate_both:
        print("Generating BOTH standard demo datasets (2yr + 5yr)...")
        print("=" * 70)

        datasets_to_generate = [
            {
                "output": "data/demo_energy_demand.csv",
                "n_hours": 17520,  # ~2 years
                "start_date": "2022-01-01",
                "description": "2-year dataset (~104 weekends)",
            },
            {
                "output": "data/demo_energy_demand_5yr.csv",
                "n_hours": 43800,  # ~5 years
                "start_date": "2020-01-01",
                "description": "5-year dataset (~260 weekends)",
            },
        ]
    else:
        # Single dataset mode
        datasets_to_generate = [
            {
                "output": args.output or "data/demo_energy_demand.csv",
                "n_hours": args.n_hours or 17520,
                "start_date": args.start_date,
                "description": "Custom dataset",
            }
        ]

    # Generate each dataset
    for idx, config in enumerate(datasets_to_generate, 1):
        if generate_both:
            print(f"\n[{idx}/{len(datasets_to_generate)}] Generating {config['description']}...")
            print("-" * 70)
        else:
            print("Generating synthetic demo dataset...")

        # Generate dataset
        df = generate_demo_dataset(
            n_hours=config["n_hours"],
            start_date=config["start_date"],
            seed=args.seed
        )

        # Create output directory if it doesn't exist
        output_path = Path(config["output"])
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Save to CSV
        df.to_csv(output_path, index=False)

        print(f"✅ Dataset saved to: {output_path}")
        print(f"\nDataset shape: {df.shape}")
        print(f"Date range: {df['date'].min()} to {df['date'].max()}")
        print(f"\nFirst 5 rows:")
        print(df.head())
        print(f"\nStatistics:")
        print(df.describe())
        print(f"\nFile size: {output_path.stat().st_size / 1024:.1f} KB")

        # Generate visualization
        if not args.no_plot:
            try:
                import matplotlib.pyplot as plt

                fig, axes = plt.subplots(3, 1, figsize=(15, 10))

                # Plot first month
                first_month_end = pd.to_datetime(config["start_date"]) + pd.DateOffset(
                    months=1
                )
                first_month = df[df["date"] < first_month_end]

                axes[0].plot(first_month["date"], first_month["energy_demand"])
                axes[0].set_title(
                    "Energy Demand - First Month (Daily + Weekly Patterns)"
                )
                axes[0].set_ylabel("Energy Demand")
                axes[0].grid(True, alpha=0.3)

                axes[1].plot(
                    first_month["date"],
                    first_month["temperature"],
                    label="Temperature",
                    alpha=0.7,
                )
                axes[1].plot(
                    first_month["date"],
                    first_month["humidity"],
                    label="Humidity",
                    alpha=0.7,
                )
                axes[1].set_title("Covariates - First Month")
                axes[1].set_ylabel("Value")
                axes[1].legend()
                axes[1].grid(True, alpha=0.3)

                # Plot full timeline (weekly averages)
                weekly = df.set_index("date").resample("W").mean()
                axes[2].plot(weekly.index, weekly["energy_demand"])
                axes[2].set_title("Energy Demand - Full Timeline (Weekly Averages)")
                axes[2].set_xlabel("Date")
                axes[2].set_ylabel("Energy Demand")
                axes[2].grid(True, alpha=0.3)

                plt.tight_layout()

                # Save plot in same directory as output CSV
                plot_path = output_path.parent / (
                    output_path.stem + "_preview.png"
                )
                plt.savefig(plot_path, dpi=150, bbox_inches="tight")
                print(f"\n✅ Preview plot saved to: {plot_path}")
                plt.close()

            except ImportError:
                print("\n⚠️  matplotlib not available - skipping visualization")

    if generate_both:
        print("\n" + "=" * 70)
        print("✅ Both datasets generated successfully!")
        print("\nGenerated files:")
        for config in datasets_to_generate:
            path = Path(config["output"])
            size_kb = path.stat().st_size / 1024
            print(f"  - {config['output']} ({size_kb:.1f} KB) - {config['description']}")
