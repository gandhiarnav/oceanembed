# Why does the 2020 dummy dataset have 365 days but end on Dec 30?
# Did we intentionally exclude one day, or did the date generation omit Feb 29?

import os
import numpy as np
import pandas as pd


def check_dataset_dates():
    # 1. Path setup
    base_dir = os.path.dirname(os.path.abspath(__file__))
    dates_path = os.path.join(base_dir, "dummy_dates.npy")

    print("=" * 65)
    print("       CHECKING 2020 DUMMY DATASET DATE GENERATION")
    print("=" * 65)

    # 2. Load dates
    if not os.path.exists(dates_path):
        print(f"Error: {dates_path} not found!")
        return

    dates = np.load(dates_path)
    dates_dt = pd.to_datetime(dates)
    total_days = len(dates)

    print(f"\n[1] DATASET OVERVIEW:")
    print(f"  • Total date entries : {total_days}")
    print(f"  • Start date         : {dates[0]}")
    print(f"  • End date           : {dates[-1]}")

    # 3. Key Date Inspections
    has_feb_29 = "2020-02-29" in dates
    has_dec_31 = "2020-12-31" in dates

    # 4. Check Date Continuity (daily steps)
    time_diffs = dates_dt[1:] - dates_dt[:-1]
    is_continuous = all(diff == pd.Timedelta(days=1) for diff in time_diffs)

    # 5. Check Generation Logic
    # 2020 is a leap year (366 days: 2020-01-01 to 2020-12-31)
    gen_365_range = pd.date_range("2020-01-01", periods=365, freq="D").strftime(
        "%Y-%m-%d"
    )
    matches_365_range = np.array_equal(dates, gen_365_range)

    print(f"\n[2] VERIFICATION CHECKS:")
    print(
        f"  • Is Feb 29, 2020 included?  : {'YES [✓]' if has_feb_29 else 'NO [X]'}"
    )
    print(
        f"  • Is Dec 31, 2020 included?  : {'YES [✓]' if has_dec_31 else 'NO [X]'}"
    )
    print(
        f"  • Are dates strictly daily?  : {'YES [✓]' if is_continuous else 'NO [X]'}"
    )
    print(
        f"  • Generated via periods=365? : {'YES [✓]' if matches_365_range else 'NO [X]'}"
    )

    # 6. Conclusion / Answer to Question
    print("\n" + "=" * 65)
    print("                     CONCLUSION & FINDINGS")
    print("=" * 65)
    if has_feb_29 and not has_dec_31 and matches_365_range:
        print("""
• Feb 29 (Leap Day) WAS INCLUDED in the date sequence (Day index 59).
• Dec 31 (Year End) WAS OMITTED because the dataset was generated using
  a fixed length of 365 periods (pd.date_range('2020-01-01', periods=365))
  rather than full year 2020 range (which requires 366 days for a leap year).
• Date continuity is perfectly intact (1-day step between all 365 entries).
""")
    print("=" * 65)


if __name__ == "__main__":
    check_dataset_dates()
