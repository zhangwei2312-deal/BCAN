"""
CN data split utilities.

A-share experiment split:
- train: 36 months
- validation: 4 months
- test: 8 months
- slide step: 8 months

For 2018-01 to 2025-12 this yields 7 phases:
1. Train 2018-01~2020-12 | Val 2021-01~2021-04 | Test 2021-05~2021-12
2. Train 2018-09~2021-08 | Val 2021-09~2021-12 | Test 2022-01~2022-08
3. Train 2019-05~2022-04 | Val 2022-05~2022-08 | Test 2022-09~2023-04
4. Train 2020-01~2022-12 | Val 2023-01~2023-04 | Test 2023-05~2023-12
5. Train 2020-09~2023-08 | Val 2023-09~2023-12 | Test 2024-01~2024-08
6. Train 2021-05~2024-04 | Val 2024-05~2024-08 | Test 2024-09~2025-04
7. Train 2022-01~2024-12 | Val 2025-01~2025-04 | Test 2025-05~2025-12
"""
from datetime import datetime

import pandas as pd
from dateutil.relativedelta import relativedelta


TRAIN_MONTHS = 36
VAL_MONTHS = 4
TEST_MONTHS = 8
SLIDE_MONTHS = 8
WINDOW_MONTHS = TRAIN_MONTHS + VAL_MONTHS + TEST_MONTHS



def _month_start(year: int, month: int) -> datetime:
    return datetime(year, month, 1)


def _month_window_end(start_dt: datetime, n_months: int) -> datetime:
    next_window_start = start_dt + relativedelta(months=n_months)
    return next_window_start.replace(day=1) - relativedelta(days=1)


def calculate_phases(all_dates):
    """Return 7 rolling phases for the CN/A-share dataset."""
    dates = pd.Series(pd.to_datetime(all_dates)).sort_values().reset_index(drop=True)
    if dates.empty:
        return []

    base_year = int(dates.min().year)
    base_month = int(dates.min().month)
    max_date = dates.max()

    phases = []
    step = 0
    while True:
        train_start = _month_start(base_year, base_month)
        train_end = _month_window_end(train_start, TRAIN_MONTHS)
        val_start = train_start + relativedelta(months=TRAIN_MONTHS)
        val_end = _month_window_end(train_start, TRAIN_MONTHS + VAL_MONTHS)
        test_start = train_start + relativedelta(months=TRAIN_MONTHS + VAL_MONTHS)
        test_end = _month_window_end(train_start, WINDOW_MONTHS)

        if test_start > max_date:
            break

        train_dates = dates[(dates >= train_start) & (dates <= train_end)]
        val_dates = dates[(dates >= val_start) & (dates <= val_end)]
        test_dates = dates[(dates >= test_start) & (dates <= test_end)]

        if len(train_dates) < 100 or len(val_dates) < 20 or len(test_dates) < 20:
            break

        phases.append({
            'phase_id': step + 1,
            'train_start': pd.Timestamp(train_start),
            'train_end': pd.Timestamp(train_end),
            'val_start': pd.Timestamp(val_start),
            'val_end': pd.Timestamp(val_end),
            'test_start': pd.Timestamp(test_start),
            'test_end': pd.Timestamp(test_end),
            'train_dates': train_dates,
            'val_dates': val_dates,
            'test_dates': test_dates,
        })

        step += 1
        next_start = train_start + relativedelta(months=SLIDE_MONTHS)
        base_year = int(next_start.year)
        base_month = int(next_start.month)

    return phases

