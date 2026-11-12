from torch import nn
import math
import random
import torch
import numpy as np
from sklearn.metrics import roc_auc_score, mean_squared_error, mean_absolute_error, r2_score
import os
import pandas as pd

def reset_parameters(named_parameters):
    for name, param in named_parameters():
        if param.dim() == 0:
            continue
        elif param.dim() == 1:
            std = 1.0 / math.sqrt(param.size(0))
            nn.init.uniform_(param, -std, std)
        else:
            nn.init.xavier_normal_(param)


#def set_seed(seed):
def set_seed(seed):
    print(f"Randomly generated seed: {seed}")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def metrics(trues, preds):
    trues = np.asarray(trues, dtype=np.float64).reshape(-1)
    preds = np.asarray(preds, dtype=np.float64).reshape(-1)
    valid = np.isfinite(trues) & np.isfinite(preds)
    if not np.any(valid):
        return float("nan"), float("nan")
    mse = mean_squared_error(trues[valid], preds[valid])
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(trues[valid], preds[valid])
    return rmse, mae


def filter_strict_next_day_labels(labels_data, market_data):
    """Keep labels only when that ticker exists on the next global market date."""
    labels = labels_data.copy()
    market = market_data[["Ticker", "Date"]].copy()
    labels["Date"] = pd.to_datetime(labels["Date"])
    market["Date"] = pd.to_datetime(market["Date"])

    market_dates = sorted(market["Date"].unique())
    next_date = {
        pd.Timestamp(current): pd.Timestamp(following)
        for current, following in zip(market_dates[:-1], market_dates[1:])
    }
    market_pairs = set(zip(market["Ticker"], market["Date"].map(pd.Timestamp)))

    valid = []
    for ticker, current_date in zip(labels["Ticker"], labels["Date"]):
        following = next_date.get(pd.Timestamp(current_date))
        valid.append(following is not None and (ticker, following) in market_pairs)

    valid = np.asarray(valid, dtype=bool)
    removed = labels.loc[~valid, ["Ticker", "Date"]].copy()
    return labels.loc[valid].reset_index(drop=True), removed.reset_index(drop=True)


def get_strict_prediction_dates(split_dates, all_market_dates):
    """Return split dates whose next global trading date stays in the split."""
    split_index = pd.DatetimeIndex(pd.to_datetime(list(split_dates))).drop_duplicates().sort_values()
    market_index = pd.DatetimeIndex(pd.to_datetime(list(all_market_dates))).drop_duplicates().sort_values()
    split_set = set(split_index)
    next_global = {
        pd.Timestamp(current): pd.Timestamp(following)
        for current, following in zip(market_index[:-1], market_index[1:])
    }
    retained = [
        current
        for current in split_index
        if next_global.get(pd.Timestamp(current)) in split_set
    ]
    return pd.DatetimeIndex(retained)


def assert_strict_prediction_dates(
    prediction_dates, split_dates, all_market_dates, split_name, require_all=True
):
    """Fail if a supervised/evaluation terminal date crosses its split boundary."""
    expected = get_strict_prediction_dates(split_dates, all_market_dates)
    actual = pd.DatetimeIndex(pd.to_datetime(list(prediction_dates))).drop_duplicates().sort_values()
    extra = actual.difference(expected)
    missing = expected.difference(actual) if require_all else pd.DatetimeIndex([])
    if len(extra) or len(missing):
        raise RuntimeError(
            f"{split_name} strict prediction dates differ from the t->t+1 rule; "
            f"missing={list(missing[:3])}, extra={list(extra[:3])}."
        )
    return expected

def createPath(path):
    if os.path.exists(path):
        pass
    else:
        os.makedirs(path)

