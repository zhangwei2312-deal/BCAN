from Model_n2m_filtered_dailyrecurrent import *
from utils import *
from data_splits import calculate_phases
from data_preprocessing import build_static_relation
from backtesting import backtest_phase, calculate_prediction_metrics_from_records
import torch
import torch.nn.functional as F
from tqdm import tqdm
from torch import optim
from scipy.stats import spearmanr
import argparse
import pandas as pd
import os
import sys
import numpy.core.numeric as numeric
import numpy as np
import json
import platform
import transformers
from project_paths import PROJECT_ROOT, TEMPORAL_CONFIG, experiment_dir, resolve_assets

sys.modules['numpy._core.numeric'] = numeric

parser = argparse.ArgumentParser()
parser.add_argument('--debug-daily-relation', action='store_true')

parser.add_argument('--max-epoch', type=int, default='300',
                    help='Training max epoch')
parser.add_argument('--wait-epoch', type=int, default='30',
                    help='Early stopping patience')
parser.add_argument('--lr', type=float, default='5e-4',
                    help='Learning rate ')
parser.add_argument('--device', type=str, default='cpu',
                    help='Device to use (cpu or cuda)')
parser.add_argument('--heads-att', type=int, default='8',
                    help='attention heads')
parser.add_argument('--hidn-att', type=int, default='512',
                    help='attention hidden nodes')
parser.add_argument('--hidn-rnn', type=int, default='256',
                    help='rnn hidden nodes')
parser.add_argument('--hidn-news', type=int, default='768', help='news output dim')
parser.add_argument('--hidn-market', type=int, default='64',
                    help='market output dim')
parser.add_argument('--weight-constraint', type=float, default='0',
                    help='L2 weight constraint')
parser.add_argument('--rnn-length', type=int, default='5',
                    help='rnn length')
parser.add_argument('--dropout', type=float, default='0.4',
                    help='dropout rate')
parser.add_argument('--clip', type=float, default='0.25',
                    help='rnn clip')
parser.add_argument('--save', type=bool, default=True,
                    help='save model')
parser.add_argument('--alpha', type=float, default='0.1',
                    help='alpha parameter')
parser.add_argument('--static-thresh', type=float, default='0.5',
                    help='Static relation threshold')
parser.add_argument('--static-mode', type=str, default='level', choices=['level', 'return'],
                    help='Static relation input: level (daily cross-sectional z-scored close) '
                         'or return (daily log returns of the raw close)')
parser.add_argument('--phase', type=int, default=-1,
                    help='Run specific phase only (1-9), -1 for all phases')
parser.add_argument('--skip-backtest', action='store_true',
                    help='Skip backtesting after training (only do train+eval)')
parser.add_argument('--lambda-align', type=float, default=1.0,
                    help='Base alignment coefficient in MSE + lambda_align * base + lambda_pcda(epoch) * PCDA')
parser.add_argument('--use-pcda-align', type=int, default=1,
                    help='Enable Positive-Consensus Dynamic Alignment (PCDA)')
parser.add_argument('--lambda-pcda-align', type=float, default=1e-4,
                    help='Weight for Positive-Consensus Dynamic Alignment loss')
parser.add_argument('--pcda-pos-k', type=int, default=6,
                    help='Minimum positive heads needed for each directed PCDA edge')
parser.add_argument('--pcda-eps-sign', type=float, default=1e-8,
                    help='Sign threshold for counting positive dynamic heads in PCDA')
parser.add_argument('--pcda-dyn-only-weight', type=float, default=1.0,
                    help='PCDA edge weight for dynamic-only positive consensus edges')
parser.add_argument('--pcda-static-overlap-weight', type=float, default=0.5,
                    help='PCDA edge weight for positive consensus edges overlapping static graph')
parser.add_argument('--pcda-row-cap', type=int, default=3,
                    help='Maximum PCDA neighbors retained per source stock')
parser.add_argument('--pcda-align-warmup-epoch', type=int, default=5,
                    help='Epochs before enabling PCDA loss')
parser.add_argument('--pcda-align-ramp-epoch', type=int, default=5,
                    help='Epochs used to linearly ramp up lambda_pcda_align after warmup')
parser.add_argument('--pcda-temperature', type=float, default=0.1,
                    help='Temperature for PCDA cross-modal InfoNCE')
parser.add_argument('--temperature', type=float, default=0.1,
                    help='Temperature tau for alignment loss similarity scaling')
parser.add_argument('--data-dir', type=str, default='.',
                    help='Directory containing data files (us_news.csv, us_price.csv, etc.)')
parser.add_argument('--exp-name', type=str, default=None,
                    help='Optional experiment name used to isolate SavedModels and result outputs')
parser.add_argument('--seed', type=int, default=11380,
                    help='Random seed used for training, evaluation, and backtesting')
parser.add_argument('--bert-path', type=str, default='./Tinybert-4layers-en',
                    help='Path to pretrained BERT model directory')
parser.add_argument('--bert-batch-size', type=int, default=32,
                    help='Batch size for BERT inference (8 for 16GB GPU, 32 for 24GB GPU)')
parser.add_argument('--batch-train', type=int, default=16,
                    help='Gradient accumulation steps (15 for 16GB GPU, 16 for 24GB GPU)')


def _format_exp_value(value):
    return str(value).replace(" ", "").replace("/", "_").replace("\\", "_").replace(":", "_")


def get_phase_model_dir(args, phase_id):
    return str(experiment_dir('SavedModels', args.exp_name) / f'phase{phase_id}')


def get_result_dir(args):
    return str(experiment_dir('result', args.exp_name))


def save_runtime_manifest(args, device, phase_id=None):
    result_dir = get_result_dir(args)
    os.makedirs(result_dir, exist_ok=True)
    cuda_available = torch.cuda.is_available()
    manifest = {
        **TEMPORAL_CONFIG,
        "batch_train": int(args.batch_train),
        "tbptt_steps": int(args.batch_train),
        "prediction_context_length": int(args.rnn_length),
        "pcda_scope": "previous_day_to_current_day_only",
        "pcda_enabled": bool(args.use_pcda_align),
        "pcda_heads": int(args.heads_att),
        "pcda_consensus_k": int(args.pcda_pos_k),
        "pcda_warmup_epochs": int(args.pcda_align_warmup_epoch),
        "pcda_ramp_epochs": int(args.pcda_align_ramp_epoch),
        "lambda_pcda": float(args.lambda_pcda_align),
        "base_alignment_direction": "news_to_market",
        "bert_language": "english",
        "bert_layers": 4,
        "bert_hidden_size": 312,
        "bert_vocab_size": 30522,
        "bert_pretrained": True,
        "bert_frozen": True,
        "bert_eval": True,
        "output_activation": "linear",
        "phase": int(phase_id if phase_id is not None else args.phase),
        "device": str(device),
        "cuda": torch.version.cuda,
        "lambda_align": float(args.lambda_align),
        "data_dir": str(args.data_dir),
        "bert_path": str(args.bert_path),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "requested_device": str(device),
        "cuda_available": bool(cuda_available),
        "torch_cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version() if cuda_available else None,
        "gpu_name": (
            torch.cuda.get_device_name(device)
            if cuda_available and str(device).startswith("cuda")
            else None
        ),
        "seed": int(args.seed),
        "exp_name": _format_exp_value(args.exp_name) if args.exp_name else "",
    }
    filename = "runtime_manifest.json" if phase_id is None else f"runtime_manifest_phase{phase_id}.json"
    path = os.path.join(result_dir, filename)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)
    print(f"Runtime manifest saved to {path}")
    return manifest


def save_phase_prediction_outputs(args, phase_id, bt_results):
    """Save the official Full-model predictions and Precision@5 audit files."""
    predictions = bt_results.get('predictions_df')
    true_returns = bt_results.get('true_returns_df')
    precision_daily = bt_results.get('precision_at_5_daily_df')
    if not isinstance(predictions, pd.DataFrame) or predictions.empty:
        raise RuntimeError(f"Phase {phase_id}: no Full-model predictions to save.")
    if not isinstance(true_returns, pd.DataFrame) or true_returns.empty:
        raise RuntimeError(f"Phase {phase_id}: no t+1 targets to save with predictions.")

    result_dir = get_result_dir(args)
    os.makedirs(result_dir, exist_ok=True)
    wide_path = os.path.abspath(os.path.join(result_dir, f'predictions_phase{phase_id}.csv'))
    records_path = os.path.abspath(os.path.join(result_dir, f'prediction_records_phase{phase_id}.csv'))
    precision_path = os.path.abspath(os.path.join(result_dir, f'precision_at_5_daily_phase{phase_id}.csv'))
    canonical_prediction_path = os.path.abspath(os.path.join(result_dir, 'prediction.csv'))
    canonical_metrics_path = os.path.abspath(os.path.join(result_dir, 'metrics.csv'))

    predictions_wide = predictions.copy().sort_index()
    predictions_wide.index.name = 'Date'
    predictions_wide.to_csv(wide_path, encoding='utf-8-sig')

    targets_wide = true_returns.reindex(
        index=predictions_wide.index,
        columns=predictions_wide.columns,
    )
    predictions_long = predictions_wide.reset_index().melt(
        id_vars='Date', var_name='Ticker', value_name='prediction'
    )
    targets_long = targets_wide.reset_index().melt(
        id_vars='Date', var_name='Ticker', value_name='target_return_t_plus_1'
    )
    records = predictions_long.merge(targets_long, on=['Date', 'Ticker'], how='left')
    metric_eligible = records['prediction'].notna() & records['target_return_t_plus_1'].notna()
    records['prediction_rank'] = np.nan
    records.loc[metric_eligible, 'prediction_rank'] = (
        records.loc[metric_eligible]
        .groupby('Date')['prediction']
        .rank(method='first', ascending=False)
    )
    records['precision_metric_eligible'] = metric_eligible
    records['is_top5'] = metric_eligible & (records['prediction_rank'] <= 5)
    records['actual_positive'] = records['target_return_t_plus_1'] > 0
    records.insert(0, 'phase_id', int(phase_id))
    records.insert(0, 'seed', int(args.seed))
    records.to_csv(records_path, index=False, encoding='utf-8-sig')

    canonical_columns = [
        'seed', 'phase_id', 'Date', 'Ticker', 'prediction',
        'target_return_t_plus_1', 'prediction_rank', 'is_top5', 'actual_positive',
    ]
    canonical_records = records[canonical_columns].copy()
    canonical_records['Date'] = pd.to_datetime(canonical_records['Date']).dt.strftime('%Y-%m-%d')
    if os.path.exists(canonical_prediction_path):
        existing = pd.read_csv(canonical_prediction_path)
        if {'seed', 'phase_id'}.issubset(existing.columns):
            keep = ~(
                (pd.to_numeric(existing['seed'], errors='coerce') == int(args.seed))
                & (pd.to_numeric(existing['phase_id'], errors='coerce') == int(phase_id))
            )
            existing = existing.loc[keep]
        canonical_records = pd.concat([existing, canonical_records], ignore_index=True)
    canonical_records = (
        canonical_records.drop_duplicates(
            subset=['seed', 'phase_id', 'Date', 'Ticker'], keep='last'
        )
        .sort_values(['seed', 'phase_id', 'Date', 'Ticker'])
    )
    canonical_records.to_csv(canonical_prediction_path, index=False, encoding='utf-8-sig')

    canonical_metrics = calculate_prediction_metrics_from_records(records)
    metric_sources = {
        'MSE': 'MSE', 'RMSE': 'RMSE', 'MAE': 'MAE', 'IC': 'IC',
        'RankIC': 'RankIC', 'NDCG@5': 'mean_ndcg5',
        'Precision@5': 'precision_at_5', 'MRR': 'mean_mrr',
    }
    for metric_name, result_name in metric_sources.items():
        expected = float(bt_results[result_name])
        actual = float(canonical_metrics[metric_name])
        if not np.isclose(actual, expected, atol=1e-10, rtol=1e-8, equal_nan=True):
            raise RuntimeError(
                f"Phase {phase_id}: canonical {metric_name}={actual} differs from "
                f"backtest {result_name}={expected}."
            )

    metrics_row = {
        'seed': int(args.seed),
        'phase_id': int(phase_id),
        **canonical_metrics,
        'Sharpe': float(bt_results['sharpe_ratio']),
        'AnnualizedReturn': float(bt_results['annualized_return']),
        'MaxDrawdown': float(bt_results['max_drawdown']),
    }
    for key, value in bt_results.items():
        if key == 'annualized_volatility' or key in metrics_row:
            continue
        if isinstance(value, (str, int, float, np.integer, np.floating, bool)):
            metrics_row[key] = value
    metrics_frame = pd.DataFrame([metrics_row])
    if os.path.exists(canonical_metrics_path):
        existing_metrics = pd.read_csv(canonical_metrics_path)
        existing_metrics = existing_metrics.drop(
            columns=['annualized_volatility', 'AnnualizedVolatility'], errors='ignore'
        )
        if {'seed', 'phase_id'}.issubset(existing_metrics.columns):
            keep = ~(
                (pd.to_numeric(existing_metrics['seed'], errors='coerce') == int(args.seed))
                & (pd.to_numeric(existing_metrics['phase_id'], errors='coerce') == int(phase_id))
            )
            existing_metrics = existing_metrics.loc[keep]
        metrics_frame = pd.concat([existing_metrics, metrics_frame], ignore_index=True)
    metrics_frame = (
        metrics_frame.drop_duplicates(subset=['seed', 'phase_id'], keep='last')
        .sort_values(['seed', 'phase_id'])
    )
    metrics_frame.to_csv(canonical_metrics_path, index=False, encoding='utf-8-sig')

    if not isinstance(precision_daily, pd.DataFrame) or precision_daily.empty:
        raise RuntimeError(f"Phase {phase_id}: no valid daily Precision@5 rows.")
    precision_daily = precision_daily.copy()
    precision_daily.insert(0, 'phase_id', phase_id)
    precision_daily.to_csv(precision_path, index=False, encoding='utf-8-sig')

    return {
        'predictions_file': wide_path,
        'prediction_records_file': records_path,
        'precision_at_5_daily_file': precision_path,
        'canonical_prediction_file': canonical_prediction_path,
        'canonical_metrics_file': canonical_metrics_path,
    }


def get_effective_lambda_pcda(args, epoch):
    if not args.use_pcda_align:
        return 0.0

    base = float(args.lambda_pcda_align)
    warmup = max(int(args.pcda_align_warmup_epoch), 0)
    if epoch < warmup:
        return 0.0

    ramp = int(args.pcda_align_ramp_epoch)
    if ramp <= 0:
        return base

    progress = (epoch - warmup + 1) / ramp
    progress = min(max(progress, 0.0), 1.0)
    return base * progress


def collect_current_head_dynamics(model, expected_heads=None, strict=False):
    heads = []
    for att in model.attentions:
        dyn = getattr(att, "prev_dynamic", None)
        if dyn is not None:
            heads.append(dyn.detach())
    if len(heads) == 0:
        return None
    if expected_heads is not None and len(heads) != int(expected_heads):
        msg = f"PCDA warning: expected {expected_heads} heads, got {len(heads)}"
        if strict:
            raise RuntimeError(msg)
        print(msg)
    return torch.stack(heads, dim=0)


def _pcda_empty_stats(num_stock=0):
    return {
        "raw_mutual_edges": 0,
        "edges_after_cap": 0,
        "dyn_only_edges": 0,
        "static_overlap_edges": 0,
        "zero_edge": 1,
        "avg_neighbors_per_stock": 0.0,
        "weight_sum": 0.0,
        "num_stock": int(num_stock),
    }


def build_pcda_weight(prev_head_dynamics, static_edge, pos_k=6, eps=1e-8,
                      dyn_only_weight=1.0, static_overlap_weight=0.5,
                      row_cap=3, static_thresh=0.5):
    if prev_head_dynamics is None:
        return None, _pcda_empty_stats()

    heads = prev_head_dynamics.detach()
    if heads.dim() != 3:
        raise RuntimeError(f"PCDA expects prev_head_dynamics [H,N,N], got {list(heads.shape)}")

    H, N, N2 = heads.shape
    if N != N2:
        raise RuntimeError(f"PCDA expects square head dynamics, got {list(heads.shape)}")

    device = heads.device
    pos_count = (heads > float(eps)).sum(dim=0)
    directed_pos = pos_count >= int(pos_k)
    mutual = directed_pos & directed_pos.t()
    diag = torch.eye(N, dtype=torch.bool, device=device)
    mutual = mutual & (~diag)
    raw_mutual_edges = int(mutual.sum().item())

    kept = torch.zeros_like(mutual)
    cap = int(row_cap)
    if cap <= 0:
        kept = mutual
    else:
        min_count = torch.minimum(pos_count, pos_count.t())
        sum_count = pos_count + pos_count.t()
        for i in range(N):
            candidates = torch.where(mutual[i])[0]
            if candidates.numel() == 0:
                continue
            cand_list = candidates.detach().cpu().tolist()
            cand_list.sort(key=lambda j: (-int(min_count[i, j].item()), -int(sum_count[i, j].item()), int(j)))
            chosen = cand_list[:cap]
            kept[i, torch.tensor(chosen, dtype=torch.long, device=device)] = True

    if static_edge is None:
        static_mask = torch.zeros((N, N), dtype=torch.bool, device=device)
    else:
        static_mask = static_edge.to(device=device)
        if static_mask.dtype != torch.bool:
            static_mask = static_mask > float(static_thresh)
        static_mask = static_mask.clone()
        static_mask.fill_diagonal_(False)

    dyn_only = kept & (~static_mask)
    overlap = kept & static_mask
    weight = torch.zeros((N, N), dtype=heads.dtype, device=device)
    weight[dyn_only] = float(dyn_only_weight)
    weight[overlap] = float(static_overlap_weight)
    weight.fill_diagonal_(0.0)

    edges_after_cap = int((weight > 0).sum().item())
    stats = {
        "raw_mutual_edges": raw_mutual_edges,
        "edges_after_cap": edges_after_cap,
        "dyn_only_edges": int(dyn_only.sum().item()),
        "static_overlap_edges": int(overlap.sum().item()),
        "zero_edge": 1 if edges_after_cap == 0 else 0,
        "avg_neighbors_per_stock": float(edges_after_cap / max(N, 1)),
        "weight_sum": float(weight.sum().item()),
        "num_stock": int(N),
    }
    return weight, stats


def pcda_alignment_loss(x_news, x_market, pcda_weight, temperature=0.1):
    if pcda_weight is None:
        return x_news.new_tensor(0.0)

    weight = pcda_weight.to(device=x_news.device, dtype=x_news.dtype).detach()
    if weight.sum().item() <= 0:
        return x_news.new_tensor(0.0)

    if x_news.dim() == 2:
        x_news = x_news.unsqueeze(0)
    if x_market.dim() == 2:
        x_market = x_market.unsqueeze(0)

    if x_news.size(-1) != x_market.size(-1):
        raise ValueError(
            "PCDA alignment features must have the same hidden dimension, "
            f"got news={tuple(x_news.shape)}, market={tuple(x_market.shape)}"
        )
    if weight.shape != (x_news.size(1), x_news.size(1)):
        raise ValueError(
            "PCDA weight shape must match stock dimension, "
            f"got weight={tuple(weight.shape)}, features={tuple(x_news.shape)}"
        )

    tau = max(float(temperature), 1e-8)
    losses = []
    for step in range(x_news.size(0)):
        valid_news = x_news[step].norm(dim=-1) > 1e-5
        n_t = F.normalize(x_news[step], p=2, dim=-1)
        m_t = F.normalize(x_market[step], p=2, dim=-1)
        direction_losses = []

        # News_i -> Market_j: only the news anchor i must be valid. Market
        # candidates are always observed, so the candidate set is unchanged.
        weight_nm = weight * valid_news.to(weight.dtype).unsqueeze(1)
        weight_nm_sum = weight_nm.sum()
        sim_nm = torch.matmul(n_t, m_t.t()) / tau
        if weight_nm_sum.item() > 0:
            log_nm = F.log_softmax(sim_nm, dim=1)
            loss_nm = -(weight_nm * log_nm).sum() / weight_nm_sum
            direction_losses.append(loss_nm)

        # Market_i -> News_j: only valid news candidates j participate. They
        # are removed from both the positive weights and the softmax candidate
        # set, preventing zero-news vectors from becoming artificial negatives.
        valid_news_index = torch.nonzero(valid_news, as_tuple=False).squeeze(1)
        if valid_news_index.numel() > 0:
            weight_mn = weight.index_select(1, valid_news_index)
            weight_mn_sum = weight_mn.sum()
            if weight_mn_sum.item() > 0:
                sim_mn = torch.matmul(m_t, n_t.index_select(0, valid_news_index).t()) / tau
                log_mn = F.log_softmax(sim_mn, dim=1)
                loss_mn = -(weight_mn * log_mn).sum() / weight_mn_sum
                direction_losses.append(loss_mn)

        if direction_losses:
            losses.append(torch.stack(direction_losses).mean())

    if not losses:
        return x_news.new_tensor(0.0)
    return torch.stack(losses).mean()


def init_pcda_epoch_stats():
    return {
        "loss_sum": 0.0,
        "days": 0,
        "edge_sum": 0.0,
        "dyn_only_sum": 0.0,
        "static_overlap_sum": 0.0,
        "zero_edge_days": 0.0,
        "avg_neighbors_sum": 0.0,
    }


def update_pcda_epoch_stats(acc, pcda_loss, pcda_stats):
    acc["loss_sum"] += float(pcda_loss.item() if torch.is_tensor(pcda_loss) else pcda_loss)
    acc["days"] += 1
    acc["edge_sum"] += float(pcda_stats.get("edges_after_cap", 0))
    acc["dyn_only_sum"] += float(pcda_stats.get("dyn_only_edges", 0))
    acc["static_overlap_sum"] += float(pcda_stats.get("static_overlap_edges", 0))
    acc["zero_edge_days"] += float(pcda_stats.get("zero_edge", 1))
    acc["avg_neighbors_sum"] += float(pcda_stats.get("avg_neighbors_per_stock", 0.0))


def finalize_pcda_epoch_stats(acc, effective_lambda_pcda):
    days = max(int(acc.get("days", 0)), 1)
    return {
        "pcda_loss": acc["loss_sum"] / days,
        "effective_lambda_pcda": float(effective_lambda_pcda),
        "pcda_edges_per_day": acc["edge_sum"] / days,
        "pcda_dyn_only_edges_per_day": acc["dyn_only_sum"] / days,
        "pcda_static_overlap_edges_per_day": acc["static_overlap_sum"] / days,
        "pcda_zero_edge_day_ratio": acc["zero_edge_days"] / days,
        "pcda_avg_neighbors_per_stock": acc["avg_neighbors_sum"] / days,
    }


def load_dataset(news_file, market_file, labels_file):
    # 新闻数据
    news_df = pd.read_csv(news_file, encoding='utf-8')
    news_df['Date'] = pd.to_datetime(news_df['Date'])
    news_df = news_df.sort_values(by='Date')
    news_df = news_df.dropna()
    news_data = news_df[['Ticker', 'Date', 'Headlines']]

    # 市场数据：归一化后的5维
    market_df = pd.read_csv(market_file, encoding='utf-8')
    market_df['Date'] = pd.to_datetime(market_df['Date'])
    market_df = market_df.sort_values(by='Date')
    market_df = market_df.dropna()
    market_data = market_df[['Ticker', 'Date', 'Open_norm', 'High_norm', 'Low_norm', 'Close_norm', 'Volume_norm']]

    # 标签数据
    labels_df = pd.read_csv(labels_file, encoding='utf-8')
    labels_df['Date'] = pd.to_datetime(labels_df['Date'])
    labels_df = labels_df.sort_values(by='Date')
    labels_df = labels_df.dropna()
    labels_data = labels_df[['Ticker', 'Date', 'target']]
    labels_data, removed_labels = filter_strict_next_day_labels(labels_data, market_data)
    if not removed_labels.empty:
        print(
            f"  Strict t->t+1 audit: masked {len(removed_labels)} labels whose ticker "
            "is absent on the next global market date."
        )

    return news_data, market_data, labels_data


def split_by_dates(news_data, market_data, labels_data, train_dates, val_dates, test_dates):
    """根据日期范围分割数据"""
    train_d = pd.Series(train_dates)
    val_d = pd.Series(val_dates)
    test_d = pd.Series(test_dates)

    x_train = market_data[market_data['Date'].isin(train_d)].sort_values(by=['Date', 'Ticker'])
    x_val = market_data[market_data['Date'].isin(val_d)].sort_values(by=['Date', 'Ticker'])
    x_test = market_data[market_data['Date'].isin(test_d)].sort_values(by=['Date', 'Ticker'])

    y_train = labels_data[labels_data['Date'].isin(train_d)].sort_values(by=['Date', 'Ticker'])
    y_val = labels_data[labels_data['Date'].isin(val_d)].sort_values(by=['Date', 'Ticker'])
    y_test = labels_data[labels_data['Date'].isin(test_d)].sort_values(by=['Date', 'Ticker'])

    x_sentiment_train = news_data[news_data['Date'].isin(train_d)].sort_values(by=['Date', 'Ticker'])
    x_sentiment_val = news_data[news_data['Date'].isin(val_d)].sort_values(by=['Date', 'Ticker'])
    x_sentiment_test = news_data[news_data['Date'].isin(test_d)].sort_values(by=['Date', 'Ticker'])

    return (x_train, x_val, x_test,
            x_sentiment_train, x_sentiment_val, x_sentiment_test,
            y_train, y_val, y_test)


def generate_sequences(df, dates, rnn_length, tickers, lookback_df=None, lookback_dates=None,
                       prediction_dates=None):
    """生成序列，支持 lookback_df 跨 split 边界借用历史窗口
    
    Args:
        df: 当前 split 的数据
        dates: 当前 split 的日期列表
        rnn_length: 序列长度
        tickers: 股票列表
        lookback_df: 前一个 split 的数据（用于跨边界 lookback）
        lookback_dates: 前一个 split 的日期列表
    """
    sequences = []
    current_dates = {pd.Timestamp(date) for date in dates}
    allowed_prediction_dates = (
        current_dates
        if prediction_dates is None
        else {pd.Timestamp(date) for date in prediction_dates}
    )
    # 构建完整的历史上下文（前split尾部 + 当前split）
    if lookback_df is not None and lookback_dates is not None:
        # 只保留 lookback 的最后 rnn_length-1 天
        lb_dates = list(lookback_dates[-(rnn_length - 1):])
        combined_df = pd.concat([lookback_df, df], ignore_index=True)
        combined_dates = lb_dates + list(dates)
    else:
        combined_df = df
        combined_dates = list(dates)
    
    for i in range(rnn_length, len(combined_dates) + 1):
        window_dates = combined_dates[i - rnn_length:i]
        window_df = combined_df[combined_df["Date"].isin(window_dates)].copy()
        # 只产出标签落在当前 split 内的序列
        last_date_in_window = window_dates[-1]
        if pd.Timestamp(last_date_in_window) not in current_dates:
            continue
        if pd.Timestamp(last_date_in_window) not in allowed_prediction_dates:
            continue
        has_all_dates = set(pd.to_datetime(window_df["Date"])) == set(pd.to_datetime(window_dates))
        has_all_tickers = set(window_df["Ticker"].unique()) == set(tickers)
        has_duplicate_pairs = window_df.duplicated(["Date", "Ticker"]).any()
        if has_all_dates and has_all_tickers and not has_duplicate_pairs:
            sequences.append(window_df.sort_values(["Date", "Ticker"]))
    return sequences


def target_tensor_from_sequence(target_df, tickers):
    """Build a ticker-aligned target vector for the sequence's final date."""
    last_date = pd.Timestamp(target_df["Date"].max())
    final_rows = target_df[target_df["Date"] == last_date]
    if final_rows["Ticker"].duplicated().any():
        raise RuntimeError(f"Duplicate target rows on {last_date.date()}.")
    values = final_rows.set_index("Ticker")["target"].reindex(tickers).to_numpy(dtype=np.float64)
    return torch.tensor(values, dtype=torch.float64)


def precompute_all_sequences(market_train, market_val, market_test,
                             news_train, news_val, news_test,
                             label_train, label_val, label_test,
                             train_dates, val_dates, test_dates, all_market_dates,
                             rnn_length, tickers):
    """预计算所有train/val/test的序列，支持跨split边界lookback"""
    print("    预计算序列...")
    strict_dates = {
        'train': get_strict_prediction_dates(train_dates, all_market_dates),
        'val': get_strict_prediction_dates(val_dates, all_market_dates),
        'test': get_strict_prediction_dates(test_dates, all_market_dates),
    }
    for split_name, split_dates in (
        ('train', train_dates), ('val', val_dates), ('test', test_dates)
    ):
        assert_strict_prediction_dates(
            strict_dates[split_name], split_dates, all_market_dates, split_name
        )

    m_train = generate_sequences(
        market_train, train_dates, rnn_length, tickers,
        prediction_dates=strict_dates['train']
    )
    # val 的 lookback 是 train 尾部
    m_val = generate_sequences(market_val, val_dates, rnn_length, tickers,
                               lookback_df=market_train, lookback_dates=train_dates,
                               prediction_dates=strict_dates['val'])
    # test 的 lookback 是 val 尾部
    m_test = generate_sequences(market_test, test_dates, rnn_length, tickers,
                                 lookback_df=market_val, lookback_dates=val_dates,
                                 prediction_dates=strict_dates['test'])
    print(f"    Train序列: {len(m_train)}, Val: {len(m_val)}, Test: {len(m_test)}")
    sequences = {
        'market_train': m_train, 'market_val': m_val, 'market_test': m_test,
        'news_train': generate_sequences(
            news_train, train_dates, rnn_length, tickers,
            prediction_dates=strict_dates['train']
        ),
        'news_val': generate_sequences(news_val, val_dates, rnn_length, tickers,
                                       lookback_df=news_train, lookback_dates=train_dates,
                                       prediction_dates=strict_dates['val']),
        'news_test': generate_sequences(news_test, test_dates, rnn_length, tickers,
                                       lookback_df=news_val, lookback_dates=val_dates,
                                       prediction_dates=strict_dates['test']),
        'label_train': generate_sequences(
            label_train, train_dates, rnn_length, tickers,
            prediction_dates=strict_dates['train']
        ),
        'label_val': generate_sequences(label_val, val_dates, rnn_length, tickers,
                                       lookback_df=label_train, lookback_dates=train_dates,
                                       prediction_dates=strict_dates['val']),
        'label_test': generate_sequences(label_test, test_dates, rnn_length, tickers,
                                        lookback_df=label_val, lookback_dates=val_dates,
                                        prediction_dates=strict_dates['test']),
    }
    for split_name in ('train', 'val', 'test'):
        end_dates = {}
        for modality in ('market', 'news', 'label'):
            end_dates[modality] = [pd.Timestamp(frame['Date'].max()) for frame in sequences[f'{modality}_{split_name}']]
        if not (end_dates['market'] == end_dates['news'] == end_dates['label']):
            raise RuntimeError(f"{split_name} market/news/label sequence dates are not aligned.")
        assert_strict_prediction_dates(
            end_dates['market'],
            {'train': train_dates, 'val': val_dates, 'test': test_dates}[split_name],
            all_market_dates,
            split_name,
            require_all=False,
        )
    return sequences


def train(model, market_seqs, news_seqs, label_seqs, tickers, relation_static, S_pre=None, precomputed_targets=None,
          market_tensors=None, news_tensors=None, epoch=0):

    model.train()
    model.clear_dynamic_state()  # Reset recurrent state at each training epoch.
    optimizer.zero_grad()
    effective_lambda_pcda = get_effective_lambda_pcda(args, epoch)

    total_loss = 0
    total_sim_loss = 0
    pcda_epoch_stats = init_pcda_epoch_stats()
    batch_train = args.batch_train
    if batch_train < 1:
        raise ValueError('batch_train must be positive.')
    chunk_losses = []
    total_loss_count = 0
    
    
    prev_head_dynamics = None
    previous_relation_date = None
    pcda_debug_printed = 0

    for i in tqdm(range(len(market_seqs)), desc="Training"):
        # Use precomputed tensors when supplied; otherwise encode DataFrames.
        if market_tensors is not None and news_tensors is not None:
            market_input = market_tensors[i]
            news_input = news_tensors[i]
        else:
            market_input = market_seqs[i]
            news_input = news_seqs[i]
        
        if precomputed_targets is not None:
            y = precomputed_targets[i].to(DEVICE)
        else:
            target_df = label_seqs[i]
            y = target_tensor_from_sequence(target_df, tickers).to(DEVICE)

        pcda_weight = None
        pcda_stats = _pcda_empty_stats(len(tickers))
        if args.use_pcda_align and prev_head_dynamics is not None:
            pcda_weight, pcda_stats = build_pcda_weight(
                prev_head_dynamics=prev_head_dynamics,
                static_edge=relation_static,
                pos_k=args.pcda_pos_k,
                eps=args.pcda_eps_sign,
                dyn_only_weight=args.pcda_dyn_only_weight,
                static_overlap_weight=args.pcda_static_overlap_weight,
                row_cap=args.pcda_row_cap,
                static_thresh=args.static_thresh,
            )
        
        window_dates = sorted(market_seqs[i]['Date'].unique())
        output, sim_loss, x_r, x_news_align, x_market_align, current_dynamic = model(
            news_input, market_input, relation_static, S_pre,
            window_dates=window_dates,
        )
        output_f64 = output.view(-1).to(torch.float64)
        y_flat = y.view(-1)
        valid_targets = torch.isfinite(y_flat)
        if not torch.any(valid_targets):
            raise RuntimeError("A training window has no valid strict t->t+1 targets.")
        mse_loss = criterion(output_f64[valid_targets], y_flat[valid_targets])
        if args.use_pcda_align and pcda_weight is not None and pcda_stats.get("edges_after_cap", 0) > 0:
            pcda_loss = pcda_alignment_loss(
                x_news=x_news_align[-1:],
                x_market=x_market_align[-1:],
                pcda_weight=pcda_weight,
                temperature=args.pcda_temperature,
            )
        else:
            pcda_loss = output_f64.new_tensor(0.0)

        total_loss_val = (
            mse_loss
            + args.lambda_align * sim_loss
            + effective_lambda_pcda * pcda_loss
        )
        # Shared daily hidden graphs require one backward per accumulation chunk.
        chunk_losses.append(total_loss_val)
        model.record_pcda_debug(previous_relation_date, prev_head_dynamics,
                                effective_lambda_pcda, pcda_stats.get('edges_after_cap', 0),
                                args.use_pcda_align)

        total_loss += mse_loss.item()
        total_sim_loss += sim_loss.item() if torch.is_tensor(sim_loss) else sim_loss
        update_pcda_epoch_stats(pcda_epoch_stats, pcda_loss, pcda_stats)
        total_loss_count += 1

        if args.use_pcda_align and epoch == 0 and pcda_debug_printed < 3 and prev_head_dynamics is not None:
            print(
                "\nPCDA sanity check:\n"
                f"  prev_head_dynamics shape = {list(prev_head_dynamics.shape)}\n"
                f"  pos_k = {args.pcda_pos_k}\n"
                f"  eps = {args.pcda_eps_sign}\n"
                f"  raw mutual edges before cap = {pcda_stats.get('raw_mutual_edges', 0)}\n"
                f"  edges after row cap = {pcda_stats.get('edges_after_cap', 0)}\n"
                f"  dynamic-only edges = {pcda_stats.get('dyn_only_edges', 0)}\n"
                f"  dynamic∩static edges = {pcda_stats.get('static_overlap_edges', 0)}\n"
                f"  weight sum = {pcda_stats.get('weight_sum', 0.0):.4f}\n"
                f"  x_news_align current-day shape = {list(x_news_align[-1:].shape)}\n"
                f"  x_market_align current-day shape = {list(x_market_align[-1:].shape)}\n"
                f"  pcda_loss = {float(pcda_loss.item() if torch.is_tensor(pcda_loss) else pcda_loss):.6f}"
            )
            pcda_debug_printed += 1
    
        with torch.no_grad():
            current_head_dynamics = collect_current_head_dynamics(
                model, expected_heads=args.heads_att, strict=True
            )
            prev_head_dynamics = None if current_head_dynamics is None else current_head_dynamics.detach()
            previous_relation_date = None if current_head_dynamics is None else pd.Timestamp(window_dates[-1])
        
        if len(chunk_losses) == batch_train or i == len(market_seqs) - 1:
            torch.stack(chunk_losses).mean().backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
            optimizer.step()
            optimizer.zero_grad()
            model.detach_relation_state()
            chunk_losses.clear()
    return (
        total_loss / total_loss_count,
        total_sim_loss / total_loss_count,
        finalize_pcda_epoch_stats(pcda_epoch_stats, effective_lambda_pcda),
    )

def evaluate(model, market_seqs, news_seqs, label_seqs, tickers, relation_static, S_pre=None, precomputed_targets=None,
             market_tensors=None, news_tensors=None, epoch=0):

    model.eval()
    model.clear_dynamic_state()  # Reset recurrent state at each evaluation pass.

    preds = []
    trues = []
    
  
    prev_head_dynamics = None
    pcda_epoch_stats = init_pcda_epoch_stats()
    previous_relation_date = None
    effective_lambda_pcda = get_effective_lambda_pcda(args, epoch)

    with torch.no_grad():
        for i in tqdm(range(len(market_seqs)), desc="Evaluating"):
            # Use precomputed tensors when supplied; otherwise encode DataFrames.
            if market_tensors is not None and news_tensors is not None:
                market_input = market_tensors[i]
                news_input = news_tensors[i]
            else:
                market_input = market_seqs[i]
                news_input = news_seqs[i]

            pcda_weight = None
            pcda_stats = _pcda_empty_stats(len(tickers))
            if args.use_pcda_align and prev_head_dynamics is not None:
                pcda_weight, pcda_stats = build_pcda_weight(
                    prev_head_dynamics=prev_head_dynamics,
                    static_edge=relation_static,
                    pos_k=args.pcda_pos_k,
                    eps=args.pcda_eps_sign,
                    dyn_only_weight=args.pcda_dyn_only_weight,
                    static_overlap_weight=args.pcda_static_overlap_weight,
                    row_cap=args.pcda_row_cap,
                    static_thresh=args.static_thresh,
                )
            
            window_dates = sorted(market_seqs[i]['Date'].unique())
            output, _, x_r, x_news_align, x_market_align, current_dynamic = model(
                news_input, market_input, relation_static, S_pre,
                window_dates=window_dates,
            )

            if args.use_pcda_align and pcda_weight is not None and pcda_stats.get("edges_after_cap", 0) > 0:
                pcda_loss = pcda_alignment_loss(
                    x_news=x_news_align[-1:],
                    x_market=x_market_align[-1:],
                    pcda_weight=pcda_weight,
                    temperature=args.pcda_temperature,
                )
            else:
                pcda_loss = output.view(-1).new_tensor(0.0)
            update_pcda_epoch_stats(pcda_epoch_stats, pcda_loss, pcda_stats)
            model.record_pcda_debug(previous_relation_date, prev_head_dynamics,
                                    effective_lambda_pcda, pcda_stats.get('edges_after_cap', 0),
                                    args.use_pcda_align)
            
            with torch.no_grad():
                current_head_dynamics = collect_current_head_dynamics(
                    model, expected_heads=args.heads_att, strict=True
                )
                prev_head_dynamics = None if current_head_dynamics is None else current_head_dynamics.detach()
                previous_relation_date = None if current_head_dynamics is None else pd.Timestamp(window_dates[-1])
            
            if precomputed_targets is not None:
                y = precomputed_targets[i].to(DEVICE)
            else:
                target_df = label_seqs[i]
                y = target_tensor_from_sequence(target_df, tickers).to(DEVICE)

            preds.append(output.view(-1).cpu().numpy())
            trues.append(y.view(-1).cpu().numpy())

    RMSE, MAE = metrics(trues, preds)

    # 计算日度横截面 RankIC（每个日期独立做股票间 Spearman，再取平均）
    # 这比全样本扁平化 Spearman 更贴近实际选股排序能力
    daily_ics = []
    for i in range(len(preds)):
        p = preds[i]  # shape: (n_stocks,)
        t = trues[i]  # shape: (n_stocks,)
        valid = np.isfinite(p) & np.isfinite(t)
        p = p[valid]
        t = t[valid]
        if len(p) > 1 and np.std(p) > 0 and np.std(t) > 0:
            rho, _ = spearmanr(p, t)
            if not np.isnan(rho):
                daily_ics.append(rho)

    if daily_ics:
        val_ic = np.mean(daily_ics)
    else:
        val_ic = 0.0

    return RMSE, MAE, val_ic, finalize_pcda_epoch_stats(pcda_epoch_stats, effective_lambda_pcda)


def run_phase(phase, news_data, market_data, labels_data,
              tickers, all_dates, DEVICE, args, skip_backtest=False):
    """
    运行单个phase的训练+验证+测试+回测

    Returns:
        dict: 包含 test_RMSE, test_MAE, 以及回测指标(sharpe_ratio, annualized_return等)
    """
    phase_id = phase['phase_id']
    save_runtime_manifest(args, DEVICE, phase_id=phase_id)
    print(f"\n{'='*70}")
    print(f"Phase {phase_id}")
    print(f"  Train: {phase['train_start'].strftime('%Y-%m-%d')} ~ {phase['train_end'].strftime('%Y-%m-%d')} ({len(phase['train_dates'])} days)")
    print(f"  Val:   {phase['val_start'].strftime('%Y-%m-%d')} ~ {phase['val_end'].strftime('%Y-%m-%d')} ({len(phase['val_dates'])} days)")
    print(f"  Test:  {phase['test_start'].strftime('%Y-%m-%d')} ~ {phase['test_end'].strftime('%Y-%m-%d')} ({len(phase['test_dates'])} days)")
    if args.exp_name:
        print(f"  Experiment name: {_format_exp_value(args.exp_name)}")
        print(f"  Model dir: {get_phase_model_dir(args, phase_id)}")
        print(f"  Result dir: {get_result_dir(args)}")
    print(f"  Seed: {args.seed}")
    print(f"  lambda_align={args.lambda_align}")
    print(f"  PCDA: use={args.use_pcda_align}, lambda={args.lambda_pcda_align}, "
          f"pos_k={args.pcda_pos_k}, eps={args.pcda_eps_sign}, row_cap={args.pcda_row_cap}, "
          f"dyn_only_w={args.pcda_dyn_only_weight}, overlap_w={args.pcda_static_overlap_weight}, "
          f"warmup={args.pcda_align_warmup_epoch}, ramp={args.pcda_align_ramp_epoch}, "
          f"temperature={args.pcda_temperature}")
    print(f"{'='*70}")

    # ===== 用训练期数据构建phase级别的静态关系矩阵（防止未来信息泄露）=====
    train_start_str = phase['train_start'].strftime('%Y-%m-%d')
    train_end_str = phase['train_end'].strftime('%Y-%m-%d')
    print(f"  构建Phase级静态关系矩阵（仅训练期: {train_start_str} ~ {train_end_str}）...")
    phase_relation_matrix, phase_tickers = build_static_relation(
        market_data, threshold=args.static_thresh,
        start_date=phase['train_start'], end_date=phase['train_end'],
        mode=getattr(args, 'static_mode', 'level'),
        raw_price_path=os.path.join(args.data_dir, 'us_price.csv')
    )
    if list(phase_tickers) != list(tickers):
        raise RuntimeError("Static-relation ticker order differs from the model ticker order.")
    # 确保矩阵维度与tickers对齐
    phase_relation_static = torch.tensor(phase_relation_matrix).to(torch.float32).to(DEVICE)
    phase_S_pre = phase_relation_static.clone().to(DEVICE)

    # 每个phase重新设置随机种子，确保跨phase可复现
    set_seed(args.seed)

    # 分割数据
    (x_train, x_val, x_test,
     x_sentiment_train, x_sentiment_val, x_sentiment_test,
     y_train, y_val, y_test) = split_by_dates(
        news_data, market_data, labels_data,
        phase['train_dates'].tolist(),
        phase['val_dates'].tolist(),
        phase['test_dates'].tolist()
    )

    print(f"  Train samples: {len(x_train)}, Val: {len(x_val)}, Test: {len(x_test)}")

    # 初始化模型
    model = DRFN(
        pretrained_model_path=pretrained_model_path,
        dates=phase['train_dates'].tolist(),
        tickers=tickers,
        num_stock=NUM_STOCK,
        hidden_news=args.hidn_news,
        hidden_market=args.hidn_market,
        d_hidden=64,
        hidn_rnn=args.hidn_rnn,
        heads_att=args.heads_att,
        hidn_att=args.hidn_att,
        dropout=args.dropout,
        alpha=args.alpha,
        static_thresh=args.static_thresh,
        temperature=args.temperature,
        bert_batch_size=args.bert_batch_size
    )

    if DEVICE == "cpu":
        model.cpu()
    else:
        model.cuda(device=DEVICE)
    model.to(torch.float32)
    model.debug_daily_relation = args.debug_daily_relation

    # 预计算BERT特征（BERT权重冻结，只需算一次）
    print(f"  预计算BERT特征...")
    # 合并train+val+test的新闻用于预计算（覆盖所有可能用到的日期）
    phase_all_news = pd.concat([x_sentiment_train, x_sentiment_val, x_sentiment_test]).drop_duplicates()
    model.news_encoder.precompute_bert_cls(phase_all_news)
    print(f"  BERT预计算完成，训练时将跳过BERT推理")

    global optimizer, criterion
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_constraint)

    # 预计算所有序列（避免每个epoch重复调用generate_sequences）
    seqs = precompute_all_sequences(
        x_train, x_val, x_test,
        x_sentiment_train, x_sentiment_val, x_sentiment_test,
        y_train, y_val, y_test,
        phase['train_dates'], phase['val_dates'], phase['test_dates'],
        sorted(market_data['Date'].unique()),
        rnn_length, tickers
    )

    # 预计算target tensor（避免每个epoch每个序列都重复sort+torch.tensor）
    print("    预计算target tensor...")
    for split_name in ['train', 'val', 'test']:
        label_seqs = seqs[f'label_{split_name}']
        precomputed_targets = []
        for target_df in label_seqs:
            y_tensor = target_tensor_from_sequence(target_df, tickers)
            precomputed_targets.append(y_tensor)
        seqs[f'target_tensor_{split_name}'] = precomputed_targets
    print(f"    target tensor预计算完成")

    # Precompute input tensors in chronological order.
    print("    [Precompute] 开始预计算 Market 和 News Tensor...")
    for split_name in ['train', 'val', 'test']:
        seqs[f'market_tensor_{split_name}'] = [model.market_encoder.preprocess(df) for df in seqs[f'market_{split_name}']]
        # Strategy B 2.0: preprocess 返回 (step_inputs, T, S) 元组，不再预跑 GRU
        seqs[f'news_tensor_{split_name}'] = [model.news_encoder.preprocess(df) for df in seqs[f'news_{split_name}']]
    print("    [Precompute] Market/News Tensor预计算完成")
    # ======================================================

    # 训练
    MAX_EPOCH = args.max_epoch
    best_model_file = 0
    start_epoch = 0
    eval_epoch_best = float('inf')  # 用Val MAE做早停（越小越好）
    wait_epoch = 0  # 早停计数器初始化
    train_losses = []
    eval_losses = []
    epoch_log_rows = []
    last_train_pcda_stats = finalize_pcda_epoch_stats(init_pcda_epoch_stats(), 0.0)
    last_eval_pcda_stats = finalize_pcda_epoch_stats(init_pcda_epoch_stats(), 0.0)
    checkpoint_start_epoch = 0
    best_epoch = None
    best_train_pcda_stats = None
    best_eval_pcda_stats = None
    print(f"  Official checkpoint selection starts at epoch {checkpoint_start_epoch}.")

    for epoch in range(start_epoch, MAX_EPOCH):
        train_loss, train_sim, train_pcda_stats = train(
            model, seqs['market_train'], seqs['news_train'], seqs['label_train'],
            tickers, phase_relation_static, phase_S_pre,
            precomputed_targets=seqs['target_tensor_train'],
            market_tensors=seqs.get('market_tensor_train'),
            news_tensors=seqs.get('news_tensor_train'),
            epoch=epoch,
        )
        eval_RMSE, eval_MAE, eval_IC, eval_pcda_stats = evaluate(
            model, seqs['market_val'], seqs['news_val'], seqs['label_val'],
            tickers, phase_relation_static, phase_S_pre,
            precomputed_targets=seqs['target_tensor_val'],
            market_tensors=seqs.get('market_tensor_val'),
            news_tensors=seqs.get('news_tensor_val'),
            epoch=epoch,
        )
        last_train_pcda_stats = train_pcda_stats
        last_eval_pcda_stats = eval_pcda_stats

        eval_str = (
            f"Phase{phase_id} epoch{epoch}, "
            f"MSE{train_loss:.6f}, "
            f"sim{train_sim:.2f}, "
            f"pcda{train_pcda_stats['pcda_loss']:.6f}, "
            f"eff_lambda_pcda{train_pcda_stats['effective_lambda_pcda']:.8f}, "
            f"pcda_edges_per_day{train_pcda_stats['pcda_edges_per_day']:.2f}, "
            f"pcda_dyn_only_edges_per_day{train_pcda_stats['pcda_dyn_only_edges_per_day']:.2f}, "
            f"pcda_static_overlap_edges_per_day{train_pcda_stats['pcda_static_overlap_edges_per_day']:.2f}, "
            f"pcda_zero_edge_day_ratio{train_pcda_stats['pcda_zero_edge_day_ratio']:.4f}, "
            f"pcda_avg_nbr{train_pcda_stats['pcda_avg_neighbors_per_stock']:.4f}, "
            f"val_pcda{eval_pcda_stats['pcda_loss']:.6f}, "
            f"val_pcda_edges_per_day{eval_pcda_stats['pcda_edges_per_day']:.2f}, "
            f"val_RMSE{eval_RMSE:.4f}, "
            f"val_MAE{eval_MAE:.4f}, "
            f"val_IC{eval_IC:.4f}"
        )
        print(eval_str)
        checkpoint_eligible = True
        epoch_log_rows.append({
            'exp_name': _format_exp_value(args.exp_name) if args.exp_name else '',
            'phase_id': phase_id,
            'epoch': epoch,
            'train_mse_loss': train_loss,
            'train_static_align_loss': train_sim,
            'train_pcda_loss': train_pcda_stats['pcda_loss'],
            'effective_lambda_pcda': train_pcda_stats['effective_lambda_pcda'],
            'train_pcda_edges_per_day': train_pcda_stats['pcda_edges_per_day'],
            'train_pcda_dyn_only_edges_per_day': train_pcda_stats['pcda_dyn_only_edges_per_day'],
            'train_pcda_static_overlap_edges_per_day': train_pcda_stats['pcda_static_overlap_edges_per_day'],
            'train_pcda_zero_edge_day_ratio': train_pcda_stats['pcda_zero_edge_day_ratio'],
            'train_pcda_avg_neighbors_per_stock': train_pcda_stats['pcda_avg_neighbors_per_stock'],
            'val_pcda_loss': eval_pcda_stats['pcda_loss'],
            'val_pcda_edges_per_day': eval_pcda_stats['pcda_edges_per_day'],
            'val_pcda_dyn_only_edges_per_day': eval_pcda_stats['pcda_dyn_only_edges_per_day'],
            'val_pcda_static_overlap_edges_per_day': eval_pcda_stats['pcda_static_overlap_edges_per_day'],
            'val_pcda_zero_edge_day_ratio': eval_pcda_stats['pcda_zero_edge_day_ratio'],
            'val_pcda_avg_neighbors_per_stock': eval_pcda_stats['pcda_avg_neighbors_per_stock'],
            'val_RMSE': eval_RMSE,
            'val_MAE': eval_MAE,
            'val_IC': eval_IC,
            'checkpoint_eligible': int(checkpoint_eligible),
        })

        train_losses.append(train_loss)
        eval_losses.append(eval_MAE)

        # 用Val MAE做早停：MAE越低，预测精度越高
        if eval_MAE < eval_epoch_best:
            eval_epoch_best = eval_MAE
            eval_best_str = eval_str
            wait_epoch = 0
            best_epoch = epoch
            best_train_pcda_stats = dict(train_pcda_stats)
            best_eval_pcda_stats = dict(eval_pcda_stats)
            if args.save:
                phase_dir = get_phase_model_dir(args, phase_id)
                os.makedirs(phase_dir, exist_ok=True)
                if best_model_file and os.path.exists(best_model_file):
                    os.remove(best_model_file)
                best_model_file = f"{phase_dir}/best_model_valMAE{eval_MAE:.4f}.pth"
                torch.save(model.state_dict(), best_model_file)
        else:
            wait_epoch += 1

        if wait_epoch >= args.wait_epoch:
            print(f"Phase {phase_id} early stop at epoch {epoch}")
            break

    if epoch_log_rows:
        log_dir = get_result_dir(args)
        os.makedirs(log_dir, exist_ok=True)
        log_path = os.path.join(log_dir, f"training_log_phase{phase_id}.csv")
        pd.DataFrame(epoch_log_rows).to_csv(log_path, index=False)
        print(f"Phase {phase_id} training log saved to {log_path}")

    if best_model_file and best_epoch is not None:
        manifest_path = os.path.join(get_result_dir(args), f"best_checkpoint_phase{phase_id}.json")
        with open(manifest_path, "w", encoding="utf-8") as handle:
            json.dump({
                **TEMPORAL_CONFIG,
                "phase_id": phase_id,
                "best_epoch": int(best_epoch),
                "best_val_mae": float(eval_epoch_best),
                "checkpoint_selection_start_epoch": int(checkpoint_start_epoch),
                "checkpoint_file": os.path.abspath(best_model_file),
                "seed": int(args.seed),
                "exp_name": _format_exp_value(args.exp_name) if args.exp_name else "",
                "effective_lambda_pcda": float(get_effective_lambda_pcda(args, best_epoch)),
            }, handle, indent=2)
        print(f"Phase {phase_id} best-checkpoint manifest saved to {manifest_path}")

    if best_train_pcda_stats is not None:
        last_train_pcda_stats = best_train_pcda_stats
    if best_eval_pcda_stats is not None:
        last_eval_pcda_stats = best_eval_pcda_stats

    # 加载最优模型，在测试集上评估 RMSE/MAE
    test_RMSE = float('nan')
    test_MAE = float('nan')
    test_RankIC = float('nan')
    test_pcda_stats = finalize_pcda_epoch_stats(init_pcda_epoch_stats(), 0.0)
    
    if best_model_file and os.path.exists(best_model_file):
        state_dict = torch.load(best_model_file, map_location=DEVICE)
        model.load_state_dict(state_dict)
        test_RMSE, test_MAE, test_RankIC, test_pcda_stats = evaluate(
            model, seqs['market_test'], seqs['news_test'], seqs['label_test'],
            tickers, phase_relation_static, phase_S_pre,
            precomputed_targets=seqs['target_tensor_test'],
            market_tensors=seqs.get('market_tensor_test'),
            news_tensors=seqs.get('news_tensor_test'),
            epoch=best_epoch,
        )
        print(f"\nPhase {phase_id} Test RMSE={test_RMSE:.4f}, MAE={test_MAE:.4f}, RankIC={test_RankIC:.4f}")
    else:
        print(f"\nPhase {phase_id}: No best model saved, skipping test eval")

    # ===== Backtest and explicit relation memory / Fusion analysis =====
    result = {
        'exp_name': _format_exp_value(args.exp_name) if args.exp_name else '',
        'seed': args.seed,
        'phase_id': phase_id,
        'best_epoch': best_epoch,
        'best_val_MAE': eval_epoch_best if best_epoch is not None else float('nan'),
        'checkpoint_selection_start_epoch': checkpoint_start_epoch,
        'test_RMSE': test_RMSE,
        'test_MAE': test_MAE,
        'test_RankIC': test_RankIC,
        'sharpe_ratio': float('nan'),
        'annualized_return': float('nan'),
        'annualized_volatility': float('nan'),
        'avg_daily_return': float('nan'),
        'max_drawdown': float('nan'),
        'mean_mrr': float('nan'),
        'mean_ndcg5': float('nan'),
        'precision_at_5': float('nan'),
        'precision_at_5_valid_dates': 0,
        'predictions_file': '',
        'prediction_records_file': '',
        'precision_at_5_daily_file': '',
        'canonical_prediction_file': '',
        'canonical_metrics_file': '',
        'MSE': float('nan'),
        'RMSE': float('nan'),
        'MAE': float('nan'),
        'IC': float('nan'),
        'RankIC': float('nan'),
        'RIC': float('nan'),
        'total_return': float('nan'),
        'final_value': float('nan'),
        'n_trade_days': 0,
        'n_trades': 0,
        'daily_returns': np.array([]),
        'portfolio_values': [],
        'dates': [],
        # Explicit Relation Memory Intervention
        'memory_full_sharpe': float('nan'),
        'memory_full_annualized_return': float('nan'),
        'memory_full_total_return': float('nan'),
        'memory_full_max_drawdown': float('nan'),
        'memory_full_final_value': float('nan'),
        'memory_no_memory_sharpe': float('nan'),
        'memory_no_memory_annualized_return': float('nan'),
        'memory_no_memory_total_return': float('nan'),
        'memory_no_memory_max_drawdown': float('nan'),
        'memory_no_memory_final_value': float('nan'),
        'memory_shuffled_sharpe': float('nan'),
        'memory_shuffled_annualized_return': float('nan'),
        'memory_shuffled_total_return': float('nan'),
        'memory_shuffled_max_drawdown': float('nan'),
        'memory_shuffled_final_value': float('nan'),
        'memory_no_memory_drop_sharpe': float('nan'),
        'memory_shuffled_drop_sharpe': float('nan'),
        'memory_no_memory_drop_total_return': float('nan'),
        'memory_shuffled_drop_total_return': float('nan'),
        'pcda_use': args.use_pcda_align,
        'pcda_lambda': args.lambda_pcda_align,
        'pcda_pos_k': args.pcda_pos_k,
        'pcda_row_cap': args.pcda_row_cap,
        'pcda_dyn_only_weight': args.pcda_dyn_only_weight,
        'pcda_static_overlap_weight': args.pcda_static_overlap_weight,
        'train_pcda_loss': last_train_pcda_stats.get('pcda_loss', float('nan')),
        'train_pcda_edges_per_day': last_train_pcda_stats.get('pcda_edges_per_day', float('nan')),
        'train_pcda_dyn_only_edges_per_day': last_train_pcda_stats.get('pcda_dyn_only_edges_per_day', float('nan')),
        'train_pcda_static_overlap_edges_per_day': last_train_pcda_stats.get('pcda_static_overlap_edges_per_day', float('nan')),
        'train_pcda_zero_edge_day_ratio': last_train_pcda_stats.get('pcda_zero_edge_day_ratio', float('nan')),
        'train_pcda_avg_neighbors_per_stock': last_train_pcda_stats.get('pcda_avg_neighbors_per_stock', float('nan')),
        'val_pcda_loss': last_eval_pcda_stats.get('pcda_loss', float('nan')),
        'val_pcda_edges_per_day': last_eval_pcda_stats.get('pcda_edges_per_day', float('nan')),
        'val_pcda_dyn_only_edges_per_day': last_eval_pcda_stats.get('pcda_dyn_only_edges_per_day', float('nan')),
        'val_pcda_static_overlap_edges_per_day': last_eval_pcda_stats.get('pcda_static_overlap_edges_per_day', float('nan')),
        'val_pcda_zero_edge_day_ratio': last_eval_pcda_stats.get('pcda_zero_edge_day_ratio', float('nan')),
        'val_pcda_avg_neighbors_per_stock': last_eval_pcda_stats.get('pcda_avg_neighbors_per_stock', float('nan')),
        'test_pcda_loss': test_pcda_stats.get('pcda_loss', float('nan')),
        'test_pcda_edges_per_day': test_pcda_stats.get('pcda_edges_per_day', float('nan')),
        'test_pcda_dyn_only_edges_per_day': test_pcda_stats.get('pcda_dyn_only_edges_per_day', float('nan')),
        'test_pcda_static_overlap_edges_per_day': test_pcda_stats.get('pcda_static_overlap_edges_per_day', float('nan')),
        'test_pcda_zero_edge_day_ratio': test_pcda_stats.get('pcda_zero_edge_day_ratio', float('nan')),
        'test_pcda_avg_neighbors_per_stock': test_pcda_stats.get('pcda_avg_neighbors_per_stock', float('nan')),
    }

    if not skip_backtest and best_model_file and os.path.exists(best_model_file):
        print(f"\nPhase {phase_id} 开始回测...")
        bt_results = backtest_phase(
            model, market_data, news_data, labels_data, phase, tickers,
            phase_relation_static, phase_S_pre, DEVICE, rnn_length=args.rnn_length,
            top_k=5, transaction_cost=0.0,
            seed=args.seed
        )
        if 'error' not in bt_results:
            # The official test values use the same stock-date rows as prediction.csv.
            result['test_RMSE'] = bt_results.get('RMSE', float('nan'))
            result['test_MAE'] = bt_results.get('MAE', float('nan'))
            result['test_RankIC'] = bt_results.get('RankIC', float('nan'))
            result['sharpe_ratio'] = bt_results['sharpe_ratio']
            result['annualized_return'] = bt_results['annualized_return']
            result['annualized_volatility'] = bt_results['annualized_volatility']
            result['avg_daily_return'] = bt_results['avg_daily_return']
            result['max_drawdown'] = bt_results['max_drawdown']
            result['mean_mrr'] = bt_results['mean_mrr']
            result['mean_ndcg5'] = bt_results['mean_ndcg5']
            result['precision_at_5'] = bt_results['precision_at_5']
            result['precision_at_5_valid_dates'] = bt_results['precision_at_5_valid_dates']
            result['MSE'] = bt_results.get('MSE', float('nan'))
            result['RMSE'] = bt_results.get('RMSE', float('nan'))
            result['MAE'] = bt_results.get('MAE', float('nan'))
            result['IC'] = bt_results.get('IC', float('nan'))
            result['RankIC'] = bt_results.get('RankIC', float('nan'))
            result['RIC'] = bt_results.get('RIC', float('nan'))
            result['total_return'] = bt_results['total_return']
            result['final_value'] = bt_results['final_value']
            result['n_trade_days'] = bt_results['n_trade_days']
            result['n_trades'] = bt_results['n_trades']
            result['daily_returns'] = bt_results.get('daily_returns', np.array([]))
            result['portfolio_values'] = bt_results.get('portfolio_values', [])
            result['dates'] = bt_results.get('dates', [])
            prediction_files = save_phase_prediction_outputs(args, phase_id, bt_results)
            result.update(prediction_files)
            # Explicit Relation Memory Intervention指标
            memory_keys = [
                'memory_full_sharpe', 'memory_full_annualized_return', 'memory_full_total_return',
                'memory_full_max_drawdown', 'memory_full_final_value',
                'memory_no_memory_sharpe', 'memory_no_memory_annualized_return', 'memory_no_memory_total_return',
                'memory_no_memory_max_drawdown', 'memory_no_memory_final_value',
                'memory_shuffled_sharpe', 'memory_shuffled_annualized_return', 'memory_shuffled_total_return',
                'memory_shuffled_max_drawdown', 'memory_shuffled_final_value',
                'memory_no_memory_drop_sharpe', 'memory_shuffled_drop_sharpe',
                'memory_no_memory_drop_total_return', 'memory_shuffled_drop_total_return',
            ]
            for k in memory_keys:
                result[k] = bt_results.get(k, float('nan'))
            
            print(f"  Sharpe={bt_results['sharpe_ratio']:.4f}, "
                  f"AnnRet={bt_results['annualized_return']*100:.2f}%, "
                  f"AvgDRet={bt_results['avg_daily_return']*100:.4f}%, "
                  f"MaxDD={bt_results['max_drawdown']*100:.2f}%, "
                  f"TotalRet={bt_results['total_return']*100:.2f}%, "
                  f"FinalValue={bt_results['final_value']:.0f}, "
                  f"MRR={bt_results['mean_mrr']:.4f}, "
                  f"NDCG@5={bt_results['mean_ndcg5']:.4f}, "
                  f"Precision@5={bt_results['precision_at_5']:.4f}, "
                  f"MSE={result['MSE']:.8f}, "
                  f"IC={result['IC']:.4f}, "
                  f"RankIC={result['RankIC']:.4f}")
            print(f"  Prediction outputs: {result['predictions_file']}")
            print(f"  Prediction audit:   {result['prediction_records_file']}")
            print(f"  Precision@5 daily:  {result['precision_at_5_daily_file']}")
            print(f"  Memory Intervention: "
                  f"Full={bt_results.get('memory_full_sharpe', float('nan')):.4f}, "
                  f"No Explicit Relation Memory={bt_results.get('memory_no_memory_sharpe', float('nan')):.4f}, "
                  f"Shuffled Explicit Relation Memory={bt_results.get('memory_shuffled_sharpe', float('nan')):.4f}, "
                  f"DropNo={bt_results.get('memory_no_memory_drop_sharpe', float('nan')):.4f}, "
                  f"DropShuffle={bt_results.get('memory_shuffled_drop_sharpe', float('nan')):.4f}")

            
        else:
            print(f"  回测失败: {bt_results['error']}")

    return result


if __name__=="__main__":
    args = parser.parse_args()
    args.data_dir, args.bert_path = map(str, resolve_assets(args.data_dir, args.bert_path))
    if args.rnn_length != 5 or args.heads_att != 8:
        raise ValueError('DailyRecurrent controlled experiment requires rnn_length=5 and heads_att=8.')
    
    if args.device == 'cpu':
        DEVICE = "cpu"
    elif args.device.startswith('cuda'):
        DEVICE = args.device
    else:
        DEVICE = "cuda:" + args.device
    
    criterion = torch.nn.MSELoss()
    rnn_length = args.rnn_length
    set_seed(args.seed)

    pretrained_model_path = args.bert_path

    print("=" * 70)
    print("DRFN Multi-Phase Training + Backtesting")
    print("Base alignment direction: News anchors -> Market candidates")
    print("News-anchor filter: enabled (zero-news rows excluded)")
    print("PCDA modality directions: News -> Market and Market -> News")
    print("PCDA news filter: applied per news-anchor direction")
    if args.exp_name:
        print(f"Experiment name: {_format_exp_value(args.exp_name)}")
        print(f"SavedModels root: {experiment_dir('SavedModels', args.exp_name)}")
        print(f"Result dir: {get_result_dir(args)}")
    print(f"Random seed: {args.seed}")
    print(f"PCDA alignment: use={args.use_pcda_align}, lambda={args.lambda_pcda_align}, "
          f"pos_k={args.pcda_pos_k}, eps={args.pcda_eps_sign}, row_cap={args.pcda_row_cap}, "
          f"dyn_only_w={args.pcda_dyn_only_weight}, overlap_w={args.pcda_static_overlap_weight}, "
          f"warmup={args.pcda_align_warmup_epoch}, ramp={args.pcda_align_ramp_epoch}, "
          f"temperature={args.pcda_temperature}")
    print("=" * 70)

    save_runtime_manifest(args, DEVICE)

    # 1. 加载数据
    print("\n[1/4] Loading datasets...")
    news_file = os.path.join(args.data_dir, "us_news.csv")
    market_file = os.path.join(args.data_dir, "us_price_normalized.csv")
    labels_file = os.path.join(args.data_dir, "us_target.csv")
    news_data, market_data, labels_data = load_dataset(news_file, market_file, labels_file)

    # 获取股票数量
    NUM_STOCK = labels_data['Ticker'].nunique()

    all_tickers = sorted(labels_data['Ticker'].unique())
    all_dates = sorted(market_data['Date'].unique())
    tickers = all_tickers
    print(f"  Stocks: {len(tickers)}, Total dates: {len(all_dates)}")
    print(f"  Date range: {min(all_dates)} ~ {max(all_dates)}")

    # 2. 计算phase划分
    print("\n[2/4] Calculating phase splits...")
    phases = calculate_phases(pd.Series(all_dates))
    print(f"  Total phases: {len(phases)}")
    for p in phases:
        print(f"    Phase {p['phase_id']}: Train {p['train_start'].year}~{p['train_end'].year}, "
              f"Val {p['val_start'].year}, Test {p['test_start'].year}")

    # 3. 逐phase训练+回测
    print("\n[3/4] Starting training + backtesting...")

    # 筛选要运行的phase
    if args.phase > 0:
        phases_to_run = [p for p in phases if p['phase_id'] == args.phase]
        if not phases_to_run:
            print(f"Phase {args.phase} not found!")
            sys.exit(1)
        print(f"  Running single phase: {args.phase}")
    else:
        phases_to_run = phases
        print(f"  Running all {len(phases)} phases")

    # 保存所有phase的结果
    all_results = []

    for phase in phases_to_run:
        result = run_phase(phase, news_data, market_data, labels_data,
                          tickers, all_dates,
                          DEVICE, args, skip_backtest=args.skip_backtest)
        all_results.append(result)

    # ===== 汇总所有Phase的结果 =====
    print("\n" + "=" * 100)
    print("ALL PHASES SUMMARY")
    print("=" * 100)

    # 打印每个phase的结果
    header = (f"{'Phase':<8} {'RMSE':<8} {'MAE':<8} {'RankIC':<8} {'Sharpe':<8} {'AnnRet%':<9} "
              f"{'AvgDRet%':<9} {'MaxDD%':<9} {'TotalRet%':<10} {'FinalVal':<12} {'MRR':<8} {'NDCG@5':<8} {'P@5':<8}")
    print(f"\n{header}")
    print("-" * 110)
    for r in all_results:
        final_val_str = f"{r['final_value']:.0f}" if not pd.isna(r['final_value']) else "N/A"
        print(f"Phase {r['phase_id']:<3} "
              f"{r['test_RMSE']:<8.4f} {r['test_MAE']:<8.4f} "
              f"{r['test_RankIC']:<8.4f} "
              f"{r['sharpe_ratio']:<8.4f} "
              f"{r['annualized_return']*100:<9.2f} "
              f"{r['avg_daily_return']*100:<9.4f} "
              f"{r['max_drawdown']*100:<9.2f} "
              f"{r['total_return']*100:<10.2f} "
              f"{final_val_str:<12} "
              f"{r['mean_mrr']:<8.4f} "
              f"{r['mean_ndcg5']:<8.4f} "
              f"{r['precision_at_5']:<8.4f}")

    # ===== 各Phase指标平均值 (mean ± std) =====
    n_valid = len([r for r in all_results if not pd.isna(r['test_RMSE'])])
    if n_valid > 0:
        print("-" * 110)
        
        # 辅助函数：计算 mean ± std
        def mean_std(values):
            vals = [v for v in values if not pd.isna(v)]
            if not vals:
                return float('nan'), float('nan')
            return np.mean(vals), np.std(vals)
        
        avg_rmse, std_rmse = mean_std([r['test_RMSE'] for r in all_results])
        avg_mae, std_mae = mean_std([r['test_MAE'] for r in all_results])
        avg_rankic, std_rankic = mean_std([r['test_RankIC'] for r in all_results])
        avg_sharpe, std_sharpe = mean_std([r['sharpe_ratio'] for r in all_results])
        avg_annret, std_annret = mean_std([r['annualized_return'] for r in all_results])
        avg_avgdret, std_avgdret = mean_std([r['avg_daily_return'] for r in all_results])
        avg_maxdd, std_maxdd = mean_std([r['max_drawdown'] for r in all_results])
        avg_totalret, std_totalret = mean_std([r['total_return'] for r in all_results])
        avg_mrr, std_mrr = mean_std([r['mean_mrr'] for r in all_results])
        avg_ndcg5, std_ndcg5 = mean_std([r['mean_ndcg5'] for r in all_results])
        avg_precision5, std_precision5 = mean_std([r['precision_at_5'] for r in all_results])

        def fmt_mean_std(m, s, pct=False):
            m_str = f"{m*100:.2f}" if pct else f"{m:.4f}"
            s_str = f"{s*100:.2f}" if pct else f"{s:.4f}"
            return f"{m_str}±{s_str}"

        print(f"{'Average':<8} "
              f"{avg_rmse:<8.4f} {avg_mae:<8.4f} "
              f"{avg_rankic:<8.4f} "
              f"{avg_sharpe:<8.4f} "
              f"{avg_annret*100:<9.2f} "
              f"{avg_avgdret*100:<9.4f} "
              f"{avg_maxdd*100:<9.2f} "
              f"{avg_totalret*100:<10.2f} "
              f"{'':12} "
              f"{avg_mrr:<8.4f} "
              f"{avg_ndcg5:<8.4f} "
              f"{avg_precision5:<8.4f}")

        print(f"\n各Phase平均值 ± 标准差（共{n_valid}个有效phase）:")
        print(f"  RMSE:            {fmt_mean_std(avg_rmse, std_rmse)}")
        print(f"  MAE:             {fmt_mean_std(avg_mae, std_mae)}")
        print(f"  Test RankIC:     {fmt_mean_std(avg_rankic, std_rankic)}")
        print(f"  Sharpe Ratio:    {fmt_mean_std(avg_sharpe, std_sharpe)}")
        print(f"  年化收益率:       {fmt_mean_std(avg_annret, std_annret, pct=True)}%")
        print(f"  平均日收益率:     {fmt_mean_std(avg_avgdret, std_avgdret, pct=True)}%")
        print(f"  最大回撤:         {fmt_mean_std(avg_maxdd, std_maxdd, pct=True)}%")
        print(f"  总收益率(avg):    {fmt_mean_std(avg_totalret, std_totalret, pct=True)}%")
        print(f"  MRR:             {fmt_mean_std(avg_mrr, std_mrr)}")
        print(f"  NDCG@5:          {fmt_mean_std(avg_ndcg5, std_ndcg5)}")
        print(f"  Precision@5:     {fmt_mean_std(avg_precision5, std_precision5)}")


    # ===== Explicit Relation Memory Intervention summary =====
    analysis_rows = []

    n_memory = len([r for r in all_results if not pd.isna(r.get('memory_full_sharpe', float('nan')))])
    if n_memory > 0:
        print("\n" + "=" * 80)
        print("EXPLICIT RELATION MEMORY INTERVENTION")
        print("=" * 80)

        mem_header = (f"{'Phase':<8} {'Variant':<34} {'Sharpe':<10} {'AnnRet%':<10} "
                      f"{'TotalRet%':<11} {'MaxDD%':<10} {'DropSR':<10}")
        print("\n" + mem_header)
        print("-" * 90)

        memory_variants = [
            ('full', 'Full'),
            ('no_memory', 'No Explicit Relation Memory'),
            ('shuffled', 'Shuffled Explicit Relation Memory'),
        ]

        for r in all_results:
            if pd.isna(r.get('memory_full_sharpe', float('nan'))):
                continue

            full_sharpe = r.get('memory_full_sharpe', float('nan'))

            for key, label in memory_variants:
                if key == 'full':
                    sharpe = r.get('memory_full_sharpe', float('nan'))
                    annret = r.get('memory_full_annualized_return', float('nan'))
                    totalret = r.get('memory_full_total_return', float('nan'))
                    maxdd = r.get('memory_full_max_drawdown', float('nan'))
                    final_value = r.get('memory_full_final_value', float('nan'))
                    drop_sr = 0.0
                elif key == 'no_memory':
                    sharpe = r.get('memory_no_memory_sharpe', float('nan'))
                    annret = r.get('memory_no_memory_annualized_return', float('nan'))
                    totalret = r.get('memory_no_memory_total_return', float('nan'))
                    maxdd = r.get('memory_no_memory_max_drawdown', float('nan'))
                    final_value = r.get('memory_no_memory_final_value', float('nan'))
                    drop_sr = r.get('memory_no_memory_drop_sharpe', float('nan'))
                else:
                    sharpe = r.get('memory_shuffled_sharpe', float('nan'))
                    annret = r.get('memory_shuffled_annualized_return', float('nan'))
                    totalret = r.get('memory_shuffled_total_return', float('nan'))
                    maxdd = r.get('memory_shuffled_max_drawdown', float('nan'))
                    final_value = r.get('memory_shuffled_final_value', float('nan'))
                    drop_sr = r.get('memory_shuffled_drop_sharpe', float('nan'))

                print(f"Phase {r['phase_id']:<3} "
                      f"{label:<34} "
                      f"{sharpe:<10.4f} "
                      f"{annret*100:<10.2f} "
                      f"{totalret*100:<11.2f} "
                      f"{maxdd*100:<10.2f} "
                      f"{drop_sr:<10.4f}")

                analysis_rows.append({
                    'section': 'memory_intervention',
                    'phase_label': f"Phase {r['phase_id']}",
                    'phase_id': r['phase_id'],
                    'variant': label,
                    'sharpe_ratio': sharpe,
                    'annualized_return': annret,
                    'total_return': totalret,
                    'max_drawdown': maxdd,
                    'final_value': final_value,
                    'drop_sharpe_vs_full': drop_sr,
                })

        # Average rows by variant
        print("-" * 90)
        for key, label in memory_variants:
            if key == 'full':
                sharpe_vals = [r.get('memory_full_sharpe', np.nan) for r in all_results]
                annret_vals = [r.get('memory_full_annualized_return', np.nan) for r in all_results]
                totalret_vals = [r.get('memory_full_total_return', np.nan) for r in all_results]
                maxdd_vals = [r.get('memory_full_max_drawdown', np.nan) for r in all_results]
                final_vals = [r.get('memory_full_final_value', np.nan) for r in all_results]
                drop_vals = [0.0 for _ in all_results if not pd.isna(_.get('memory_full_sharpe', np.nan))]
            elif key == 'no_memory':
                sharpe_vals = [r.get('memory_no_memory_sharpe', np.nan) for r in all_results]
                annret_vals = [r.get('memory_no_memory_annualized_return', np.nan) for r in all_results]
                totalret_vals = [r.get('memory_no_memory_total_return', np.nan) for r in all_results]
                maxdd_vals = [r.get('memory_no_memory_max_drawdown', np.nan) for r in all_results]
                final_vals = [r.get('memory_no_memory_final_value', np.nan) for r in all_results]
                drop_vals = [r.get('memory_no_memory_drop_sharpe', np.nan) for r in all_results]
            else:
                sharpe_vals = [r.get('memory_shuffled_sharpe', np.nan) for r in all_results]
                annret_vals = [r.get('memory_shuffled_annualized_return', np.nan) for r in all_results]
                totalret_vals = [r.get('memory_shuffled_total_return', np.nan) for r in all_results]
                maxdd_vals = [r.get('memory_shuffled_max_drawdown', np.nan) for r in all_results]
                final_vals = [r.get('memory_shuffled_final_value', np.nan) for r in all_results]
                drop_vals = [r.get('memory_shuffled_drop_sharpe', np.nan) for r in all_results]

            avg_sh = np.nanmean(sharpe_vals)
            avg_ar = np.nanmean(annret_vals)
            avg_tr = np.nanmean(totalret_vals)
            avg_md = np.nanmean(maxdd_vals)
            avg_fv = np.nanmean(final_vals)
            avg_drop = np.nanmean(drop_vals)

            print(f"{'Average':<8} "
                  f"{label:<34} "
                  f"{avg_sh:<10.4f} "
                  f"{avg_ar*100:<10.2f} "
                  f"{avg_tr*100:<11.2f} "
                  f"{avg_md*100:<10.2f} "
                  f"{avg_drop:<10.4f}")

            analysis_rows.append({
                'section': 'memory_intervention',
                'phase_label': 'Average',
                'phase_id': np.nan,
                'variant': label,
                'sharpe_ratio': avg_sh,
                'annualized_return': avg_ar,
                'total_return': avg_tr,
                'max_drawdown': avg_md,
                'final_value': avg_fv,
                'drop_sharpe_vs_full': avg_drop,
            })

    # ===== 连贯资金模式：拼接所有phase的日收益率 =====
    cumulative_row = None
    all_daily_returns = []
    for r in all_results:
        dr = r.get('daily_returns', np.array([]))
        if len(dr) > 0:
            all_daily_returns.append(dr)

    if all_daily_returns:
        combined_daily_returns = np.concatenate(all_daily_returns)
        initial_cash = 1000000
        cumulative = initial_cash * np.cumprod(1 + combined_daily_returns)
        final_cumulative_value = cumulative[-1]
        cumulative_total_return = (final_cumulative_value / initial_cash) - 1
        total_days = len(combined_daily_returns)
        cumulative_annualized_return = (1 + cumulative_total_return) ** (252 / total_days) - 1
        cumulative_annualized_vol = combined_daily_returns.std() * np.sqrt(252)
        cumulative_sharpe = (cumulative_annualized_return - 0.02) / cumulative_annualized_vol if cumulative_annualized_vol > 0 else 0
        peak = np.maximum.accumulate(cumulative)
        drawdowns = (peak - cumulative) / peak
        cumulative_max_drawdown = np.max(drawdowns) if len(drawdowns) > 0 else 0
        cumulative_avg_daily_return = combined_daily_returns.mean()
        profit = final_cumulative_value - initial_cash

        print("\n" + "=" * 100)
        print("CUMULATIVE BACKTEST (连贯资金，100万起始)")
        print("=" * 100)
        print(f"  总交易天数:       {total_days}")
        print(f"  起始资金:         CNY {initial_cash:,.0f}")
        print(f"  最终资金:         CNY {final_cumulative_value:,.2f}")
        print(f"  总盈亏:           CNY {profit:,.2f}")
        print(f"  总收益率:         {cumulative_total_return * 100:.2f}%")
        print(f"  年化收益率:       {cumulative_annualized_return * 100:.2f}%")
        print(f"  夏普比率:         {cumulative_sharpe:.4f}")
        print(f"  最大回撤:         {cumulative_max_drawdown * 100:.2f}%")
        print(f"  平均日收益率:     {cumulative_avg_daily_return * 100:.4f}%")
        print("=" * 100)

        cumulative_row = {
            'exp_name': _format_exp_value(args.exp_name) if args.exp_name else '',
            'seed': args.seed,
            'phase_id': np.nan,
            'test_RMSE': np.nan,
            'test_MAE': np.nan,
            'test_RankIC': np.nan,
            'sharpe_ratio': cumulative_sharpe,
            'annualized_return': cumulative_annualized_return,
            'annualized_volatility': cumulative_annualized_vol,
            'avg_daily_return': cumulative_avg_daily_return,
            'max_drawdown': cumulative_max_drawdown,
            'mean_mrr': np.nan,
            'mean_ndcg5': np.nan,
            'precision_at_5': np.nan,
            'precision_at_5_valid_dates': np.nan,
            'total_return': cumulative_total_return,
            'final_value': final_cumulative_value,
            'n_trade_days': total_days,
            'n_trades': np.nan,
            'memory_full_sharpe': np.nan,
            'memory_no_memory_sharpe': np.nan,
            'memory_shuffled_sharpe': np.nan,
            'memory_no_memory_drop_sharpe': np.nan,
            'memory_shuffled_drop_sharpe': np.nan,
            'phase_label': 'Cumulative',
        }

    # 保存结果到CSV（主结果表 + 分析摘要表）
    results_df = pd.DataFrame(all_results)
    results_df['phase_label'] = results_df['phase_id'].apply(lambda x: f"Phase {int(x)}")

    avg_row = {
        'exp_name': _format_exp_value(args.exp_name) if args.exp_name else '',
        'seed': args.seed,
        'phase_id': np.nan,
        'phase_label': 'Average',
        'test_RMSE': np.nanmean([r['test_RMSE'] for r in all_results]),
        'test_MAE': np.nanmean([r['test_MAE'] for r in all_results]),
        'test_RankIC': np.nanmean([r['test_RankIC'] for r in all_results]),
        'sharpe_ratio': np.nanmean([r['sharpe_ratio'] for r in all_results]),
        'annualized_return': np.nanmean([r['annualized_return'] for r in all_results]),
        'annualized_volatility': np.nanmean([r['annualized_volatility'] for r in all_results]),
        'avg_daily_return': np.nanmean([r['avg_daily_return'] for r in all_results]),
        'max_drawdown': np.nanmean([r['max_drawdown'] for r in all_results]),
        'mean_mrr': np.nanmean([r['mean_mrr'] for r in all_results]),
        'mean_ndcg5': np.nanmean([r['mean_ndcg5'] for r in all_results]),
        'precision_at_5': np.nanmean([r['precision_at_5'] for r in all_results]),
        'precision_at_5_valid_dates': np.nansum([r['precision_at_5_valid_dates'] for r in all_results]),
        'MSE': np.nanmean([r['MSE'] for r in all_results]),
        'RMSE': np.nanmean([r['RMSE'] for r in all_results]),
        'MAE': np.nanmean([r['MAE'] for r in all_results]),
        'IC': np.nanmean([r['IC'] for r in all_results]),
        'RankIC': np.nanmean([r['RankIC'] for r in all_results]),
        'RIC': np.nanmean([r['RIC'] for r in all_results]),
        'total_return': np.nanmean([r['total_return'] for r in all_results]),
        'final_value': np.nan,
        'n_trade_days': np.nanmean([r['n_trade_days'] for r in all_results]),
        'n_trades': np.nanmean([r['n_trades'] for r in all_results]),
        'memory_full_sharpe': np.nanmean([r.get('memory_full_sharpe', np.nan) for r in all_results]),
        'memory_no_memory_sharpe': np.nanmean([r.get('memory_no_memory_sharpe', np.nan) for r in all_results]),
        'memory_shuffled_sharpe': np.nanmean([r.get('memory_shuffled_sharpe', np.nan) for r in all_results]),
        'memory_no_memory_drop_sharpe': np.nanmean([r.get('memory_no_memory_drop_sharpe', np.nan) for r in all_results]),
        'memory_shuffled_drop_sharpe': np.nanmean([r.get('memory_shuffled_drop_sharpe', np.nan) for r in all_results]),
        'pcda_use': args.use_pcda_align,
        'pcda_lambda': args.lambda_pcda_align,
        'pcda_pos_k': args.pcda_pos_k,
        'pcda_row_cap': args.pcda_row_cap,
        'pcda_dyn_only_weight': args.pcda_dyn_only_weight,
        'pcda_static_overlap_weight': args.pcda_static_overlap_weight,
        'train_pcda_loss': np.nanmean([r.get('train_pcda_loss', np.nan) for r in all_results]),
        'train_pcda_edges_per_day': np.nanmean([r.get('train_pcda_edges_per_day', np.nan) for r in all_results]),
        'train_pcda_dyn_only_edges_per_day': np.nanmean([r.get('train_pcda_dyn_only_edges_per_day', np.nan) for r in all_results]),
        'train_pcda_static_overlap_edges_per_day': np.nanmean([r.get('train_pcda_static_overlap_edges_per_day', np.nan) for r in all_results]),
        'train_pcda_zero_edge_day_ratio': np.nanmean([r.get('train_pcda_zero_edge_day_ratio', np.nan) for r in all_results]),
        'train_pcda_avg_neighbors_per_stock': np.nanmean([r.get('train_pcda_avg_neighbors_per_stock', np.nan) for r in all_results]),
        'val_pcda_loss': np.nanmean([r.get('val_pcda_loss', np.nan) for r in all_results]),
        'val_pcda_edges_per_day': np.nanmean([r.get('val_pcda_edges_per_day', np.nan) for r in all_results]),
        'val_pcda_dyn_only_edges_per_day': np.nanmean([r.get('val_pcda_dyn_only_edges_per_day', np.nan) for r in all_results]),
        'val_pcda_static_overlap_edges_per_day': np.nanmean([r.get('val_pcda_static_overlap_edges_per_day', np.nan) for r in all_results]),
        'val_pcda_zero_edge_day_ratio': np.nanmean([r.get('val_pcda_zero_edge_day_ratio', np.nan) for r in all_results]),
        'val_pcda_avg_neighbors_per_stock': np.nanmean([r.get('val_pcda_avg_neighbors_per_stock', np.nan) for r in all_results]),
        'test_pcda_loss': np.nanmean([r.get('test_pcda_loss', np.nan) for r in all_results]),
        'test_pcda_edges_per_day': np.nanmean([r.get('test_pcda_edges_per_day', np.nan) for r in all_results]),
        'test_pcda_dyn_only_edges_per_day': np.nanmean([r.get('test_pcda_dyn_only_edges_per_day', np.nan) for r in all_results]),
        'test_pcda_static_overlap_edges_per_day': np.nanmean([r.get('test_pcda_static_overlap_edges_per_day', np.nan) for r in all_results]),
        'test_pcda_zero_edge_day_ratio': np.nanmean([r.get('test_pcda_zero_edge_day_ratio', np.nan) for r in all_results]),
        'test_pcda_avg_neighbors_per_stock': np.nanmean([r.get('test_pcda_avg_neighbors_per_stock', np.nan) for r in all_results]),
    }

    result_rows = [results_df]
    result_rows.append(pd.DataFrame([avg_row]))
    if cumulative_row is not None:
        result_rows.append(pd.DataFrame([cumulative_row]))
    final_results_df = pd.concat(result_rows, ignore_index=True, sort=False)

    main_cols = ['exp_name', 'seed', 'phase_label', 'phase_id', 'test_RMSE', 'test_MAE', 'test_RankIC', 'sharpe_ratio', 'annualized_return',
                 'avg_daily_return', 'max_drawdown', 'mean_mrr',
                 'mean_ndcg5', 'precision_at_5', 'precision_at_5_valid_dates',
                 'MSE', 'RMSE', 'MAE', 'IC', 'RankIC', 'RIC',
                 'total_return', 'final_value', 'n_trade_days', 'n_trades',
                 'predictions_file', 'prediction_records_file', 'precision_at_5_daily_file',
                 'canonical_prediction_file', 'canonical_metrics_file',
                 'pcda_use', 'pcda_lambda', 'pcda_pos_k', 'pcda_row_cap',
                 'pcda_dyn_only_weight', 'pcda_static_overlap_weight',
                 'train_pcda_loss', 'train_pcda_edges_per_day',
                 'train_pcda_dyn_only_edges_per_day', 'train_pcda_static_overlap_edges_per_day',
                 'train_pcda_zero_edge_day_ratio', 'train_pcda_avg_neighbors_per_stock',
                 'val_pcda_loss', 'val_pcda_edges_per_day',
                 'val_pcda_dyn_only_edges_per_day', 'val_pcda_static_overlap_edges_per_day',
                 'val_pcda_zero_edge_day_ratio', 'val_pcda_avg_neighbors_per_stock',
                 'test_pcda_loss', 'test_pcda_edges_per_day',
                 'test_pcda_dyn_only_edges_per_day', 'test_pcda_static_overlap_edges_per_day',
                 'test_pcda_zero_edge_day_ratio', 'test_pcda_avg_neighbors_per_stock']
    output_tag = f"phase{args.phase}" if args.phase > 0 else "all_phases"
    result_dir = get_result_dir(args)
    os.makedirs(result_dir, exist_ok=True)
    results_output_path = os.path.join(result_dir, f'all_phases_results_{output_tag}.csv')
    analysis_output_path = os.path.join(result_dir, f'analysis_summary_{output_tag}.csv')

    final_results_df[main_cols].to_csv(results_output_path, index=False)
    print(f"\nResults saved to {results_output_path}")

    analysis_df = pd.DataFrame(analysis_rows)
    if not analysis_df.empty:
        analysis_cols = ['section', 'phase_label', 'phase_id', 'variant',
                         'sharpe_ratio', 'annualized_return',
                         'total_return', 'max_drawdown', 'final_value',
                         'drop_sharpe_vs_full']
        for col in analysis_cols:
            if col not in analysis_df.columns:
                analysis_df[col] = np.nan
        analysis_df[analysis_cols].to_csv(analysis_output_path, index=False)
        print(f"Analysis summary saved to {analysis_output_path}")
