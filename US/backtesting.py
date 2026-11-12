"""
回测模块
功能：
1. 模型推理：在测试集上生成预测值
2. 日内回测：Top-K选股，算日收益
3. 计算指标：夏普比率、年化收益率、最大回撤、MRR、NDCG@5
4. 多phase汇总：每个phase独立回测，最后取平均

可以独立运行（用于加载已训练模型回测），也可以被main调用
"""

import argparse
from pathlib import Path
import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
from Model_n2m_filtered_dailyrecurrent import DRFN

DRFN_KWARGS = {"static_thresh": 0.5}
from utils import (
    set_seed,
    filter_strict_next_day_labels,
    get_strict_prediction_dates,
    assert_strict_prediction_dates,
)
from data_splits import calculate_phases
import sys
import numpy.core.numeric as numeric
import os

sys.modules['numpy._core.numeric'] = numeric
from typing import Dict
from sklearn.metrics import ndcg_score
from scipy.stats import spearmanr


def calculate_prediction_metrics_from_records(records):
    """Recompute canonical prediction metrics from one stock-date table."""
    required = {'Date', 'Ticker', 'prediction', 'target_return_t_plus_1'}
    missing = required.difference(records.columns)
    if missing:
        raise RuntimeError(f"Prediction records are missing columns: {sorted(missing)}")

    frame = records.copy()
    frame['Date'] = pd.to_datetime(frame['Date'])
    frame['prediction'] = pd.to_numeric(frame['prediction'], errors='coerce')
    frame['target_return_t_plus_1'] = pd.to_numeric(
        frame['target_return_t_plus_1'], errors='coerce'
    )
    valid = np.isfinite(frame['prediction']) & np.isfinite(frame['target_return_t_plus_1'])
    frame = frame.loc[valid].copy()
    if frame.empty:
        raise RuntimeError("No valid prediction-target pairs are available for canonical metrics.")

    errors = frame['prediction'].to_numpy() - frame['target_return_t_plus_1'].to_numpy()
    mse = float(np.mean(np.square(errors)))
    mae = float(np.mean(np.abs(errors)))
    daily_ic = []
    daily_rank_ic = []
    daily_ndcg = []
    daily_precision = []
    daily_mrr = []

    for _, daily in frame.groupby('Date', sort=True):
        daily = daily.sort_values('Ticker')
        predictions = daily['prediction'].to_numpy(dtype=float)
        targets = daily['target_return_t_plus_1'].to_numpy(dtype=float)
        if len(daily) > 5 and np.std(predictions) > 0 and np.std(targets) > 0:
            daily_ic.append(float(np.corrcoef(predictions, targets)[0, 1]))
            rank_ic = spearmanr(predictions, targets).correlation
            daily_rank_ic.append(0.0 if np.isnan(rank_ic) else float(rank_ic))

        ranked = daily.sort_values('prediction', ascending=False, kind='mergesort')
        if len(ranked) >= 5:
            top5 = ranked.head(5)
            daily_precision.append(
                float((top5['target_return_t_plus_1'] > 0).sum() / 5.0)
            )
            shifted = targets - min(float(np.min(targets)), 0.0)
            daily_ndcg.append(
                float(ndcg_score(shifted.reshape(1, -1), predictions.reshape(1, -1), k=5))
            )

        best_ticker = daily.loc[daily['target_return_t_plus_1'].idxmax(), 'Ticker']
        ranked_tickers = ranked['Ticker'].tolist()
        daily_mrr.append(1.0 / (ranked_tickers.index(best_ticker) + 1))

    return {
        'MSE': mse,
        'RMSE': float(np.sqrt(mse)),
        'MAE': mae,
        'IC': float(np.mean(daily_ic)) if daily_ic else 0.0,
        'RankIC': float(np.mean(daily_rank_ic)) if daily_rank_ic else 0.0,
        'NDCG@5': float(np.mean(daily_ndcg)) if daily_ndcg else 0.0,
        'Precision@5': float(np.mean(daily_precision)) if daily_precision else 0.0,
        'MRR': float(np.mean(daily_mrr)) if daily_mrr else 0.0,
        'valid_pairs': int(len(frame)),
        'valid_dates': int(frame['Date'].nunique()),
    }


# ==================== 推理函数 ====================

# ==================== Explicit Relation Memory Intervention ====================

def _clear_dynamic_state_if_possible(model):
    """Reset recurrent state before or after an independent inference sequence."""
    if hasattr(model, "clear_dynamic_state"):
        model.clear_dynamic_state()
    elif hasattr(model, "attentions"):
        for att in model.attentions:
            if hasattr(att, "prev_dynamic"):
                att.prev_dynamic = None
        if hasattr(model, "last_current_dynamic_signed"):
            model.last_current_dynamic_signed = None


def _shuffle_dynamic_memory(model, rng):
    """Permute both axes of every head's explicit memory with one stock order."""
    memory = next((att.prev_dynamic for att in model.attentions
                   if att.prev_dynamic is not None), None)
    if memory is None or memory.size(0) <= 1:
        return
    perm = torch.as_tensor(rng.permutation(memory.size(0)), dtype=torch.long, device=memory.device)
    for att in model.attentions:
        if att.prev_dynamic is not None:
            att.prev_dynamic = att.prev_dynamic[perm][:, perm].detach()


def intervention_inference(model, market_data, news_data, dates, tickers,
                           relation_static, S_pre=None, rnn_length=5,
                           mode="full", seed=11380):
    """
    动态关系记忆干预推理。

    mode:
      full:
        Full: retain previous-day explicit relation matrices and GRU_r hidden.
      no_memory:
        No Explicit Relation Memory: clear D_prev before each day; retain GRU_r hidden.
      shuffled_memory:
        Shuffled Explicit Relation Memory: permute D_prev rows/columns; retain GRU_r hidden.

    Returns:
        predictions_df: DataFrame, index=Date, columns=Ticker
        eval_dates: list, 实际产生预测的日期
    """
    assert mode in {"full", "no_memory", "shuffled_memory"}

    model.eval()
    _clear_dynamic_state_if_possible(model)

    all_predictions = []
    all_dates = []
    rng = np.random.default_rng(seed)

    desc_map = {
        "full": "Full inference",
        "no_memory": "No Explicit Relation Memory推理中",
        "shuffled_memory": "Shuffled Explicit Relation Memory推理中",
    }

    with torch.no_grad():
        for i in tqdm(range(rnn_length - 1, len(dates)), desc=desc_map[mode]):
            current_date = dates[i]
            start_idx = i - rnn_length + 1
            end_idx = i + 1
            window_dates = dates[start_idx:end_idx]
            market_window = market_data[market_data['Date'].isin(window_dates)]
            news_window = news_data[news_data['Date'].isin(window_dates)]

            if mode == "no_memory":
                model.clear_relation_memory_only()
            elif mode == "shuffled_memory":
                _shuffle_dynamic_memory(model, rng)

            fwd_out = model(news_window, market_window, relation_static, S_pre,
                            window_dates=window_dates)
            output = fwd_out[0]

            predictions = output.view(-1).cpu().numpy()
            all_predictions.append(predictions)
            all_dates.append(current_date)


    predictions_df = pd.DataFrame(all_predictions, index=all_dates, columns=tickers)
    _clear_dynamic_state_if_possible(model)
    return predictions_df, all_dates


def _nan_backtest_result():
    return {
        'final_value': float('nan'),
        'total_return': float('nan'),
        'annualized_return': float('nan'),
        'annualized_volatility': float('nan'),
        'sharpe_ratio': float('nan'),
        'max_drawdown': float('nan'),
        'avg_daily_return': float('nan'),
        'mean_mrr': float('nan'),
        'mean_ndcg5': float('nan'),
        'precision_at_5': float('nan'),
        'mean_precision_at_k_up': float('nan'),
        'precision_at_5_valid_dates': 0,
        'precision_at_5_daily_df': pd.DataFrame(),
        'MSE': float('nan'),
        'RMSE': float('nan'),
        'MAE': float('nan'),
        'IC': float('nan'),
        'RankIC': float('nan'),
        'RIC': float('nan'),
        'n_trade_days': 0,
        'n_trades': 0,
        'portfolio_values': [],
        'daily_returns': np.array([]),
        'dates': [],
        'trade_log': [],
    }


def _safe_backtest(predictions_df, true_returns_df, initial_cash=1000000,
                   top_k=5, transaction_cost=0.0):
    if predictions_df is None or predictions_df.empty:
        return _nan_backtest_result()
    out = run_day_trading_backtest(
        predictions_df, true_returns_df,
        initial_cash=initial_cash,
        top_k=top_k,
        transaction_cost=transaction_cost
    )
    return out if out is not None else _nan_backtest_result()



# ==================== 回测函数 ====================

def run_day_trading_backtest(predictions_df: pd.DataFrame, true_returns_df: pd.DataFrame,
                             initial_cash: float = 1000000, top_k: int = 10,
                             transaction_cost: float = 0.001) -> Dict:
    """
    日内回测：每天选预测值最高的Top-K只股票，等权买入，次日卖出
    
    Args:
        predictions_df: DataFrame, index=Date, columns=Ticker, 模型预测值
        true_returns_df: DataFrame, index=Date, columns=Ticker, 真实收益率
        initial_cash: float, 初始资金
        top_k: int, 每天选几只股票
        transaction_cost: float, 单边交易成本
    
    Returns:
        dict: 回测结果（含夏普、年化收益、回撤、MRR、NDCG等）
    """
    prediction_dates = predictions_df.index
    true_return_dates = true_returns_df.index

    # us_target.csv 的 Date=t 已经表示 Close_t -> Close_{t+1} 的收益。
    # 因此 prediction Date=t 必须直接和 true_returns_df.loc[t] 对齐。
    tradeable_dates = sorted(prediction_dates.intersection(true_return_dates))

    if not tradeable_dates:
        return None

    cash = initial_cash
    portfolio_value = initial_cash
    portfolio_values = [initial_cash]
    trade_dates_log = []
    daily_returns = []
    trade_log = []
    daily_mrr_scores = []
    daily_ndcg5_scores = []
    daily_precision5_scores = []
    precision5_daily_rows = []
    ic_list = []
    rankic_list = []
    sq_error_values = []
    abs_error_values = []

    prev_weights = {}
    turnover_values = []

    for current_date in tqdm(tradeable_dates, desc="执行回测"):
        current_predictions = predictions_df.loc[current_date].dropna()
        actual_returns_all = true_returns_df.loc[current_date].dropna()

        # 计算排序指标
        if not current_predictions.empty and not actual_returns_all.empty:
            common_tickers = current_predictions.index.intersection(actual_returns_all.index)
            preds_aligned = current_predictions[common_tickers]
            actuals_aligned = actual_returns_all[common_tickers]
            if len(common_tickers) > 5:
                pred_vals = preds_aligned.values.astype(float)
                actual_vals = actuals_aligned.values.astype(float)
                valid = ~(np.isnan(pred_vals) | np.isnan(actual_vals) | np.isinf(pred_vals) | np.isinf(actual_vals))
                pred_vals = pred_vals[valid]
                actual_vals = actual_vals[valid]
                if len(pred_vals) > 5:
                    errors = pred_vals - actual_vals
                    sq_error_values.extend(np.square(errors).tolist())
                    abs_error_values.extend(np.abs(errors).tolist())
                    if np.std(pred_vals) > 0 and np.std(actual_vals) > 0:
                        ic_list.append(float(np.corrcoef(pred_vals, actual_vals)[0, 1]))
                        rank_ic = spearmanr(pred_vals, actual_vals).correlation
                        rankic_list.append(0.0 if np.isnan(rank_ic) else float(rank_ic))
            if not actuals_aligned.empty:
                best_stock_ticker = actuals_aligned.idxmax()
                predicted_rank_list = preds_aligned.sort_values(ascending=False).index.tolist()
                try:
                    rank = predicted_rank_list.index(best_stock_ticker) + 1
                    daily_mrr_scores.append(1.0 / rank)
                except ValueError:
                    daily_mrr_scores.append(0.0)
            if len(common_tickers) >= 5:
                selected = preds_aligned.nlargest(5).index
                selected_actuals = actuals_aligned.loc[selected]
                positive_count = int((selected_actuals > 0).sum())
                precision5 = float(positive_count / 5.0)
                daily_precision5_scores.append(precision5)
                precision5_daily_rows.append({
                    'date': str(pd.Timestamp(current_date).date()),
                    'precision_at_5': precision5,
                    'positive_count': positive_count,
                    'top_k': 5,
                    'selected_tickers': '|'.join(str(item) for item in selected),
                })

                true_scores_raw = actuals_aligned.values
                min_return = true_scores_raw.min()
                true_scores_shifted = true_scores_raw - min_return if min_return < 0 else true_scores_raw
                true_scores = true_scores_shifted.reshape(1, -1)
                pred_scores = preds_aligned.values.reshape(1, -1)
                ndcg5 = ndcg_score(true_scores, pred_scores, k=5)
                daily_ndcg5_scores.append(ndcg5)

        tradable_predictions = current_predictions[
            current_predictions.index.intersection(actual_returns_all.index)
        ]
        if tradable_predictions.empty:
            daily_returns.append(0.0)
            portfolio_values.append(portfolio_value)
            trade_dates_log.append(current_date)
            continue

        top_k_preds = tradable_predictions.nlargest(top_k)
        stocks_to_buy = top_k_preds[top_k_preds > 0].index.tolist()

        if not stocks_to_buy:
            # 空仓日：若前一日有持仓，这一天要把持仓全部卖出，应收卖出单边成本；
            # 之后资金留作现金，当日收益记为 0（仍计入 D，见论文 4.1 节定义）。
            liquidated = sum(prev_weights.values())
            if liquidated > 0:
                turnover_values.append(liquidated / 2.0)
                daily_return = -liquidated * transaction_cost
            else:
                daily_return = 0.0
        else:
            actual_returns = true_returns_df.loc[current_date, stocks_to_buy].dropna()
            if actual_returns.empty:
                daily_return = 0.0
            else:
                avg_gross_return = actual_returns.mean()
                # 交易成本按实际换手计费：只对当日实际变动的名义金额收单边成本。
                # traded = sum_i |w_i,t - w_i,t-1|，等于买入名义 + 卖出名义；
                # 单向换手 tau_t = traded / 2。若当日 5 只全部换掉，traded = 2、
                # 成本 = 2 * transaction_cost，与原实现一致。
                cur_weights = {tk: 1.0 / len(stocks_to_buy) for tk in stocks_to_buy}
                all_names = set(cur_weights) | set(prev_weights)
                traded = sum(abs(cur_weights.get(tk, 0.0) - prev_weights.get(tk, 0.0))
                             for tk in all_names)
                turnover_oneway = traded / 2.0
                turnover_values.append(turnover_oneway)
                daily_return = avg_gross_return - traded * transaction_cost
                trade_log.append({'date': current_date, 'action': 'buy', 'tickers': stocks_to_buy,
                                  'count': len(stocks_to_buy), 'actual_count': len(actual_returns),
                                  'capital_invested': portfolio_value,
                                  'turnover': turnover_oneway, 'cost_charged': traded * transaction_cost,
                                  'avg_gross_return': avg_gross_return, 'daily_return': daily_return})

        prev_weights = ({tk: 1.0 / len(stocks_to_buy) for tk in stocks_to_buy}
                        if stocks_to_buy else {})
        portfolio_value *= (1 + daily_return)
        daily_returns.append(daily_return)
        portfolio_values.append(portfolio_value)
        trade_dates_log.append(current_date)

    daily_returns = np.array(daily_returns)
    n_days = max(len(daily_returns), 1)
    total_return = (portfolio_value / initial_cash) - 1

    # 年化收益率
    annualized_return = (1 + total_return) ** (252 / n_days) - 1
    annualized_volatility = daily_returns.std() * np.sqrt(252)

    # 夏普比率 (无风险利率2%)
    sharpe_ratio = (annualized_return - 0.02) / annualized_volatility if annualized_volatility > 0 else 0

    # 最大回撤
    portfolio_array = np.array(portfolio_values)
    peak = np.maximum.accumulate(portfolio_array)
    drawdowns = (peak - portfolio_array) / peak
    max_drawdown = np.max(drawdowns) if len(drawdowns) > 0 else 0

    # MRR和NDCG@5
    mean_mrr = np.mean(daily_mrr_scores) if daily_mrr_scores else 0
    mean_ndcg5 = np.mean(daily_ndcg5_scores) if daily_ndcg5_scores else 0
    precision_at_5 = np.mean(daily_precision5_scores) if daily_precision5_scores else 0
    precision_at_5_daily_df = pd.DataFrame(precision5_daily_rows)

    metric_predictions = predictions_df.loc[tradeable_dates].copy()
    metric_targets = true_returns_df.reindex(
        index=metric_predictions.index, columns=metric_predictions.columns
    )
    prediction_records = metric_predictions.reset_index().melt(
        id_vars=metric_predictions.index.name or 'index',
        var_name='Ticker', value_name='prediction',
    ).rename(columns={metric_predictions.index.name or 'index': 'Date'})
    target_records = metric_targets.reset_index().melt(
        id_vars=metric_targets.index.name or 'index',
        var_name='Ticker', value_name='target_return_t_plus_1',
    ).rename(columns={metric_targets.index.name or 'index': 'Date'})
    metric_records = prediction_records.merge(
        target_records, on=['Date', 'Ticker'], how='left', validate='one_to_one'
    )
    canonical_metrics = calculate_prediction_metrics_from_records(metric_records)
    if not np.isclose(
        canonical_metrics['Precision@5'], precision_at_5, atol=1e-12, rtol=1e-10
    ):
        raise RuntimeError("Daily Precision@5 rows differ from canonical prediction records.")

    return {
        'final_value': portfolio_value,
        'total_return': total_return,
        'annualized_return': annualized_return,
        'annualized_volatility': annualized_volatility,
        'sharpe_ratio': sharpe_ratio,
        'max_drawdown': max_drawdown,
        'avg_daily_return': daily_returns.mean(),
        'mean_turnover': float(np.mean(turnover_values)) if turnover_values else 0.0,
        'turnover_values': turnover_values,
        'mean_mrr': canonical_metrics['MRR'],
        'mean_ndcg5': canonical_metrics['NDCG@5'],
        'precision_at_5': canonical_metrics['Precision@5'],
        'mean_precision_at_k_up': canonical_metrics['Precision@5'],
        'precision_at_5_valid_dates': len(daily_precision5_scores),
        'precision_at_5_daily_df': precision_at_5_daily_df,
        'MSE': canonical_metrics['MSE'],
        'RMSE': canonical_metrics['RMSE'],
        'MAE': canonical_metrics['MAE'],
        'IC': canonical_metrics['IC'],
        'RankIC': canonical_metrics['RankIC'],
        'RIC': canonical_metrics['RankIC'],
        'n_trade_days': len(daily_returns),
        'n_trades': len(trade_log),
        'portfolio_values': portfolio_values,
        'daily_returns': daily_returns,
        'dates': trade_dates_log,
        'trade_log': trade_log,
    }


# ==================== Phase回测 ====================

def backtest_phase(model, market_data, news_data, labels_data, phase, tickers,
                   relation_static, S_pre, device, rnn_length=5,
                   top_k=5, transaction_cost=0.0,
                   seed=11380):
    """
    对单个 phase 执行预测、回测和显式关系记忆干涉。
    
    Args:
        model: DRFN模型（已加载该phase的最优权重）
        market_data: 全量归一化市场数据
        news_data: 全量新闻数据
        labels_data: 全量target数据
        phase: dict, phase信息（含train/val/test日期）
        tickers: list, 股票列表
        relation_static: torch.Tensor
        S_pre: torch.Tensor
        device: str
        rnn_length: int
        top_k: int, 回测选股数量
        transaction_cost: float, 交易成本
    
    Returns:
        dict: 回测指标、预测和显式关系记忆干涉
    """
    all_market_dates = sorted(pd.to_datetime(market_data['Date'].unique()))
    strict_test_dates = get_strict_prediction_dates(phase['test_dates'], all_market_dates)
    assert_strict_prediction_dates(
        strict_test_dates, phase['test_dates'], all_market_dates, f"phase{phase['phase_id']}/test"
    )
    all_test_dates = strict_test_dates.tolist()
    if not all_test_dates:
        return {'error': 'no strict test prediction dates'}
    if model.news_encoder._precomputed_cls is None or len(model.news_encoder._precomputed_cls) == 0:
        raise RuntimeError(
            "TinyBERT CLS cache is empty before backtest inference. "
            "Call model.news_encoder.precompute_bert_cls(relevant_news_data) first."
        )
    all_dates_sorted = all_market_dates
    all_dates_dt = pd.to_datetime(all_dates_sorted)

    # 推理日期：测试首日前 rnn_length-1 个交易日 + 测试期。
    # 每个 prediction Date=t 的窗口为 [t-rnn_length+1, ..., t]。
    test_dates_ts = pd.to_datetime(pd.Series(all_test_dates)).sort_values().reset_index(drop=True)
    first_test_pos = all_dates_dt.get_loc(pd.Timestamp(test_dates_ts.iloc[0]))
    last_test_pos = all_dates_dt.get_loc(pd.Timestamp(test_dates_ts.iloc[-1]))
    start_pos = max(0, first_test_pos - rnn_length + 1)
    inference_dates = all_dates_dt[start_pos:last_test_pos + 1].tolist()
    
    if len(inference_dates) <= rnn_length:
        return {'error': 'inference dates too short'}
    
    # ===== 1. 动态关系记忆干预推理 =====
    print(f"\nPhase {phase['phase_id']} Explicit Relation Memory Intervention中...")

    predictions_df, eval_dates = intervention_inference(
        model, market_data, news_data, inference_dates, tickers,
        relation_static, S_pre, rnn_length, mode="full", seed=seed
    )
    predictions_no_memory_df, _ = intervention_inference(
        model, market_data, news_data, inference_dates, tickers,
        relation_static, S_pre, rnn_length, mode="no_memory", seed=seed
    )
    predictions_shuffled_df, _ = intervention_inference(
        model, market_data, news_data, inference_dates, tickers,
        relation_static, S_pre, rnn_length, mode="shuffled_memory", seed=seed
    )

    # 只保留测试期内的预测
    test_dates_ts = [pd.Timestamp(d) for d in all_test_dates]
    predictions_test = predictions_df[predictions_df.index.isin(test_dates_ts)]
    predictions_no_memory_test = predictions_no_memory_df[predictions_no_memory_df.index.isin(test_dates_ts)]
    predictions_shuffled_test = predictions_shuffled_df[predictions_shuffled_df.index.isin(test_dates_ts)]

    if predictions_test.empty:
        return {'error': 'no predictions in test period'}

    # ===== 2. 回测 =====
    # 构建真实收益率矩阵：Date=t 已经是 Close_t -> Close_{t+1}。
    test_labels = labels_data[labels_data['Date'].isin(all_test_dates)]
    true_returns_df = test_labels.pivot(index='Date', columns='Ticker', values='target').sort_index()
    
    # 回测：Full 主模型 + 两种动态记忆干预
    results = _safe_backtest(
        predictions_test, true_returns_df,
        initial_cash=1000000, top_k=top_k, transaction_cost=transaction_cost
    )
    results_no_memory = _safe_backtest(
        predictions_no_memory_test, true_returns_df,
        initial_cash=1000000, top_k=top_k, transaction_cost=transaction_cost
    )
    results_shuffled = _safe_backtest(
        predictions_shuffled_test, true_returns_df,
        initial_cash=1000000, top_k=top_k, transaction_cost=transaction_cost
    )

    if pd.isna(results.get('sharpe_ratio', float('nan'))):
        return {'error': 'backtest returned None'}

    print(f"  Memory Intervention: "
          f"Full Sharpe={results['sharpe_ratio']:.4f}, "
          f"No Explicit Relation Memory Sharpe={results_no_memory['sharpe_ratio']:.4f}, "
          f"Shuffled Explicit Relation Memory Sharpe={results_shuffled['sharpe_ratio']:.4f}")
    
    return {
        'phase_id': phase['phase_id'],
        'sharpe_ratio': results['sharpe_ratio'],
        'annualized_return': results['annualized_return'],
        'annualized_volatility': results['annualized_volatility'],
        'avg_daily_return': results['avg_daily_return'],
        'max_drawdown': results['max_drawdown'],
        'mean_mrr': results['mean_mrr'],
        'mean_ndcg5': results['mean_ndcg5'],
        'precision_at_5': results['precision_at_5'],
        'mean_precision_at_k_up': results['mean_precision_at_k_up'],
        'precision_at_5_valid_dates': results['precision_at_5_valid_dates'],
        'precision_at_5_daily_df': results['precision_at_5_daily_df'],
        'predictions_df': predictions_test,
        'true_returns_df': true_returns_df,
        'MSE': results.get('MSE', float('nan')),
        'RMSE': results.get('RMSE', float('nan')),
        'MAE': results.get('MAE', float('nan')),
        'IC': results.get('IC', float('nan')),
        'RankIC': results.get('RankIC', float('nan')),
        'RIC': results.get('RIC', float('nan')),
        'total_return': results['total_return'],
        'final_value': results['final_value'],
        'n_trade_days': results['n_trade_days'],
        'n_trades': results['n_trades'],
        # 原始数据，main用来拼接做连贯计算
        'daily_returns': results['daily_returns'],
        'portfolio_values': results['portfolio_values'],
        'dates': results['dates'],
        # Explicit Relation Memory Intervention
        'memory_full_sharpe': results['sharpe_ratio'],
        'memory_full_annualized_return': results['annualized_return'],
        'memory_full_total_return': results['total_return'],
        'memory_full_max_drawdown': results['max_drawdown'],
        'memory_full_final_value': results['final_value'],

        'memory_no_memory_sharpe': results_no_memory['sharpe_ratio'],
        'memory_no_memory_annualized_return': results_no_memory['annualized_return'],
        'memory_no_memory_total_return': results_no_memory['total_return'],
        'memory_no_memory_max_drawdown': results_no_memory['max_drawdown'],
        'memory_no_memory_final_value': results_no_memory['final_value'],

        'memory_shuffled_sharpe': results_shuffled['sharpe_ratio'],
        'memory_shuffled_annualized_return': results_shuffled['annualized_return'],
        'memory_shuffled_total_return': results_shuffled['total_return'],
        'memory_shuffled_max_drawdown': results_shuffled['max_drawdown'],
        'memory_shuffled_final_value': results_shuffled['final_value'],

        'memory_no_memory_drop_sharpe': results['sharpe_ratio'] - results_no_memory['sharpe_ratio'],
        'memory_shuffled_drop_sharpe': results['sharpe_ratio'] - results_shuffled['sharpe_ratio'],
        'memory_no_memory_drop_total_return': results['total_return'] - results_no_memory['total_return'],
        'memory_shuffled_drop_total_return': results['total_return'] - results_shuffled['total_return'],
    }


# ==================== 独立运行入口 ====================

def main():
    """
    独立运行回测（加载已训练好的模型，在单个phase上回测）
    用法: python backtesting.py <模型路径> [phase_id]
    """
    parser = argparse.ArgumentParser(description="Standalone Daily-Recurrent checkpoint backtest")
    parser.add_argument('checkpoint', type=Path)
    parser.add_argument('phase_id', type=int, nargs='?', default=9, choices=range(1, 10))
    parser.add_argument('--data-dir', default='.')
    parser.add_argument('--bert-path', default='./Tinybert-4layers-en')
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--seed', type=int, default=11380)
    opts = parser.parse_args()
    if not opts.checkpoint.is_file():
        raise FileNotFoundError(f'Checkpoint not found: {opts.checkpoint}')
    set_seed(opts.seed)

    # 参数
    RNN_LENGTH = 5
    TOP_K = 5
    TRANSACTION_COST = 0.0

    # 文件路径
    from project_paths import resolve_assets
    DATA_DIR, resolved_bert = resolve_assets(opts.data_dir, opts.bert_path)
    news_file = os.path.join(DATA_DIR, 'us_news.csv')
    market_file = os.path.join(DATA_DIR, 'us_price_normalized.csv')
    labels_file = os.path.join(DATA_DIR, 'us_target.csv')

    # 加载数据
    print("加载数据...")
    news_df = pd.read_csv(news_file, encoding='utf-8')
    news_df['Date'] = pd.to_datetime(news_df['Date'])
    news_df = news_df.sort_values(by='Date').dropna()
    news_data = news_df[['Ticker', 'Date', 'Headlines']]

    market_df = pd.read_csv(market_file, encoding='utf-8')
    market_df['Date'] = pd.to_datetime(market_df['Date'])
    market_df = market_df.sort_values(by='Date').dropna()
    market_data = market_df[['Ticker', 'Date', 'Open_norm', 'High_norm', 'Low_norm', 'Close_norm', 'Volume_norm']]

    labels_df = pd.read_csv(labels_file, encoding='utf-8')
    labels_df['Date'] = pd.to_datetime(labels_df['Date'])
    labels_df = labels_df.sort_values(by='Date').dropna()
    labels_data = labels_df[['Ticker', 'Date', 'target']]
    labels_data, _ = filter_strict_next_day_labels(labels_data, market_data)

    all_tickers = sorted(labels_data['Ticker'].unique())
    all_dates = sorted(market_data['Date'].unique())

    # 计算phases
    phases = calculate_phases(pd.Series(all_dates))

    # 确定回测phase
    phase_id = opts.phase_id
    phase = None
    for p in phases:
        if p['phase_id'] == phase_id:
            phase = p
            break
    if phase is None:
        print(f"Phase {phase_id} 不存在！可选: {[p['phase_id'] for p in phases]}")
        sys.exit(1)

    # Build the static graph from this phase's training period only.
    # 避免使用全量pkl（全量pkl包含未来年份数据，在量化回测中属于严重的数据泄露）
    print(f"正在基于 Phase {phase_id} 训练集构建静态关系矩阵（防止未来函数）...")
    from data_preprocessing import build_static_relation
    phase_relation_matrix, _ = build_static_relation(
        market_data,
        threshold=DRFN_KWARGS.get('static_thresh', 0.5),
        start_date=phase['train_start'],
        end_date=phase['train_end'],
    )
    relation_static = torch.tensor(phase_relation_matrix, dtype=torch.float32, device=opts.device)
    num_stock = relation_static.size(1)
    S_pre = relation_static.clone()

    print(f"\n回测 Phase {phase_id}")
    print(f"  Train: {phase['train_start'].strftime('%Y-%m-%d')} ~ {phase['train_end'].strftime('%Y-%m-%d')}")
    print(f"  Val:   {phase['val_start'].strftime('%Y-%m-%d')} ~ {phase['val_end'].strftime('%Y-%m-%d')}")
    print(f"  Test:  {phase['test_start'].strftime('%Y-%m-%d')} ~ {phase['test_end'].strftime('%Y-%m-%d')}")

    # 加载模型
    bert_path = str(resolved_bert)
    if not os.path.isdir(bert_path):
        raise FileNotFoundError(
            "English TinyBERT directory not found: ./Tinybert-4layers-en"
        )

    model = DRFN(
        pretrained_model_path=bert_path,
        dates=phase['train_dates'].tolist(),
        tickers=all_tickers,
        num_stock=num_stock,
        hidden_news=768,
        hidden_market=64,
        d_hidden=64,
        hidn_rnn=256,
        heads_att=8,
        hidn_att=512,
        dropout=0.4,
        alpha=0.1,
        **DRFN_KWARGS,
    )
    model.to(device=opts.device, dtype=torch.float32)

    model_path = opts.checkpoint

    print(f"加载模型: {model_path}")
    state = torch.load(model_path, map_location=opts.device)
    model.load_state_dict(state)
    global_market_dates = pd.DatetimeIndex(sorted(market_data['Date'].unique()))
    first_test_date = pd.Timestamp(min(phase['test_dates']))
    last_test_date = pd.Timestamp(max(phase['test_dates']))
    first_test_pos = global_market_dates.get_loc(first_test_date)
    cache_start = global_market_dates[max(0, first_test_pos - RNN_LENGTH + 1)]
    relevant_news_data = news_data[
        (news_data['Date'] >= cache_start) & (news_data['Date'] <= last_test_date)
    ]
    model.news_encoder.precompute_bert_cls(relevant_news_data)
    if model.news_encoder._precomputed_cls is None or len(model.news_encoder._precomputed_cls) == 0:
        raise RuntimeError("Standalone backtest TinyBERT CLS cache is empty.")
    model.eval()

    results = backtest_phase(
        model, market_data, news_data, labels_data, phase, all_tickers,
        relation_static, S_pre, opts.device, rnn_length=RNN_LENGTH,
        top_k=TOP_K, transaction_cost=TRANSACTION_COST,
        seed=opts.seed,
    )

    if 'error' in results:
        print(f"回测失败: {results['error']}")
        sys.exit(1)

    print('\n' + '=' * 60)
    print(f'Phase {phase_id} 回测结果')
    print('=' * 60)
    print(f"夏普比率 (SR):      {results['sharpe_ratio']:.4f}")
    print(f"年化收益率:         {results['annualized_return'] * 100:.2f}%")
    print(f"平均日收益率:       {results['avg_daily_return'] * 100:.4f}%")
    print(f"总收益率:           {results['total_return'] * 100:.2f}%")
    print(f"最大回撤:           {results['max_drawdown'] * 100:.2f}%")
    print(f"最终资金:           {results['final_value']:,.2f}")
    print(f"MRR:                {results['mean_mrr']:.4f}")
    print(f"NDCG@5:             {results['mean_ndcg5']:.4f}")
    print(f"Precision@5:        {results['precision_at_5']:.4f}")
    print(f"MSE:                {results.get('MSE', float('nan')):.8f}")
    print(f"IC:                 {results.get('IC', float('nan')):.4f}")
    print(f"RankIC:             {results.get('RankIC', float('nan')):.4f}")
    print(f"回测天数:           {results['n_trade_days']}")
    print(f"交易次数:           {results['n_trades']}")

    # Explicit Relation Memory Intervention结果
    print(f"\n{'=' * 60}")
    print(f'Phase {phase_id} Explicit Relation Memory Intervention结果')
    print('=' * 60)
    print(f"Full Sharpe:          {results.get('memory_full_sharpe', float('nan')):.4f}")
    print(f"No Explicit Relation Memory Sharpe:      {results.get('memory_no_memory_sharpe', float('nan')):.4f}")
    print(f"Shuffled Explicit Relation Memory Sharpe:      {results.get('memory_shuffled_sharpe', float('nan')):.4f}")
    print(f"No Explicit Relation Memory ΔSharpe:     {results.get('memory_no_memory_drop_sharpe', float('nan')):.4f}")
    print(f"Shuffled Explicit Relation Memory ΔSharpe:     {results.get('memory_shuffled_drop_sharpe', float('nan')):.4f}")



if __name__ == '__main__':
    main()
