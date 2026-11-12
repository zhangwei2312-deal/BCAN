"""
数据预处理脚本
功能：
1. 读取原始数据（cn_price/cn_news/cn_target）
2. 对市场数据进行日内Z-Score归一化
3. 构建股票关系矩阵
4. 保存处理好的数据

注意：日内归一化已由 normalize_price.py 单独完成
本脚本主要负责：构建关系矩阵、验证数据、打印phase划分信息

运行方式：python data_preprocessing.py
"""

import pandas as pd
import numpy as np
import os
import pickle
from datetime import datetime

# ==================== 配置 ====================
DATA_DIR = '.'  # 当前目录


# ==================== 构建静态关系矩阵 ====================
def build_static_relation(market_norm_df, threshold=0.5, start_date=None, end_date=None,
                          mode="level", raw_price_path=None):
    """
    基于归一化后收盘价的相关性构建静态关系矩阵

    Args:
        market_norm_df: 归一化后的市场数据 (cn_price_normalized.csv)
        threshold: 相关性阈值
        start_date: 可选，只使用此日期之后的数据（用于phase级构建，防止未来信息泄露）
        end_date: 可选，只使用此日期之前的数据
        mode: "level"  = 逐日截面 z-score 后的收盘价（Close_norm），论文的参考配置
              "return" = 原始收盘价的日对数收益，用于与 level 图做对比
        raw_price_path: mode="return" 时必填，指向原始行情 CSV（含 Ticker/Date/Close）

    Returns:
        relation_matrix: numpy array, shape (num_stocks, num_stocks)
        tickers: list of stock tickers
    """
    desc = ""
    if start_date or end_date:
        desc = f" (时间范围: {start_date or '最早'} ~ {end_date or '最晚'})"
    print(f"正在构建静态关系矩阵{desc}...")

    # 按时间范围筛选数据（防止未来信息泄露）
    df = market_norm_df.copy()
    if start_date is not None:
        df = df[df['Date'] >= pd.Timestamp(start_date)]
    if end_date is not None:
        df = df[df['Date'] <= pd.Timestamp(end_date)]

    # ===== 输入序列选择 =====
    if mode == "return":
        if raw_price_path is None:
            raise ValueError("mode='return' requires raw_price_path pointing at the raw price CSV")
        raw = pd.read_csv(raw_price_path, usecols=['Ticker', 'Date', 'Close'])
        raw['Date'] = pd.to_datetime(raw['Date'])
        if start_date is not None:
            raw = raw[raw['Date'] >= pd.Timestamp(start_date)]
        if end_date is not None:
            raw = raw[raw['Date'] <= pd.Timestamp(end_date)]
        pivot = raw.pivot(index='Date', columns='Ticker', values='Close').sort_index()
        series = np.log(pivot.where(pivot > 0)).diff()
        corr_matrix = series.corr(min_periods=30)
        print(f"  输入模式: return（日对数收益，{series.shape[0]} 个交易日）")
    else:
        pivot = df.pivot(index='Date', columns='Ticker', values='Close_norm')
        corr_matrix = pivot.corr()
        print(f"  输入模式: level（逐日截面 z-score 收盘价，{pivot.shape[0]} 个交易日）")
    
    tickers = corr_matrix.columns.tolist()
    num_stocks = len(tickers)
    
    relation_matrix = (corr_matrix.values > threshold).astype(float)
    np.fill_diagonal(relation_matrix, 1)
    
    print(f"  关系矩阵: {num_stocks} x {num_stocks}")
    print(f"  正相关关系比例: {(relation_matrix.sum() - num_stocks) / (num_stocks * (num_stocks - 1)) * 100:.2f}%")
    
    return relation_matrix, tickers


# ==================== 主函数 ====================
def main():
    print("=" * 60)
    print("DRFN 数据预处理")
    print("=" * 60)
    
    # 1. 检查数据文件是否存在
    print("\n[1/4] 检查数据文件...")
    files_needed = {
        'cn_price.csv': '原始行情',
        'cn_price_normalized.csv': '归一化行情（由normalize_price.py生成）',
        'cn_target.csv': '目标收益率',
        'cn_news.csv': '新闻标题',
    }
    
    for fname, desc in files_needed.items():
        fpath = os.path.join(DATA_DIR, fname)
        if os.path.exists(fpath):
            size = os.path.getsize(fpath)
            print(f"  ✅ {fname} ({size:,} bytes) - {desc}")
        else:
            if fname == 'cn_price_normalized.csv':
                print(f"  ❌ {fname} 不存在! 请先运行: python normalize_price.py")
            else:
                print(f"  ❌ {fname} 不存在!")
    
    # 2. 读取并验证数据
    print("\n[2/4] 读取并验证数据...")
    
    price_df = pd.read_csv(f'{DATA_DIR}/cn_price.csv')
    norm_df = pd.read_csv(f'{DATA_DIR}/cn_price_normalized.csv')
    target_df = pd.read_csv(f'{DATA_DIR}/cn_target.csv')
    news_df = pd.read_csv(f'{DATA_DIR}/cn_news.csv')
    
    print(f"  原始行情: {len(price_df):,} 行, {price_df['Ticker'].nunique()} 只股票")
    print(f"  归一化行情: {len(norm_df):,} 行, {norm_df['Ticker'].nunique()} 只股票")
    print(f"  目标数据: {len(target_df):,} 行, {target_df['Ticker'].nunique()} 只股票")
    print(f"  新闻数据: {len(news_df):,} 行, {news_df['Ticker'].nunique()} 只股票")
    
    # 验证对齐
    price_pairs = set(zip(price_df['Ticker'], price_df['Date']))
    norm_pairs = set(zip(norm_df['Ticker'], norm_df['Date']))
    target_pairs = set(zip(target_df['Ticker'], target_df['Date']))
    news_pairs = set(zip(news_df['Ticker'], news_df['Date']))
    
    if price_pairs == norm_pairs == target_pairs == news_pairs:
        print("  ✅ 四个文件 (Ticker,Date) 完全对齐")
    else:
        print("  ❌ 文件不对齐!")
    
    # 3. 构建关系矩阵
    print("\n[3/4] 构建静态关系矩阵...")
    relation_matrix, tickers = build_static_relation(norm_df, threshold=0.5)
    
    # 保存关系矩阵
    rel_path = f'{DATA_DIR}/cn_relationships.pkl'
    with open(rel_path, 'wb') as f:
        pickle.dump({
            'relation_matrix': relation_matrix,
            'tickers': tickers
        }, f)
    print(f"  已保存: {rel_path}")
    
    # 4. 打印phase划分信息
    print("\n[4/4] Phase划分预览...")
    from data_splits import calculate_phases
    
    all_dates = pd.to_datetime(price_df['Date']).sort_values().unique()
    phases = calculate_phases(pd.Series(all_dates))
    
    print(f"  总共 {len(phases)} 个phase:")
    for p in phases:
        print(f"    Phase {p['phase_id']}: "
              f"Train {p['train_start'].year}~{p['train_end'].year} ({len(p['train_dates'])}d) | "
              f"Val {p['val_start'].year} ({len(p['val_dates'])}d) | "
              f"Test {p['test_start'].year} ({len(p['test_dates'])}d)")
    
    print("\n" + "=" * 60)
    print("数据预处理完成！可以运行训练:")
    print("  python run_n2m_allstocks_dailyrecurrent_cn.py --phases 1 --device cuda:0")
    print("=" * 60)


if __name__ == '__main__':
    main()
