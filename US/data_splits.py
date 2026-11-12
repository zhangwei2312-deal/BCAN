"""
数据集划分：滑动窗口多阶段划分
训练集3年 + 验证集1年 + 测试集1年，滑动步长1年
适用于13年数据（如2010-2022），产生9个phase
"""
import pandas as pd
from datetime import datetime


def calculate_phases(all_dates):
    """
    根据实际交易日计算所有phase的日期范围
    
    训练3年 + 验证1年 + 测试1年 = 5年窗口
    滑动步长1年
    
    Args:
        all_dates: pd.Series of datetime, 所有交易日（已排序）
    
    Returns:
        phases: list of dict, 每个包含 phase_id, train_dates, val_dates, test_dates (pd.Series)
    """
    dates = pd.Series(pd.to_datetime(all_dates)).sort_values().reset_index(drop=True)
    
    TRAIN_YEARS = 3
    VAL_YEARS = 1
    TEST_YEARS = 1
    SLIDE_YEARS = 1
    WINDOW_YEARS = TRAIN_YEARS + VAL_YEARS + TEST_YEARS  # 5
    
    min_date = dates.min()
    max_date = dates.max()
    
    # 计算总年数
    total_years = (max_date.year - min_date.year) + 1  # 13 for 2010-2022
    
    num_phases = total_years - WINDOW_YEARS + 1  # 13 - 5 + 1 = 9
    
    phases = []
    
    for i in range(num_phases):
        # 每个phase的起始年份偏移
        train_start_year = min_date.year + i
        
        # 训练集：起始年 ~ 起始年+2（3年）
        train_start = datetime(train_start_year, 1, 1)
        train_end = datetime(train_start_year + TRAIN_YEARS - 1, 12, 31)
        
        # 验证集：起始年+3（1年）
        val_start = datetime(train_start_year + TRAIN_YEARS, 1, 1)
        val_end = datetime(train_start_year + TRAIN_YEARS + VAL_YEARS - 1, 12, 31)
        
        # 测试集：起始年+4（1年）
        test_start = datetime(train_start_year + TRAIN_YEARS + VAL_YEARS, 1, 1)
        test_end = datetime(train_start_year + TRAIN_YEARS + VAL_YEARS + TEST_YEARS - 1, 12, 31)
        
        # 在实际交易日中筛选
        train_dates = dates[(dates >= train_start) & (dates <= train_end)]
        val_dates = dates[(dates >= val_start) & (dates <= val_end)]
        test_dates = dates[(dates >= test_start) & (dates <= test_end)]
        
        # 跳过交易日不足的phase
        if len(train_dates) < 100 or len(val_dates) < 20 or len(test_dates) < 20:
            print(f"  Phase {i+1}: 交易日不足，跳过 (train={len(train_dates)}, val={len(val_dates)}, test={len(test_dates)})")
            continue
        
        phases.append({
            'phase_id': i + 1,
            'train_start': train_start,
            'train_end': train_end,
            'val_start': val_start,
            'val_end': val_end,
            'test_start': test_start,
            'test_end': test_end,
            'train_dates': train_dates,
            'val_dates': val_dates,
            'test_dates': test_dates,
        })
    
    return phases
