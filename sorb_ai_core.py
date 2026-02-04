"""
SORB AI - Core Strategy Engine
==============================

This is the base SORB strategy that the AI will enhance.
Copied from the proven working strategy with optimized parameters.

Author: Quant Research AI
"""

import pandas as pd
import numpy as np
from datetime import datetime, time, timedelta
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')


@dataclass
class StrategyConfig:
    """Strategy parameters - proven optimal values"""

    # Timezone offset: UTC+2 -> ET
    timezone_offset_hours: int = -7

    # Opening Range (ET)
    or_start_hour: int = 9
    or_start_min: int = 30
    or_duration_min: int = 45  # Optimized

    # Entry Filters
    volume_mult: float = 1.2
    rsi_period: int = 14
    rsi_long_low: int = 50
    rsi_long_high: int = 70  # Optimized
    adx_period: int = 14
    adx_threshold: int = 25  # Optimized

    # Targets
    tp1_mult: float = 0.8  # Optimized
    tp2_mult: float = 1.2  # Optimized

    # Direction
    long_only: bool = True  # Critical - shorts have negative edge

    # Risk Management
    risk_per_trade_pct: float = 1.0  # Increased for challenge
    max_trades_per_day: int = 3
    daily_loss_guard_pct: float = 2.0
    total_dd_guard_pct: float = 5.0

    # Session Times (ET)
    entry_start_hour: int = 9
    entry_start_min: int = 45
    entry_end_hour: int = 14
    entry_end_min: int = 30
    flat_hour: int = 15
    flat_min: int = 45

    # HTF Filter
    ema_period: int = 20

    # Execution Costs
    commission_pct: float = 0.01
    slippage_ticks: float = 1.0
    spread_ticks: float = 1.5
    tick_size: float = 0.25


@dataclass
class Trade:
    """Trade record"""
    entry_time: datetime
    exit_time: datetime
    direction: int
    entry_price: float
    exit_price: float
    stop_loss: float
    tp1: float
    tp2: float
    position_size: float
    pnl: float
    pnl_r: float
    exit_reason: str
    signal_type: str  # 'SORB' or 'AI_ENHANCED'
    confidence: float  # 0-1 confidence score


class Indicators:
    """Technical indicator calculations"""

    @staticmethod
    def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
        high, low, close = df['high'], df['low'], df['close']
        tr = pd.concat([
            high - low,
            abs(high - close.shift(1)),
            abs(low - close.shift(1))
        ], axis=1).max(axis=1)
        return tr.rolling(window=period).mean()

    @staticmethod
    def rsi(df: pd.DataFrame, period: int = 14) -> pd.Series:
        delta = df['close'].diff()
        gain = delta.where(delta > 0, 0).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))

    @staticmethod
    def ema(series: pd.Series, period: int) -> pd.Series:
        return series.ewm(span=period, adjust=False).mean()

    @staticmethod
    def sma(series: pd.Series, period: int) -> pd.Series:
        return series.rolling(window=period).mean()

    @staticmethod
    def adx(df: pd.DataFrame, period: int = 14) -> pd.Series:
        high, low, close = df['high'], df['low'], df['close']

        plus_dm = high.diff()
        minus_dm = -low.diff()
        plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0)
        minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0)

        tr = pd.concat([
            high - low,
            abs(high - close.shift(1)),
            abs(low - close.shift(1))
        ], axis=1).max(axis=1)

        atr = tr.rolling(window=period).mean()
        plus_di = 100 * (plus_dm.rolling(window=period).mean() / atr)
        minus_di = 100 * (minus_dm.rolling(window=period).mean() / atr)

        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
        return dx.rolling(window=period).mean()

    @staticmethod
    def bollinger_bands(df: pd.DataFrame, period: int = 20, std_dev: float = 2.0):
        middle = df['close'].rolling(period).mean()
        std = df['close'].rolling(period).std()
        upper = middle + std_dev * std
        lower = middle - std_dev * std
        return middle, upper, lower

    @staticmethod
    def macd(df: pd.DataFrame, fast: int = 12, slow: int = 26, signal: int = 9):
        fast_ema = df['close'].ewm(span=fast, adjust=False).mean()
        slow_ema = df['close'].ewm(span=slow, adjust=False).mean()
        macd_line = fast_ema - slow_ema
        signal_line = macd_line.ewm(span=signal, adjust=False).mean()
        histogram = macd_line - signal_line
        return macd_line, signal_line, histogram


def load_data(filepath: str) -> pd.DataFrame:
    """Load OHLCV data from CSV"""
    try:
        df = pd.read_csv(filepath, sep='\t')
        if len(df.columns) <= 2:
            df = pd.read_csv(filepath)
    except:
        df = pd.read_csv(filepath)

    df.columns = df.columns.str.lower().str.strip()

    # Find datetime column
    dt_col = None
    for col in ['datetime', 'date', 'time', 'timestamp']:
        if col in df.columns:
            dt_col = col
            break
    if dt_col is None:
        dt_col = df.columns[0]

    # Parse datetime
    try:
        df[dt_col] = pd.to_datetime(df[dt_col])
    except:
        df[dt_col] = pd.to_datetime(df[dt_col], format='%Y.%m.%d %H:%M:%S')

    df.set_index(dt_col, inplace=True)
    df.sort_index(inplace=True)

    # Handle volume
    if 'volume' not in df.columns or df['volume'].sum() == 0:
        if 'tickvolume' in df.columns:
            df['volume'] = df['tickvolume']
        else:
            df['volume'] = 1

    return df[['open', 'high', 'low', 'close', 'volume']]


def prepare_data(df: pd.DataFrame, config: StrategyConfig) -> pd.DataFrame:
    """Add all indicators and markers"""
    df = df.copy()

    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)

    # Timezone conversion
    if config.timezone_offset_hours != 0:
        df.index = df.index + pd.Timedelta(hours=config.timezone_offset_hours)

    # Basic indicators
    df['atr'] = Indicators.atr(df, 14)
    df['rsi'] = Indicators.rsi(df, config.rsi_period)
    df['adx'] = Indicators.adx(df, config.adx_period)
    df['ema20'] = Indicators.ema(df['close'], config.ema_period)
    df['ema50'] = Indicators.ema(df['close'], 50)
    df['vol_sma'] = Indicators.sma(df['volume'], 20)

    # Bollinger Bands
    df['bb_mid'], df['bb_upper'], df['bb_lower'] = Indicators.bollinger_bands(df)
    df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])

    # MACD
    df['macd'], df['macd_signal'], df['macd_hist'] = Indicators.macd(df)

    # Returns
    df['ret_1'] = df['close'].pct_change(1) * 100
    df['ret_6'] = df['close'].pct_change(6) * 100
    df['ret_12'] = df['close'].pct_change(12) * 100

    # Volatility
    df['volatility'] = df['ret_1'].rolling(20).std()

    # Session markers
    df['hour'] = df.index.hour
    df['minute'] = df.index.minute
    df['date'] = df.index.date
    df['day_of_week'] = df.index.dayofweek
    df['time_min'] = df['hour'] * 60 + df['minute']

    # Opening range period
    or_start = config.or_start_hour * 60 + config.or_start_min
    or_end = or_start + config.or_duration_min
    df['is_or_period'] = (df['time_min'] >= or_start) & (df['time_min'] < or_end)

    # Entry window
    entry_start = config.entry_start_hour * 60 + config.entry_start_min
    entry_end = config.entry_end_hour * 60 + config.entry_end_min
    df['is_entry_window'] = (df['time_min'] >= entry_start) & (df['time_min'] <= entry_end)

    # Flat time
    flat_time = config.flat_hour * 60 + config.flat_min
    df['is_flat_time'] = df['time_min'] >= flat_time

    # Daily high/low for context
    df['daily_high'] = df.groupby('date')['high'].transform('max')
    df['daily_low'] = df.groupby('date')['low'].transform('min')
    df['daily_range'] = df['daily_high'] - df['daily_low']
    df['daily_position'] = (df['close'] - df['daily_low']) / df['daily_range'].replace(0, np.nan)

    return df


def calculate_opening_range(df: pd.DataFrame, date, config: StrategyConfig) -> Optional[Dict]:
    """Calculate opening range for a specific date"""
    day_data = df[df['date'] == date]
    or_data = day_data[day_data['is_or_period']]

    if len(or_data) < 2:
        return None

    or_high = or_data['high'].max()
    or_low = or_data['low'].min()
    or_size = or_high - or_low

    # Get daily ATR (scale 5min ATR by ~78 bars per day, sqrt for volatility)
    # Or just use the OR size itself as reference - more practical
    atr_5min = day_data['atr'].iloc[-1] if len(day_data) > 0 else None
    if atr_5min is None or pd.isna(atr_5min) or atr_5min == 0:
        return None

    # Estimate daily ATR from 5min (roughly sqrt(78) * 5min ATR for scaling)
    # But simpler: just use a reasonable absolute range check
    # NAS100 typical daily range is 100-400 points
    # OR should be 20-150 points typically

    # More lenient check: OR size should be reasonable (10-200 points for NAS100)
    if or_size < 10 or or_size > 300:
        return None

    # Also check that OR isn't too small relative to recent volatility
    if or_size < atr_5min * 2:  # At least 2x the 5-min ATR
        return None

    # Use OR size as the reference for targets
    atr = or_size

    # Bias at OR end
    or_end_bar = or_data.iloc[-1]
    close_at_end = or_end_bar['close']
    ema_at_end = or_end_bar['ema20']
    adx_at_end = or_end_bar['adx']

    # Determine bias
    if close_at_end > ema_at_end:
        bias = 1
    elif close_at_end < ema_at_end:
        bias = -1
    else:
        bias = 0

    is_trending = adx_at_end > config.adx_threshold if not pd.isna(adx_at_end) else False

    return {
        'or_high': or_high,
        'or_low': or_low,
        'or_size': or_size,
        'or_mid': (or_high + or_low) / 2,
        'bias': bias,
        'is_trending': is_trending,
        'atr': atr
    }


def check_sorb_signal(bar, prev_bar, or_info: Dict, config: StrategyConfig) -> Tuple[Optional[int], float]:
    """
    Check for standard SORB entry signal
    Returns (direction, confidence) or (None, 0)
    """
    if or_info is None or or_info['bias'] == 0 or not or_info['is_trending']:
        return None, 0

    # Volume filter
    if bar['volume'] <= bar['vol_sma'] * config.volume_mult:
        return None, 0

    # Long breakout
    if or_info['bias'] == 1:
        if bar['close'] > or_info['or_high'] and prev_bar['close'] <= or_info['or_high']:
            if config.rsi_long_low < bar['rsi'] < config.rsi_long_high:
                return 1, 1.0  # Full confidence for standard SORB

    # Short breakout (if not long_only)
    if not config.long_only and or_info['bias'] == -1:
        if bar['close'] < or_info['or_low'] and prev_bar['close'] >= or_info['or_low']:
            rsi_short_low = 100 - config.rsi_long_high
            rsi_short_high = 100 - config.rsi_long_low
            if rsi_short_low < bar['rsi'] < rsi_short_high:
                return -1, 1.0

    return None, 0
