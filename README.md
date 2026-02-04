# SORB-AI: Machine Learning Enhanced Trading Strategy for NAS100

An ML-enhanced Session Opening Range Breakout (SORB) strategy for NAS100/US100 CFD trading, designed for prop firm challenges.

## Strategy Overview

This strategy combines classical technical analysis (Opening Range Breakout) with machine learning to identify high-probability trade setups.

### Core Concept
- **Opening Range (OR)**: 9:30 - 10:15 AM ET (first 45 minutes of US session)
- **Entry**: Breakout above OR high with ML confirmation
- **Direction**: Long-only (shorts have negative historical edge)
- **Targets**: Based on OR size (0.8x for TP1, 1.2x for TP2)

### ML Enhancement
The Gradient Boosting model analyzes 35 features to filter signals:
- Price momentum & mean reversion indicators
- Volatility regime detection
- Volume patterns
- Session timing optimization

## Backtest Results

| Metric | Value |
|--------|-------|
| Total Trades | 136 |
| Win Rate | 57.4% |
| Expectancy | 1.008R |
| Profit Factor | 2.52 |
| Max Drawdown | ~5.35% |

*Results from 50/50 train/test split on 9 years of data (2016-2025)*

## Project Structure

```
SORB-AI-NAS100/
├── sorb_ai_core.py        # Core strategy engine & indicators
├── ml_signal_generator.py # ML model training & prediction
├── sorb_ai_strategy.py    # Combined strategy execution
├── backtest_with_charts.py # Backtesting with visualization
└── README.md
```

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/SORB-AI-NAS100.git
cd SORB-AI-NAS100

# Install dependencies
pip install pandas numpy scikit-learn matplotlib seaborn
```

## Usage

### 1. Prepare Your Data

Export 5-minute NAS100 data from your broker (MT4/MT5/TradingView) with columns:
- DateTime, Open, High, Low, Close, Volume

Save as `5m_data.csv` in the project folder.

### 2. Run Backtest

```python
python backtest_with_charts.py
```

This generates:
- `backtest_report.png` - Visual analysis charts
- `trade_log.csv` - Detailed trade history
- Console output with statistics

### 3. Analyze Results

The backtest produces:
- Equity curve
- Drawdown analysis
- Monthly returns heatmap
- Performance by signal type
- Best trading hours/days

## Strategy Parameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| OR Duration | 45 min | Opening range period |
| Risk/Trade | 1.0% | Position sizing |
| TP1 | 0.8x OR | First profit target |
| TP2 | 1.2x OR | Final profit target |
| RSI Filter | 50-70 | Momentum confirmation |
| ADX Threshold | 25 | Trend strength filter |
| ML Confidence | 0.55+ | Minimum signal threshold |

## Risk Management

- **Max Daily Loss**: 3%
- **Max Drawdown Guard**: 5%
- **Max Trades/Day**: 3
- **Long-Only**: Short signals disabled

## Important Notes

1. **This is research code** - Not production-ready for live trading
2. **Past performance ≠ future results** - Markets change
3. **Paper trade first** - Validate on current market conditions
4. **Update data regularly** - ML models degrade with stale data

## ML Model Details

### Features Used (Top 10 by importance)
1. `daily_position` (22.4%) - Price position in daily range
2. `atr_pct` (12.4%) - ATR as percentage of price
3. `dist_ma50` (9.7%) - Distance from 50-period MA
4. `atr` (6.0%) - Absolute ATR value
5. `vol_ratio` (5.2%) - Current vs historical volatility
6. `volatility` (4.6%) - Rolling return std dev
7. `adx` (4.0%) - Trend strength
8. `time_since_open` (3.5%) - Minutes since market open
9. `macd_hist` (3.1%) - MACD histogram
10. `vol_trend` (2.9%) - Volume trend

### Model Architecture
- **Type**: Gradient Boosting Classifier
- **Estimators**: 100
- **Max Depth**: 10
- **Train/Test Split**: Time-series aware (no lookahead bias)
