"""
This strategy combines:
1. Original SORB signals (high confidence)
2. ML-enhanced signals (when SORB conditions are close)
3. Near-miss detection (pattern matching)
"""

import pandas as pd
import numpy as np
from datetime import datetime
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

from sorb_ai_core import (
    StrategyConfig, Trade, Indicators,
    load_data, prepare_data, calculate_opening_range, check_sorb_signal
)
from ml_signal_generator import (
    MLConfig, MLSignalModel, NearMissDetector, FeatureEngineer,
    print_training_report
)


@dataclass
class AIStrategyConfig:
    """Configuration for AI-enhanced strategy"""

    # Base SORB config
    sorb_config: StrategyConfig = None

    # ML config
    ml_config: MLConfig = None

    # Signal weighting
    sorb_weight: float = 1.0      # Full weight for SORB signals
    ml_weight: float = 0.7        # Reduced weight for pure ML signals
    near_miss_weight: float = 0.8  # Weight for near-miss signals

    # Risk adjustment based on signal type
    sorb_risk_mult: float = 1.0
    ml_risk_mult: float = 0.7     # Less risk for ML-only signals
    near_miss_risk_mult: float = 0.8

    # Minimum combined confidence to trade
    min_confidence: float = 0.55

    # Maximum signals per day (across all types)
    max_signals_per_day: int = 4

    def __post_init__(self):
        if self.sorb_config is None:
            self.sorb_config = StrategyConfig()
        if self.ml_config is None:
            self.ml_config = MLConfig()


class SORBAIStrategy:
    """AI-Enhanced SORB Strategy"""

    def __init__(self, config: AIStrategyConfig = None):
        self.config = config or AIStrategyConfig()
        self.ml_model = MLSignalModel(self.config.ml_config)
        self.near_miss_detector = NearMissDetector()
        self.feature_engineer = FeatureEngineer(self.config.ml_config)
        self.trades: List[Trade] = []
        self.signals_log: List[Dict] = []

    def train_ml_model(self, df: pd.DataFrame) -> Dict:
        """Train the ML model on historical data"""
        print("\n" + "="*60)
        print("TRAINING ML MODEL")
        print("="*60)

        # Prepare data with all indicators
        prepared_df = prepare_data(df, self.config.sorb_config)

        # Train model for long signals
        metrics = self.ml_model.train(prepared_df, direction=1)

        print_training_report(metrics)

        return metrics

    def generate_combined_signal(self, bar, prev_bar, or_info: Dict,
                                  features: pd.DataFrame) -> Tuple[Optional[int], float, str]:
        """
        Generate combined signal from all sources

        Returns:
            (direction, confidence, signal_type)
        """
        signals = []

        # 1. Check standard SORB signal
        sorb_direction, sorb_conf = check_sorb_signal(
            bar, prev_bar, or_info, self.config.sorb_config
        )

        if sorb_direction is not None:
            signals.append({
                'direction': sorb_direction,
                'confidence': sorb_conf * self.config.sorb_weight,
                'type': 'SORB',
                'risk_mult': self.config.sorb_risk_mult
            })

        # 2. Check ML signal (only if no SORB signal)
        if sorb_direction is None and self.ml_model.is_trained:
            try:
                # Make sure features align with training
                feature_cols = self.ml_model.feature_engineer.feature_names
                if len(feature_cols) > 0:
                    # Ensure features has same columns as training
                    features_aligned = features.reindex(columns=feature_cols, fill_value=0)
                    should_trade, ml_prob = self.ml_model.generate_signal(features_aligned)

                    if should_trade:
                        signals.append({
                            'direction': 1,  # Long only for now
                            'confidence': ml_prob * self.config.ml_weight,
                            'type': 'ML',
                            'risk_mult': self.config.ml_risk_mult
                        })
            except Exception as e:
                pass  # Silently skip on errors

        # 3. Near-miss signals DISABLED - they have negative expectancy
        # if len(signals) == 0 and or_info is not None:
        #     is_near_miss, reason, nm_conf = self.near_miss_detector.check_near_miss(
        #         bar, or_info, self.config.sorb_config
        #     )
        #     if is_near_miss:
        #         signals.append({
        #             'direction': 1,
        #             'confidence': nm_conf * self.config.near_miss_weight,
        #             'type': f'NEAR_MISS:{reason}',
        #             'risk_mult': self.config.near_miss_risk_mult
        #         })

        # Combine signals
        if not signals:
            return None, 0, ""

        # Take the highest confidence signal
        best_signal = max(signals, key=lambda x: x['confidence'])

        if best_signal['confidence'] >= self.config.min_confidence:
            return best_signal['direction'], best_signal['confidence'], best_signal['type']

        return None, 0, ""

    def run_backtest(self, df: pd.DataFrame, initial_capital: float = 100000,
                     train_pct: float = 0.5) -> Dict:
        """
        Run backtest with ML training on first portion of data

        Args:
            df: Raw OHLCV data
            initial_capital: Starting capital
            train_pct: Percentage of data to use for ML training
        """
        print("\n" + "="*60)
        print("SORB AI BACKTEST")
        print("="*60)

        # Split data for training and testing
        split_idx = int(len(df) * train_pct)
        train_df = df.iloc[:split_idx]
        test_df = df.iloc[split_idx:]

        print(f"\nData split:")
        print(f"  Training: {len(train_df):,} bars ({train_pct*100:.0f}%)")
        print(f"  Testing:  {len(test_df):,} bars ({(1-train_pct)*100:.0f}%)")

        # Train ML model on training data
        train_prepared = prepare_data(train_df, self.config.sorb_config)
        training_metrics = self.ml_model.train(train_prepared, direction=1)

        if 'error' in training_metrics:
            print(f"Training error: {training_metrics['error']}")
            return {'error': training_metrics['error']}

        print_training_report(training_metrics)

        # Run backtest on test data
        print("\n" + "="*60)
        print("RUNNING BACKTEST ON OUT-OF-SAMPLE DATA")
        print("="*60)

        prepared_df = prepare_data(test_df, self.config.sorb_config)

        # Create fresh feature engineer for test data
        test_feature_engineer = FeatureEngineer(self.config.ml_config)
        features_df = test_feature_engineer.create_features(prepared_df)

        # Use the scaler from trained model
        test_feature_engineer.scaler = self.feature_engineer.scaler
        test_feature_engineer.feature_names = self.feature_engineer.feature_names

        print(f"Features created: {len(features_df)} rows")
        print(f"Feature columns: {len(features_df.columns)}")

        # State variables
        equity = initial_capital
        hwm = equity
        position = None
        signals_today = 0
        daily_start = equity
        current_date = None

        self.trades = []
        self.signals_log = []
        equity_curve = []

        config = self.config.sorb_config
        dates = prepared_df['date'].unique()

        for i, date in enumerate(dates):
            if i % 100 == 0:
                print(f"  Processing day {i}/{len(dates)}...")

            day_data = prepared_df[prepared_df['date'] == date]
            day_features = features_df[features_df.index.isin(day_data.index)]

            if len(day_data) < 10:
                continue

            # New day reset
            if current_date != date:
                current_date = date
                daily_start = equity
                signals_today = 0

            # Calculate opening range
            or_info = calculate_opening_range(prepared_df, date, config)

            # Update features with OR info
            if or_info is not None:
                day_features = day_features.copy()
                day_features['dist_from_or_high'] = (or_info['or_high'] - day_data['close']) / or_info['or_size']
                day_features['dist_from_or_low'] = (day_data['close'] - or_info['or_low']) / or_info['or_size']
                day_features['or_size_vs_atr'] = or_info['or_size'] / day_data['atr']

            # Iterate through bars
            for j in range(1, len(day_data)):
                bar = day_data.iloc[j]
                prev_bar = day_data.iloc[j-1]
                bar_time = day_data.index[j]

                # Get features for this bar
                try:
                    if bar_time in day_features.index:
                        bar_features = day_features.loc[[bar_time]].copy()
                    else:
                        # Create features from the bar itself
                        bar_features = features_df.iloc[[j]].copy() if j < len(features_df) else None
                        if bar_features is None:
                            continue
                except:
                    continue

                equity_curve.append({'datetime': bar_time, 'equity': equity})
                hwm = max(hwm, equity)

                # Check guards
                daily_pnl_pct = (equity - daily_start) / daily_start * 100
                total_dd = (hwm - equity) / hwm * 100

                guards_triggered = (
                    daily_pnl_pct <= -config.daily_loss_guard_pct or
                    total_dd >= config.total_dd_guard_pct
                )

                # Manage existing position
                if position is not None:
                    exit_price = None
                    exit_reason = None

                    if position['direction'] == 1:
                        if bar['low'] <= position['stop_loss']:
                            exit_price = position['stop_loss']
                            exit_reason = 'stop_loss'
                        elif bar['high'] >= position['tp1'] and not position.get('tp1_hit'):
                            position['tp1_hit'] = True
                            position['stop_loss'] = position['entry_price'] + 0.1 * or_info['or_size'] if or_info else position['entry_price']
                        elif position.get('tp1_hit') and bar['high'] >= position['tp2']:
                            exit_price = position['tp2']
                            exit_reason = 'tp2'

                    # EOD flat
                    if bar['is_flat_time'] and exit_price is None:
                        exit_price = bar['close']
                        exit_reason = 'eod_flat'

                    # Guardrail exit
                    if guards_triggered and exit_price is None:
                        exit_price = bar['close']
                        exit_reason = 'guardrail'

                    if exit_price is not None:
                        # Calculate P&L
                        gross_pnl = (exit_price - position['entry_price']) * position['size']

                        # Costs
                        costs = (
                            position['entry_price'] * position['size'] *
                            config.commission_pct / 100 * 2 +
                            (config.slippage_ticks + config.spread_ticks) *
                            config.tick_size * position['size'] * 2
                        )
                        net_pnl = gross_pnl - costs

                        risk_amount = abs(position['entry_price'] - position['initial_stop']) * position['size']
                        pnl_r = net_pnl / risk_amount if risk_amount > 0 else 0

                        self.trades.append(Trade(
                            entry_time=position['entry_time'],
                            exit_time=bar_time,
                            direction=position['direction'],
                            entry_price=position['entry_price'],
                            exit_price=exit_price,
                            stop_loss=position['initial_stop'],
                            tp1=position['tp1'],
                            tp2=position['tp2'],
                            position_size=position['size'],
                            pnl=net_pnl,
                            pnl_r=pnl_r,
                            exit_reason=exit_reason,
                            signal_type=position['signal_type'],
                            confidence=position['confidence']
                        ))

                        equity += net_pnl
                        position = None

                # Check for new entry
                if position is None and or_info is not None:
                    can_trade = (
                        bar['is_entry_window'] and
                        not bar['is_flat_time'] and
                        not guards_triggered and
                        signals_today < self.config.max_signals_per_day
                    )

                    if can_trade:
                        direction, confidence, signal_type = self.generate_combined_signal(
                            bar, prev_bar, or_info, bar_features
                        )

                        # Debug: track signal attempts
                        if i == 0 and j < 50:  # First few days, first 50 bars
                            pass  # Removed verbose logging

                        if direction is not None:
                            # Log signal
                            self.signals_log.append({
                                'time': bar_time,
                                'type': signal_type,
                                'confidence': confidence,
                                'price': bar['close']
                            })

                            # Calculate position
                            if direction == 1:
                                stop_loss = or_info['or_low'] - 0.1 * or_info['or_size']
                                stop_dist = bar['close'] - stop_loss
                                tp1 = bar['close'] + or_info['or_size'] * config.tp1_mult
                                tp2 = bar['close'] + or_info['or_size'] * config.tp2_mult
                            else:
                                stop_loss = or_info['or_high'] + 0.1 * or_info['or_size']
                                stop_dist = stop_loss - bar['close']
                                tp1 = bar['close'] - or_info['or_size'] * config.tp1_mult
                                tp2 = bar['close'] - or_info['or_size'] * config.tp2_mult

                            # Risk adjustment based on signal type
                            if 'SORB' in signal_type:
                                risk_mult = self.config.sorb_risk_mult
                            elif 'ML' in signal_type:
                                risk_mult = self.config.ml_risk_mult
                            else:
                                risk_mult = self.config.near_miss_risk_mult

                            # Also scale by confidence
                            risk_mult *= confidence

                            risk_amount = equity * (config.risk_per_trade_pct / 100) * risk_mult
                            size = risk_amount / stop_dist if stop_dist > 0 else 0

                            if size > 0:
                                position = {
                                    'entry_time': bar_time,
                                    'entry_price': bar['close'],
                                    'direction': direction,
                                    'size': size,
                                    'stop_loss': stop_loss,
                                    'initial_stop': stop_loss,
                                    'tp1': tp1,
                                    'tp2': tp2,
                                    'tp1_hit': False,
                                    'signal_type': signal_type,
                                    'confidence': confidence
                                }
                                signals_today += 1

        # Build equity curve
        equity_df = pd.DataFrame(equity_curve)
        if len(equity_df) > 0:
            equity_df.set_index('datetime', inplace=True)

        return {
            'trades': self.trades,
            'signals_log': self.signals_log,
            'equity_curve': equity_df,
            'final_equity': equity,
            'return_pct': (equity - initial_capital) / initial_capital * 100,
            'training_metrics': training_metrics
        }

    def analyze_results(self) -> Dict:
        """Analyze backtest results"""
        if not self.trades:
            return {'error': 'No trades'}

        # Overall metrics
        pnl_r = [t.pnl_r for t in self.trades]
        wins = [r for r in pnl_r if r > 0]
        losses = [r for r in pnl_r if r < 0]

        total = len(self.trades)
        win_rate = len(wins) / total if total > 0 else 0
        expectancy = np.mean(pnl_r)

        gross_profit = sum(wins) if wins else 0
        gross_loss = abs(sum(losses)) if losses else 1
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0

        # By signal type
        signal_types = {}
        for t in self.trades:
            st = t.signal_type.split(':')[0]  # Get main type
            if st not in signal_types:
                signal_types[st] = {'trades': 0, 'wins': 0, 'total_r': 0}
            signal_types[st]['trades'] += 1
            signal_types[st]['total_r'] += t.pnl_r
            if t.pnl > 0:
                signal_types[st]['wins'] += 1

        # Exit reasons
        exit_stats = {}
        for t in self.trades:
            r = t.exit_reason
            if r not in exit_stats:
                exit_stats[r] = {'count': 0, 'total_r': 0}
            exit_stats[r]['count'] += 1
            exit_stats[r]['total_r'] += t.pnl_r

        # Confidence analysis
        high_conf = [t for t in self.trades if t.confidence >= 0.7]
        med_conf = [t for t in self.trades if 0.55 <= t.confidence < 0.7]
        low_conf = [t for t in self.trades if t.confidence < 0.55]

        confidence_analysis = {
            'high': {
                'count': len(high_conf),
                'win_rate': len([t for t in high_conf if t.pnl > 0]) / len(high_conf) if high_conf else 0,
                'expectancy': np.mean([t.pnl_r for t in high_conf]) if high_conf else 0
            },
            'medium': {
                'count': len(med_conf),
                'win_rate': len([t for t in med_conf if t.pnl > 0]) / len(med_conf) if med_conf else 0,
                'expectancy': np.mean([t.pnl_r for t in med_conf]) if med_conf else 0
            },
            'low': {
                'count': len(low_conf),
                'win_rate': len([t for t in low_conf if t.pnl > 0]) / len(low_conf) if low_conf else 0,
                'expectancy': np.mean([t.pnl_r for t in low_conf]) if low_conf else 0
            }
        }

        return {
            'total_trades': total,
            'win_rate': win_rate,
            'expectancy': expectancy,
            'profit_factor': profit_factor,
            'avg_win': np.mean(wins) if wins else 0,
            'avg_loss': abs(np.mean(losses)) if losses else 0,
            'signal_types': signal_types,
            'exit_stats': exit_stats,
            'confidence_analysis': confidence_analysis
        }

    def print_report(self, results: Dict, metrics: Dict):
        """Print comprehensive report"""
        print("\n" + "="*70)
        print("SORB AI STRATEGY REPORT")
        print("="*70)

        print(f"\n--- OVERALL PERFORMANCE ---")
        print(f"Total Trades:    {metrics['total_trades']}")
        print(f"Win Rate:        {metrics['win_rate']*100:.1f}%")
        print(f"Expectancy:      {metrics['expectancy']:.3f}R")
        print(f"Profit Factor:   {metrics['profit_factor']:.2f}")
        print(f"Avg Win:         {metrics['avg_win']:.2f}R")
        print(f"Avg Loss:        {metrics['avg_loss']:.2f}R")
        print(f"Total Return:    {results['return_pct']:.2f}%")

        print(f"\n--- BY SIGNAL TYPE ---")
        for st, stats in sorted(metrics['signal_types'].items(),
                                 key=lambda x: x[1]['total_r'], reverse=True):
            wr = stats['wins'] / stats['trades'] * 100 if stats['trades'] > 0 else 0
            exp = stats['total_r'] / stats['trades'] if stats['trades'] > 0 else 0
            print(f"  {st}:")
            print(f"    Trades: {stats['trades']}, WR: {wr:.1f}%, Exp: {exp:.3f}R")

        print(f"\n--- BY CONFIDENCE LEVEL ---")
        for level, stats in metrics['confidence_analysis'].items():
            if stats['count'] > 0:
                print(f"  {level.upper()}: {stats['count']} trades, "
                      f"WR: {stats['win_rate']*100:.1f}%, Exp: {stats['expectancy']:.3f}R")

        print(f"\n--- EXIT REASONS ---")
        for reason, stats in metrics['exit_stats'].items():
            avg_r = stats['total_r'] / stats['count'] if stats['count'] > 0 else 0
            print(f"  {reason}: {stats['count']} ({avg_r:.3f}R avg)")

        # Trading frequency
        if self.trades:
            first = min(t.entry_time for t in self.trades)
            last = max(t.entry_time for t in self.trades)
            days = (last - first).days or 1
            tpd = metrics['total_trades'] / days
            tpm = tpd * 22

            print(f"\n--- TRADING FREQUENCY ---")
            print(f"Test Period:    {first.date()} to {last.date()} ({days} days)")
            print(f"Trades/Day:     {tpd:.2f}")
            print(f"Trades/Month:   {tpm:.1f}")

            # Compare to baseline SORB
            sorb_trades = [t for t in self.trades if 'SORB' in t.signal_type]
            ai_trades = [t for t in self.trades if 'SORB' not in t.signal_type]

            print(f"\n--- SIGNAL CONTRIBUTION ---")
            print(f"SORB signals:   {len(sorb_trades)} ({len(sorb_trades)/days*22:.1f}/month)")
            print(f"AI signals:     {len(ai_trades)} ({len(ai_trades)/days*22:.1f}/month)")
            print(f"Frequency boost: {(len(ai_trades)/max(len(sorb_trades),1)):.1f}x more trades from AI")

            if metrics['expectancy'] > 0:
                print(f"\n--- CHALLENGE PROJECTION ---")
                for risk in [0.5, 1.0, 1.5, 2.0]:
                    monthly = tpm * metrics['expectancy'] * risk
                    if monthly > 0:
                        days_to_10 = (10 / monthly) * 22
                        print(f"  {risk}% risk: {monthly:.2f}%/month, ~{days_to_10:.0f} days to 10%")

        print("="*70)


def main():
    import sys

    print("="*70)
    print("SORB AI - Machine Learning Enhanced Strategy")
    print("="*70)

    data_file = sys.argv[1] if len(sys.argv) > 1 else "5m_data.csv"

    try:
        # Load data
        df = load_data(data_file)
        print(f"\nLoaded {len(df):,} bars")
        print(f"Range: {df.index[0]} to {df.index[-1]}")

        # Create strategy
        config = AIStrategyConfig(
            sorb_config=StrategyConfig(
                risk_per_trade_pct=1.0,
                long_only=True
            ),
            ml_config=MLConfig(
                min_probability=0.55,
                high_confidence_threshold=0.65
            ),
            min_confidence=0.55,
            max_signals_per_day=4
        )

        strategy = SORBAIStrategy(config)

        # Run backtest (50% train, 50% test)
        results = strategy.run_backtest(df, train_pct=0.5)

        if 'error' not in results:
            metrics = strategy.analyze_results()
            if 'error' not in metrics:
                strategy.print_report(results, metrics)
            else:
                print(f"\nNo trades generated. Possible issues:")
                print(f"  - ML signals not meeting confidence threshold")
                print(f"  - SORB conditions too strict")
                print(f"\nSignals logged: {len(results.get('signals_log', []))}")
        else:
            print(f"Error: {results['error']}")

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
