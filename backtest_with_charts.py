import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.gridspec import GridSpec
import warnings
warnings.filterwarnings('ignore')

# Import strategy components
from sorb_ai_core import (
    StrategyConfig, Trade, load_data, prepare_data, calculate_opening_range, check_sorb_signal
)
from ml_signal_generator import (
    MLConfig, MLSignalModel, FeatureEngineer, print_training_report
)
from sorb_ai_strategy import SORBAIStrategy, AIStrategyConfig


def run_full_backtest(data_file: str = '5m_data.csv', train_pct: float = 0.5):
    """Run complete backtest and return results"""

    print("="*70)
    print("SORB AI - FULL BACKTEST WITH ANALYSIS")
    print("="*70)

    # Load data
    df = load_data(data_file)
    print(f"\nLoaded {len(df):,} bars")
    print(f"Date range: {df.index[0]} to {df.index[-1]}")

    # Create strategy with optimized config
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

    # Run backtest
    results = strategy.run_backtest(df, train_pct=train_pct)

    if 'error' in results:
        print(f"Error: {results['error']}")
        return None, None

    metrics = strategy.analyze_results()

    if 'error' in metrics:
        print("No trades generated")
        return None, None

    # Print detailed report
    strategy.print_report(results, metrics)

    return results, metrics, strategy


def create_charts(results: dict, metrics: dict, strategy, save_path: str = None):
    """Create comprehensive analysis charts"""

    trades = results['trades']
    equity_curve = results['equity_curve']

    if len(trades) == 0:
        print("No trades to chart")
        return

    # Set up the figure with subplots
    fig = plt.figure(figsize=(16, 20))
    gs = GridSpec(4, 2, figure=fig, hspace=0.3, wspace=0.25)

    # Style settings
    plt.style.use('seaborn-v0_8-darkgrid')
    colors = {
        'equity': '#2ecc71',
        'drawdown': '#e74c3c',
        'win': '#27ae60',
        'loss': '#c0392b',
        'ml': '#3498db',
        'sorb': '#9b59b6'
    }

    # =========================================
    # 1. Equity Curve (top left)
    # =========================================
    ax1 = fig.add_subplot(gs[0, 0])

    equity_curve.plot(ax=ax1, color=colors['equity'], linewidth=1.5, legend=False)
    ax1.fill_between(equity_curve.index, 100000, equity_curve['equity'],
                     alpha=0.3, color=colors['equity'])

    ax1.set_title('Equity Curve', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Equity ($)')
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))

    # Add performance annotation
    final_equity = results['final_equity']
    total_return = results['return_pct']
    ax1.annotate(f'Final: ${final_equity:,.0f}\nReturn: {total_return:.1f}%',
                 xy=(0.02, 0.98), xycoords='axes fraction',
                 fontsize=10, verticalalignment='top',
                 bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    # =========================================
    # 2. Drawdown Chart (top right)
    # =========================================
    ax2 = fig.add_subplot(gs[0, 1])

    # Calculate drawdown
    hwm = equity_curve['equity'].cummax()
    drawdown = (equity_curve['equity'] - hwm) / hwm * 100

    drawdown.plot(ax=ax2, color=colors['drawdown'], linewidth=1)
    ax2.fill_between(drawdown.index, 0, drawdown, alpha=0.3, color=colors['drawdown'])

    ax2.set_title('Drawdown', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Date')
    ax2.set_ylabel('Drawdown (%)')

    max_dd = drawdown.min()
    ax2.annotate(f'Max DD: {max_dd:.2f}%',
                 xy=(0.02, 0.02), xycoords='axes fraction',
                 fontsize=10, verticalalignment='bottom',
                 bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    # =========================================
    # 3. Monthly Returns Heatmap (middle left)
    # =========================================
    ax3 = fig.add_subplot(gs[1, 0])

    # Create monthly returns
    trades_df = pd.DataFrame([{
        'date': t.exit_time,
        'pnl_pct': t.pnl / 100000 * 100  # Approximate % return
    } for t in trades])
    trades_df['date'] = pd.to_datetime(trades_df['date'])
    trades_df['year'] = trades_df['date'].dt.year
    trades_df['month'] = trades_df['date'].dt.month

    monthly = trades_df.groupby(['year', 'month'])['pnl_pct'].sum().unstack(fill_value=0)

    # Plot heatmap
    im = ax3.imshow(monthly.values, cmap='RdYlGn', aspect='auto', vmin=-5, vmax=5)

    ax3.set_xticks(range(12))
    ax3.set_xticklabels(['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                         'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
    ax3.set_yticks(range(len(monthly.index)))
    ax3.set_yticklabels(monthly.index)

    ax3.set_title('Monthly Returns (%)', fontsize=14, fontweight='bold')

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax3, shrink=0.8)
    cbar.set_label('Return %')

    # Add text annotations
    for i in range(len(monthly.index)):
        for j in range(12):
            if j + 1 in monthly.columns:
                val = monthly.iloc[i, monthly.columns.get_loc(j + 1)]
                if val != 0:
                    color = 'white' if abs(val) > 2.5 else 'black'
                    ax3.text(j, i, f'{val:.1f}', ha='center', va='center',
                            color=color, fontsize=8)

    # =========================================
    # 4. Trade Distribution (middle right)
    # =========================================
    ax4 = fig.add_subplot(gs[1, 1])

    pnl_r = [t.pnl_r for t in trades]

    # Create histogram
    bins = np.linspace(-3, 5, 30)
    n, bins_edges, patches = ax4.hist(pnl_r, bins=bins, edgecolor='black', alpha=0.7)

    # Color bars based on win/loss
    for i, patch in enumerate(patches):
        if bins_edges[i] < 0:
            patch.set_facecolor(colors['loss'])
        else:
            patch.set_facecolor(colors['win'])

    ax4.axvline(x=0, color='black', linestyle='--', linewidth=2)
    ax4.axvline(x=np.mean(pnl_r), color='blue', linestyle='-', linewidth=2, label=f'Mean: {np.mean(pnl_r):.2f}R')

    ax4.set_title('Trade Distribution (R-multiples)', fontsize=14, fontweight='bold')
    ax4.set_xlabel('R-multiple')
    ax4.set_ylabel('Frequency')
    ax4.legend()

    # =========================================
    # 5. Win Rate by Signal Type (bottom left)
    # =========================================
    ax5 = fig.add_subplot(gs[2, 0])

    signal_stats = metrics['signal_types']
    signal_names = list(signal_stats.keys())
    win_rates = [signal_stats[s]['wins'] / signal_stats[s]['trades'] * 100
                 for s in signal_names]
    trade_counts = [signal_stats[s]['trades'] for s in signal_names]
    expectancies = [signal_stats[s]['total_r'] / signal_stats[s]['trades']
                    for s in signal_names]

    x = np.arange(len(signal_names))
    width = 0.35

    bars1 = ax5.bar(x - width/2, win_rates, width, label='Win Rate %', color=colors['win'])

    ax5_twin = ax5.twinx()
    bars2 = ax5_twin.bar(x + width/2, expectancies, width, label='Expectancy (R)', color=colors['ml'])

    ax5.set_ylabel('Win Rate (%)', color=colors['win'])
    ax5_twin.set_ylabel('Expectancy (R)', color=colors['ml'])
    ax5.set_xticks(x)
    ax5.set_xticklabels(signal_names)
    ax5.set_title('Performance by Signal Type', fontsize=14, fontweight='bold')

    # Add trade count labels
    for i, (bar, count) in enumerate(zip(bars1, trade_counts)):
        ax5.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f'n={count}', ha='center', va='bottom', fontsize=9)

    ax5.axhline(y=50, color='gray', linestyle='--', alpha=0.5)
    ax5_twin.axhline(y=0, color='gray', linestyle='--', alpha=0.5)

    # =========================================
    # 6. Cumulative R-multiple (bottom right)
    # =========================================
    ax6 = fig.add_subplot(gs[2, 1])

    # Sort trades by time and calculate cumulative R
    sorted_trades = sorted(trades, key=lambda t: t.exit_time)
    cumulative_r = np.cumsum([t.pnl_r for t in sorted_trades])
    trade_numbers = range(1, len(sorted_trades) + 1)

    ax6.plot(trade_numbers, cumulative_r, color=colors['equity'], linewidth=2)
    ax6.fill_between(trade_numbers, 0, cumulative_r, alpha=0.3,
                     where=[r >= 0 for r in cumulative_r], color=colors['win'])
    ax6.fill_between(trade_numbers, 0, cumulative_r, alpha=0.3,
                     where=[r < 0 for r in cumulative_r], color=colors['loss'])

    ax6.axhline(y=0, color='black', linestyle='-', linewidth=1)
    ax6.set_title('Cumulative R-multiple', fontsize=14, fontweight='bold')
    ax6.set_xlabel('Trade Number')
    ax6.set_ylabel('Cumulative R')

    # =========================================
    # 7. Performance Summary Table (bottom)
    # =========================================
    ax7 = fig.add_subplot(gs[3, :])
    ax7.axis('off')

    # Create summary table data
    summary_data = [
        ['OVERALL PERFORMANCE', '', 'RISK METRICS', ''],
        ['Total Trades', f"{metrics['total_trades']}", 'Max Drawdown', f"{drawdown.min():.2f}%"],
        ['Win Rate', f"{metrics['win_rate']*100:.1f}%", 'Profit Factor', f"{metrics['profit_factor']:.2f}"],
        ['Expectancy', f"{metrics['expectancy']:.3f}R", 'Avg Win', f"{metrics['avg_win']:.2f}R"],
        ['Total Return', f"{results['return_pct']:.2f}%", 'Avg Loss', f"{metrics['avg_loss']:.2f}R"],
        ['', '', '', ''],
        ['TRADING FREQUENCY', '', 'CHALLENGE PROJECTION', ''],
        ['Test Period Days', f"{(sorted_trades[-1].exit_time - sorted_trades[0].entry_time).days}",
         '1.0% Risk/Month', f"{metrics['total_trades']/(sorted_trades[-1].exit_time - sorted_trades[0].entry_time).days*22*metrics['expectancy']*1:.2f}%"],
        ['Trades/Month', f"{metrics['total_trades']/((sorted_trades[-1].exit_time - sorted_trades[0].entry_time).days/30):.1f}",
         '1.5% Risk/Month', f"{metrics['total_trades']/(sorted_trades[-1].exit_time - sorted_trades[0].entry_time).days*22*metrics['expectancy']*1.5:.2f}%"],
        ['ML Signals', f"{metrics['signal_types'].get('ML', {}).get('trades', 0)}",
         'Days to 10% (1% risk)', f"~{10/(metrics['total_trades']/(sorted_trades[-1].exit_time - sorted_trades[0].entry_time).days*22*metrics['expectancy']*1)*22:.0f}"],
    ]

    table = ax7.table(cellText=summary_data, loc='center', cellLoc='center',
                      colWidths=[0.2, 0.15, 0.2, 0.15])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.8)

    # Style header rows
    for i in [0, 6]:
        for j in range(4):
            table[(i, j)].set_text_props(fontweight='bold')
            table[(i, j)].set_facecolor('#3498db')
            table[(i, j)].set_text_props(color='white')

    # Main title
    fig.suptitle('SORB AI Strategy - Backtest Analysis Report',
                 fontsize=18, fontweight='bold', y=0.98)

    plt.tight_layout(rect=[0, 0, 1, 0.97])

    # Save or show
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
        print(f"\n📊 Chart saved to: {save_path}")

    plt.show()

    return fig


def create_trade_log(trades: list, save_path: str = None):
    """Create detailed trade log CSV"""

    trade_data = []
    for i, t in enumerate(sorted(trades, key=lambda x: x.entry_time)):
        trade_data.append({
            'Trade #': i + 1,
            'Entry Time': t.entry_time,
            'Exit Time': t.exit_time,
            'Signal Type': t.signal_type,
            'Confidence': f"{t.confidence:.2f}",
            'Direction': 'LONG' if t.direction == 1 else 'SHORT',
            'Entry Price': f"{t.entry_price:.2f}",
            'Exit Price': f"{t.exit_price:.2f}",
            'Stop Loss': f"{t.stop_loss:.2f}",
            'P&L ($)': f"{t.pnl:.2f}",
            'P&L (R)': f"{t.pnl_r:.2f}",
            'Exit Reason': t.exit_reason,
            'Result': 'WIN' if t.pnl > 0 else 'LOSS'
        })

    df = pd.DataFrame(trade_data)

    if save_path:
        df.to_csv(save_path, index=False)
        print(f"📝 Trade log saved to: {save_path}")

    return df


def print_detailed_stats(trades: list, metrics: dict):
    """Print additional detailed statistics"""

    print("\n" + "="*70)
    print("DETAILED STATISTICS")
    print("="*70)

    # Win/Loss streaks
    results = ['W' if t.pnl > 0 else 'L' for t in sorted(trades, key=lambda x: x.exit_time)]

    max_win_streak = 0
    max_loss_streak = 0
    current_streak = 0
    current_type = None

    for r in results:
        if r == current_type:
            current_streak += 1
        else:
            if current_type == 'W':
                max_win_streak = max(max_win_streak, current_streak)
            elif current_type == 'L':
                max_loss_streak = max(max_loss_streak, current_streak)
            current_type = r
            current_streak = 1

    # Final streak
    if current_type == 'W':
        max_win_streak = max(max_win_streak, current_streak)
    elif current_type == 'L':
        max_loss_streak = max(max_loss_streak, current_streak)

    print(f"\n--- STREAKS ---")
    print(f"Max Winning Streak: {max_win_streak} trades")
    print(f"Max Losing Streak:  {max_loss_streak} trades")

    # Best and worst trades
    sorted_by_pnl = sorted(trades, key=lambda x: x.pnl_r, reverse=True)

    print(f"\n--- BEST TRADES ---")
    for t in sorted_by_pnl[:5]:
        print(f"  {t.entry_time.strftime('%Y-%m-%d %H:%M')} | {t.signal_type} | +{t.pnl_r:.2f}R | {t.exit_reason}")

    print(f"\n--- WORST TRADES ---")
    for t in sorted_by_pnl[-5:]:
        print(f"  {t.entry_time.strftime('%Y-%m-%d %H:%M')} | {t.signal_type} | {t.pnl_r:.2f}R | {t.exit_reason}")

    # Exit reason analysis
    print(f"\n--- EXIT REASON ANALYSIS ---")
    exit_stats = {}
    for t in trades:
        r = t.exit_reason
        if r not in exit_stats:
            exit_stats[r] = {'count': 0, 'wins': 0, 'total_r': 0}
        exit_stats[r]['count'] += 1
        exit_stats[r]['total_r'] += t.pnl_r
        if t.pnl > 0:
            exit_stats[r]['wins'] += 1

    print(f"{'Exit Reason':<15} {'Count':<8} {'Win Rate':<10} {'Avg R':<10}")
    print("-"*43)
    for reason, stats in sorted(exit_stats.items(), key=lambda x: x[1]['total_r'], reverse=True):
        wr = stats['wins'] / stats['count'] * 100
        avg_r = stats['total_r'] / stats['count']
        print(f"{reason:<15} {stats['count']:<8} {wr:<10.1f}% {avg_r:<10.2f}")

    # Time analysis
    print(f"\n--- TIME ANALYSIS ---")
    hour_stats = {}
    for t in trades:
        h = t.entry_time.hour
        if h not in hour_stats:
            hour_stats[h] = {'count': 0, 'wins': 0, 'total_r': 0}
        hour_stats[h]['count'] += 1
        hour_stats[h]['total_r'] += t.pnl_r
        if t.pnl > 0:
            hour_stats[h]['wins'] += 1

    print(f"{'Hour (ET)':<12} {'Trades':<8} {'Win Rate':<10} {'Avg R':<10}")
    print("-"*40)
    for hour in sorted(hour_stats.keys()):
        stats = hour_stats[hour]
        wr = stats['wins'] / stats['count'] * 100
        avg_r = stats['total_r'] / stats['count']
        print(f"{hour:02d}:00{'':<7} {stats['count']:<8} {wr:<10.1f}% {avg_r:<10.2f}")

    # Day of week analysis
    print(f"\n--- DAY OF WEEK ANALYSIS ---")
    dow_stats = {}
    dow_names = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
    for t in trades:
        dow = t.entry_time.weekday()
        if dow not in dow_stats:
            dow_stats[dow] = {'count': 0, 'wins': 0, 'total_r': 0}
        dow_stats[dow]['count'] += 1
        dow_stats[dow]['total_r'] += t.pnl_r
        if t.pnl > 0:
            dow_stats[dow]['wins'] += 1

    print(f"{'Day':<12} {'Trades':<8} {'Win Rate':<10} {'Avg R':<10}")
    print("-"*40)
    for dow in sorted(dow_stats.keys()):
        stats = dow_stats[dow]
        wr = stats['wins'] / stats['count'] * 100
        avg_r = stats['total_r'] / stats['count']
        print(f"{dow_names[dow]:<12} {stats['count']:<8} {wr:<10.1f}% {avg_r:<10.2f}")


def main():
    import sys
    import os

    # Get data file
    data_file = sys.argv[1] if len(sys.argv) > 1 else '5m_data.csv'

    # Check if data exists
    if not os.path.exists(data_file):
        print(f"Error: {data_file} not found")
        print("Please provide your 5m NAS100 data file")
        return

    # Run backtest
    results, metrics, strategy = run_full_backtest(data_file, train_pct=0.5)

    if results is None:
        return

    # Print detailed stats
    print_detailed_stats(results['trades'], metrics)

    # Create trade log
    trade_log = create_trade_log(results['trades'], 'trade_log.csv')

    # Create and save charts
    print("\n" + "="*70)
    print("GENERATING CHARTS...")
    print("="*70)

    try:
        fig = create_charts(results, metrics, strategy, 'backtest_report.png')
        print("\n✅ Backtest complete!")
        print(f"   📊 Charts saved to: backtest_report.png")
        print(f"   📝 Trade log saved to: trade_log.csv")
    except Exception as e:
        print(f"Error creating charts: {e}")
        print("Charts require matplotlib. Install with: pip install matplotlib")


if __name__ == "__main__":
    main()
