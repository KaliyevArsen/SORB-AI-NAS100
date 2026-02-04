"""
This module trains ML models to:
1. Identify "near-miss" SORB setups that still have edge
2. Predict which setups are most likely to succeed
3. Generate additional signals when conditions are favorable
Models:
Random Forest for pattern classification
Gradient Boosting for probability estimation
Simple Neural Network for complex patterns
"""

import pandas as pd
import numpy as np
from datetime import datetime
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import pickle
import os


@dataclass
class MLConfig:
    # Feature engineering
    lookback_bars: int = 20
    forward_bars: int = 12  # ~1 hour for target

    # Target definition
    min_profit_pct: float = 0.15  # Minimum move to count as "win"

    # Model parameters
    n_estimators: int = 100
    max_depth: int = 10
    min_samples_leaf: int = 50

    # Signal generation
    min_probability: float = 0.55  # Minimum prob to generate signal
    high_confidence_threshold: float = 0.65

    # Training
    train_test_split: float = 0.7
    n_cv_folds: int = 5


class FeatureEngineer:
    """Generate features for ML models"""

    def __init__(self, config: MLConfig):
        self.config = config
        self.scaler = StandardScaler()
        self.feature_names = []

    def create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create all features for ML model"""
        df = df.copy()

        features = pd.DataFrame(index=df.index)

        # ============================================
        # Price-based features
        # ============================================

        # Returns at various lookbacks
        for lb in [1, 3, 6, 12, 24]:
            features[f'ret_{lb}'] = df['close'].pct_change(lb) * 100

        # Distance from moving averages
        for period in [10, 20, 50]:
            ma = df['close'].rolling(period).mean()
            features[f'dist_ma{period}'] = (df['close'] - ma) / ma * 100

        # Bollinger Band position
        features['bb_position'] = df['bb_position']

        # RSI
        features['rsi'] = df['rsi']
        features['rsi_change'] = df['rsi'].diff(3)

        # ADX
        features['adx'] = df['adx']

        # MACD
        features['macd_hist'] = df['macd_hist']
        features['macd_hist_change'] = df['macd_hist'].diff(3)

        # ============================================
        # Volatility features
        # ============================================

        features['volatility'] = df['volatility']
        features['atr'] = df['atr']
        features['atr_pct'] = df['atr'] / df['close'] * 100

        # Volatility ratio (current vs historical)
        vol_20 = df['ret_1'].rolling(20).std()
        vol_50 = df['ret_1'].rolling(50).std()
        features['vol_ratio'] = vol_20 / vol_50

        # Range expansion/contraction
        bar_range = df['high'] - df['low']
        avg_range = bar_range.rolling(20).mean()
        features['range_ratio'] = bar_range / avg_range

        # ============================================
        # Volume features
        # ============================================

        features['vol_ratio_20'] = df['volume'] / df['vol_sma']

        # Volume trend
        vol_5 = df['volume'].rolling(5).mean()
        vol_20 = df['volume'].rolling(20).mean()
        features['vol_trend'] = vol_5 / vol_20

        # ============================================
        # Pattern features
        # ============================================

        # Bar characteristics
        body = df['close'] - df['open']
        total_range = df['high'] - df['low']
        features['body_ratio'] = body / total_range.replace(0, np.nan)

        # Upper/lower wick
        upper_wick = df['high'] - df[['open', 'close']].max(axis=1)
        lower_wick = df[['open', 'close']].min(axis=1) - df['low']
        features['upper_wick_pct'] = upper_wick / total_range.replace(0, np.nan)
        features['lower_wick_pct'] = lower_wick / total_range.replace(0, np.nan)

        # Consecutive moves
        up_bars = (df['close'] > df['open']).astype(int)
        features['up_streak'] = up_bars.rolling(10).sum()

        # ============================================
        # Session/Time features
        # ============================================

        features['hour'] = df['hour']
        features['time_since_open'] = (df['time_min'] - 9*60 - 30).clip(lower=0)
        features['day_of_week'] = df['day_of_week']

        # Position in daily range
        features['daily_position'] = df['daily_position']

        # ============================================
        # Opening Range features (if available)
        # ============================================

        # Distance from OR levels (will be set during prediction)
        features['dist_from_or_high'] = np.nan
        features['dist_from_or_low'] = np.nan
        features['or_size_vs_atr'] = np.nan

        # ============================================
        # Trend features
        # ============================================

        # EMA alignment
        features['ema_trend'] = (df['ema20'] > df['ema50']).astype(int)

        # Price vs EMAs
        features['above_ema20'] = (df['close'] > df['ema20']).astype(int)
        features['above_ema50'] = (df['close'] > df['ema50']).astype(int)

        # Store feature names
        self.feature_names = list(features.columns)

        return features

    def create_target(self, df: pd.DataFrame, direction: int = 1) -> pd.Series:
        """
        Create target variable for supervised learning
        1 = profitable move in expected direction
        0 = loss or no significant move
        """
        # Forward return
        forward_ret = df['close'].pct_change(self.config.forward_bars).shift(-self.config.forward_bars) * 100

        if direction == 1:
            target = (forward_ret >= self.config.min_profit_pct).astype(int)
        else:
            target = (forward_ret <= -self.config.min_profit_pct).astype(int)

        return target

    def fit_scaler(self, features: pd.DataFrame):
        """Fit the scaler on training data"""
        clean_features = features.dropna()
        if len(clean_features) > 0:
            self.scaler.fit(clean_features)

    def transform(self, features: pd.DataFrame) -> np.ndarray:
        """Transform features using fitted scaler"""
        # Fill NaN with 0 for prediction
        features_filled = features.fillna(0)
        return self.scaler.transform(features_filled)


class MLSignalModel:
    """Machine Learning model for signal generation"""

    def __init__(self, config: MLConfig):
        self.config = config
        self.feature_engineer = FeatureEngineer(config)
        self.model = None
        self.is_trained = False
        self.training_metrics = {}

    def train(self, df: pd.DataFrame, direction: int = 1) -> Dict:
        """
        Train the ML model on historical data

        Args:
            df: Prepared dataframe with indicators
            direction: 1 for long signals, -1 for short

        Returns:
            Training metrics
        """
        print("Creating features...")
        features = self.feature_engineer.create_features(df)
        target = self.feature_engineer.create_target(df, direction)

        # Combine and drop NaN
        data = features.copy()
        data['target'] = target

        # Only use trading hours - check if columns exist
        if 'is_entry_window' in df.columns and 'is_flat_time' in df.columns:
            trading_hours = df['is_entry_window'] & ~df['is_flat_time']
            data = data[trading_hours]

        # Drop rows with too many NaN values, then fill remaining
        data = data.dropna(thresh=len(data.columns) - 5)  # Allow up to 5 NaN columns
        data = data.fillna(0)

        if len(data) < 1000:
            print(f"Warning: Only {len(data)} samples after filtering")
            return {'error': f'Not enough data for training (only {len(data)} samples)'}

        print(f"Training samples: {len(data)}")

        # Split data (time-series aware)
        split_idx = int(len(data) * self.config.train_test_split)
        train_data = data.iloc[:split_idx]
        test_data = data.iloc[split_idx:]

        X_train = train_data.drop('target', axis=1)
        y_train = train_data['target']
        X_test = test_data.drop('target', axis=1)
        y_test = test_data['target']

        # Fit scaler
        self.feature_engineer.fit_scaler(X_train)
        X_train_scaled = self.feature_engineer.transform(X_train)
        X_test_scaled = self.feature_engineer.transform(X_test)

        print("Training model...")

        # Train Gradient Boosting (usually better for financial data)
        self.model = GradientBoostingClassifier(
            n_estimators=self.config.n_estimators,
            max_depth=self.config.max_depth,
            min_samples_leaf=self.config.min_samples_leaf,
            learning_rate=0.1,
            subsample=0.8,
            random_state=42
        )

        self.model.fit(X_train_scaled, y_train)

        # Evaluate
        train_pred = self.model.predict(X_train_scaled)
        test_pred = self.model.predict(X_test_scaled)
        test_prob = self.model.predict_proba(X_test_scaled)[:, 1]

        # Calculate metrics
        self.training_metrics = {
            'train_accuracy': accuracy_score(y_train, train_pred),
            'test_accuracy': accuracy_score(y_test, test_pred),
            'test_precision': precision_score(y_test, test_pred, zero_division=0),
            'test_recall': recall_score(y_test, test_pred, zero_division=0),
            'test_f1': f1_score(y_test, test_pred, zero_division=0),
            'positive_rate_train': y_train.mean(),
            'positive_rate_test': y_test.mean(),
            'n_train': len(y_train),
            'n_test': len(y_test)
        }

        # Feature importance
        importance = pd.DataFrame({
            'feature': self.feature_engineer.feature_names,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)

        self.training_metrics['top_features'] = importance.head(10).to_dict('records')

        # Analyze predictions at different probability thresholds
        threshold_analysis = []
        for thresh in [0.5, 0.55, 0.6, 0.65, 0.7]:
            pred_at_thresh = (test_prob >= thresh).astype(int)
            n_signals = pred_at_thresh.sum()
            if n_signals > 0:
                precision_at_thresh = precision_score(y_test, pred_at_thresh, zero_division=0)
                threshold_analysis.append({
                    'threshold': thresh,
                    'n_signals': int(n_signals),
                    'precision': precision_at_thresh
                })

        self.training_metrics['threshold_analysis'] = threshold_analysis

        self.is_trained = True
        return self.training_metrics

    def predict_probability(self, features: pd.DataFrame) -> float:
        """Get probability of profitable trade"""
        if not self.is_trained or self.model is None:
            return 0.5

        features_scaled = self.feature_engineer.transform(features)
        prob = self.model.predict_proba(features_scaled)[:, 1]
        return float(prob[0]) if len(prob) > 0 else 0.5

    def generate_signal(self, bar_features: pd.DataFrame) -> Tuple[bool, float]:
        """
        Generate trading signal based on ML prediction

        Returns:
            (should_trade, confidence)
        """
        prob = self.predict_probability(bar_features)

        if prob >= self.config.high_confidence_threshold:
            return True, prob
        elif prob >= self.config.min_probability:
            return True, prob
        else:
            return False, prob

    def save(self, filepath: str):
        """Save model to file"""
        with open(filepath, 'wb') as f:
            pickle.dump({
                'model': self.model,
                'scaler': self.feature_engineer.scaler,
                'feature_names': self.feature_engineer.feature_names,
                'config': self.config,
                'metrics': self.training_metrics
            }, f)

    def load(self, filepath: str):
        """Load model from file"""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
            self.model = data['model']
            self.feature_engineer.scaler = data['scaler']
            self.feature_engineer.feature_names = data['feature_names']
            self.config = data['config']
            self.training_metrics = data['metrics']
            self.is_trained = True


class NearMissDetector:
    """
    Detects "near-miss" SORB setups that don't meet all criteria
    but historically still had edge
    """

    def __init__(self):
        self.historical_near_misses = []

    def check_near_miss(self, bar, or_info: Dict, config) -> Tuple[bool, str, float]:
        """
        Check if current bar is a near-miss SORB setup

        Returns:
            (is_near_miss, reason, confidence_adjustment)
        """
        if or_info is None:
            return False, "", 0

        reasons = []
        confidence = 0.5  # Start at 50%

        # Check each SORB condition and see how close we are

        # 1. Price near OR high but not broken yet
        dist_to_or_high = (or_info['or_high'] - bar['close']) / or_info['or_size']
        if 0 < dist_to_or_high < 0.2:  # Within 20% of breaking
            reasons.append(f"price_near_breakout_{dist_to_or_high:.1%}")
            confidence += 0.1

        # 2. Bias is positive but not confirmed by EMA
        if or_info['bias'] == 0:  # Neutral bias
            if bar['close'] > bar['ema20']:  # But above EMA
                reasons.append("neutral_bias_but_above_ema")
                confidence += 0.05

        # 3. ADX below threshold but rising
        if not or_info['is_trending']:
            adx_change = bar['adx'] - bar.get('adx_prev', bar['adx'])
            if adx_change > 0:
                reasons.append("adx_rising")
                confidence += 0.05

        # 4. Volume slightly below threshold
        vol_ratio = bar['volume'] / bar['vol_sma']
        if 1.0 < vol_ratio < config.volume_mult:
            reasons.append(f"vol_slightly_low_{vol_ratio:.2f}")
            confidence += 0.05

        # 5. RSI slightly outside range but trending
        rsi = bar['rsi']
        if config.rsi_long_low - 5 < rsi < config.rsi_long_low:
            reasons.append("rsi_slightly_low")
            confidence += 0.03
        elif config.rsi_long_high < rsi < config.rsi_long_high + 5:
            reasons.append("rsi_slightly_high")
            confidence += 0.03

        # 6. Already broke out but pulled back
        if bar['close'] < or_info['or_high'] and bar['high'] > or_info['or_high']:
            reasons.append("pullback_after_breakout")
            confidence += 0.15  # This is actually a good setup

        # 7. Strong momentum into OR high
        if bar['ret_6'] > 0.1 and dist_to_or_high < 0.3:
            reasons.append("momentum_into_level")
            confidence += 0.1

        is_near_miss = len(reasons) >= 2 and confidence >= 0.6

        return is_near_miss, ", ".join(reasons), confidence


def print_training_report(metrics: Dict):
    """Print formatted training report"""
    print("\n" + "="*60)
    print("ML MODEL TRAINING REPORT")
    print("="*60)

    if 'error' in metrics:
        print(f"Error: {metrics['error']}")
        return

    print(f"\n--- DATASET ---")
    print(f"Training samples: {metrics['n_train']:,}")
    print(f"Test samples:     {metrics['n_test']:,}")
    print(f"Positive rate (train): {metrics['positive_rate_train']*100:.1f}%")
    print(f"Positive rate (test):  {metrics['positive_rate_test']*100:.1f}%")

    print(f"\n--- MODEL PERFORMANCE ---")
    print(f"Train Accuracy: {metrics['train_accuracy']*100:.1f}%")
    print(f"Test Accuracy:  {metrics['test_accuracy']*100:.1f}%")
    print(f"Test Precision: {metrics['test_precision']*100:.1f}%")
    print(f"Test Recall:    {metrics['test_recall']*100:.1f}%")
    print(f"Test F1 Score:  {metrics['test_f1']*100:.1f}%")

    print(f"\n--- THRESHOLD ANALYSIS ---")
    print(f"{'Threshold':<12}{'Signals':<12}{'Precision':<12}")
    print("-"*36)
    for t in metrics.get('threshold_analysis', []):
        print(f"{t['threshold']:<12.2f}{t['n_signals']:<12}{t['precision']*100:.1f}%")

    print(f"\n--- TOP FEATURES ---")
    for i, feat in enumerate(metrics.get('top_features', [])[:10]):
        print(f"  {i+1}. {feat['feature']}: {feat['importance']:.4f}")

    print("="*60)
