"""
K-Means Unsupervised Learning — Market Regime Detection & Anomaly Filter
Adapted from ATS_US30_NAS into AQRS

Clusters historical price bars into regimes (bull/bear/sideways/volatile)
and flags anomaly bars that deviate from normal market behavior.
"""
import numpy as np
import pandas as pd
from pathlib import Path
import pickle
import logging
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

logger = logging.getLogger(__name__)


class UnsupervisedRegimeDetector:
    """
    Unsupervised ML for market regime detection using K-Means clustering.
    
    Features used for clustering:
    - Returns, volatility, volume, range, RSI
    - Clusters into 4 regimes: BULL, BEAR, SIDEWAYS, VOLATILE
    - Detects anomaly bars (outliers from all clusters)
    """
    
    def __init__(self, config=None):
        self.config = config
        self.model_path = Path("intelligence/models/regime_kmeans.pkl")
        self.scaler_path = Path("intelligence/models/regime_scaler.pkl")
        self.pca_path = Path("intelligence/models/regime_pca.pkl")
        self.model = None
        self.scaler = None
        self.pca = None
        self.n_clusters = 4  # BULL, BEAR, SIDEWAYS, VOLATILE
        self.regime_labels = {0: "BULL", 1: "BEAR", 2: "SIDEWAYS", 3: "VOLATILE"}
        self.feature_cols = [
            "returns_5", "returns_10", "returns_20",
            "volatility_10", "volatility_20",
            "volume_ratio", "range_ratio", "range_atr_ratio",
            "rsi_14", "close_vs_ma20_pct",
        ]
        
        # Create model directory
        self.model_path.parent.mkdir(parents=True, exist_ok=True)
    
    def _extract_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract feature matrix from OHLCV data for clustering."""
        features = pd.DataFrame(index=df.index)
        
        # Returns
        features["returns_5"] = df["close"].pct_change(5)
        features["returns_10"] = df["close"].pct_change(10)
        features["returns_20"] = df["close"].pct_change(20)
        
        # Volatility (standard deviation of returns)
        features["volatility_10"] = df["close"].pct_change().rolling(10).std()
        features["volatility_20"] = df["close"].pct_change().rolling(20).std()
        
        # Volume ratio (current vs average)
        vol_avg = df.get("tick_volume", df.get("volume", df["close"] * 0 + 1)).rolling(20).mean()
        volume = df.get("tick_volume", df.get("volume", df["close"] * 0 + 1))
        features["volume_ratio"] = volume / vol_avg.replace(0, np.nan)
        
        # Range ratio (current range vs average)
        candle_range = df["high"] - df["low"]
        avg_range = candle_range.rolling(20).mean()
        features["range_ratio"] = candle_range / avg_range.replace(0, np.nan)
        
        # Range vs ATR
        atr = self._compute_atr(df)
        features["range_atr_ratio"] = candle_range / atr.replace(0.001, 0.001)
        
        # RSI
        features["rsi_14"] = self._compute_rsi(df["close"], 14)
        
        # Close vs MA20 (% distance)
        ma20 = df["close"].rolling(20).mean()
        features["close_vs_ma20_pct"] = ((df["close"] - ma20) / ma20.replace(0, np.nan)) * 100
        
        return features.fillna(0).replace([np.inf, -np.inf], 0)
    
    def _compute_rsi(self, series: pd.Series, period: int = 14) -> pd.Series:
        delta = series.diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        avg_gain = gain.rolling(period).mean()
        avg_loss = loss.rolling(period).mean()
        rs = avg_gain / avg_loss.replace(0, np.nan)
        return 100 - (100 / (1 + rs)).fillna(50)
    
    def _compute_atr(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        high, low, close = df["high"], df["low"], df["close"]
        prev_close = close.shift(1)
        tr = np.maximum(
            high - low,
            np.maximum((high - prev_close).abs(), (low - prev_close).abs())
        ).fillna(0.0)
        return tr.rolling(period).mean()
    
    def train(self, df: pd.DataFrame, retrain: bool = False) -> dict:
        """
        Train K-Means model on historical data.
        
        Args:
            df: OHLCV DataFrame (must have: open, high, low, close, tick_volume)
            retrain: Force retrain even if model exists
        
        Returns:
            dict with training results
        """
        features = self._extract_features(df)
        
        # Remove rows with NaN (first 20 rows due to lookback)
        valid_mask = features.notna().all(axis=1)
        X = features[valid_mask].values
        
        if len(X) < self.n_clusters * 10:
            logger.warning(f"Not enough data to train ({len(X)} rows). Need at least {self.n_clusters * 10}.")
            return {"trained": False, "reason": "insufficient_data", "rows": len(X)}
        
        # Standardize
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)
        
        # PCA for visualization (optional, helps with clustering stability)
        self.pca = PCA(n_components=min(5, X_scaled.shape[1]))
        X_pca = self.pca.fit_transform(X_scaled)
        
        # Train K-Means
        self.model = KMeans(
            n_clusters=self.n_clusters,
            random_state=42,
            n_init=10,
            max_iter=300
        )
        self.model.fit(X_pca)
        
        # Map cluster labels to regime names
        # We determine mapping by looking at cluster centroids' return and volatility
        centroids = self.model.cluster_centers_
        cluster_map = {}
        for i in range(self.n_clusters):
            ret_val = centroids[i, 0]  # PCA component 1 (mostly returns)
            vol_val = centroids[i, 3] if centroids.shape[1] > 3 else 0  # PCA component 4 (mostly volatility)
            
            if ret_val > 0.5 and vol_val < 0.5:
                cluster_map[i] = "BULL"
            elif ret_val < -0.5 and vol_val < 0.5:
                cluster_map[i] = "BEAR"
            elif vol_val > 0.5:
                cluster_map[i] = "VOLATILE"
            else:
                cluster_map[i] = "SIDEWAYS"
        
        self.regime_labels = cluster_map
        
        # Save models
        self._save_models()
        
        # Get cluster distribution
        labels = self.model.predict(X_pca)
        regime_counts = pd.Series([self.regime_labels[l] for l in labels]).value_counts()
        
        logger.info(f"✅ K-Means trained on {len(X)} bars")
        logger.info(f"   Regime distribution: {regime_counts.to_dict()}")
        
        return {
            "trained": True,
            "samples": len(X),
            "regime_distribution": regime_counts.to_dict(),
            "cluster_map": cluster_map,
        }
    
    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Predict regimes and detect anomalies for each row.
        
        Args:
            df: OHLCV DataFrame
        
        Returns:
            DataFrame with added columns: regime_cluster, regime_label, is_anomaly, anomaly_score
        """
        result = df.copy()
        result["regime_cluster"] = -1
        result["regime_label"] = "UNKNOWN"
        result["is_anomaly"] = False
        result["anomaly_score"] = 0.0
        
        if self.model is None:
            loaded = self._load_models()
            if not loaded:
                logger.warning("No trained model found. Training on provided data...")
                self.train(df)
                if self.model is None:
                    return result
        
        features = self._extract_features(df)
        valid_mask = features.notna().all(axis=1)
        
        if valid_mask.sum() == 0:
            return result
        
        X = features[valid_mask].values
        
        if self.scaler:
            X_scaled = self.scaler.transform(X)
        else:
            X_scaled = X
        
        if self.pca:
            X_pca = self.pca.transform(X_scaled)
        else:
            X_pca = X_scaled
        
        # Predict clusters
        clusters = self.model.predict(X_pca)
        
        # Calculate anomaly scores (distance to nearest centroid)
        distances = self.model.transform(X_pca)
        min_distances = distances.min(axis=1)
        
        # Anomaly threshold: 95th percentile of distances
        anomaly_threshold = np.percentile(min_distances, 95)
        
        # Assign results
        result.loc[valid_mask, "regime_cluster"] = clusters
        result.loc[valid_mask, "regime_label"] = [self.regime_labels.get(c, "UNKNOWN") for c in clusters]
        result.loc[valid_mask, "anomaly_score"] = min_distances
        result.loc[valid_mask, "is_anomaly"] = min_distances > anomaly_threshold
        
        return result
    
    def get_regime_signal(self, df: pd.DataFrame) -> dict:
        """
        Get current market regime signal from the latest bar.
        
        Returns:
            dict with regime info and anomaly status
        """
        predicted = self.predict(df)
        if predicted.empty:
            return {"regime": "UNKNOWN", "is_anomaly": False, "anomaly_score": 0.0}
        
        latest = predicted.iloc[-1]
        return {
            "regime": latest.get("regime_label", "UNKNOWN"),
            "is_anomaly": latest.get("is_anomaly", False),
            "anomaly_score": latest.get("anomaly_score", 0.0),
            "cluster": latest.get("regime_cluster", -1),
        }
    
    def _save_models(self):
        """Save trained models to disk."""
        if self.model:
            with open(self.model_path, "wb") as f:
                pickle.dump(self.model, f)
        if self.scaler:
            with open(self.scaler_path, "wb") as f:
                pickle.dump(self.scaler, f)
        if self.pca:
            with open(self.pca_path, "wb") as f:
                pickle.dump(self.pca, f)
        logger.info(f"Models saved to {self.model_path.parent}")
    
    def _load_models(self) -> bool:
        """Load trained models from disk."""
        try:
            if self.model_path.exists():
                with open(self.model_path, "rb") as f:
                    self.model = pickle.load(f)
            if self.scaler_path.exists():
                with open(self.scaler_path, "rb") as f:
                    self.scaler = pickle.load(f)
            if self.pca_path.exists():
                with open(self.pca_path, "rb") as f:
                    self.pca = pickle.load(f)
            return self.model is not None
        except Exception as e:
            logger.error(f"Failed to load models: {e}")
            return False
    
    def enrich_pipeline(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add regime and anomaly columns to the pipeline DataFrame.
        To be called during pipeline execution.
        """
        predicted = self.predict(df)
        
        # Add columns if they don't exist
        for col in ["regime_label", "is_anomaly", "anomaly_score"]:
            if col in predicted.columns:
                df[col] = predicted[col].values
        
        return df