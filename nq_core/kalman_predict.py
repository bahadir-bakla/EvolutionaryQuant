# NQ Quant Bot - Kalman Filter Prediction with Visualization
# Predicts future price using Kalman filter

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import Optional, Tuple, List
import yfinance as yf
from datetime import datetime, timedelta

try:
    from filterpy.kalman import KalmanFilter
    FILTERPY_AVAILABLE = True
except ImportError:
    FILTERPY_AVAILABLE = False
    print("filterpy not installed. Run: pip install filterpy")


@dataclass
class KalmanPrediction:
    """Kalman filter prediction result"""
    current_price: float
    predicted_price: float
    predicted_change_pct: float
    prediction_direction: str  # 'UP', 'DOWN', 'FLAT'
    confidence: float
    velocity: float  # Price velocity/momentum
    acceleration: float  # Rate of change of velocity
    prediction_band_upper: float
    prediction_band_lower: float


class KalmanPredictor:
    """
    Kalman Filter for price prediction
    
    State: [price, velocity, acceleration]
    - Predicts future price based on current state
    - Provides confidence bands
    """
    
    def __init__(
        self,
        process_noise: float = 0.01,
        measurement_noise: float = 0.1,
        prediction_steps: int = 5
    ):
        self.process_noise = process_noise
        self.measurement_noise = measurement_noise
        self.prediction_steps = prediction_steps
        
        self.kf = None
        self.is_initialized = False
        self.history = []
        
    def initialize(self, initial_price: float):
        """Initialize Kalman filter"""
        if not FILTERPY_AVAILABLE:
            raise ImportError("filterpy is required")
        
        # State: [price, velocity, acceleration]
        self.kf = KalmanFilter(dim_x=3, dim_z=1)
        
        # Initial state
        self.kf.x = np.array([[initial_price], [0.0], [0.0]])
        
        # State transition matrix (physics-based)
        dt = 1  # Time step
        self.kf.F = np.array([
            [1, dt, 0.5*dt**2],  # price = price + velocity*dt + 0.5*accel*dt^2
            [0, 1, dt],          # velocity = velocity + accel*dt
            [0, 0, 1]            # acceleration stays constant
        ])
        
        # Measurement matrix (we only observe price)
        self.kf.H = np.array([[1, 0, 0]])
        
        # Process noise
        self.kf.Q = np.eye(3) * self.process_noise
        self.kf.Q[0, 0] = self.process_noise * 0.1
        self.kf.Q[1, 1] = self.process_noise * 0.5
        self.kf.Q[2, 2] = self.process_noise
        
        # Measurement noise
        self.kf.R = np.array([[self.measurement_noise]])
        
        # Initial covariance
        self.kf.P = np.eye(3) * 100
        
        self.is_initialized = True
        
    def update(self, price: float) -> KalmanPrediction:
        """Update with new price and get prediction"""
        if not self.is_initialized:
            self.initialize(price)
        
        # Predict step
        self.kf.predict()
        
        # Update step with measurement
        self.kf.update(np.array([[price]]))
        
        # Extract state
        current_state = self.kf.x.flatten()
        filtered_price = current_state[0]
        velocity = current_state[1]
        acceleration = current_state[2]
        
        # Multi-step prediction
        predicted_state = current_state.copy()
        for _ in range(self.prediction_steps):
            predicted_state = self.kf.F @ predicted_state
        
        predicted_price = predicted_state[0]
        
        # Confidence from covariance
        uncertainty = np.sqrt(self.kf.P[0, 0])
        confidence = 1 / (1 + uncertainty / filtered_price * 100)
        
        # Prediction bands (2-sigma)
        band_width = uncertainty * 2 * self.prediction_steps
        
        # Direction
        change_pct = (predicted_price - filtered_price) / filtered_price * 100
        if change_pct > 0.1:
            direction = 'UP'
        elif change_pct < -0.1:
            direction = 'DOWN'
        else:
            direction = 'FLAT'
        
        # Store history
        self.history.append({
            'price': price,
            'filtered': filtered_price,
            'predicted': predicted_price,
            'velocity': velocity,
            'acceleration': acceleration,
            'confidence': confidence
        })
        
        return KalmanPrediction(
            current_price=filtered_price,
            predicted_price=predicted_price,
            predicted_change_pct=change_pct,
            prediction_direction=direction,
            confidence=confidence,
            velocity=velocity,
            acceleration=acceleration,
            prediction_band_upper=predicted_price + band_width,
            prediction_band_lower=predicted_price - band_width
        )
    
    def predict_n_steps(self, n_steps: int) -> np.ndarray:
        """Predict n steps into the future"""
        if not self.is_initialized:
            return np.array([])
        
        predictions = []
        state = self.kf.x.flatten().copy()
        
        for _ in range(n_steps):
            state = self.kf.F @ state
            predictions.append(state[0])
        
        return np.array(predictions)


def create_prediction_chart(
    df: pd.DataFrame,
    predictions: List[dict],
    future_predictions: np.ndarray,
    save_path: Optional[str] = None
) -> str:
    """Create visualization of Kalman predictions"""
    
    fig, axes = plt.subplots(3, 1, figsize=(14, 10))
    
    # Extract data
    prices = [p['price'] for p in predictions]
    filtered = [p['filtered'] for p in predictions]
    velocity = [p['velocity'] for p in predictions]
    acceleration = [p['acceleration'] for p in predictions]
    confidence = [p['confidence'] for p in predictions]
    
    # === Chart 1: Price with Kalman filter ===
    ax1 = axes[0]
    ax1.plot(prices, label='Actual Price', alpha=0.7, color='blue')
    ax1.plot(filtered, label='Kalman Filtered', linewidth=2, color='red')
    
    # Add future predictions
    if len(future_predictions) > 0:
        future_x = range(len(prices), len(prices) + len(future_predictions))
        ax1.plot(future_x, future_predictions, label='Prediction', 
                 linestyle='--', linewidth=2, color='green')
        
        # Prediction band
        uncertainty_growth = np.linspace(10, 50, len(future_predictions))
        upper = future_predictions + uncertainty_growth
        lower = future_predictions - uncertainty_growth
        ax1.fill_between(future_x, lower, upper, alpha=0.2, color='green',
                        label='Prediction Band')
    
    ax1.set_title('Price Prediction with Kalman Filter', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Price')
    ax1.legend(loc='upper left')
    ax1.grid(True, alpha=0.3)
    
    # === Chart 2: Velocity (Momentum) ===
    ax2 = axes[1]
    colors = ['green' if v > 0 else 'red' for v in velocity]
    ax2.bar(range(len(velocity)), velocity, color=colors, alpha=0.7)
    ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax2.set_title('Price Velocity (Momentum)', fontsize=12)
    ax2.set_ylabel('Velocity')
    ax2.grid(True, alpha=0.3)
    
    # === Chart 3: Confidence ===
    ax3 = axes[2]
    ax3.fill_between(range(len(confidence)), confidence, alpha=0.5, color='purple')
    ax3.set_title('Prediction Confidence', fontsize=12)
    ax3.set_ylabel('Confidence')
    ax3.set_ylim(0, 1)
    ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Chart saved to: {save_path}")
    
    plt.show()
    return save_path


def analyze_with_kalman(symbol: str = "NQ=F", interval: str = "15m", period: str = "7d"):
    """Analyze and visualize Kalman predictions"""
    
    print(f"\n{'='*60}")
    print(f"KALMAN FILTER PRICE PREDICTION - {symbol}")
    print(f"{'='*60}")
    
    # Fetch data
    print(f"\nFetching {interval} data...")
    ticker = yf.Ticker(symbol)
    df = ticker.history(period=period, interval=interval)
    df.columns = df.columns.str.lower()
    print(f"Bars: {len(df)}")
    
    # Run Kalman filter
    print("\nRunning Kalman filter...")
    predictor = KalmanPredictor(
        process_noise=0.01,
        measurement_noise=0.1,
        prediction_steps=5
    )
    
    for price in df['close']:
        prediction = predictor.update(price)
    
    # Future predictions
    print("\nGenerating future predictions...")
    future = predictor.predict_n_steps(20)  # Predict 20 bars ahead
    
    # Current state
    last = predictor.history[-1]
    print(f"\nCurrent Analysis:")
    print(f"  Price: {last['price']:.2f}")
    print(f"  Filtered: {last['filtered']:.2f}")
    print(f"  Velocity: {last['velocity']:.4f}")
    print(f"  Acceleration: {last['acceleration']:.6f}")
    print(f"  Confidence: {last['confidence']:.1%}")
    
    # Prediction
    print(f"\nPrediction ({predictor.prediction_steps} bars ahead):")
    print(f"  Predicted Price: {prediction.predicted_price:.2f}")
    print(f"  Expected Change: {prediction.predicted_change_pct:.2f}%")
    print(f"  Direction: {prediction.prediction_direction}")
    
    # Create chart
    print("\nGenerating chart...")
    chart_path = "kalman_prediction.png"
    create_prediction_chart(df, predictor.history, future, chart_path)
    
    return predictor, df


def get_kalman_signal(predictor: KalmanPredictor) -> Tuple[str, float]:
    """Get trading signal from Kalman prediction"""
    if not predictor.history:
        return 'NEUTRAL', 0.0
    
    last = predictor.history[-1]
    velocity = last['velocity']
    acceleration = last['acceleration']
    confidence = last['confidence']
    
    # Strong signal: velocity and acceleration aligned
    if velocity > 0.5 and acceleration > 0:
        return 'LONG', confidence * 1.5
    elif velocity < -0.5 and acceleration < 0:
        return 'SHORT', confidence * 1.5
    elif velocity > 0.2:
        return 'LONG', confidence
    elif velocity < -0.2:
        return 'SHORT', confidence
    else:
        return 'NEUTRAL', 0.0


# === TEST ===
if __name__ == "__main__":
    predictor, df = analyze_with_kalman("NQ=F", "15m", "7d")
