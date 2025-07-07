# prediction_logger.py

### This module (`prediction_logger.py`) is used to **track and store the accuracy of live candlestick signal predictions**.

### Records each prediction with timestamp, predicted direction (UP / DOWN), actual return, and whether it was correct.
### Computes hit rate across all logged predictions.
### Supports exporting prediction history to CSV (`prediction_log.csv`).

### Integrated with live Binance API signal detection. Every time a new 5-minute candle completes:
### 1. Our stratgy signal makes a prediction (Bullish/ Bearish / None).
### 2. On the next candle, the actual return is compared to check if the prediction was accurate.
### 3. Result is logged and used to update the hit rate in real time.


import pandas as pd
from datetime import datetime

class PredictionLogger:
    def __init__(self):
        self.log = []

    def record_prediction(self, timestamp, prediction, close_now, close_prev):
        """
        Record a single prediction result including direction, return, and hit status.
        """
        if prediction.startswith("UP") or prediction.startswith("DOWN"):
            try:
                ret = (close_now - close_prev) / close_prev
                hit = (
                    (prediction == "UP" and ret > 0) or
                    (prediction == "DOWN" and ret < 0)
                )
                self.log.append({
                    'timestamp': timestamp,
                    'prediction': prediction,
                    'return': ret,
                    'hit': int(hit)
                })
            except Exception as e:
                print(f"[Logger] Error recording prediction: {e}")

    def get_hit_rate(self):
        """
        Calculate the current prediction hit rate.
        """
        if not self.log:
            return 0.0
        return sum(entry['hit'] for entry in self.log) / len(self.log)

    def to_dataframe(self):
        """
        Convert the internal log list to a pandas DataFrame.
        """
        return pd.DataFrame(self.log)

    def save_to_csv(self, path='prediction_log.csv'):
        """
        Save the prediction log to a CSV file.
        """
        df = self.to_dataframe()
        df.to_csv(path, index=False)
        print(f"[Logger] Saved {len(df)} predictions to {path}")
