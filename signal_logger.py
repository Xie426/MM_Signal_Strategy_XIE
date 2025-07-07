import pandas as pd
import os

class SignalHistoryLogger:
    """
    A class to log and manage trading signals in a CSV file.

    Attributes:
        filename (str): Path to the CSV file for storing signals.
        df (pd.DataFrame): In-memory DataFrame holding signals with columns:
            - timestamp: Signal timestamp (Asia/Singapore timezone).
            - type: Signal type ('bullish' or 'bearish').
            - price: Close price at the signal time.
            - trigger: Conditions that triggered the signal (e.g., 'BullishEngulfing + MA5>MA20').
    """

    def __init__(self, filename='signal_history.csv'):
        """
        Initialize the logger with a CSV file.

        If the file exists, it is loaded into memory. Otherwise, an empty DataFrame
        is created with the required columns.

        Args:
            filename (str): Path to the CSV file (default: 'signal_history.csv').
        """
        self.filename = filename
        if os.path.exists(self.filename):
            self.df = pd.read_csv(self.filename, parse_dates=['timestamp'])
        else:
            self.df = pd.DataFrame(columns=['timestamp', 'type', 'price', 'trigger'])

    def add_signal(self, signal_type, timestamp, price, trigger=''):
        """
        Add a new signal to the logger and save to CSV.

        Appends the signal to the in-memory DataFrame and immediately writes to the
        CSV file to ensure real-time persistence. Used in `generate_combined_signals`
        when a confirmed signal (bullish_combined=1 or bearish_combined=-1) is detected.

        Args:
            signal_type (str): Type of signal ('bullish' or 'bearish').
            timestamp (pd.Timestamp): Timestamp of the signal (Asia/Singapore).
            price (float): Close price at the signal time.
            trigger (str): Conditions that triggered the signal (default: '').
        """
        new_row = pd.DataFrame([{
            'timestamp': timestamp,
            'type': signal_type,
            'price': price,
            'trigger': trigger
        }])
        if self.df.empty:
            self.df = new_row
        else:
            self.df = pd.concat([self.df, new_row], ignore_index=True)
        # Save to CSV immediately for real-time persistence
        self.df.to_csv(self.filename, index=False)

    def get_history(self):
        """
        Retrieve the signal history from the CSV file.

        Used in `plot_realtime_signals` to load historical signals for plotting.
        Only signals within the chart's time range (e.g., last 50 candles) are
        plotted after filtering by timestamp.

        Returns:
            pd.DataFrame: DataFrame with columns ['timestamp', 'type', 'price', 'trigger'].
                         Returns an empty DataFrame if the file doesn’t exist.
        """
        if os.path.exists(self.filename):
            return pd.read_csv(self.filename, parse_dates=['timestamp'])
        return pd.DataFrame(columns=['timestamp', 'type', 'price', 'trigger'])

    def save_to_csv(self, filename=None):
        """
        Save the in-memory signal data to a CSV file.

        If filename is provided, saves to the specified path; otherwise, uses the
        default filename.

        Args:
            filename (str, optional): Path to save the CSV file.
        """
        filename = filename or self.filename
        self.df.to_csv(filename, index=False)