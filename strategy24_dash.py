# -*- coding: utf-8 -*-
"""
Created on Sat Jun  7 00:16:33 2025

@author: XIE
"""
import pandas as pd
import numpy as np
import talib
import joblib
from binance.client import Client
from dash import Dash, dcc, html
from dash.dependencies import Output, Input
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import dash_bootstrap_components as dbc

# ============ Binance Setup ============
client = Client(api_key='your_api_key', api_secret='your_api_secret')

# ============ Feature Construction ============
def fetch_and_process_data(symbol="BTCUSDT", interval=Client.KLINE_INTERVAL_5MINUTE, limit=1000):
    klines = client.get_klines(symbol=symbol, interval=interval, limit=limit)
    df = pd.DataFrame(klines, columns=[
        'timestamp', 'open', 'high', 'low', 'close', 'volume',
        'close_time', 'quote_asset_volume', 'number_of_trades',
        'taker_buy_base_volume', 'taker_buy_quote_volume', 'ignore'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.set_index('timestamp', inplace=True)
    df = df[['open', 'high', 'low', 'close', 'volume']].astype(float)

    # Technical indicators
    df['RSI'] = talib.RSI(df['close'], timeperiod=14)
    macd, macdsignal, _ = talib.MACD(df['close'])
    df['MACD'] = macd
    df['MACD_Signal'] = macdsignal
    df['ATR'] = talib.ATR(df['high'], df['low'], df['close'])
    df['mean_ATR'] = df['ATR'].rolling(20).mean()
    df['Volume_MA20'] = df['volume'].rolling(20).mean()
    df['CDLSHOOTINGSTAR'] = talib.CDLSHOOTINGSTAR(df['open'], df['high'], df['low'], df['close'])
    df['OBV'] = talib.OBV(df['close'], df['volume'])
    df['AD'] = talib.AD(df['high'], df['low'], df['close'], df['volume'])
    df['ADOSC'] = talib.ADOSC(df['high'], df['low'], df['close'], df['volume'])
    df['MFI'] = talib.MFI(df['high'], df['low'], df['close'], df['volume'])
    df['BOP'] = (df['close'] - df['open']) / (df['high'] - df['low'] + 1e-9)

    # Lag features
    base_features = [
        'close', 'volume', 'RSI', 'MACD', 'MACD_Signal',
        'ATR', 'mean_ATR', 'Volume_MA20', 'CDLSHOOTINGSTAR',
        'OBV', 'AD', 'ADOSC', 'MFI', 'BOP'
    ]
    for lag in range(1, 3):
        for col in base_features:
            df[f"{col}_t-{lag}"] = df[col].shift(lag)

    df.dropna(inplace=True)
    return df

# ============ Load Model ============
model = joblib.load("lgb_model_strategy24.pkl")

# ============ Dash App ============
app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
app.layout = html.Div([
    html.H4("[Strategy 2+4] BTCUSDT 5-Minute Live Chart + Signal Probability"),
    dcc.Interval(id='interval-component', interval=300 * 1000, n_intervals=0),
    dcc.Graph(id='live-chart')
])

@app.callback(
    Output('live-chart', 'figure'),
    Input('interval-component', 'n_intervals')
)
def update_chart(n):
    df = fetch_and_process_data()
    df_plot = df.iloc[-50:].copy()
    latest = df.iloc[[-1]]  # 最后一行用于预测

    # 特征列表（训练时的 42 个）
    features = [col for col in df.columns if col not in ['open', 'high', 'low']]
    X_pred = latest[features]

    prob = model.predict_proba(X_pred)[0][0]  # class -1 的概率

    # ==== Plot ====
    fig = make_subplots(rows=4, cols=1, shared_xaxes=True, vertical_spacing=0.1,
                        subplot_titles=['Candlestick', 'RSI', 'ATR', 'Volume'],
                        row_heights=[0.4, 0.2, 0.2, 0.2])

    fig.add_trace(go.Candlestick(x=df_plot.index, open=df_plot['open'], high=df_plot['high'],
                                 low=df_plot['low'], close=df_plot['close'], name='Candlestick'), row=1, col=1)
    fig.add_annotation(x=latest.index[0], y=latest['close'].values[0], text=f"P↓={prob:.2%}",
                       showarrow=True, arrowhead=1, row=1, col=1)

    fig.add_trace(go.Scatter(x=df_plot.index, y=df_plot['RSI'], mode='lines', name='RSI'), row=2, col=1)
    fig.add_hline(y=50, line_dash='dash', line_color='gray', row=2, col=1)

    fig.add_trace(go.Scatter(x=df_plot.index, y=df_plot['ATR'], mode='lines', name='ATR'), row=3, col=1)
    fig.add_trace(go.Scatter(x=df_plot.index, y=df_plot['mean_ATR'], mode='lines', name='mean_ATR', line=dict(dash='dash')), row=3, col=1)

    fig.add_trace(go.Bar(x=df_plot.index, y=df_plot['volume'], name='Volume'), row=4, col=1)
    fig.add_trace(go.Scatter(x=df_plot.index, y=df_plot['Volume_MA20'], mode='lines', name='Volume_MA20', line=dict(dash='dash')), row=4, col=1)

    fig.update_layout(height=800, template='plotly_white', xaxis_rangeslider_visible=False)
    return fig

if __name__ == '__main__':
    app.run(debug=True, port=8051)



# http://127.0.0.1:8051

