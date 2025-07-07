# -*- coding: utf-8 -*-
"""
Created on Sat Jun  7 04:55:37 2025

@author: XIE
"""
import pandas as pd
import talib
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils import resample
import lightgbm as lgb
import joblib

# === 读取原始数据并转为 5 分钟 ===
df_1min = pd.read_csv("BTCUSDT_1min_2024-05-01_to_2025-05-01.csv", parse_dates=['timestamp'])
df_1min.set_index('timestamp', inplace=True)

df = df_1min.resample('5min').agg({
    'open': 'first',
    'high': 'max',
    'low': 'min',
    'close': 'last',
    'volume': 'sum'
}).dropna()

# === 技术指标（策略 2 + 5）
df['RSI'] = talib.RSI(df['close'], timeperiod=14)
macd, macdsignal, _ = talib.MACD(df['close'])
df['MACD'] = macd
df['MACD_Signal'] = macdsignal
df['BullishEngulfing'] = talib.CDLENGULFING(df['open'], df['high'], df['low'], df['close'])

# === 策略信号构造（预测下一根是否出现上涨信号）
df['Signal_Strategy25'] = 0
cond2 = (df['MACD'] > df['MACD_Signal']) & (df['BullishEngulfing'] > 0)
cond5 = (df['RSI'] > 65) & (df['MACD'] > df['MACD_Signal']) & (df['BullishEngulfing'] > 0)
df.loc[cond2 | cond5, 'Signal_Strategy25'] = 1
df['target'] = df['Signal_Strategy25'].shift(-1)  # 避免未来信息泄露

# === 特征构造（含滞后特征）
df['MACD_diff'] = df['MACD'] - df['MACD_Signal']
df['RSI_shift1'] = df['RSI'].shift(1)
df['RSI_shift2'] = df['RSI'].shift(2)
df['MACD_diff_shift1'] = df['MACD_diff'].shift(1)
df['MACD_diff_shift2'] = df['MACD_diff'].shift(2)

features = [
    'close', 'volume', 'RSI', 'MACD', 'MACD_Signal', 'MACD_diff',
    'RSI_shift1', 'RSI_shift2', 'MACD_diff_shift1', 'MACD_diff_shift2', 'BullishEngulfing'
]
df.dropna(inplace=True)

X = df[features]
y = df['target']

# === 下采样平衡类别
df_all = pd.concat([X, y], axis=1)
df_1 = df_all[df_all['target'] == 1]
df_0 = df_all[df_all['target'] == 0]
df_0_down = resample(df_0, replace=False, n_samples=len(df_1)*3, random_state=42)
df_bal = pd.concat([df_1, df_0_down]).sample(frac=1, random_state=42)

X_bal = df_bal[features]
y_bal = df_bal['target']

# === 划分训练集测试集
X_train, X_test, y_train, y_test = train_test_split(X_bal, y_bal, test_size=0.2, shuffle=False)

# === 模型训练
clf = lgb.LGBMClassifier()
clf.fit(X_train, y_train)

# === 评估
y_pred = clf.predict(X_test)
print("[Classification Report]")
print(classification_report(y_test, y_pred))
print("[Confusion Matrix]")
print(confusion_matrix(y_test, y_pred))

# === 保存模型
joblib.dump(clf, "lgb_model_strategy25.pkl")
print("Model saved as lgb_model_strategy25.pkl")
