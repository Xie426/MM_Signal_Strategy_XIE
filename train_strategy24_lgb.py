# -*- coding: utf-8 -*-
"""
Created on Sat Jun  7 04:55:37 2025

@author: XIE
"""
import pandas as pd
import talib
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import lightgbm as lgb
import joblib

# 读取数据（1分钟合成5分钟）
df_1min = pd.read_csv("BTCUSDT_1min_2024-05-01_to_2025-05-01.csv", parse_dates=['timestamp'])
df_1min.set_index('timestamp', inplace=True)
df = df_1min.resample('5min').agg({
    'open': 'first',
    'high': 'max',
    'low': 'min',
    'close': 'last',
    'volume': 'sum'
}).dropna()

# === 添加技术指标（策略 2 + 4 所需） ===
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

# 构建策略信号
df['strat2'] = (df['MACD'] < df['MACD_Signal']) & (df['CDLSHOOTINGSTAR'] < 0)
df['strat4'] = (df['ATR'] > df['mean_ATR'] * 1.1) & \
               (df['volume'] > df['Volume_MA20'] * 1.2) & (df['RSI'] < 35)
df['Signal_Strategy24'] = 0
df.loc[df['strat2'] | df['strat4'], 'Signal_Strategy24'] = -1

# 预测目标 = 下一根K线的信号
df['target'] = df['Signal_Strategy24'].shift(-1)

# 拉平历史指标：t, t-1, t-2
base_features = [
    'close', 'volume', 'RSI', 'MACD', 'MACD_Signal',
    'ATR', 'mean_ATR', 'Volume_MA20', 'CDLSHOOTINGSTAR',
    'OBV', 'AD', 'ADOSC', 'MFI', 'BOP'
]
for lag in range(1, 3):
    for col in base_features:
        df[f"{col}_t-{lag}"] = df[col].shift(lag)

df.dropna(inplace=True)
features = [col for col in df.columns if any(col.startswith(f) for f in base_features)]
X = df[features]
y = df['target']

# 只保留 -1 和 0 样本
X = X[y.isin([-1, 0])]
y = y[y.isin([-1, 0])]

# 分训练测试（时间顺序）
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# === 欠采样：将 0 类降采样与 -1 类均衡 ===
df_train = X_train.copy()
df_train['target'] = y_train
minority = df_train[df_train['target'] == -1]
majority = df_train[df_train['target'] == 0].sample(len(minority), random_state=42)
df_balanced = pd.concat([minority, majority]).sample(frac=1, random_state=42)  # shuffle

X_train_bal = df_balanced.drop(columns='target')
y_train_bal = df_balanced['target']

# === 模型训练 ===
clf = lgb.LGBMClassifier(random_state=42)
clf.fit(X_train_bal, y_train_bal)

# === 模型评估 ===
y_pred = clf.predict(X_test)
print("[Classification Report]")
print(classification_report(y_test, y_pred))
print("[Confusion Matrix]")
print(confusion_matrix(y_test, y_pred))

# === 保存模型 ===
joblib.dump(clf, "lgb_model_strategy24.pkl")




