# MM_Signal_Strategy_XIE
Part of a 5-member MM635 group project. This repository contains only my individual contribution, including signal strategies, model training, real-time dashboards, and the final individual report.

📎 [Download Final Report – 635_MM_XIE.pdf](./635_MM_XIE.pdf)

> If the PDF preview does not render, click the filename above to download and view locally.

## 🗂️ Folder Structure
MM XIE/
├── 635_MM_XIE.pdf # Final report (in English)
├── BTCUSDT_1min_2024-05-01_to_2025-05-01.xlsx # Raw 1-min Binance data
├── lgb_model_strategy24.pkl # Trained LightGBM model for Strategy 2+4 (bearish)
├── lgb_model_strategy25.pkl # Trained LightGBM model for Strategy 2+5 (bullish)
├── prediction_logger.py # Real-time model prediction logger
├── signal_logger.py # Real-time strategy signal recorder
├── signal_history_xie.xlsx # Recorded signal history (evaluation results)
├── strategy24_dash.py # Dash app for visualizing Strategy 2+4
├── strategy25_dash.py # Dash app for visualizing Strategy 2+5
├── strategy34_dash.py # Dash app for Strategy 2+4 + 2+5 combined view
├── train_strategy24_lgb.py # LightGBM training script for Strategy 2+4
├── train_strategy25_lgb.py # LightGBM training script for Strategy 2+5
├── volume_signals.ipynb # Signal generation using OBV/AD/MFI/etc.
├── XIE_5Min_LiveAPI.ipynb # Live signal generation on 5-minute K-line
├── Show video.mp4 # Optional: Demo video for real-time interface (Dash app)

## 📘 Project Background

- 📊 Data: 1-minute historical BTCUSDT data (Binance) from May 1, 2024 to May 1, 2025
- 🧠 Goal: Build real-time signal strategies and prediction models for deployment
- 👤 This folder documents **my individual work**, from signal logic to model deployment

---

## 🔍 Signal Strategies Overview

### ✅ Strategy 2 + 4 (Bearish Reversal)
- MACD histogram flips negative
- ATR spike + RSI < 30
- Predicts: `-1` (short signal) or `0` (no signal)

### ✅ Strategy 2 + 5 (Bullish Rebound)
- RSI > 65 + MACD bullish crossover + Bullish Engulfing
- Predicts: `+1` (buy signal) or `0` (no signal)

> 📌 Strategy 2 + 5 is the **main strategy**, and Strategy 2 + 4 is used as a **supportive strategy**.

---

## 🧠 Model Training (LightGBM)

| Strategy      | Model File                    | Script                  |
|---------------|-------------------------------|--------------------------|
| Strategy 2+4  | `lgb_model_strategy24.pkl`     | `train_strategy24_lgb.py` |
| Strategy 2+5  | `lgb_model_strategy25.pkl`     | `train_strategy25_lgb.py` |

Models are trained using labeled signals from 1-minute data resampled to 5-minute intervals.  
Only two classes are used: signal vs. no-signal (binary classification).

---

## 📊 Real-Time Visualization (Dash Apps)

Use the following `.py` files to launch interactive dashboards:

```bash
python strategy24_dash.py     
python strategy25_dash.py     
python strategy34_dash.py

📓 Notebooks
XIE_5Min_LiveAPI.ipynb: Real-time signal generation logic using 5-min K-line

volume_signals.ipynb: Evaluation of 5 volume-based indicators (OBV, AD, ADOSC, MFI, BOP)
with backtesting results and net value curves.

📄 Final Report
📎 635_MM_XIE.pdf: The full project write-up (English)

Signal logic, strategy comparison, model explanation

Backtest charts and live signal screenshots

Conclusions and next steps

📁 Logs and Output
signal_logger.py: Saves model predictions in real-time

prediction_logger.py: Records signal ID, model probability, time

signal_history_xie.xlsx: Exported results of real-time strategy signal generation

▶️ Optional: Demo Video
Show video.mp4: Optional presentation video showing real-time signal system (Dash UI)
