#!/usr/bin/env python
"""
scripts/prepare_indicators.py

Run from project root:
python scripts/prepare_indicators.py --data-dir "Data" --out-dir "data/processed/price_indicators"
"""

import argparse
from pathlib import Path
import pandas as pd
import numpy as np

def rsi_wilder(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=period, min_periods=period).mean()
    avg_loss = loss.rolling(window=period, min_periods=period).mean()
    avg_gain = avg_gain.copy()
    avg_loss = avg_loss.copy()
    for i in range(period, len(series)):
        if i == period:
            prev_gain = avg_gain.iloc[i]
            prev_loss = avg_loss.iloc[i]
        else:
            prev_gain = (prev_gain * (period - 1) + gain.iloc[i]) / period
            prev_loss = (prev_loss * (period - 1) + loss.iloc[i]) / period
        avg_gain.iloc[i] = prev_gain
        avg_loss.iloc[i] = prev_loss
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

def macd_pandas(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    macd = ema_fast - ema_slow
    macd_signal = macd.ewm(span=signal, adjust=False).mean()
    macd_hist = macd - macd_signal
    return macd, macd_signal, macd_hist

def process_file(path: Path, out_dir: Path, use_talib: bool = False):
    symbol = path.stem.upper()
    df = pd.read_csv(path, parse_dates=["Date"])
    df = df.sort_values("Date").reset_index(drop=True)
    df["return"] = df["Close"].pct_change()
    df["MA20"] = df["Close"].rolling(20).mean()
    df["MA50"] = df["Close"].rolling(50).mean()
    df["volatility_20d"] = df["return"].rolling(20).std()
    if use_talib:
        try:
            import talib  # type: ignore
        except Exception:
            print("ta-lib not available; falling back to pandas implementations")
            use_talib = False

    if use_talib:
        df["RSI14"] = talib.RSI(df["Close"].values, timeperiod=14)
        macd, signal, hist = talib.MACD(df["Close"].values, fastperiod=12, slowperiod=26, signalperiod=9)
        df["MACD"], df["MACD_signal"], df["MACD_hist"] = macd, signal, hist
    else:
        df["RSI14"] = rsi_wilder(df["Close"], 14)
        macd, signal, hist = macd_pandas(df["Close"])
        df["MACD"], df["MACD_signal"], df["MACD_hist"] = macd, signal, hist

    out_file = out_dir / f"{symbol}_processed.csv"
    df.to_csv(out_file, index=False)
    return out_file

def main(data_dir, out_dir, use_talib=False):
    data_dir = Path(data_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    csvs = sorted(data_dir.glob("*.csv"))
    for csv in csvs:
        print("Processing:", csv)
        out_file = process_file(csv, out_dir, use_talib)
        print("Saved:", out_file)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True, help="Path to directory containing price CSVs")
    parser.add_argument("--out-dir", required=True, help="Output directory for processed CSVs")
    parser.add_argument("--use-talib", action="store_true", help="Use ta-lib if available")
    args = parser.parse_args()
    main(args.data_dir, args.out_dir, args.use_talib)
