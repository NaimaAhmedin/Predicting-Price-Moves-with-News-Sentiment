#!/usr/bin/env python3
"""
scripts/sentiment_analysis.py
Usage:
  python scripts/sentiment_analysis.py \
    --news ../data/processed/news_cleaned.csv \
    --prices ../data/processed/price_indicators.csv \
    --out ../data/processed/final_merged_dataset.csv
"""

import argparse
from pathlib import Path
import pandas as pd
import numpy as np
import re
# Try to import TextBlob; if unavailable, provide a lightweight fallback shim
try:
    from textblob import TextBlob  # type: ignore
except Exception:
    class _Sentiment:
        def __init__(self, polarity):
            self.polarity = float(polarity)
    class TextBlob:
        # small positive/negative token sets for a simple heuristic fallback
        _POS = {
            'good', 'great', 'positive', 'up', 'bull', 'profit', 'gain',
            'beat', 'outperform', 'strong', 'improve', 'increase', 'surge',
            'rise', 'grow', 'growth'
        }
        _NEG = {
            'bad', 'terrible', 'negative', 'down', 'bear', 'loss', 'drop',
            'miss', 'weak', 'decline', 'decrease', 'fall', 'plunge'
        }
        def __init__(self, text):
            self._text = '' if text is None else str(text).lower()
        @property
        def sentiment(self):
            words = re.findall(r"\w+", self._text)
            if not words:
                return _Sentiment(0.0)
            pos = sum(1 for w in words if w in TextBlob._POS)
            neg = sum(1 for w in words if w in TextBlob._NEG)
            # simple normalized polarity in [-1,1]
            polarity = (pos - neg) / max(1, len(words))
            return _Sentiment(polarity)
from scipy.stats import pearsonr

def compute_polarity(text):
    try:
        return TextBlob(str(text)).sentiment.polarity
    except Exception:
        return 0.0

def aggregate_daily_sentiment(news_df):
    # ensure datetime
    news_df['date'] = pd.to_datetime(news_df['date'], errors='coerce')
    news_df['ticker'] = news_df['stock'].astype(str).str.upper()
    news_df['polarity'] = news_df['headline'].apply(compute_polarity)

    news_df['date_only'] = news_df['date'].dt.date
    agg = (news_df.groupby(['ticker', 'date_only'])
           .agg(avg_sentiment=('polarity', 'mean'),
                min_sentiment=('polarity', 'min'),
                max_sentiment=('polarity', 'max'),
                news_count=('headline', 'count'))
           .reset_index()
           .rename(columns={'date_only': 'Date'}))
    agg['Date'] = pd.to_datetime(agg['Date'])
    return agg

def prepare_price_df(price_df):
    price_df['Date'] = pd.to_datetime(price_df['Date'])
    price_df['ticker'] = price_df['ticker'].astype(str).str.upper()
    price_df = price_df.sort_values(['ticker','Date']).reset_index(drop=True)
    # compute daily return (close pct change)
    price_df['daily_return'] = price_df.groupby('ticker')['Close'].pct_change()
    return price_df

def merge_and_analyze(price_df, sentiment_df):
    merged = price_df.merge(sentiment_df, on=['ticker','Date'], how='left')
    # fill missing sentiment with 0 and news_count with 0
    merged['avg_sentiment'] = merged['avg_sentiment'].fillna(0.0)
    merged['news_count'] = merged['news_count'].fillna(0).astype(int)
    # correlation per ticker (Pearson) and global
    per_ticker_corr = {}
    for t, grp in merged.groupby('ticker'):
        tmp = grp.dropna(subset=['daily_return'])
        if tmp.shape[0] >= 10:
            r, p = pearsonr(tmp['avg_sentiment'], tmp['daily_return'])
            per_ticker_corr[t] = {'pearson_r': float(r), 'p_value': float(p), 'n': int(tmp.shape[0])}
        else:
            per_ticker_corr[t] = {'pearson_r': None, 'p_value': None, 'n': int(tmp.shape[0])}

    # global correlation (drop NaNs)
    g = merged.dropna(subset=['daily_return'])
    if g.shape[0] >= 10:
        global_r, global_p = pearsonr(g['avg_sentiment'], g['daily_return'])
    else:
        global_r, global_p = None, None

    return merged, per_ticker_corr, (global_r, global_p)

def save_outputs(merged_df, per_ticker_corr, global_stats, out_path, corr_path):
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    merged_df.to_csv(out_path, index=False)
    # save correlations
    corr_df = pd.DataFrame.from_dict(per_ticker_corr, orient='index').reset_index().rename(columns={'index':'ticker'})
    corr_df.to_csv(corr_path, index=False)
    return out_path, corr_path

def main(args):
    news_path = Path(args.news)
    price_path = Path(args.prices)
    out_path = Path(args.out)
    corr_path = out_path.parent / 'sentiment_correlations.csv'

    if not news_path.exists():
        raise FileNotFoundError(f"News file not found: {news_path}")
    if not price_path.exists():
        raise FileNotFoundError(f"Price file not found: {price_path}")

    news = pd.read_csv(news_path)
    prices = pd.read_csv(price_path)

    print("Aggregating daily sentiment...")
    sentiment_daily = aggregate_daily_sentiment(news)
    print("Preparing price data...")
    prices_prepared = prepare_price_df(prices)
    print("Merging and computing correlations...")
    merged, per_ticker_corr, global_stats = merge_and_analyze(prices_prepared, sentiment_daily)

    print("Global Pearson r:", global_stats)
    out_file, corr_file = save_outputs(merged, per_ticker_corr, global_stats, out_path, corr_path)
    print("Saved merged dataset to:", out_file)
    print("Saved per-ticker correlations to:", corr_file)
    if global_stats[0] is not None:
        print("Global Pearson r = {:.4f}, p={:.3g}".format(global_stats[0], global_stats[1]))
    else:
        print("Not enough data for global Pearson correlation.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sentiment analysis + merge with price data")
    parser.add_argument("--news", required=True, help="Path to cleaned news CSV (news_cleaned.csv)")
    parser.add_argument("--prices", required=True, help="Path to price indicators CSV (price_indicators.csv)")
    parser.add_argument("--out", required=True, help="Path to output merged CSV")
    args = parser.parse_args()
    main(args)
