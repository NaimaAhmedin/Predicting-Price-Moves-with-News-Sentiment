# 📊 Stock Market Data Analysis & Technical Indicators (YFinance Dataset)

This project performs **exploratory data analysis (EDA)**, **technical indicator engineering**, and **visualization** on historical stock price data downloaded from **Yahoo Finance**.

The goal is to explore price behavior, detect patterns, generate financial indicators, and prepare clean datasets for further modeling or machine-learning tasks.

---

## 🚀 Project Overview

### **1️⃣ Data Collection**

- Stocks analyzed: **AAPL, MSFT, GOOG, META, AMZN, NVDA**
- Data includes: Open, High, Low, Close, Volume, Adjusted Close

### **2️⃣ Data Cleaning & Preparation**

- Date sorting
- Handling missing values
- Creating daily returns
- Outlier detection

### **3️⃣ Feature Engineering**

Computed indicators:

- **MA20**, **MA50** (Moving Averages)
- **Daily Returns**
- **RSI-14** (Relative Strength Index)

### **4️⃣ Visualization**

- Close price trend plots
- Moving averages
- RSI charts
- Return correlation heatmaps

### **5️⃣ Outputs**

- Clean CSVs (processed datasets)
- Plots
- Ready-to-use time-series features for ML models

---

## 📁 Project Structure

project/
│── Data/ # Raw yfinance CSV files
│── notebooks/ # EDA & feature engineering notebooks
│── scripts/ # Python scripts for reusable functions
│── processed/ # Cleaned & feature-engineered datasets
│── README.md # Main project documentation

---

## 🛠️ Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Scikit-Learn
- TextBlob (optional sentiment)
- Jupyter Notebook

---

## 🎯 Project Goal

To build an **end-to-end pipeline** that:

- Cleans stock price data
- Computes technical indicators
- Visualizes trends
- Prepares data for machine-learning models

---

# Create virtual environment

python -m venv venv

# Activate (Windows)

venv\Scripts\activate

# Activate (Mac/Linux)

source venv/bin/activate
