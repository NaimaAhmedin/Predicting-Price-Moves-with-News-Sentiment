# 🧩 Scripts Documentation

This folder contains Python scripts used to modularize and automate the data cleaning and feature-engineering pipeline.

---

## 📌 Included Scripts

### **1️⃣ data_loader.py**

Loads raw CSV files and returns pandas DataFrames.

### **2️⃣ preprocessing.py**

Handles:

- Missing values
- Date sorting
- Outlier detection
- Return calculations

### **3️⃣ indicators.py**

Computes technical indicators:

- MA20
- MA50
- RSI-14
- (You can add more indicators here)

### **4️⃣ visualization.py**

Generates:

- Price trend plots
- Moving averages
- RSI charts
- Correlation plots

---

## 🎯 Purpose

- Avoid repeating code in notebooks
- Maintain cleaner workflow
- Allow building a scalable data pipeline
- Prepare data for ML models in a structured way
