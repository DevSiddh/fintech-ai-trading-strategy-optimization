# Machine Learning-Based Financial Time-Series Forecasting and Strategy Evaluation System

## Project Report — German MSc Application Supplement

**Author**: Challa Yagnesh Sai Siddhardha (22NG1A6112)  
**Institution**: Usha Rama College of Engineering and Technology (Autonomous), JNTUK, Kakinada  
**Programme**: B.Tech in Artificial Intelligence and Machine Learning (2022–2026)  
**GitHub**: [github.com/DevSiddh/fintech-ai-trading-strategy-optimization](https://github.com/DevSiddh/fintech-ai-trading-strategy-optimization)

*Note: This document supplements the official JNTUK project report. It is structured to highlight competencies relevant to German MSc programmes in AI, ML, and Data Science.*

---

## Abstract

Financial time series are inherently non-stationary and noisy, posing significant challenges for predictive modelling. This project develops an end-to-end supervised learning framework that forecasts one-day-ahead cryptocurrency returns using eleven hand-engineered technical indicators as input features. Three model architectures — XGBoost, Random Forest, and Long Short-Term Memory (LSTM) networks — are trained on historical OHLCV data, evaluated through time-preserving validation, and deployed via a dual-interface web dashboard (Streamlit and Flask). The system incorporates transaction-cost-aware backtesting with risk-adjusted performance metrics including the Sharpe ratio, maximum drawdown, directional accuracy, and win rate. Experimental results demonstrate that gradient-boosted tree ensembles consistently achieve directional accuracy exceeding 55% — a statistically meaningful improvement over random guessing in daily-frequency financial data.

**Keywords**: Machine Learning, Financial Time Series, XGBoost, LSTM, Ensemble Methods, Backtesting, Feature Engineering

---

## 1. Introduction

### 1.1 Problem Statement

Predicting financial asset price movements is a fundamental challenge in quantitative finance. Traditional econometric approaches — ARIMA, GARCH, and vector autoregression — assume linear relationships and stationary distributions, assumptions routinely violated by real market data. Machine learning offers an alternative paradigm: learn complex, nonlinear patterns directly from historical data without explicit parametric assumptions.

### 1.2 Research Questions

1. Can supervised machine learning models trained on technical indicators predict next-day return direction with statistically significant accuracy?
2. Which model architecture (tree ensemble vs. recurrent neural network) performs best on structured tabular financial data?
3. Does a threshold-based signal filter improve strategy performance after accounting for realistic transaction costs?

### 1.3 Objectives

- Construct a complete ML pipeline from raw data ingestion through model deployment
- Implement and compare three model architectures (XGBoost, Random Forest, LSTM)
- Design a transaction-cost-aware backtesting engine with standardised risk metrics
- Build interactive dashboards for real-time prediction and model interpretability

---

## 2. Literature Review

The intersection of machine learning and financial forecasting has attracted significant academic attention. Fischer and Krauss (2018) demonstrated that LSTM networks outperform traditional classifiers on S&P 500 constituent data, achieving 54.4% directional accuracy. Chen and Guestrin (2016) introduced XGBoost, which has since become the dominant algorithm for structured tabular data in both competition and production settings. Wilder (1978) formalised the Relative Strength Index, one of the most widely used technical oscillators.

A key methodological insight from prior work is the importance of **stationarity-aware target design** — predicting returns rather than raw prices eliminates the non-stationarity that causes most financial ML models to overfit on price trends rather than learn genuine predictive signals.

---

## 3. Methodology

### 3.1 Data Pipeline

```
Yahoo Finance API (yfinance)
    ↓
Raw OHLCV data [Open, High, Low, Close, Volume]
    ↓
MultiIndex column flattening + field selection
    ↓
11 Technical Indicators computed (SMA, WMA, MOM, STCK, STCD, RSI, MACD, MACD_SIGNAL, LWR, ADO, CCI)
    ↓
Supervised dataset: X = normalised indicators, y = (Close_{t+1} − Close_t) / Close_t
    ↓
Time-preserving split: Train 72% → Validation 8% → Test 20%
```

### 3.2 Feature Engineering

| Category | Indicators | Economic Interpretation |
|----------|-----------|------------------------|
| Trend | SMA₁₄, WMA₁₄, MOM₁₄ | Direction and strength of price movement |
| Oscillator | RSI₁₄, STCK, STCD, LWR₁₄, CCI₂₀ | Overbought/oversold conditions, mean reversion |
| Trend-Momentum | MACD, MACD_SIGNAL | Convergence/divergence of moving averages |
| Volume | ADO (Accumulation/Distribution) | Money flow confirmation of price trends |

### 3.3 Model Architectures

**XGBoost** (n_estimators=200, learning_rate=0.05, max_depth=6): A gradient-boosted tree ensemble that sequentially adds weak learners to minimise a regularised objective function. Chosen for its robustness to outliers and automatic feature interaction discovery.

**Random Forest** (n_estimators=200, max_depth=10): A bagging ensemble of de-correlated decision trees. Provides out-of-bag feature importance estimates and naturally resists overfitting through bootstrap aggregation.

**LSTM** (64 units, sequence_length=10, EarlyStopping patience=8): A recurrent neural network with gating mechanisms designed for sequence data. Each training sample is a 10-day × 11-feature tensor. Trained with Adam optimiser and MSE loss.

### 3.4 Evaluation Protocol

Predictions are evaluated on a time-preserving holdout set (most recent 20% of data) — no shuffling, to prevent look-ahead bias. The signal generation rule uses a ±0.1% threshold (calibrated to typical exchange transaction costs) to filter noise:

| Predicted Return | Signal | Position |
|-----------------|--------|----------|
| > +0.1% | BUY (+1) | Long |
| < −0.1% | SELL (−1) | Short |
| within ±0.1% | HOLD (0) | No position |

---

## 4. Results

### 4.1 Model Performance

| Metric | XGBoost | Random Forest | LSTM |
|--------|---------|---------------|------|
| Directional Accuracy | ~56–60% | ~54–58% | ~52–56% |
| RMSE | ~0.025 | ~0.026 | ~0.028 |
| Training Time | Seconds | Seconds | Minutes |

XGBoost consistently achieves the highest directional accuracy, supporting the finding that gradient boosting on well-engineered features outperforms both bagging ensembles and recurrent architectures on daily-frequency financial data.

### 4.2 Backtest Performance (XGBoost Strategy)

| Metric | Value |
|--------|-------|
| Sharpe Ratio | 0.8–1.4 (regime-dependent) |
| Maximum Drawdown | 15–35% |
| Win Rate (active trades) | 50–58% |
| Signal Frequency | ~40–60% HOLD (noise-filtered) |

### 4.3 Feature Importance

XGBoost's built-in feature importance consistently ranks RSI, MACD, and Stochastic %K as the most predictive indicators — validating the indicator selection through the model itself rather than assumption alone.

---

## 5. Discussion

### Key Findings

1. **Tree ensembles outperform deep learning on tabular indicator data** at daily frequency, consistent with recent literature showing gradient boosting matches or exceeds deep learning on structured problems.

2. **The signal threshold is critical** — without the ±0.1% filter, transaction costs erode any predictive edge.

3. **Returns prediction beats price prediction** — models trained to predict raw prices simply learn yesterday's price; models trained to predict returns must learn the actual signal.

### Limitations

- Single-asset focus (no cross-asset portfolio optimisation)
- Daily frequency only (no intraday microstructure)
- Reliance on technical indicators alone (no fundamental or sentiment data)

### Future Work

Multi-asset portfolio optimisation, incorporation of on-chain metrics and NLP sentiment, reinforcement learning for dynamic position sizing, and probabilistic forecasting using quantile regression.

---

## 6. Software Implementation

### Technology Stack

| Layer | Technologies |
|-------|-------------|
| Data | yfinance, pandas, numpy |
| ML | scikit-learn, XGBoost, TensorFlow/Keras |
| Backend | Flask (REST API), joblib (model persistence) |
| Frontend | Streamlit, Plotly.js, vanilla JavaScript, CSS |
| Reproducibility | requirements.txt, argparse CLI |

### Deployment Options

1. **CLI**: `python main.py --ticker BTC-USD --start 2018-01-01 --model XGBoost`
2. **Streamlit Dashboard**: `streamlit run app.py`
3. **Flask Web App**: `python flask_app.py`

---

## 7. Competencies Demonstrated

| Area | Evidence |
|------|----------|
| Machine Learning | XGBoost, Random Forest, LSTM — trained, compared, evaluated |
| Deep Learning | LSTM architecture design, sequence modelling, EarlyStopping |
| Data Science | Feature engineering, stationarity analysis, normalisation |
| Statistics | Sharpe ratio, drawdown analysis, return distribution analysis |
| Software Engineering | Dual-interface architecture, async pipeline, REST API |
| Research Methodology | Time-preserving validation, hypothesis testing, literature review |

---

## 8. Academic Alignment with German MSc Programmes

This project intersects with active research areas at:

- **TU Munich / Tübingen**: Time series forecasting, deep learning for sequential data
- **KIT / Goethe Frankfurt**: Financial machine learning, risk modelling
- **Fraunhofer IAIS / TU Berlin**: Explainable AI, feature importance analysis
- **TU Darmstadt / Freiburg**: Reinforcement learning, autonomous trading agents

---

## References

1. Chen, T., & Guestrin, C. (2016). XGBoost: A Scalable Tree Boosting System. *KDD 2016*.
2. Fischer, T., & Krauss, C. (2018). Deep learning with LSTM networks for financial market predictions. *EJOR*, 270(2), 654–669.
3. Wilder, J. W. (1978). *New Concepts in Technical Trading Systems*. Trend Research.
4. Sharpe, W. F. (1994). The Sharpe Ratio. *Journal of Portfolio Management*, 21(1), 49–58.
5. Murphy, J. J. (1999). *Technical Analysis of the Financial Markets*. NYIF.

---

*This document supplements the official JNTUK project submission. Use alongside the GitHub repository for German MSc applications.*
