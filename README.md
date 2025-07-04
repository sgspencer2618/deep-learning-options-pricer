# Deep Learning for US Options Pricing ($AAPL Equity Options)

## Table of Contents

- [Project Description](#project-description)
- [Motivation & Background](#motivation--background)
- [Data Description](#data-description)
- [Repository Structure](#repository-structure)
- [Modeling Approach](#modeling-approach)
- [Evaluation & Results](#evaluation--results)
- [Limitations & Future Work](#limitations--future-work)
- [How to Run](#how-to-run)
- [References](#references)
- [Contact / Contribution / License](#contact--contribution--license)

## Project Description

Inspired by techniques from recent academic literature and proprietary research, This repository explores, implements, and evaluates different Machine Learning models to emperically price American equity Options for the $AAPL stock. Comparisons are made between 3 different Machine Learning models, namely:

1. XGBoost (Gradient-Boosted Trees) with Bayesian Hyperparameter Tuning
2. Gated Recurrent Unit (GRU)
3. Gated Recurrent Unit with Weighted Loss Function

## Motivation & Background
Market prices for options often deviate from theoretical values (determined by Black-Scholes, etc.) due to liquidity and market regime shifts.

We know that Machine Learning models can fit complex, nonlinear relationships between variables, and exploit additional features that are not necessarily captured by other methods of estimation.

This repo is based on recent research (See [[1]][ref1], [[2]][ref2]) and looks to bridge the gap between quant finance and modern ML.

## Data Description
For this project, I sourced recent historical options data from [Alpha Vantage](https://www.alphavantage.co/)'s Options Data API, with columns for expiration, strike, greeks, as well as integrating daily OHLC to try to capture market regime behaviour.

For this, I implemented a modular automated data ingestion pipeline using AWS S3 and Github Actions for scalibility and continuous integration during development.

In training, a time-based train/val/test split was used to avoid leakage; never shuffling entries across time in training, testing, or validation data.

### Feature Engineering
Here is the breakdown of features used in model training:
#### Features Used
1. 'strike'
2. 'option_type_encoded'
3. 'date'
4. 'implied_volatility'
5. 'delta'
6. 'gamma'
7. 'theta'
8. 'vega'
9. 'rho'
10. 'log_moneyness'
11. 'time_to_maturity'
12. 'log_moneyness_norm'
13. 'intrinsic_value_norm'

#### Greek product Features
I also feature-engineered several 'compound' features, calculated as simple products of greeks, namely:
1. 'delta_x_iv',
2. 'vega_x_ttm',
3. 'gamma_x_logm',
4. 'theta_x_intrinsic',

## Repository Structure
```
├───.github             # GitHub Actions/workflows for CI/CD automation
│   └───workflows           # CI/CD pipeline YAMLs
├───/dataIngest         # Automated data ingestion (config, helpers, schedulers, scripts)
│   ├───/config             # Ingestion configuration (API keys, settings)
│   ├───/helpers          # Utility functions for metadata, S3, etc.
│   ├───/logs               # Logs for ingestion processes
│   ├───/scheduler          # Scheduler scripts for timed data pulls
│   ├───/scripts            # Standalone scripts for fetching data
│   └───/src                # Core ingestion pipeline modules
├───/notebooks          # Jupyter notebooks for exploration and prototyping
├───/scripts            # Utility or run scripts (real-time, batch jobs, etc.)
└───/src                # Main source code for modeling and features
    ├───/features           # Feature engineering, indicator, and preprocessing code
    ├───/model              # XGBoost Model training, config, and evaluation
    └───/neural             # Neural network (GRU, attention) modules and experiments
```

## Modeling Approach
### Baseline
XGBoost regression was used for speed and feature importance as a baseline to compare the other models' performance.

### Advanced
Gated Recurrent Unit (GRU) neural networks with attention for sequence modeling, implementing with and without weighted loss functions to target fat tails and rare, high-value contracts.

### Tuning and Feature Scaling
Optuna was used for hyperparameter tuning. Feature was scaling handled using numpy and pandas.

## Evaluation & Results

### Metrics
- RMSE
- MAE
- MedAE
- R^2
## Limitations & Future Work
Based on observations and evaluation metrics:

### Limitations
- The model tends to **consistently underestimate** ultra-expensive illiquid contracts due to training data imbalance
- Only considers vanilla options (no spreads or multileg support)
- Would benefit from more exogenous data (macro, rates, VIX, etc)

- 
### Next Steps
- ensemble model exploration
- more features
- production-grade deployment
- Opportunities for transfer learning, real-time inference, and online retraining.

## How to Run
...
## References
[ref1]:https://arxiv.org/abs/2409.03204
[ref2]:https://arxiv.org/abs/2409.06724
## License
MIT License 2025
