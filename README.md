# Deep Learning for US Options Pricing ($AAPL Equity Options)

## Table of Contents

1. [Project Description](#project-description)
2. [Motivation & Background](#motivation--background)
3. [Data Description](#data-description)
    - [Feature Engineering](#feature-engineering)
4. [Repository Structure](#repository-structure)
5. [Modeling Approach](#modeling-approach)
    - [Baseline](#baseline)
    - [Advanced](#advanced)
    - [Tuning and Feature Scaling](#tuning-and-feature-scaling)
6. [Evaluation & Results](#evaluation--results)
    - [Models Compared](#models-compared)
    - [Performance Highlights](#performance-highlights)
    - [Key Plots](#key-plots)
    - [Analysis](#analysis)
7. [Limitations & Future Work](#limitations--future-work)
    - [Limitations](#limitations)
    - [Next Steps](#next-steps)
8. [How to Run](#how-to-run)
9. [References](#references)
10. [Contact / Contribution / License](#contact--contribution--license)

## Project Description

Inspired by techniques from recent academic literature and proprietary research, This repository explores, implements, and evaluates different Machine Learning models to emperically price American equity Options for the $AAPL stock. Comparisons are made between different Machine Learning models, namely:

1. XGBoost (Gradient-Boosted Trees) with Bayesian Hyperparameter Tuning
2. Gated Recurrent Unit (GRU)
3. Multilayer Perceptron (MLP)

## Motivation & Background
Market prices for options often deviate from theoretical values (determined by Black-Scholes, etc.) due to liquidity and market regime shifts.

We know that Machine Learning models can fit complex, nonlinear relationships between variables, and exploit additional features that are not necessarily captured by other methods of estimation.

This repo is based on recent research (See [[1]][ref1], [[2]][ref2]) and looks to bridge the gap between quant finance and modern ML.

## Data Description
For this project, I sourced recent historical options data from [Alpha Vantage](https://www.alphavantage.co/)'s Options Data API, with columns for expiration, strike, greeks, as well as integrating daily OHLC to try to capture market regime behaviour.

For this, I implemented a modular automated data ingestion pipeline using AWS S3 and Github Actions for scalibility and continuous integration during development.

In training, a time-based train/val/test split was used to avoid leakage; never shuffling entries across time in training, testing, or validation data - with certain features (such as greeks, IV) being shifted one day to prevent data leakage.

### Feature Engineering
Here is the breakdown of features used in model training:
#### Features Used
1. 'strike'
2. 'option_type_encoded'
3. 'date'
4. 'implied_volatility'
5. 'iv_change'
6. 'delta'
7. 'gamma'
8. 'theta'
9. 'vega'
10. 'rho'
11. 'log_moneyness'
12. 'time_to_maturity'
13. 'log_moneyness_norm'

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
Gated Recurrent Unit (GRU) neural networks with attention for sequence modeling, implementing with and without weighted loss functions to target fat tails and rare, high-value contracts. Multilayer Perceptron with 3 hidden layers, batch normalization, dropout, and L2 regularization (weight_decay).

### Tuning and Feature Scaling
Optuna was used for GRU hyperparameter tuning. Feature was scaling handled using numpy and pandas. The MLP used an Adam optimizer.

## Evaluation & Results

### Models Compared
- **XGBoost** Regressor as a **Baseline**
- **GRU (Gated Recurrent Unit)** neural network with attention
- **MLP (Multilayer Perceptron)**
- all trained and evaluated on identical engineered theoretical feature set (Greeks, moneyness, TTM, etc.)

### Performance Highlights

### Metrics

#### XGBoost Metrics
On a smaller test set (15%) segmented from the full data, the XGBoost model performed as follows:
- RMSE: 11.65148
- MAE: 7.56345
- MedAE: 3.79191
- R2: 0.90826

#### GRU Metrics
The GRU was trained on rolling windows taken from the same training data. the model performed as follows:
- RMSE: 3.01269
- MAE: 2.57142
- MedAE: 0.51145
- R2: 0.96596

#### MLP Metrics
RMSE: 10.81831
MAE: 8.26400
MedAE: 7.04856
R2: 0.92091

### Key plots

#### XGBoost
![XGBoost True vs. Predicted](assets/XGB_tvp.png) | ![XGBoost Feature Importance](assets/XGB_featimp.png) |
|:-----------------------------------------------:|:-----------------------------------------------------:|
| True vs Predicted Price                         | XGBoost Feature Importance                            |

#### GRU
| ![True vs Predicted Price](assets/gru_tvp.png) | ![GRU Norm QQ Plot](assets/gru_qq.png) |
|:--------------:|:--------------:|
|  True vs Predicted Price     | GRU QQ Plot of Residuals (against normal dist.)      |

Price Data Distribution:

![Data Distribution](assets/gru_ct.png)

#### MLP
![MLP True vs. Predicted](assets/mlp_tvp.png) | ![MLP Feature Importance](assets/mlp_resid.png) |
|:-----------------------------------------------:|:-----------------------------------------------------:|
| True vs Predicted Price                         | MLP Feature Importance                            |

### Analysis

#### Model Evaluation Table
Below is a table comparing the predictions of the GRU models, with the $\Delta$-values representing the difference between the model prediction and the true price.

| True Price ($) | XBoost ($) | GRU ($)     | MLP ($)    | $\Delta$ XGBoost (Abs) | $\Delta$ GRU (Abs) | $\Delta$ MLP (Abs)|
| -------------- | -----------|-------------|------------|------------------------|--------------------|-------------------|
| 122.93         | 121.92339  | 119.491135  | 122.65782  | 1.00661                | 3.43887            | 0.27218           |                   
| 63.32          | 64.440445  | 60.955967   | 67.25442   | 1.12044                | 2.36403            | 3.93442           |
| 62.82          | 61.433544  | 59.486553   | 61.650814  | 1.38646                | 3.33345            | 1.16919           |
| 146.52         | 148.74052  | 144.68823   | 141.23569  | 2.22052                | 1.83177            | 5.28431           |
| 126.52         | 124.18036  | 127.01633   | 137.10991  | 2.33964                | 0.49633            | 10.58991          |
## Limitations & Future Work

### Limitations
1. Model underestimates ultra-expensive, illiquid contracts due to data imbalance
2. Only considers vanilla options; no spreads/multileg.

### Next Steps
- Implement LSTM (Long Short-Term Memory) model to compare
- Ensemble models
- More features
- Opportunities for transfer learning, real-time inference, and online retraining.
## How to Run
...
## References
See Papers:
1. [Pricing American Options using Machine Learning Algorithms](https://arxiv.org/abs/2409.03204)
2. [MLP, XGBoost, KAN, TDNN, and LSTM-GRU Hybrid RNN with Attention for SPX and NDX European Call Option Pricing](https://arxiv.org/abs/2409.06724)

[ref1]:https://arxiv.org/abs/2409.03204
[ref2]:https://arxiv.org/abs/2409.06724
## License
MIT License 2025
