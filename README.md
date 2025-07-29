# Deep Learning for US Options Pricing ($AAPL Equity Options)

<p align="center">
  <img src="https://github.com/user-attachments/assets/5ef3adce-335f-445c-bc03-2d84f804db39" width="500"/>
</p>

## Summary

This project explores machine learning models for options price prediction using engineered financial features. We compare XGBoost, a Multilayer Perceptron (MLP), and a Gated Recurrent Unit (GRU) model on historical market data. While MLP had the lowest average error in the full test set, this is partly due to data imbalance; the GRU actually outperformed all models on the most common price range (under $140), as shown in the low-value test sample. Our results highlight that model choice should align with data characteristics and feature structure.

## Table of Contents
1. [Summary](#summary)
2. [Project Description](#project-description)
3. [Motivation & Background](#motivation--background)
4. [Data Description](#data-description)
    - [Feature Engineering](#feature-engineering)
5. [Repository Structure](#repository-structure)
6. [Modeling Approach](#modeling-approach)
    - [Baseline](#baseline)
    - [Advanced](#advanced)
    - [Tuning and Feature Scaling](#tuning-and-feature-scaling)
7. [Evaluation & Results](#evaluation--results)
    - [Models Compared](#models-compared)
    - [Performance Highlights](#performance-highlights)
    - [Key Plots](#key-plots)
    - [Analysis](#analysis)
    - [Overall Performance Evaluation](#overall-performance-evaluation)
8. [Limitations & Future Work](#limitations--future-work)
    - [Limitations](#limitations)
    - [Next Steps](#next-steps)
9. [How to Run](#how-to-run)
10. [References](#references)
11. [License](#license)

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
├── .github             # GitHub Actions/workflows for CI/CD automation
│   └── workflows       # CI/CD pipeline YAMLs
├── assets              # Static assets (eval/metric graphs)
├── data                # Local data directory for final model inputs/outputs
│   └── mlp                 # Data specifically used for MLP model training
├── dataIngest          # Automated data ingestion pipeline
│   ├── config              # Config files (API keys, settings, parameters)
│   ├── helpers             # Utility functions (file handling, S3, etc.)
│   ├── logs                # Runtime logs for ingestion jobs
│   ├── scheduler           # Scheduler scripts for timed ingestion
│   ├── scripts             # One-off or batch ingestion scripts
│   └── src                 # Core pipeline modules for ingestion logic
├── logs                # General logs for model training, debugging, etc.
├── scripts             # Executable scripts for training, evaluation, or batch jobs
├── src                 # Main source code for modeling pipeline
│   ├── datasets            # Dataset loaders and preprocessing
│   ├── evaluation          # Model evaluation, metrics, and visualization tools
│   ├── features            # Feature engineering, technical indicators, etc.
│   ├── models              # Model definitions and configuration: GRU, XGBoost, MLP, etc.
│   ├── model_files         # Serialized models, configs, and artifacts
│   └── training            # Training loops and files (XGB, GRU, MLP)
```

## Modeling Approach
Models were trained on-device with identical training conditions.
### Baseline
XGBoost regression was used for speed and feature importance as a baseline to compare the other models' performance.

### Advanced
Gated Recurrent Unit (GRU) neural networks with attention for sequence modeling, implementing with and without weighted loss functions to target fat tails and rare, high-value contracts. Multilayer Perceptron with 3 hidden layers, batch normalization, dropout, and L2 regularization (weight_decay).

### MLP Configuration
<p align="center">
  <img src="assets\MLP_config.png" width="500"/>
</p>

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
Data was split (70:15:25) train-val-test, and models were trained and evaluated on the identical training, validation, and testing data to ensure comparable results.

#### XGBoost Metrics
- RMSE: 11.65148
- MAE: 7.56345
- MedAE: 3.79191
- R2: 0.90826

#### GRU Metrics
The GRU was trained on rolling windows taken from the same training data, and tested on the same segmented data set, process into rolling windows. the model performed as follows:
- RMSE: 3.01269
- MAE: 2.57142
- MedAE: 0.51145
- R2: 0.96596

#### MLP Metrics
- RMSE: 10.81831
- MAE: 8.26400
- MedAE: 7.04856
- R2: 0.92091

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

## Analysis

#### Performance on Test Dataset
Below are tables comparing the predictions of the GRU models with a test data sample, with the $\Delta$-values representing the difference between the model prediction and the true price. Since the GRU was trained and tested on rolling windows, the metrics below are taken **at random** from only **matching rows** across the test data (non-windowed for MLP, XGB and windowed for GRU). **This may skew analyses for GRU performance against the other models**.

#### Absolute Error

| True Price ($) | XBoost ($) | GRU ($)   | MLP ($)   | $\Delta$ XGBoost (Abs)  | $\Delta$ GRU (Abs) | $\Delta$ MLP (Abs) |
|----------------|------------|-----------|-----------|-------------------------|--------------------|--------------------|
| 122.93         | 121.92339  | 119.49114 | 122.65782 | 1.00661                 | 3.43887            | 0.27218            |
| 137.23         | 141.28304  | 133.08649 | 140.21288 | 4.05304                 | 4.14351            | 2.98288            |
| 166.73         | 162.74054  | 163.80864 | 153.64980 | 3.98946                 | 2.92136            | 13.08020           |
| 146.23         | 126.39964  | 136.40060 | 141.14595 | 19.83036                | 9.82939            | 5.08405            |
| 209.47         | 167.93335  | 132.48540 | 163.00860 | 41.53665                | 76.98460           | 46.46139           |
| 225.57         | 148.08543  | 134.25928 | 218.50418 | 77.48457                | 91.31073           | 7.06582            |


Here are the percentage errors represented similarly:

#### Percentage Error
| True Price ($) | XBoost ($) | GRU ($)   | MLP ($)   | $\Delta$ XGBoost (%) | $\Delta$ GRU (%) | $\Delta$ MLP (%) |
|----------------|------------|-----------|-----------|----------------------|------------------|------------------|
| 122.93         | 121.92339  | 119.49114 | 122.65782 | 0.82%                | 2.80%            | 0.22%            |
| 137.23         | 141.28304  | 133.08649 | 140.21288 | 2.95%                | 3.02%            | 2.17%            |
| 166.73         | 162.74054  | 163.80864 | 153.64980 | 2.39%                | 1.75%            | 7.85%            |
| 146.23         | 126.39964  | 136.40060 | 141.14595 | 13.56%               | 6.72%            | 3.48%            |
| 209.47         | 167.93335  | 132.48540 | 163.00860 | 19.83%               | 36.75%           | 22.18%           |
| 225.57         | 148.08543  | 134.25928 | 218.50418 | 34.35%               | 40.48%           | 3.13%            |

Here are the average percentage errors for each model **from this test sample**:

| Average $\Delta$ XGBoost (%) | Average $\Delta$ GRU (%) | Average $\Delta$ MLP (%)|
|------------------------------|--------------------------|-------------------------|
| 12.317                       | 15.253                   | **6.505**                   |

#### Overall Average Error Metrics

And here are the **overall** approximate average percentage errors for each model **from only the matching rows of the test data**:

| Average $\Delta$ XGBoost (%) | Average $\Delta$ GRU (%) | Average $\Delta$ MLP (%)|
|------------------------------|--------------------------|-------------------------|
| 15.56                        | 28.23                    | **13.11**                   |

### Performance on Undersampled/Low-Mid-Value Contracts
#### Addressing Test Data Imbalance
The metrics above show the more complex GRU model's underperformance in relation to the MLP and XGB models.

However, analyzing the GRU's metrics and test data show that the GRU **hugely outperforms** the MLP and XGB in pricing **mid-to-lower-value contracts (value < 140)** and that the dataset of matching rows from which the above contracts were taken is **skewed such that only ~2% of the contracts are of value < 140**. Sampling this 2% of the test data, we can see that the **GRU outperforms the other two models** in almost all cases.

| True Price ($) | XBoost ($) | GRU ($)   | MLP ($)   | $\Delta$ XGBoost (%) | $\Delta$ GRU (%) | $\Delta$ MLP (%) |
|----------------|------------|-----------|-----------|----------------------|------------------|------------------|
| 62.82          | 61.43354   | 59.48655  | 61.65081  | 2.21%                | 5.31%            | 1.86%            |
| 126.52         | 124.18036  | 127.01633 | 137.10991 | 1.85%                | 0.39%            | 8.37%            |
| 62.68          | 70.34517   | 63.12310  | 80.28888  | 12.23%               | 0.71%            | 28.09%           |
| 127.02         | 105.78838  | 128.68573 | 132.18924 | 16.72%               | 1.31%            | 4.07%            |
| 126.73         | 100.01202  | 126.66743 | 128.86774 | 21.08%               | 0.05%            | 1.69%            |
| 137.97         | 115.07793  | 133.05862 | 136.33330 | 16.59%               | 3.56%            | 1.19%            |

Here are the average percentage errors for each model **from the undersampled rows**:

| Average $\Delta$ XGBoost (%) | Average $\Delta$ GRU (%) | Average $\Delta$ MLP (%) |
|------------------------------|--------------------------|--------------------------|
| 7.82                         | **2.40**                     | 6.10                     |


## Overall Performance Evaluation

The performance metrics above highlight each model's characteristics influence options price prediction accuracy.

### Best Performance on Test Sample: MLP
Overall, the MLP (fully-connected neural network) had the lowest average error across the test sample, benefiting from its ability to model the complex nonlinear relationships present in options data, performing particularly well on mid-priced contracts. However, the model struggles with extreme values due to data imbalance. The flexibility of the MLP along with its respectable performance distinguishes as the best model option out of the three models compared we have.

### Best Overall Performance: GRU
In my analysis, I believe that due to the data imbalance in the test data (addressed in the above section), the GRU's performance was misrepresented in the test sample. Also, with the skewness of the training dataset (right skewed, many more training examples for lower prices contracts), I feel that the GRU's strong performance in the mid-low priced options puts it above the other two models, with the unbalanced data being its limiting factor.

### Takeaways
- **XGBoost remains competitive**, offering strong baseline results, however struggling on outliers and extreme price values.
- **Model selection should be data-driven**, Rather than defaulting to more sophisticated models, it’s important to validate which architecture aligns best with the underlying data and task.
- **Proper feature engineering and preprocessing may enable simpler models to outperform more sophisticated architectures when data structure favors them.**
- Large errors for all models on high-priced options highlight the need for **specialized treatment or feature engineering for outliers/extremes** in (unbalanced) financial datasets.


## Limitations & Future Work

### Limitations
1. Model underestimates ultra-expensive, illiquid contracts due to data imbalance
2. Only considers vanilla options; no spreads/multileg.

### Next Steps
- Take steps to improve GRU performance through over/undersampling training data for training
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
