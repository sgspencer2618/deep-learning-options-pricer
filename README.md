<a target="_blank" href="https://colab.research.google.com/github/sgspencer2618/xgboost-options-pricing/blob/main/options_pricer.ipynb">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>

# XGBoost Options Pricing Model for $AAPL stock and Generation of Buy/Sell Signals Based on Model Predictions

This project is an implementation of an option pricer using gradient-boosted trees (XGBoost), that predicts american options last prices using historical market options data. It cleans and merges features such as greeks (Delta, Gamma, Vega, Theta, Rho), and moneyness measures, then trains a model to forecast option prices for both calls and puts. The code also demonstrates how to:

- Perform time-series–aware splitting to avoid data leakage.
- Tune hyperparameters (e.g., max_depth, learning_rate) with Bayesian optimization.
- Generate and overlay Buy/Sell signals on a price chart to visualize model-driven trading decisions.

In future versions, I plan to add functionality for other stocks, as well as provide additional insights to users - potentially implementing a UI and integrating stock price dashboards for real-time functionality and useability.

## How To Use

The project was implemented in Google Colab, and the link is available in the 'Code' section along with the source code of the project. All libraries and dependencies should install when the first cell is run. Data files are contained in this Github repostiory. To see the model's findings as displayed in the Github code, do the following:

1. Click the link to view the program in Google Colab's environment
2. Download the files titled *"aapl_2016_2020.csv"*, *"aapl_2021_2023.csv"*, *"aapl_stock_prices.csv"*, and *"options_model.json"*. The first 3 files are data files (found at https://www.kaggle.com/datasets/kylegraupe/aapl-options-data-2016-2020/data and https://www.kaggle.com/datasets/muhammadbilalhaneef/-apple-stock-prices-from-1981-to-2023) if you would like to retrain the model, and the last (json) file (zipped in the repository) contains the data for the model as trained and displayed in the code in this repo.
3. On the left sidebar, upload the files into a folder in Colab's file directory, and name the new folder **"options_data"**
4. In the toolbar, go to **Runtime > Run all**

The pretrained model has error metrics as follows:
Mean Absolute Error: 4.4466
Mean Squared Error: 206.5355
RMSE: 14.3713
R^2: 0.9006

The $AAPL stock price candle chart with overlaid model predictions should look like this:

<img width="651" alt="Overlaid Candle Chart" src="https://github.com/user-attachments/assets/20e3a587-b9b7-4a10-93c8-f3c04a69c671" />

The green upward-pointing arrows on each candle are "Buy" signals, and the red downward-pointing arrows are "Sell" signals as determined by the model's findings, with a price change threshold of +/- 1.2%.

## Credits
This project was inspired by the project outlined in: https://www.tidy-finance.org/python/option-pricing-via-machine-learning.html#:~:text=All%20ML%20methods%20seem%20to,in%2Dthe%2Dmoney%20options.

The databases can be found on Kaggle.com at:
- https://www.kaggle.com/datasets/kylegraupe/aapl-options-data-2016-2020/data (options chains)
- https://www.kaggle.com/datasets/muhammadbilalhaneef/-apple-stock-prices-from-1981-to-2023 ($AAPL stock price data)

## License
MIT License Copyright (c) 2024 Sean Gywnn Spencer
