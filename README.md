# XGBoost Options Pricing Model for $AAPL stock and Generation of Buy/Sell Signals Based on Model Predictions

This project is an implementation of an option pricer using gradient-boosted trees (XGBoost), that predicts american options last prices using historical market options data. It cleans and merges features such as greeks (Delta, Gamma, Vega, Theta, Rho), and moneyness measures, then trains a model to forecast option prices for both calls and puts. The code also demonstrates how to:

- Perform time-series–aware splitting to avoid data leakage.
- Tune hyperparameters (e.g., max_depth, learning_rate) with Bayesian optimization.
- Generate and overlay Buy/Sell signals on a price chart to visualize model-driven trading decisions.

In future projects, I would like to add functionality for other stocks, as well as provide additional insights to users - potentially implementing a UI and integrating stock price dashboards for real-time functionality and useability.

## How To Use

The project was implemented in Google Colab, and the link is available in the 'Code' section along with the source code of the project. All libraries and dependencies should install when the first cell is run. Data files are contained in this Github repostiory. To see the model's findings as displayed in the Github code, do the following:

1. Click the link to view the program in Google Colab's environment
2. Download the files titled *"aapl_2016_2020.csv"*, *"aapl_2021_2023.csv"*, *"aapl_stock_prices.csv"*, and *"options_model.json"*. The first 3 files are data files (found at https://www.kaggle.com/datasets/kylegraupe/aapl-options-data-2016-2020/data) if you would like to retrain the model, and the last (json) file (zipped in the repository) contains the data for the model as trained and displayed in the code in this repo.
3. On the left sidebar, upload the files into a folder in Colab's file directory, and name the new folder **"options_data"**
4. In the toolbar, go to **Runtime > Run all**
