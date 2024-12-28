# XGBoost Options Pricing Model for $AAPL stock and Generation of Buy/Sell Signals Based on Model Predictions

This project is an implementation of an option pricer using gradient-boosted trees (XGBoost), that predicts american options last prices using historical market options data. It cleans and merges features such as greeks (Delta, Gamma, Vega, Theta, Rho), and moneyness measures, then trains a model to forecast option prices for both calls and puts. The code also demonstrates how to:

- Perform time-series–aware splitting to avoid data leakage.
- Tune hyperparameters (e.g., max_depth, learning_rate) with Bayesian optimization.
- Generate and overlay Buy/Sell signals on a price chart to visualize model-driven trading decisions.

In future projects, I would like to add functionality for other stocks, as well as provide additional insights to users - potentially implementing a UI and integrating stock price dashboards for real-time functionality and useability.
