# Options Trading Analysis

## Overview

As a comprehensive platform designed for advanced financial modeling, options pricing, and portfolio risk analysis, this project integrates quantitative models such as Black-Scholes, Monte Carlo Simulation, and Binomial Trees to evaluate derivatives pricing and optimize portfolio strategies. Built using Streamlit, the platform also provides tools for pricing and visualizing metrics to help traders and investors make data-driven analysis tailored for dynamic trading strategies.

Project Demo: [options-trading-analysis.streamlit.app](https://options-trading-analysis.streamlit.app/)

---

## Features

- **Option Pricing Models**:

  - Black-Scholes Model (closed-form solution for European options).
  - Monte Carlo Simulation (flexible for complex payoffs).
  - Binomial Tree Model (discrete time-step approach).

- **P&L Heatmap Analysis**:

  - Visualizes profit and loss across spot price and volatility scenarios.
  - Includes 2D heatmaps and 3D interactive visualizations.

- **Risk Analysis and Portfolio Optimization**:
  - Simulates portfolio returns for selected strategies.
  - Provides risk metrics such as VaR (Value at Risk) and CVaR (Conditional Value at Risk).
  - Implements mean-variance optimization to maximize the Sharpe Ratio.

---

## Technologies

- **Python Libraries**:

  - `numpy`, `pandas` for data manipulation.
  - `scipy` for statistical and optimization functions.
  - `matplotlib`, `seaborn` for 2D visualizations.
  - `plotly` for interactive 3D visualizations.

- **Streamlit**: Interactive web-based UI for financial modeling and analysis.

---

## Future Enhancements

- Support for American option pricing.
- Additional strategy types (e.g., butterfly spreads, calendar spreads).
- Incorporation of real-time market data for dynamic analysis.
- Integration with APIs for trading and portfolio management.

---

## License

Copyright © 2024 [Ryan Sun] (https://github.com/Ryii).<br />
This project is [MIT](https://github.com/Ryii/Options-Trading-Analysis/blob/main/LICENSE.md) licensed.
