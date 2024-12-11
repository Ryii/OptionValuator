import numpy as np
import matplotlib.pyplot as plt

def binomial_tree_pricer(current_price, strike, time_to_maturity, volatility, interest_rate, option_type='Call', steps=100):
    dt = time_to_maturity / steps
    u = np.exp(volatility * np.sqrt(dt))
    d = 1/u
    p = (np.exp(interest_rate*dt)-d)/(u-d)
    prices = current_price*(u**(np.arange(steps,-1,-1)))*(d**(np.arange(0,steps+1,1)))
    if option_type == 'Call':
        values = np.maximum(prices - strike,0)
    else:
        values = np.maximum(strike - prices,0)

    for i in range(steps-1,-1,-1):
        values = np.exp(-interest_rate*dt)*(p*values[0:i+1]+(1-p)*values[1:i+2])
    return values[0]

def sharpe_ratio_objective(weights, returns):
    port_returns = returns @ weights
    mean_ret = np.mean(port_returns)
    vol = np.std(port_returns)
    if vol < 1e-6:
        return 1e6
    return -mean_ret/vol


def plot_binomial_tree(current_price, strike, time_to_maturity, volatility, interest_rate, steps, option_type='Call'):
    dt = time_to_maturity / steps
    u = np.exp(volatility * np.sqrt(dt))
    d = 1/u

    fig, ax = plt.subplots(figsize=(8, 6))
    prices = np.zeros((steps+1, steps+1))
    for i in range(steps+1):
        for j in range(i+1):
            prices[j,i] = current_price * (u**(i-j))*(d**j)

    for i in range(steps+1):
        for j in range(i+1):
            ax.plot(i, prices[j,i], 'o', color='black')
            ax.text(i, prices[j,i], f"{prices[j,i]:.2f}", fontsize=8, ha='center', va='bottom')

    # ax.set_title("Binomial Tree (Prices)", fontsize=16, fontweight='bold')
    ax.set_xlabel("Step", fontsize=12)
    ax.set_ylabel("Underlying Price", fontsize=12)
    plt.tight_layout()
    return fig

def plot_binomial_convergence(current_price, strike, time_to_maturity, volatility, interest_rate, option_type='Call'):
    steps_range = range(10, 210, 20)
    prices = []
    for s in steps_range:
        val = binomial_tree_pricer(current_price, strike, time_to_maturity, volatility, interest_rate, option_type, steps=s)
        prices.append(val)

    fig, ax = plt.subplots(figsize=(10,7))
    ax.plot(steps_range, prices, marker='o')
    # ax.set_title("Convergence of Binomial Tree Pricing", fontsize=16, fontweight='bold')
    ax.set_xlabel("Number of Steps", fontsize=12)
    ax.set_ylabel("Option Price", fontsize=12)
    return fig
