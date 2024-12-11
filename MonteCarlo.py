import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def monte_carlo_pricer(current_price, strike, time_to_maturity, volatility, interest_rate, option_type='Call', n_sims=10000):
    drift = (interest_rate - 0.5 * volatility**2) * time_to_maturity
    diffusion = volatility * np.sqrt(time_to_maturity)
    sim_end_prices = current_price * np.exp(drift + diffusion * np.random.randn(n_sims))
    if option_type == 'Call':
        payoffs = np.maximum(sim_end_prices - strike, 0)
    else:
        payoffs = np.maximum(strike - sim_end_prices, 0)
    price = np.mean(payoffs) * np.exp(-interest_rate * time_to_maturity)
    return price


def plot_monte_carlo_distribution(current_price, strike, time_to_maturity, volatility, interest_rate, n_sims=10000, option_type='Call'):
    drift = (interest_rate - 0.5 * volatility**2) * time_to_maturity
    diffusion = volatility * np.sqrt(time_to_maturity)
    sim_end_prices = current_price * np.exp(drift + diffusion * np.random.randn(n_sims))

    if option_type == 'Call':
        payoffs = np.maximum(sim_end_prices - strike, 0)
    else:
        payoffs = np.maximum(strike - sim_end_prices, 0)

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.histplot(sim_end_prices, bins=50, kde=True, color='blue', label='Simulated End Prices', ax=ax)
    ax.set_xlabel('Simulated End Price', fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)

    sorted_prices = np.sort(sim_end_prices)
    if option_type == 'Call':
        payoff_curve = np.maximum(sorted_prices - strike, 0)
    else:
        payoff_curve = np.maximum(strike - sorted_prices, 0)

    ax.plot(sorted_prices, payoff_curve, color='red', linewidth=2, label='Payoff Curve')
    ax.legend()
    plt.tight_layout()
    return fig