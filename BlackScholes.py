import numpy as np
from scipy.stats import norm

class BlackScholes:
    def __init__(self, time_to_maturity, strike, current_price, volatility, interest_rate):
        self.time_to_maturity = time_to_maturity
        self.strike = strike
        self.current_price = current_price
        self.volatility = volatility
        self.interest_rate = interest_rate

    def d1_d2(self):
        d1 = (np.log(self.current_price / self.strike) +
              (self.interest_rate + 0.5 * self.volatility**2) * self.time_to_maturity) / (self.volatility * np.sqrt(self.time_to_maturity))
        d2 = d1 - self.volatility * np.sqrt(self.time_to_maturity)
        return d1, d2

    def calculate_prices(self):
        d1, d2 = self.d1_d2()
        call_price = self.current_price * norm.cdf(d1) - self.strike * np.exp(-self.interest_rate * self.time_to_maturity) * norm.cdf(d2)
        put_price = self.strike * np.exp(-self.interest_rate * self.time_to_maturity) * norm.cdf(-d2) - self.current_price * norm.cdf(-d1)
        return call_price, put_price

    def greeks(self):
        d1, d2 = self.d1_d2()
        delta_call = norm.cdf(d1)
        delta_put = delta_call - 1
        gamma = norm.pdf(d1) / (self.current_price * self.volatility * np.sqrt(self.time_to_maturity))
        vega = (self.current_price * norm.pdf(d1) * np.sqrt(self.time_to_maturity)) / 100
        theta_call = (-self.current_price * norm.pdf(d1) * self.volatility / (2 * np.sqrt(self.time_to_maturity)) -
                      self.interest_rate * self.strike * np.exp(-self.interest_rate * self.time_to_maturity) * norm.cdf(d2)) / 365
        theta_put = (-self.current_price * norm.pdf(d1) * self.volatility / (2 * np.sqrt(self.time_to_maturity)) +
                     self.interest_rate * self.strike * np.exp(-self.interest_rate * self.time_to_maturity) * norm.cdf(-d2)) / 365
        rho_call = (self.strike * self.time_to_maturity * np.exp(-self.interest_rate * self.time_to_maturity) * norm.cdf(d2)) / 100
        rho_put = (-self.strike * self.time_to_maturity * np.exp(-self.interest_rate * self.time_to_maturity) * norm.cdf(-d2)) / 100

        return {
            "Call Delta": delta_call,
            "Put Delta": delta_put,
            "Gamma": gamma,
            "Vega": vega,
            "Call Theta": theta_call,
            "Put Theta": theta_put,
            "Call Rho": rho_call,
            "Put Rho": rho_put
        }