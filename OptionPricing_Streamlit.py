import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
from scipy.optimize import minimize
from BlackScholes import BlackScholes
from MonteCarlo import monte_carlo_pricer, plot_monte_carlo_distribution
from BinomialTree import binomial_tree_pricer, sharpe_ratio_objective, plot_binomial_convergence, plot_binomial_tree

st.set_page_config(
    page_title="Advanced Option Pricing, Strategies & Risk Analysis",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --------------------------------------
# Utility Classes and Functions
# --------------------------------------

def price_option_leg(pricing_method, option_type, current_price, strike, time_to_maturity, volatility, interest_rate, n_sims=10000, steps=100):
    if pricing_method == "Black-Scholes":
        bs = BlackScholes(time_to_maturity, strike, current_price, volatility, interest_rate)
        call_price, put_price = bs.calculate_prices()
        return call_price if option_type == 'Call' else put_price
    elif pricing_method == "Monte Carlo":
        return monte_carlo_pricer(current_price, strike, time_to_maturity, volatility, interest_rate, option_type=option_type, n_sims=n_sims)
    elif pricing_method == "Binomial Tree":
        return binomial_tree_pricer(current_price, strike, time_to_maturity, volatility, interest_rate, option_type=option_type, steps=steps)

def plot_pnl_heatmap(strike, time_to_maturity, interest_rate, spot_range, vol_range, current_price, volatility,
                     pricing_method, option_type_selection, n_sims=None, steps=None, use_theoretical=True, custom_price=None):
    pnl_matrix = np.zeros((len(vol_range), len(spot_range)))

    for i, vol_scenario in enumerate(vol_range):
        for j, spot_scenario in enumerate(spot_range):
            option_price = price_option_leg(
                pricing_method=pricing_method,
                option_type=option_type_selection,
                current_price=spot_scenario,
                strike=strike,
                time_to_maturity=time_to_maturity,
                volatility=vol_scenario,
                interest_rate=interest_rate,
                n_sims=n_sims,
                steps=steps
            ) if use_theoretical else custom_price

            if option_type_selection == "Call":
                pnl_matrix[i, j] = max(spot_scenario - strike, 0) - option_price
            else:
                pnl_matrix[i, j] = max(strike - spot_scenario, 0) - option_price

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        pnl_matrix,
        xticklabels=np.round(spot_range, 2),
        yticklabels=np.round(vol_range, 2),
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        ax=ax,
    )
    ax.set_xlabel("Spot Price at Maturity", fontsize=12)
    ax.set_ylabel("Volatility at Inception", fontsize=12)
    plt.tight_layout()

    return fig, pnl_matrix


# --------------------------------------
# Sidebar Inputs
# --------------------------------------
page = st.selectbox("Page", ["Option Pricing", "P&L Heatmap Analysis", "Risk Analysis & Portfolio Optimization"])

with st.sidebar:
    st.title("Advanced Options & Strategies")
    st.subheader("Pricing Method")
    pricing_method = st.radio("Choose Pricing Method", ["Black-Scholes", "Monte Carlo", "Binomial Tree"], label_visibility="collapsed")
    option_type_selection = st.radio("Option Type:", ["Call", "Put"])

    if pricing_method == "Monte Carlo":
        n_sims = st.number_input("Number of Simulations", value=10000, step=1000)
        steps = None
    elif pricing_method == "Binomial Tree":
        steps = st.number_input("Number of Steps in Binomial Tree", value=100, step=50)
        n_sims = None
    else:
        n_sims = None
        steps = None

    if page == "P&L Heatmap Analysis":
        st.markdown("---")
        st.subheader("Heatmap Parameters")

        if pricing_method == "Black-Scholes":
            price_choice = st.radio("Option Price Source:", ["Theoretical Black-Scholes Price", "Custom Price"])
        elif pricing_method == "Monte Carlo":
            price_choice = st.radio("Option Price Source:", ["Theoretical Monte Carlo Price", "Custom Price"])
        else:
            price_choice = st.radio("Option Price Source:", ["Theoretical Binomial Tree Price", "Custom Price"])

        if price_choice == "Custom Price":
            custom_price = st.number_input("Enter Custom Option Price", min_value=0.0, value=150.0)
            use_theoretical = False
        else:
            st.write("")
            custom_price = None
            use_theoretical = True

        min_percent = st.slider("Min Spot Price (%):", min_value=0.01, max_value=2.0, value=0.8, step=0.01)
        max_percent = st.slider("Max Spot Price (%):", min_value=0.01, max_value=2.0, value=1.2, step=0.01)
        vol_min = st.slider('Min Volatility:', min_value=0.01, max_value=1.0, value=0.1, step=0.01)
        vol_max = st.slider('Max Volatility:', min_value=0.01, max_value=1.0, value=0.5, step=0.01)

    if page == "Risk Analysis & Portfolio Optimization":
        st.markdown("---")
        st.subheader("Portfolio Construction (Risk & Optimization)")
        st.write("Build a small portfolio of strategies to analyze risk & optimize:")
        available_strats = ["Long Call", "Long Put", "Long Straddle"]
        portfolio_strategies = [st.selectbox(f"Strategy {i+1}", ["None"]+available_strats) for i in range(3)]
        portfolio_initial_weights = [st.number_input(f"Initial Weight {i+1}", value=0.0, step=0.1) for i in range(3)]
        nonnegative = st.checkbox("Non-negative Weights", value=True)

    st.markdown("---")
    st.subheader("Input Parameters")
    current_price = st.number_input("Underlying Asset Price", value=150.0)
    strike = st.number_input("Strike Price", value=150.0)
    time_to_maturity = st.number_input("Time to Maturity (Years)", value=1.0)
    volatility = st.number_input("Volatility (σ)", value=0.3)
    interest_rate = st.number_input("Risk-Free Interest Rate", value=0.05)

    if page == "P&L Heatmap Analysis":
        spot_range = np.linspace(min_percent * current_price, max_percent * current_price)
        vol_range = np.linspace(vol_min, vol_max, 10)


# --------------------------------------
# CSS Styles
# --------------------------------------

box_color = "#ffcccc" if option_type_selection == "Put" else "#90ee99"
st.markdown(
    f"""
    <style>
        .custom-container {{
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 20px;
        }}
        .custom-table {{
            display: flex;
            flex-direction: column;
            justify-content: center;
            border-collapse: collapse;
            margin: 0;
            padding: 0;
        }}
        .custom-table th, .custom-table td {{
            border: 1px solid #ddd;
            padding: 8px;
            text-align: center;
        }}
        .custom-arrow {{
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            font-size: 1.2rem;
            gap: 5px;
            min-width: 5rem;
        }}
        .custom-box {{
            display: flex;
            align-items: center;
            justify-content: center;
            background-color: {box_color};
            color: black;
            border-radius: 10px;
            padding: 20px;
            min-height: 3.1rem;
        }}
        .custom-value {{
            font-size: 1.5rem;
            font-weight: bold;
        }}
        .centered-metrics {{
            display: flex;
            justify-content: center;
            align-items: center;
            gap: 20px;
            max-width: 800px;
            margin: 0 auto;
        }}
        .metric-box {{
            text-align: center;
            flex: 1;
            padding: 10px;
            background-color: #f5f5f5;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
        }}
        .metric-title {{
            font-size: 1rem;
            font-weight: bold;
            margin-bottom: 5px;
        }}
        .metric-value {{
            font-weight: normal;
            font-size: 2rem;
            color: #333;
        }}
        hr {{
            margin: 6px 0px 15px 0px;
        }}
    </style>
    """,
    unsafe_allow_html=True
)


# --------------------------------------
# Main Pages
# --------------------------------------


if page == "Option Pricing":
    st.title("Option Pricing")
    
    if pricing_method == "Black-Scholes":
        st.markdown("#### Black-Scholes")
        st.write("""
        The Black-Scholes model provides a closed-form solution for European-style options
        (assuming no dividends, constant volatility, and risk-free interest rates). The estimated option 
        price can be calculated directly from the underlying asset price, strike price, time to maturity, 
        volatility, and risk-free rate.
        """)
        if option_type_selection == "Call":
            st.latex(r"""
            \Large C = N(d_1) S - K e^{-r(T-t)} N(d_2)
            """)
        else:
            st.latex(r"""
            \Large P = K e^{-r(T-t)} N(-d_2) - S N(-d_1)
            """)

        st.markdown("###### With:")
        st.latex(r"""
        d_1 = \frac{\ln\left(\frac{S}{K}\right) + \left(r + \frac{\sigma^2}{2}\right)(T-t)}{\sigma \sqrt{T-t}}, \quad d_2 = d_1 - \sigma \sqrt{T-t}
        """)
        st.write("")

        bs = BlackScholes(time_to_maturity, strike, current_price, volatility, interest_rate)
        call_price, put_price = bs.calculate_prices()
        option_price = call_price if option_type_selection == 'Call' else put_price
        
        st.markdown(
            f"""
            <div class="custom-container">
                <div class="custom-table">
                    <table style="margin: 0">
                        <thead>
                            <tr>
                                <th>Current Stock Price (S)</th>
                                <th>Strike Price (K)</th>
                                <th>Time to Maturity (T-t)</th>
                                <th>Volatility (σ)</th>
                                <th>Risk-Free Rate (r)</th>
                            </tr>
                        </thead>
                        <tbody>
                            <tr>
                                <td>{current_price:.2f}</td>
                                <td>{strike:.2f}</td>
                                <td>{time_to_maturity:.2f}</td>
                                <td>{volatility:.2f}</td>
                                <td>{interest_rate:.2f}</td>
                            </tr>
                        </tbody>
                    </table>
                </div>
                <div class="custom-arrow">
                    <div>{option_type_selection} Value</div>
                    <div>➜</div>
                </div>
                <div class="custom-box">
                    <div class="custom-value">
                        ${option_price:.2f}
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

        greeks = bs.greeks()
        st.write("")
        st.write("")
        st.markdown("#### Options Greeks")
        
        metrics_html = """
        <div class="centered-metrics">
            <div class="metric-box">
                <div class="metric-title">Delta (Δ)</div>
                <div class="metric-value">{:.3f}</div>
            </div>
            <div class="metric-box">
                <div class="metric-title">Gamma (Γ)</div>
                <div class="metric-value">{:.3f}</div>
            </div>
            <div class="metric-box">
                <div class="metric-title">Theta (Θ)</div>
                <div class="metric-value">{:.3f}</div>
            </div>
            <div class="metric-box">
                <div class="metric-title">Vega (v)</div>
                <div class="metric-value">{:.3f}</div>
            </div>
            <div class="metric-box">
                <div class="metric-title">Rho (ρ)</div>
                <div class="metric-value">{:.3f}</div>
            </div>
        </div>
        """.format(
            greeks["Call Delta"] if option_type_selection=='Call' else greeks["Put Delta"],
            greeks["Gamma"],
            greeks["Call Theta"] if option_type_selection=='Call' else greeks["Put Theta"],
            greeks["Vega"],
            greeks["Call Rho"] if option_type_selection=='Call' else greeks["Put Rho"],
        )
        st.markdown(metrics_html, unsafe_allow_html=True)
        st.write("")

        spot_range_for_greeks = np.linspace(current_price * 0.5, current_price * 1.5, 50)
        deltas_list, gammas_list, thetas_list, vegas_list, rhos_list, prices_list = [], [], [], [], [], []

        for s_spot in spot_range_for_greeks:
            bs_temp = BlackScholes(time_to_maturity, strike, s_spot, volatility, interest_rate)
            gr = bs_temp.greeks()

            if option_type_selection == 'Call':
                deltas_list.append(gr["Call Delta"])
                thetas_list.append(gr["Call Theta"])
                rhos_list.append(gr["Call Rho"])
            else:
                deltas_list.append(gr["Put Delta"])
                thetas_list.append(gr["Put Theta"])
                rhos_list.append(gr["Put Rho"])

            gammas_list.append(gr["Gamma"])
            vegas_list.append(gr["Vega"])
            prices_list.append(s_spot)

        sns.set_style("whitegrid")

        fig_delta, ax_delta = plt.subplots()
        sns.lineplot(x=prices_list, y=deltas_list, ax=ax_delta)
        ax_delta.set_xlabel("Underlying Asset Price")
        ax_delta.set_ylabel("Delta")
        ax_delta.set_title("Delta vs Underlying Price")
        fig_delta.tight_layout()

        fig_gamma, ax_gamma = plt.subplots()
        sns.lineplot(x=prices_list, y=gammas_list, ax=ax_gamma)
        ax_gamma.set_xlabel("Underlying Asset Price")
        ax_gamma.set_ylabel("Gamma")
        ax_gamma.set_title("Gamma vs Underlying Price")
        fig_gamma.tight_layout()

        fig_theta, ax_theta = plt.subplots()
        sns.lineplot(x=prices_list, y=thetas_list, ax=ax_theta)
        ax_theta.set_xlabel("Underlying Asset Price")
        ax_theta.set_ylabel("Theta")
        ax_theta.set_title("Theta vs Underlying Price")
        fig_theta.tight_layout()

        fig_vega, ax_vega = plt.subplots()
        sns.lineplot(x=prices_list, y=vegas_list, ax=ax_vega)
        ax_vega.set_xlabel("Underlying Asset Price")
        ax_vega.set_ylabel("Vega")
        ax_vega.set_title("Vega vs Underlying Price")
        fig_vega.tight_layout()

        fig_rho, ax_rho = plt.subplots()
        sns.lineplot(x=prices_list, y=rhos_list, ax=ax_rho)
        ax_rho.set_xlabel("Underlying Asset Price")
        ax_rho.set_ylabel("Rho")
        ax_rho.set_title("Rho vs Underlying Price")
        fig_rho.tight_layout()

        row1_col1, row1_col2 = st.columns(2)
        with row1_col1:
            st.pyplot(fig_delta)
        with row1_col2:
            st.pyplot(fig_gamma)

        row2_col1, row2_col2 = st.columns(2)
        with row2_col1:
            st.pyplot(fig_theta)
        with row2_col2:
            st.pyplot(fig_vega)

        row3_col1, row3_col2, row3_col3 = st.columns([1,2,1])
        with row3_col2:
            st.pyplot(fig_rho)


    elif pricing_method == "Monte Carlo":
        st.markdown("#### Monte Carlo Simulation Method")
        st.write("""
        Monte Carlo methods price options by simulating the underlying asset price 
        many times and taking the averaged discounted payoffs. Unlike Black-Scholes, this approach does not 
        rely on closed-form solutions and can better handle a wide variety of payoff structures.
        """)

        if option_type_selection == "Call":
            st.latex(r"""
            \Large C = e^{-r(T-t)} \frac{1}{N}\sum_{i=1}^{N}\max(S_{T}^{(i)} - K,\,0)
            """)
        else:
            st.latex(r"""
            \Large P = e^{-rT} \frac{1}{N}\sum_{i=1}^{N}\max(K - S_{T}^{(i)},\,0)
            """)

        st.markdown("###### With:")
        st.latex(r"""
        \large S_T^{(i)} = S e^{(r-\frac{\sigma^2}{2})(T-t) + \sigma\sqrt{T-t}Z_i}, \quad Z_i \approx N(0,1)
        """)
        st.write("")

        option_price = monte_carlo_pricer(current_price, strike, time_to_maturity, volatility, interest_rate,
                                          option_type=option_type_selection, n_sims=n_sims)

        st.markdown(
            f"""
            <div class="custom-container">
                <div class="custom-table">
                    <table style="margin: 0">
                        <thead>
                            <tr>
                                <th>Current Stock Price (S)</th>
                                <th>Strike Price (K)</th>
                                <th>Time to Maturity (T-t)</th>
                                <th>Volatility (σ)</th>
                                <th>Risk-Free Rate (r)</th>
                                <th>Simulations (N)</th>
                            </tr>
                        </thead>
                        <tbody>
                            <tr>
                                <td>{current_price:.2f}</td>
                                <td>{strike:.2f}</td>
                                <td>{time_to_maturity:.2f}</td>
                                <td>{volatility:.2f}</td>
                                <td>{interest_rate:.2f}</td>
                                <td>{n_sims}</td>
                            </tr>
                        </tbody>
                    </table>
                </div>
                <div class="custom-arrow">
                    <div>{option_type_selection} Value</div>
                    <div>➜</div>
                </div>
                <div class="custom-box">
                    <div class="custom-value">
                        ${option_price:.2f}
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

        for _ in range(2):
            st.write("")
        st.markdown("#### Monte Carlo Distribution Visualization")
        st.write("")
        fig_mc = plot_monte_carlo_distribution(
            current_price, strike, time_to_maturity, volatility, interest_rate, n_sims=n_sims, option_type=option_type_selection
        )
        col1, col2, col3 = st.columns([1, 4, 1])
        with col2:
            st.pyplot(fig_mc)


    elif pricing_method == "Binomial Tree":
        st.markdown("#### Binomial Tree Model")
        st.write("""
        The Binomial Tree model prices options by discretizing the option's lifetime 
        into a price tree. At each node, the underlying price either moves 
        up or down. By working backward from the terminal payoffs, we can compute a fair 
        option price at the initial node, which is our expected fair option price.
        """)

        if option_type_selection == "Call":
            st.latex(r"""
            \Large V_0 = e^{-r\Delta t} \big[ p V_{u} + (1 - p) V_{d} \big]
            """)
        else:
            st.latex(r"""
            \Large V_0 = e^{-r\Delta t} \big[ p V_{u} + (1 - p) V_{d} \big]
            """)

        st.markdown("###### With:")

        if option_type_selection == "Call":
            st.latex(r"""
            u = e^{\sigma \sqrt{\Delta t}}, \quad d = e^{-\sigma \sqrt{\Delta t}} = \frac{1}{u}, \quad p = \frac{e^{r \Delta t} - d}{u - d}, \quad V_u = \textrm{max}(Su - K, 0), \quad V_d = \textrm{max}(Sd - K, 0)
            """)
        else:
            st.latex(r"""
            u = e^{\sigma \sqrt{\Delta t}}, \quad d = e^{-\sigma \sqrt{\Delta t}} = \frac{1}{u}, \quad p = \frac{e^{r \Delta t} - d}{u - d}, \quad V_u = \textrm{max}(K - Su, 0), \quad V_d = \textrm{max}(K - Sd, 0)
            """)
        st.write("")

        if steps is None:
            steps = 100
        option_price = binomial_tree_pricer(current_price, strike, time_to_maturity, volatility, interest_rate,
                                            option_type=option_type_selection, steps=steps)

        st.markdown(
            f"""
            <div class="custom-container">
                <div class="custom-table">
                    <table style="margin: 0">
                        <thead>
                            <tr>
                                <th>Current Stock Price (S)</th>
                                <th>Strike Price (K)</th>
                                <th>Time to Maturity (T-t)</th>
                                <th>Volatility (σ)</th>
                                <th>Risk-Free Rate (r)</th>
                                <th>Number of Steps</th>
                            </tr>
                        </thead>
                        <tbody>
                            <tr>
                                <td>{current_price:.2f}</td>
                                <td>{strike:.2f}</td>
                                <td>{time_to_maturity:.2f}</td>
                                <td>{volatility:.2f}</td>
                                <td>{interest_rate:.2f}</td>
                                <td>{steps}</td>
                            </tr>
                        </tbody>
                    </table>
                </div>
                <div class="custom-arrow">
                    <div>{option_type_selection} Value</div>
                    <div>➜</div>
                </div>
                <div class="custom-box">
                    <div class="custom-value">
                        ${option_price:.2f}
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

        for _ in range(2):
            st.write("")
        st.markdown("#### Binomial Tree Visualization")

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Binomial Tree Visualization (Max 6 Steps)**")
            fig_tree = plot_binomial_tree(
                current_price, strike, time_to_maturity, volatility, interest_rate, steps=min(6, steps), option_type=option_type_selection
            )
            st.pyplot(fig_tree)

        with col2:
            st.markdown("**Binomial Tree Convergence**")
            fig_convergence = plot_binomial_convergence(
                current_price, strike, time_to_maturity, volatility, interest_rate, option_type=option_type_selection
            )
            st.pyplot(fig_convergence)



elif page == "P&L Heatmap Analysis":
    st.title("P&L Heatmap Analysis")

    if pricing_method == "Monte Carlo" and n_sims is None:
        n_sims = 10000
    if pricing_method == "Binomial Tree" and steps is None:
        steps = 100

    spot_range = np.linspace(min_percent * current_price, max_percent * current_price, 10)
    vol_range = np.linspace(vol_min, vol_max, 10)

    st.write("The P&L (Profit and Loss) Heatmap provides a visual representation of how an option's profitability varies under different market conditions. It evaluates the option's performance across a range of spot prices at maturity and implied volatilities at inception.")

    col1, col2 = st.columns(2, gap="medium")

    with col1:
        st.subheader("2D Heatmap")
        st.markdown(f"**P&L for {option_type_selection} Option ({pricing_method})**")
        fig_pnl, pnl_matrix = plot_pnl_heatmap(
            strike=strike,
            time_to_maturity=time_to_maturity,
            interest_rate=interest_rate,
            spot_range=spot_range,
            vol_range=vol_range,
            current_price=current_price,
            volatility=volatility,
            pricing_method=pricing_method,
            option_type_selection=option_type_selection,
            n_sims=n_sims,
            steps=steps,
            use_theoretical=use_theoretical,
            custom_price=custom_price
        )
        st.pyplot(fig_pnl)

    with col2:
        st.subheader("3D Heatmap")
        st.markdown(f"**P&L Surface for {option_type_selection} Option ({pricing_method})**")
        X, Y = np.meshgrid(spot_range, vol_range)
        fig_3d = go.Figure(data=[go.Surface(z=pnl_matrix, x=X, y=Y, colorscale='RdBu', reversescale=True)])
        fig_3d.update_layout(
            scene=dict(
                xaxis_title='Spot Price',
                yaxis_title='Volatility',
                zaxis_title='P&L'
            ),
            autosize=True,
            margin=dict(l=0, r=0, b=0, t=0),
        )
        st.plotly_chart(fig_3d, use_container_width=True)

elif page == "Risk Analysis & Portfolio Optimization":
    st.title("Risk Analysis & Portfolio Optimization")
    st.write(f"""
        By selecting from a set of strategies and calculating key risk metrics, this page is designed to help traders and
        investors construct and refine a portfolio of options strategies by provideing tools to assess portfolio performance
        and optimize strategy allocation to achieve better returns relative to risk.
    """)

    chosen_strats = [(s, w) for s, w in zip(portfolio_strategies, portfolio_initial_weights) if s != "None"]

    if len(chosen_strats) == 0:
        st.warning("No strategies selected from sidebar.")
    else:
        st.subheader("Simulating Underlying & Strategy Payoffs")

        n_sims_risk = 10000
        drift = (interest_rate - 0.5 * volatility**2)*time_to_maturity
        diffusion = volatility * np.sqrt(time_to_maturity)
        end_prices = current_price * np.exp(drift + diffusion*np.random.randn(n_sims_risk))

        strat_payoffs = []
        strat_names = []
        initial_costs = []

        for s, w in chosen_strats:
            if s == "Long Call":
                c = price_option_leg(pricing_method, 'Call', current_price, strike, time_to_maturity, volatility, interest_rate, n_sims=10000, steps=100)
                payoffs = np.maximum(end_prices - strike,0) - c
                initial_cost = c
            elif s == "Long Put":
                p = price_option_leg(pricing_method, 'Put', current_price, strike, time_to_maturity, volatility, interest_rate, n_sims=10000, steps=100)
                payoffs = np.maximum(strike - end_prices,0) - p
                initial_cost = p
            elif s == "Long Straddle":
                c = price_option_leg(pricing_method, 'Call', current_price, strike, time_to_maturity, volatility, interest_rate, n_sims=10000, steps=100)
                p = price_option_leg(pricing_method, 'Put', current_price, strike, time_to_maturity, volatility, interest_rate, n_sims=10000, steps=100)
                payoffs = (np.maximum(end_prices - strike,0) + np.maximum(strike - end_prices,0))-(c+p)
                initial_cost = c+p

            returns = payoffs / initial_cost - 1.0
            strat_payoffs.append(returns)
            strat_names.append(s)
            initial_costs.append(initial_cost)

        returns_matrix = np.column_stack(strat_payoffs)

        init_w = np.array([w for _, w in chosen_strats])
        if init_w.sum() == 0:
            init_w = np.ones_like(init_w)/len(init_w)
        else:
            init_w /= init_w.sum()

        initial_port_returns = returns_matrix @ init_w
        alpha = 0.95
        var_threshold = np.percentile(initial_port_returns, (1-alpha)*100)
        cvar = initial_port_returns[initial_port_returns < var_threshold].mean() if np.any(initial_port_returns < var_threshold) else var_threshold

        st.markdown(f"**Initial Portfolio VaR (95%)**: {var_threshold:.2f}")
        st.markdown(f"**Initial Portfolio CVaR**: {cvar:.2f}")

        fig_dist, ax_dist = plt.subplots(figsize=(8,5))
        sns.histplot(initial_port_returns, kde=True, ax=ax_dist, color='blue')
        ax_dist.set_title("Distribution of Initial Portfolio Returns")
        ax_dist.set_xlabel("Return")
        plt.tight_layout()
        st.pyplot(fig_dist)

        st.subheader("Mean-Variance Optimization: Maximize Sharpe Ratio")

        def constraint_sum_weights(w):
            return np.sum(w) - 1.0

        cons = [{'type': 'eq', 'fun': constraint_sum_weights}]
        bounds = [(0.0, 1.0)]*returns_matrix.shape[1] if nonnegative else None

        res = minimize(sharpe_ratio_objective, init_w, args=(returns_matrix,), method='SLSQP', constraints=cons, bounds=bounds)
        if res.success and (not nonnegative or np.all(res.x >= -1e-9)):
            opt_w = res.x
            st.write("**Optimal Weights (Max Sharpe):**")
            if len(strat_names) == 1:
                st.write(f"{strat_names[0]}: {opt_w[0]:.2f}")
            elif len(strat_names) == 2:
                co1, col2, col3, col4, col5 = st.columns(5)
                with col2:
                    st.write(f"{strat_names[0]}: {opt_w[0]:.2f}")
                with col4:
                    st.write(f"{strat_names[1]}: {opt_w[1]:.2f}")
            else:
                co1, col2, col3, col4, col5, col6, col7 = st.columns(7)
                with col2:
                    st.write(f"{strat_names[0]}: {opt_w[0]:.2f}")
                with col4:
                    st.write(f"{strat_names[1]}: {opt_w[1]:.2f}")
                with col6:
                    st.write(f"{strat_names[2]}: {opt_w[2]:.2f}")

            opt_port_returns = returns_matrix @ opt_w
            opt_mean = np.mean(opt_port_returns)
            opt_vol = np.std(opt_port_returns)
            opt_sharpe = opt_mean/opt_vol if opt_vol>0 else np.nan

            st.write(f"**Optimal Portfolio Mean Return:** {opt_mean:.2f}")
            st.write(f"**Optimal Portfolio Volatility:** {opt_vol:.2f}")
            st.write(f"**Optimal Portfolio Sharpe Ratio:** {opt_sharpe:.2f}")

            fig_opt, ax_opt = plt.subplots(figsize=(8,5))
            sns.histplot(opt_port_returns, kde=True, ax=ax_opt, color='green')
            ax_opt.set_title("Distribution of Optimal Portfolio Returns")
            ax_opt.set_xlabel("Return")
            plt.tight_layout()
            st.pyplot(fig_opt)
        else:
            st.error("Optimization failed - try adjusting constraints or initial weights.")
