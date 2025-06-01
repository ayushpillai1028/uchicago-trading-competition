import pandas as pd
import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


data = pd.read_csv('Case2.csv')


TRAIN, TEST = train_test_split(data, test_size = 0.2, shuffle = False)

class Allocator():
    def __init__(self, train_data, window=4, min_periods=10, amplification_exponent=1.5):
        self.running_price_paths = train_data.copy()
        self.window = window
        self.min_periods = min_periods
        # The exponent >1 amplifies extremes
        self.amplification_exponent = amplification_exponent

    def allocate_portfolio(self, asset_prices):
        # Update historical prices
        new_prices_df = pd.DataFrame([asset_prices], columns=self.running_price_paths.columns)
        self.running_price_paths = pd.concat([self.running_price_paths, new_prices_df], ignore_index=True)
        num_assets = len(asset_prices)
        epsilon = 1e-8

        if len(self.running_price_paths) < self.min_periods:
            return np.ones(num_assets) / num_assets

        # Calculate returns
        returns_df = self.running_price_paths.pct_change().dropna()

        # Compute baseline signals
        avg_returns = returns_df.rolling(window=self.window).mean().iloc[-1]
        vol = returns_df.rolling(window=self.window).std().iloc[-1] + epsilon
        sharpe_score = avg_returns / vol

        # Momentum (last 3-day average return, volatility normalized)
        raw_momentum = returns_df.iloc[-3:].mean().values
        vol_momentum = raw_momentum / (vol.values + epsilon)
        momentum_norm = vol_momentum / (np.linalg.norm(vol_momentum) + epsilon)

        # Contrarian signal via z-scores (using rolling window)
        z_scores = (returns_df - returns_df.rolling(self.window).mean()) / (returns_df.rolling(self.window).std() + epsilon)
        contrarian_signal = -z_scores.iloc[-1].values
        contrarian_norm = contrarian_signal / (np.linalg.norm(contrarian_signal) + epsilon)

        # Up-Down Volatility Asymmetry Signal
        downside_vol = returns_df[returns_df < 0].rolling(self.window).std().iloc[-1]
        upside_vol = returns_df[returns_df > 0].rolling(self.window).std().iloc[-1]
        asymmetry_signal = (upside_vol - downside_vol).fillna(0).values
        asymmetry_norm = asymmetry_signal / (np.linalg.norm(asymmetry_signal) + epsilon)

        # Cross-Sectional Rank Signal
        latest_returns = returns_df.iloc[-1].values
        rank_signal = pd.Series(latest_returns).rank().values
        rank_norm = rank_signal / (np.linalg.norm(rank_signal) + epsilon)

        sharpe_norm = sharpe_score / (np.linalg.norm(sharpe_score) + epsilon)

        # Base signal blending (using the original weights)
        regime_vol = returns_df.rolling(window=self.window).std().mean().mean()
        current_vol = returns_df.iloc[-self.window:].std().mean()
        if current_vol > regime_vol:
            signal_weights = np.array([0.3, 0.4, 0.1, 0.1, 0.1])
        else:
            signal_weights = np.array([0.4, 0.3, 0.1, 0.1, 0.1])
        
        blended_signal = (
            signal_weights[0] * sharpe_norm +
            signal_weights[1] * momentum_norm +
            signal_weights[2] * contrarian_norm +
            signal_weights[3] * asymmetry_norm +
            signal_weights[4] * rank_norm
        )
        
        # --- Adaptive Nonlinear Signal Amplification ---
        # Adjust amplification exponent based on volatility regime
        if current_vol > 1.25 * regime_vol:
            self.amplification_exponent = 1.2  # soften amplification in volatile regime
        else:
            self.amplification_exponent = 1.5  # default aggressive amplification

        aggressive_signal = np.sign(blended_signal) * (np.abs(blended_signal) ** self.amplification_exponent)
        weights = aggressive_signal / (np.sum(np.abs(aggressive_signal)) + epsilon)

        # --- Volatility Targeting Overlay ---
        portfolio_vol = np.sqrt(np.dot(weights, np.dot(returns_df.cov().values, weights)))
        target_vol = 0.015  # target daily volatility (e.g., 1.5%)
        scaler = target_vol / (portfolio_vol + epsilon)
        weights = weights * min(scaler, 1.5)  # cap scaler to avoid over-leveraging
        weights = weights / (np.sum(np.abs(weights)) + epsilon)
        return weights
    

def grading(train_data, test_data): 
    '''
    Grading Script
    '''
    weights = np.full(shape=(len(test_data.index),6), fill_value=0.0)
    alloc = Allocator(train_data)
    for i in range(0,len(test_data)):
        weights[i,:] = alloc.allocate_portfolio(test_data.iloc[i,:])
        if np.sum(weights < -1) or np.sum(weights > 1):
            raise Exception("Weights Outside of Bounds")
    
    capital = [1]
    for i in range(len(test_data) - 1):
        shares = capital[-1] * weights[i] / np.array(test_data.iloc[i,:])
        balance = capital[-1] - np.dot(shares, np.array(test_data.iloc[i,:]))
        net_change = np.dot(shares, np.array(test_data.iloc[i+1,:]))
        capital.append(balance + net_change)
    capital = np.array(capital)
    returns = (capital[1:] - capital[:-1]) / capital[:-1]
    
    if np.std(returns) != 0:
        sharpe = np.mean(returns) / np.std(returns)
    else:
        sharpe = 0
        
    return sharpe, capital, weights

sharpe, capital, weights = grading(TRAIN, TEST)
#Sharpe gets printed to command line
print(sharpe)

plt.figure(figsize=(10, 6), dpi=80)
plt.title("Capital")
plt.plot(np.arange(len(TEST)), capital)
plt.show()

plt.figure(figsize=(10, 6), dpi=80)
plt.title("Weights")
plt.plot(np.arange(len(TEST)), weights)
plt.legend(TEST.columns)
plt.show()