# Required imports
import os
import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces
import matplotlib.pyplot as plt
from stable_baselines3 import PPO, SAC  
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import BaseCallback, CallbackList
from sb3_contrib import RecurrentPPO
from stable_baselines3.sac.policies import SACPolicy
from stable_baselines3.common.distributions import SquashedDiagGaussianDistribution
import torch
import torch.nn as nn
from stable_baselines3.common.torch_layers import create_mlp
import plotly.graph_objects as go
#50
#10
#30 + 1.5
#20
#only for SAC & SACLSTM 30
#30 for SAC, SACLSTM, PPOLSTM

My_TrainingSteps = 20000
LSTMMultiplier = 1
My_TestSteps = 500 

TrainingModifiers = {
    'PPO Agent': 1,     
    'PPO-LSTM Agent': 1, 
    'SAC Agent': 1,      
    'SAC-LSTM Agent': 1  
}

# Set working directory
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)

### 1. Download and Prepare Data
tickers = ['AAPL', 'GS', 'IBM', 'JPM', 'MSFT', 'NKE', 'AMZN', 'NVDA', 'ORCL', 'TSLA']

stock_data = {}
for ticker in tickers:
    try:
        df = pd.read_csv(f'{ticker}.csv', index_col='Date', parse_dates=True)
        if not df.empty:
            stock_data[ticker] = df
            print(f"{ticker}: Start {df.index.min()}, End {df.index.max()}")
    except FileNotFoundError:
        print(f"Warning: {ticker}.csv not found, skipping.")
        continue

fixed_start = pd.to_datetime('2020-01-02')
fixed_end = pd.to_datetime('2025-02-26')
valid_tickers = [t for t in stock_data if stock_data[t].index.min() <= fixed_start and stock_data[t].index.max() >= fixed_end]
if not valid_tickers:
    print("No tickers match the range. Available tickers:")
    for t in stock_data:
        print(f"{t}: {stock_data[t].index.min()} to {stock_data[t].index.max()}")
    raise ValueError("No tickers match the fixed date range")

common_start, common_end = fixed_start, fixed_end
total_days = (common_end - common_start).days
train_days = int(total_days * 0.7)
val_days = int(total_days * 0.15)
train_end = common_start + pd.Timedelta(days=train_days)
val_end = train_end + pd.Timedelta(days=val_days)
My_TestSteps = max(train_days, val_days, total_days - train_days - val_days)

print(f"Common date range: {common_start} to {common_end}")
print(f"Training: {common_start} to {train_end}")
print(f"Validation: {train_end} to {val_end}")
print(f"Test: {val_end} to {common_end}")
print(f"Valid tickers: {len(valid_tickers)}")

### 2. Add Technical Indicators
def calculate_technical_indicators(df):
    df['EMA12_VW'] = (df['Close'] * df['Volume']).ewm(span=12, adjust=False).mean() / df['Volume'].ewm(span=12, adjust=False).mean()
    df['EMA26_VW'] = (df['Close'] * df['Volume']).ewm(span=26, adjust=False).mean() / df['Volume'].ewm(span=26, adjust=False).mean()
    df['VWMACD'] = df['EMA12_VW'] - df['EMA26_VW']
    price_bins = np.arange(df['Close'].min().round(), df['Close'].max().round() + 1, 1)
    df['Price_Bin'] = pd.cut(df['Close'], bins=price_bins)
    volume_profile = df.groupby('Price_Bin')['Volume'].sum()
    df['PoC'] = volume_profile.idxmax().mid
    df.drop(columns=['Price_Bin'], inplace=True)
    df['Range'] = df['High'] - df['Low']
    df['Large_Candle'] = df['Range'] > df['Range'].rolling(20).mean() * 1.5
    df['Order_Block_Bull'] = np.where((df['Large_Candle'] & (df['Close'] > df['Open'])) & 
                                      (df['Range'].shift(-5).rolling(5).mean() < df['Range'] * 0.5), 
                                      df['Low'], np.nan)
    df['Order_Block_Bear'] = np.where((df['Large_Candle'] & (df['Close'] < df['Open'])) & 
                                      (df['Range'].shift(-5).rolling(5).mean() < df['Range'] * 0.5), 
                                      df['High'], np.nan)
    print("Order_Block_Bull non-NaN count before fill:", df['Order_Block_Bull'].notna().sum())
    print("Order_Block_Bear non-NaN count before fill:", df['Order_Block_Bear'].notna().sum())
    df['Order_Block_Bull'].fillna(method='ffill', inplace=True)
    df['Order_Block_Bear'].fillna(method='ffill', inplace=True)
    df['Order_Block_Bull'].fillna(method='bfill', inplace=True)
    df['Order_Block_Bear'].fillna(method='bfill', inplace=True)
    print("Order_Block_Bull non-NaN count after fill:", df['Order_Block_Bull'].notna().sum())
    print("Order_Block_Bear non-NaN count after fill:", df['Order_Block_Bear'].notna().sum())
    df['Delta'] = np.where(df['Close'] > df['Open'], df['Volume'], -df['Volume'])
    df['Cum_Delta'] = df['Delta'].cumsum()
    df['BB_MA'] = df['Close'].rolling(window=20).mean()
    df['BB_Std'] = df['Close'].rolling(window=20).std()
    df['BB_Upper'] = df['BB_MA'] + 2 * df['BB_Std']
    print("Before dropna:", len(df))
    print("NaN counts before dropna:")
    print(df[['Large_Candle', 'Order_Block_Bull', 'Order_Block_Bear', 'BB_MA', 'BB_Std', 'BB_Upper']].isna().sum())
    df.dropna(inplace=True)
    print("After dropna:", len(df))
    return df[['Open', 'High', 'Low', 'Close', 'Volume', 'VWMACD', 'PoC', 
               'Order_Block_Bull', 'Order_Block_Bear', 'Cum_Delta', 'BB_Upper']]

processed_data = {t: calculate_technical_indicators(stock_data[t].loc[common_start:common_end].copy()) for t in valid_tickers}
training_data = {t: processed_data[t].loc[common_start:train_end] for t in valid_tickers}
validation_data = {t: processed_data[t].loc[train_end:val_end] for t in valid_tickers}
test_data = {t: processed_data[t].loc[val_end:common_end] for t in valid_tickers}

expected_columns = {'Open', 'High', 'Low', 'Close', 'Volume', 'VWMACD', 'PoC', 
                    'Order_Block_Bull', 'Order_Block_Bear', 'Cum_Delta', 'BB_Upper'}

for ticker in valid_tickers:
    for data_set, data_name in [(training_data, "training"), (validation_data, "validation"), (test_data, "test")]:
        if set(data_set[ticker].columns) != expected_columns:
            raise ValueError(f"Inconsistent features for {ticker} in {data_name} data")

### 3. Define Stock Trading Environment with Sequence Output
class GymTrading(gym.Env):
    def __init__(self, stock_data, window_size=1, render_mode=None):
        super().__init__()
        self.stock_data = {ticker: df for ticker, df in stock_data.items() if not df.empty}
        self.tickers = list(self.stock_data.keys())
        if not self.tickers:
            raise ValueError("All provided stock data is empty")

        self.window_size = window_size  # Now configurable (1 for PPO/SAC, 5 for LSTM)
        sample_df = next(iter(self.stock_data.values()))
        self.n_features = len(sample_df.columns)
        self.obs_dim = (self.n_features * len(self.tickers) + len(self.tickers) + 4 + len(self.tickers))

        self.action_space = spaces.Box(low=-1, high=1, shape=(len(self.tickers),), dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf,
                                            shape=(self.window_size * self.obs_dim,),  # Updated obs_dim
                                            dtype=np.float32)

        # Portfolio initialization (unchanged)
        self.initial_balance = 1000.0
        self.balance = self.initial_balance
        self.net_worth = self.initial_balance
        self.max_net_worth = self.initial_balance
        self.shares_held = {ticker: 0 for ticker in self.tickers}
        self.cost_basis = {ticker: 0.0 for ticker in self.tickers}
        self.total_shares_sold = {ticker: 0 for ticker in self.tickers}
        self.total_sales_value = {ticker: 0.0 for ticker in self.tickers}
        self.net_worth_history = []

        self.current_step = 0
        self.num_cycles = 1
        self.data_length = min(len(df) for df in self.stock_data.values()) - 1

        print(f"Data lengths for tickers: {[len(df) for df in self.stock_data.values()]}")
        print(f"Computed data_length: {self.data_length}")

        self.max_steps = self.data_length 
        print(f"Computed max_steps: {self.max_steps}")
        self.render_mode = render_mode if render_mode in self.metadata['render_modes'] else 'human'

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.balance = self.initial_balance
        self.net_worth = self.initial_balance
        self.max_net_worth = self.initial_balance
        self.shares_held = {ticker: 0 for ticker in self.tickers}
        self.cost_basis = {ticker: 0.0 for ticker in self.tickers}
        self.total_shares_sold = {ticker: 0 for ticker in self.tickers}
        self.total_sales_value = {ticker: 0.0 for ticker in self.tickers}
        self.current_step = self.window_size - 1  # Start at window_size - 1 (e.g., 4 for window_size=5)
        self.net_worth_history = [self.initial_balance]

        # Initialize history with window_size timesteps
        self.history = np.zeros((self.window_size, self.obs_dim))
        for i in range(self.window_size):
            self.history[i] = self._get_observation_at(max(0, i))

        return self.history.flatten(), {}  # Shape: (window_size * obs_dim,)


    def _get_observation_at(self, step):
        """
        Get the observation at a specific step, including unrealized P&L per ticker.
        """
        data_step = (step % self.data_length) if self.data_length > 0 else 0
        frame = np.zeros(self.obs_dim)
        idx = 0
        for ticker in self.tickers:
            df = self.stock_data[ticker]
            frame[idx:idx + self.n_features] = df.iloc[min(data_step, len(df) - 1)].values
            idx += self.n_features
        frame[idx:idx + len(self.tickers)] = [self.shares_held[ticker] for ticker in self.tickers]
        idx += len(self.tickers)
        frame[idx] = self.balance
        frame[idx + 1] = self.net_worth
        frame[idx + 2] = self.max_net_worth
        frame[idx + 3] = step
        idx += 4
        # Add unrealized P&L for each ticker
        for ticker in self.tickers:
            current_price = self.stock_data[ticker].iloc[min(data_step, len(df) - 1)]['Close']
            avg_cost = (self.cost_basis[ticker] / self.shares_held[ticker] if self.shares_held[ticker] > 0 else 0.0)
            unrealized_pl = (current_price - avg_cost) * self.shares_held[ticker]
            frame[idx] = unrealized_pl
            idx += 1
        return frame

    def _next_observation(self):
        """
        Update and return the next observation (now just the current state).
        """
        self.history = np.zeros((self.window_size, self.obs_dim))  # Shape: (1, obs_dim)
        self.history[-1] = self._get_observation_at(self.current_step)
        return self.history.flatten()  # Shape: (obs_dim,)

    def step(self, actions):
        """
        Execute one time step within the environment.
        """
        # Debug: Check the shape of actions
        # print(f"Step {self.current_step}: Received actions shape: {np.array(actions).shape}, actions: {actions}")

        if isinstance(actions, (list, np.ndarray)) and len(actions.shape) > 1:
            actions = actions[0]

        self.current_step += 1
        if self.current_step > self.max_steps:
            return self._next_observation(), 0.0, True, False, {}

        # Map the current step to the data index
        data_step = (self.current_step % self.data_length) if self.data_length > 0 else 0

        current_prices = {}
        realized_pl_step = 0.0
        self.last_actions = {ticker: 0 for ticker in self.tickers}
        transaction_fee_flat = 0     #for simplicity of testing and learning

        for i, ticker in enumerate(self.tickers):
            current_prices[ticker] = self.stock_data[ticker].iloc[data_step]['Close']
            action = actions[i]

            if action > 0:  # Buy
                shares_to_buy = int(self.balance * action / current_prices[ticker])
                cost = shares_to_buy * current_prices[ticker]
                if cost > 0:
                    fee = transaction_fee_flat
                    total_cost = cost + fee
                    if total_cost <= self.balance:
                        self.balance -= total_cost
                        self.shares_held[ticker] += shares_to_buy
                        self.cost_basis[ticker] += cost
                        self.last_actions[ticker] = shares_to_buy

            elif action < 0 and self.shares_held[ticker] > 0:  # Sell
                shares_to_sell = max(1, int(self.shares_held[ticker] * abs(action)))
                shares_to_sell = min(shares_to_sell, self.shares_held[ticker])
                print(f"Selling {shares_to_sell} shares of {ticker}")

                if shares_to_sell > 0:
                    average_cost = (self.cost_basis[ticker] / self.shares_held[ticker]
                                   if self.shares_held[ticker] > 0 else 0.0)
                    sale_proceeds = shares_to_sell * current_prices[ticker]
                    fee = transaction_fee_flat
                    net_proceeds = max(sale_proceeds - fee, 0)
                    pl = (current_prices[ticker] - average_cost) * shares_to_sell - fee
                    realized_pl_step += pl
                    self.balance += net_proceeds
                    self.shares_held[ticker] -= shares_to_sell
                    self.cost_basis[ticker] -= average_cost * shares_to_sell
                    if self.shares_held[ticker] == 0:
                        self.cost_basis[ticker] = 0.0
                    self.total_shares_sold[ticker] += shares_to_sell
                    self.total_sales_value[ticker] += net_proceeds
                    self.last_actions[ticker] = -shares_to_sell

        self.balance = max(self.balance, 0)
        holdings_value = sum(self.shares_held[ticker] * current_prices[ticker] for ticker in self.tickers)
        self.net_worth = self.balance + holdings_value
        self.net_worth = max(self.net_worth, 1e-6)
        self.max_net_worth = max(self.net_worth, self.max_net_worth)
        self.net_worth_history.append(self.net_worth)

        step_sharpe_ratio = 0.0
        window_size = min(5, len(self.net_worth_history))
        if len(self.net_worth_history) > 1:
            window = self.net_worth_history[-window_size:]
            returns = np.diff(window) / window[:-1]
            if len(returns) > 0:
                mean_returns = np.mean(returns)
                mean_returns = 0.0 if not np.isfinite(mean_returns) else mean_returns
                std_returns = (np.std(returns, ddof=1) if len(returns) > 1
                               else max(np.abs(mean_returns) * 0.1, 1e-4))
                std_returns = 1e-4 if not np.isfinite(std_returns) or std_returns == 0 else std_returns
                step_sharpe_ratio = mean_returns / std_returns
                step_sharpe_ratio = 0.0 if not np.isfinite(step_sharpe_ratio) else step_sharpe_ratio

        reward = 0.9 * realized_pl_step + 0.1 * step_sharpe_ratio
        reward = 0.0 if not np.isfinite(reward) else reward

        done = self.net_worth <= 0 or self.current_step >= self.max_steps
        obs = self._next_observation()
        info = {}

        return obs, reward, done, False, info

    def display_state(self, mode=None, agent_name=None):
        """
        Display the current state of the environment.
        """
        if mode is None:
            mode = self.render_mode
        profit = self.net_worth - self.initial_balance
        if mode == 'human':
            base_str = f"Step: {self.current_step}\tBalance: {self.balance:.2f}"
            if agent_name:
                base_str += f"\tAgent: {agent_name}"
            print(base_str)

            actions_str = ", ".join(f"{ticker}({self.last_actions[ticker]:+d})"
                                    for ticker in self.tickers if self.last_actions[ticker] != 0)
            if actions_str:
                print(f"Actions ({agent_name if agent_name else 'Unknown'}): {actions_str}")

            holdings_str = ", ".join(f"{ticker}({self.shares_held[ticker]})"
                                     for ticker in self.tickers if self.shares_held[ticker] > 0)
            if holdings_str:
                print(f"Holdings ({agent_name if agent_name else 'Unknown'}): {holdings_str}")

            print(f"Net worth ({agent_name if agent_name else 'Unknown'}): {self.net_worth:.2f}\t"
                  f"Profit: {profit:.2f}")
        else:
            raise NotImplementedError(f"display state {mode} not supported")

### 4. Define Models
class ActionLoggerCallback(BaseCallback):
    def __init__(self, verbose=0):
        super(ActionLoggerCallback, self).__init__(verbose)
        self.actions = []  # Store all actions
        self.buy_counts = {}  # Count buy actions per ticker
        self.sell_counts = {}  # Count sell actions per ticker
        self.tickers = None

    def _on_training_start(self):
        # Initialize ticker list and counters when training starts
        self.tickers = self.training_env.envs[0].tickers
        for ticker in self.tickers:
            self.buy_counts[ticker] = 0
            self.sell_counts[ticker] = 0

    def _on_step(self):
        action = self.locals['actions']
        if len(action.shape) > 1:  # Handle (1, n_tickers) shape
            action = action[0]
        self.actions.append(action.copy())

        # Count buy and sell actions
        for i, ticker in enumerate(self.tickers):
            if action[i] > 0:
                self.buy_counts[ticker] += 1
            elif action[i] < 0 and self.training_env.envs[0].shares_held[ticker] > 0:
                self.sell_counts[ticker] += 1  # Only count sell if shares are held

        if self.n_calls % 100 == 0:
            print(f"Step {self.n_calls}: Action = {action}")
        return True

    def _on_training_end(self):
        # Summarize buy/sell counts
        print("\n--- SAC-LSTM Training Action Summary ---")
        total_buys = sum(self.buy_counts.values())
        total_sells = sum(self.sell_counts.values())
        print(f"Total Buy Actions: {total_buys}")
        print(f"Total Sell Actions: {total_sells}")
        for ticker in self.tickers:
            print(f"{ticker}: Buys = {self.buy_counts[ticker]}, Sells = {self.sell_counts[ticker]}")
        if total_sells == 0:
            print("WARNING: SAC-LSTM did not execute any sell trades during training.")
        else:
            print("SAC-LSTM executed sell trades during training.")
        print("---------------------------------------\n")


class TrainingMonitorCallback(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.policy_losses = []
        self.value_losses = []
        self.step_count = 0

    def _on_step(self):
        if hasattr(self.model, 'logger'):
            if 'train/policy_loss' in self.model.logger.name_to_value:
                policy_loss = self.model.logger.name_to_value['train/policy_loss']
                self.policy_losses.append(policy_loss)
            elif 'train/actor_loss' in self.model.logger.name_to_value:
                policy_loss = self.model.logger.name_to_value['train/actor_loss']
                self.policy_losses.append(policy_loss)
            if 'train/value_loss' in self.model.logger.name_to_value:
                self.value_losses.append(self.model.logger.name_to_value['train/value_loss'])

        self.step_count += 1
        if self.step_count % 1000 == 0 and self.policy_losses:  # Check if policy_losses is non-empty
            print(f"Step {self.step_count}: Policy Loss = {self.policy_losses[-1]:.4f}")
        return True

    def _on_training_end(self):
        if self.policy_losses:  # Only print and plot if there’s data
            print(f"\n--- Training Summary for {self.model.__class__.__name__} ---")
            print(f"Average Policy Loss: {np.mean(self.policy_losses):.4f}")
            plt.plot(range(len(self.policy_losses)), self.policy_losses, label="Policy Loss")
            if self.value_losses:
                plt.plot(range(len(self.value_losses)), self.value_losses, label="Value Loss")
            plt.xlabel("Training Steps")
            plt.ylabel("Loss")
            plt.legend()
            plt.title(f"Loss Curves for {self.model.__class__.__name__}")
            plt.show()

class SAC_trading_model:
    def __init__(self, env, total_timesteps, model_path="sac_model.zip"):
        self.model_path = model_path
        policy_kwargs = {"net_arch": [256, 256, 256]}
        self.model = SAC("MlpPolicy", env, learning_rate=0.0001, ent_coef=0.5, batch_size=256, 
                         verbose=1, tensorboard_log="./sac_logs/", policy_kwargs=policy_kwargs)
        self.total_timesteps = total_timesteps

    def train(self):
        action_callback = ActionLoggerCallback()
        monitor_callback = TrainingMonitorCallback()
        callbacks = CallbackList([action_callback, monitor_callback])
        self.model.learn(total_timesteps=self.total_timesteps, callback=callbacks)
        self.model.save(self.model_path)
        print(f"Saved SAC model to {self.model_path}")

    def load(self, env):
        if os.path.exists(self.model_path):
            print(f"Loading SAC model from {self.model_path}")
            self.model = SAC.load(self.model_path, env=env)
        else:
            raise FileNotFoundError(f"No SAC model found at {self.model_path}")

    def predict(self, obs):
        action, _ = self.model.predict(obs)
        print("SAC Action shape:", action.shape)
        return action

class PPO_trading_model:
    def __init__(self, env, total_timesteps, model_path="ppo_model.zip"):
        self.model_path = model_path
        policy_kwargs = {"net_arch": [128, 64]}
        self.model = PPO("MlpPolicy", env, verbose=1, device="cpu", tensorboard_log="./ppo_logs/", policy_kwargs=policy_kwargs)
        self.total_timesteps = total_timesteps

    def train(self):
        action_callback = ActionLoggerCallback()
        monitor_callback = TrainingMonitorCallback()
        callbacks = CallbackList([action_callback, monitor_callback])
        self.model.learn(total_timesteps=self.total_timesteps, callback=callbacks)
        self.model.save(self.model_path)
        print(f"Saved PPO model to {self.model_path}")

    def load(self, env):
        if os.path.exists(self.model_path):
            print(f"Loading PPO model from {self.model_path}")
            self.model = PPO.load(self.model_path, env=env)
        else:
            raise FileNotFoundError(f"No PPO model found at {self.model_path}")

    def predict(self, obs):
        action, _ = self.model.predict(obs)
        print("PPO Action shape:", action.shape)
        return action
    
class PPO_LSTM_trading_model:
    def __init__(self, env, total_timesteps, model_path="ppo_lstm_model.zip",
                 learning_rate=0.0003, n_steps=2048, batch_size=128, n_epochs=20,
                 gamma=0.95, gae_lambda=0.9, clip_range=0.1, ent_coef=0.5,
                 vf_coef=0.7, lstm_hidden_size=128):
        self.model_path = model_path
        self.window_size = 5  # Add window_size=5
        policy_kwargs = {
            "lstm_hidden_size": lstm_hidden_size,
            "net_arch": dict(pi=[128, 64], vf=[128, 64])
        }
        self.model = RecurrentPPO(
            "MlpLstmPolicy",
            env,
            learning_rate=learning_rate,
            n_steps=n_steps,
            batch_size=batch_size,
            n_epochs=n_epochs,
            gamma=gamma,
            gae_lambda=gae_lambda,
            clip_range=clip_range,
            ent_coef=ent_coef,
            vf_coef=vf_coef,
            policy_kwargs=policy_kwargs,
            tensorboard_log="./ppo_lstm_logs/",
            verbose=1,
            device="cuda"
            
        )
        self.total_timesteps = total_timesteps * LSTMMultiplier  

    def train(self):
        action_callback = ActionLoggerCallback()
        monitor_callback = TrainingMonitorCallback()
        callbacks = CallbackList([action_callback, monitor_callback])
        self.model.learn(total_timesteps=self.total_timesteps, callback=callbacks)
        self.model.save(self.model_path)
        print(f"Saved PPO-LSTM model to {self.model_path}")

    def load(self, env):
        if os.path.exists(self.model_path):
            print(f"Loading PPO-LSTM model from {self.model_path}")
            self.model = RecurrentPPO.load(self.model_path, env=env)
        else:
            raise FileNotFoundError(f"No PPO-LSTM model found at {self.model_path}")

    def predict(self, obs, state=None, episode_start=None, deterministic=True):
        # Expect obs shape: (window_size * obs_dim,) = (620,)
        if not isinstance(obs, np.ndarray):
            obs = np.array(obs)
        if len(obs.shape) == 1 and obs.shape[0] != self.window_size * 124:
            raise ValueError(f"Unexpected obs shape: {obs.shape}. Expected ({self.window_size * 124},)")
        action, state = self.model.predict(obs, state=state, episode_start=episode_start, deterministic=deterministic)
        print("PPOLSTM Action shape:", action.shape)
        return action, state
    

class DebugSAC(SAC):
    def collect_rollouts(self, env, callback, train_freq, replay_buffer, action_noise=None, learning_starts=0, log_interval=None):
        print(f"DebugSAC collect_rollouts: Starting rollout")
        rollout = super().collect_rollouts(
            env=env,
            callback=callback,
            train_freq=train_freq,
            replay_buffer=replay_buffer,
            action_noise=action_noise,
            learning_starts=learning_starts,
            log_interval=log_interval
        )
        print(f"DebugSAC collect_rollouts: Rollout completed")
        return rollout

    def _sample_action(self, learning_starts, action_noise, n_envs):
        actions, buffer_actions = super()._sample_action(learning_starts, action_noise, n_envs)
        # print(f"DebugSAC _sample_action: actions shape before reshape: {actions.shape}, actions: {actions}")

        # Ensure actions have shape (n_envs, action_dim)
        if len(actions.shape) == 1:  # Shape (10,) when using policy predict
            actions = actions.reshape(n_envs, -1)  # Reshape to (1, 10)
        elif actions.shape[0] != n_envs or actions.shape[1] != self.action_space.shape[0]:
            raise ValueError(f"Unexpected action shape: {actions.shape}. Expected ({n_envs}, {self.action_space.shape[0]})")

        # Ensure buffer_actions have the same shape
        if len(buffer_actions.shape) == 1:
            buffer_actions = buffer_actions.reshape(n_envs, -1)

       # print(f"DebugSAC _sample_action: actions shape after reshape: {actions.shape}, actions: {actions}")
        return actions, buffer_actions

class SAC_LSTM_trading_model:
    def __init__(self, env, total_timesteps, model_path="sac_lstm_model.zip",
                 learning_rate=0.0001, batch_size=512, ent_coef=0.5,
                 lstm_hidden_size=256, n_lstm_layers=1, dropout_rate=0.05):
        self.model_path = model_path
        self.window_size = 5  # Add window_size=5
        policy_kwargs = {
            "lstm_hidden_size": lstm_hidden_size,
            "n_lstm_layers": n_lstm_layers,
            "net_arch": [256, 256, 256],
            "activation_fn": nn.Tanh,
            "dropout_rate": dropout_rate,
            "window_size": self.window_size  # Pass to policy
        }
        self.model = DebugSAC(
            policy=MlpLstmSACPolicy,
            env=env,
            learning_rate=learning_rate,
            ent_coef=ent_coef,
            batch_size=batch_size,
            buffer_size=1200000,
            train_freq=(20, "step"),
            policy_kwargs=policy_kwargs,
            verbose=1,
            tensorboard_log="./sac_lstm_logs/",
            device="cuda"
        )
        self.total_timesteps = total_timesteps * LSTMMultiplier

    def train(self):
        action_callback = ActionLoggerCallback()
        monitor_callback = TrainingMonitorCallback()
        callbacks = CallbackList([action_callback, monitor_callback])
        self.model.learn(total_timesteps=self.total_timesteps, callback=callbacks)
        self.model.save(self.model_path)
        print(f"Saved SAC-LSTM model to {self.model_path}")

    def load(self, env):
        if os.path.exists(self.model_path):
            print(f"Loading SAC-LSTM model from {self.model_path}")
            self.model = DebugSAC.load(self.model_path, env=env)
        else:
            raise FileNotFoundError(f"No SAC-LSTM model found at {self.model_path}")

    def predict(self, obs, state=None, episode_start=None, deterministic=True):
        print(f"SACLSTM predict: Entering predict method")
        if not isinstance(obs, np.ndarray):
            obs = np.array(obs)
        if len(obs.shape) == 1:  # (620,)
            if obs.shape[0] != self.window_size * 124:
                raise ValueError(f"Unexpected obs shape: {obs.shape}. Expected ({self.window_size * 124},)")
            obs = obs.reshape(1, -1)  # (1, 620)
        action, state = self.model.predict(obs, state=state, episode_start=episode_start, deterministic=deterministic)
        if not isinstance(action, np.ndarray):
            action = np.array(action)
        if len(action.shape) == 1:  # (10,)
            action = action.reshape(1, -1)  # (1, 10)
        expected_shape = (1, 10)
        if action.shape != expected_shape:
            raise ValueError(f"Unexpected action shape: {action.shape}. Expected {expected_shape}")
        return action, state

class MlpLstmSACPolicy(SACPolicy):
    def __init__(self, observation_space, action_space, lr_schedule, 
                 lstm_hidden_size=128, net_arch=None, activation_fn=nn.Tanh, 
                 n_lstm_layers=1, dropout_rate=0.2, window_size=5, *args, **kwargs):
        super().__init__(observation_space, action_space, lr_schedule, *args, **kwargs)
        self.lstm_hidden_size = lstm_hidden_size
        self.n_lstm_layers = n_lstm_layers
        self.net_arch = net_arch or [256, 256]
        self.activation_fn = activation_fn
        self.dropout_rate = dropout_rate
        self.window_size = window_size  # Update to 5
        input_size = observation_space.shape[0] // self.window_size  # e.g., 620 / 5 = 124

        self.lstm = nn.LSTM(input_size, lstm_hidden_size, n_lstm_layers, batch_first=True)
        self.dropout = nn.Dropout(p=dropout_rate)

        actor_layers = []
        prev_size = lstm_hidden_size
        for size in self.net_arch:
            actor_layers.append(nn.Linear(prev_size, size))
            actor_layers.append(activation_fn())
            actor_layers.append(nn.Dropout(p=dropout_rate))
            prev_size = size
        actor_layers.append(nn.Linear(prev_size, action_space.shape[0] * 2))
        self.actor_net = nn.Sequential(*actor_layers)

        critic_input_dim = lstm_hidden_size + action_space.shape[0]
        def create_critic_net():
            layers = []
            prev_size = critic_input_dim
            for size in self.net_arch:
                layers.append(nn.Linear(prev_size, size))
                layers.append(activation_fn())
                layers.append(nn.Dropout(p=dropout_rate))
                prev_size = size
            layers.append(nn.Linear(prev_size, 1))
            return nn.Sequential(*layers)
        self.critic_1 = create_critic_net()
        self.critic_2 = create_critic_net()

        self.action_dist = SquashedDiagGaussianDistribution(action_space.shape[0])
        self.lstm_states = None

    def reset_lstm_states(self, batch_size=1):
        self.lstm_states = (
            torch.zeros(self.n_lstm_layers, batch_size, self.lstm_hidden_size).to(self.device),
            torch.zeros(self.n_lstm_layers, batch_size, self.lstm_hidden_size).to(self.device)
        )

    def forward(self, obs, lstm_states=None, episode_starts=None, deterministic=False):
        if not isinstance(obs, torch.Tensor):
            obs = torch.tensor(obs, dtype=torch.float32).to(self.device)
        if len(obs.shape) == 1:  # (620,)
            obs = obs.unsqueeze(0)  # (1, 620)
        batch_size = obs.shape[0]
        obs = obs.view(batch_size, self.window_size, -1)  # (batch_size, 5, 124)

        if lstm_states is None:
            lstm_states = (
                torch.zeros(self.n_lstm_layers, batch_size, self.lstm_hidden_size).to(self.device),
                torch.zeros(self.n_lstm_layers, batch_size, self.lstm_hidden_size).to(self.device)
            )
        if episode_starts is not None:
            episode_starts = torch.tensor(episode_starts, dtype=torch.float32).to(self.device)
            if episode_starts.dim() == 1:
                episode_starts = episode_starts.unsqueeze(0)
            lstm_states = tuple(state * (1 - episode_starts.view(1, -1, 1)) for state in lstm_states)

        lstm_out, (hidden, cell) = self.lstm(obs, lstm_states)
        features = self.dropout(lstm_out[:, -1, :])  # Last timestep output

        actor_output = self.actor_net(features)
        mean_actions, log_std = actor_output.chunk(2, dim=-1)
        log_std = log_std.clamp(-20, 2)
        self.action_dist.proba_distribution(mean_actions, log_std)
        actions = self.action_dist.get_actions(deterministic=deterministic)

        q1 = self.critic_1(torch.cat([features, actions], dim=-1))
        q2 = self.critic_2(torch.cat([features, actions], dim=-1))
        return actions, (hidden, cell), q1, q2

    def _predict(self, observation, lstm_states=None, episode_starts=None, deterministic=False):
        actions, lstm_states_new, _, _ = self.forward(observation, lstm_states, episode_starts, deterministic)
        self.lstm_states = lstm_states_new
        actions_np = actions.detach().cpu().numpy()
        if len(actions_np.shape) == 2:  # (1, 10)
            actions_np = actions_np[0]  # (10,)
        elif len(actions_np.shape) != 1 or actions_np.shape[0] != self.action_space.shape[0]:
            raise ValueError(f"Unexpected action shape in _predict: {actions_np.shape}. Expected ({self.action_space.shape[0]},)")
        return actions_np

    def evaluate_actions(self, obs, actions, lstm_states=None, episode_starts=None):
        if not isinstance(obs, torch.Tensor):
            obs = torch.tensor(obs, dtype=torch.float32).to(self.device)
        if not isinstance(actions, torch.Tensor):
            actions = torch.tensor(actions, dtype=torch.float32).to(self.device)
        if len(obs.shape) == 1:
            obs = obs.unsqueeze(0)
        obs = obs.view(obs.shape[0], self.window_size, -1)  # (batch_size, 1, 124)
        
        _, next_lstm_states, q1, q2 = self.forward(obs, lstm_states, episode_starts)
        log_prob = self.action_dist.log_prob(actions)
        return q1, q2, log_prob

    def set_training_mode(self, mode: bool):
        self.train(mode)

    def predict(self, observation, state=None, episode_starts=None, deterministic=False):
        if state is None and self.lstm_states is None:
            self.reset_lstm_states(batch_size=1)
        elif state is not None:
            self.lstm_states = state
        if episode_starts is not None and self.lstm_states is not None:
            episode_starts = np.array(episode_starts) if not isinstance(episode_starts, np.ndarray) else episode_starts
            if episode_starts.size == 1:  # Scalar or single-element array
                episode_starts = np.array([episode_starts.item()])
            self.lstm_states = tuple(s * (1 - torch.tensor(episode_starts, dtype=torch.float32).view(1, -1, 1).to(self.device)) for s in self.lstm_states)
        actions = self._predict(observation, self.lstm_states, episode_starts, deterministic)
        return actions, self.lstm_states
    
    
### 5. Training Function, call in Main()
def train_arena(data, total_timesteps):
    # Define window sizes for each agent
    window_sizes = {
        'PPO Agent': 1,
        'PPO-LSTM Agent': 5,
        'SAC Agent': 1,
        'SAC-LSTM Agent': 5
    }

    agents = {}
    model_paths = {
        'PPO Agent': "ppo_model.zip",
        'PPO-LSTM Agent': "ppo_lstm_model.zip",
        'SAC Agent': "sac_model.zip",
        'SAC-LSTM Agent': "sac_lstm_model.zip"
    }
    models_exist = {name: os.path.exists(path) for name, path in model_paths.items()}
    any_models_exist = any(models_exist.values())
    adjusted_timesteps = {name: int(total_timesteps * modifier) for name, modifier in TrainingModifiers.items()}

    # Create a dictionary to store environments for each agent
    envs = {}
    for agent_name in model_paths.keys():
        env = DummyVecEnv([lambda: GymTrading(data, window_size=window_sizes[agent_name])])
        env = VecNormalize(env, norm_obs=True, norm_reward=True)
        envs[agent_name] = env
        print(f"{agent_name} env observation shape: {env.observation_space.shape}")

        # Initialize agents with their specific environments
        if agent_name == 'PPO Agent':
            agents[agent_name] = PPO_trading_model(envs[agent_name], adjusted_timesteps[agent_name], model_path=model_paths[agent_name])
        elif agent_name == 'PPO-LSTM Agent':
            agents[agent_name] = PPO_LSTM_trading_model(
                envs[agent_name], adjusted_timesteps[agent_name], model_path=model_paths[agent_name],
                learning_rate=0.0003, n_steps=1024, batch_size=128, n_epochs=20,
                gamma=0.95, gae_lambda=0.9, clip_range=0.1, ent_coef=0.02,
                vf_coef=0.7, lstm_hidden_size=128
            )
        elif agent_name == 'SAC Agent':
            agents[agent_name] = SAC_trading_model(envs[agent_name], adjusted_timesteps[agent_name], model_path=model_paths[agent_name])
        elif agent_name == 'SAC-LSTM Agent':
            agents[agent_name] = SAC_LSTM_trading_model(
                envs[agent_name], adjusted_timesteps[agent_name], model_path=model_paths[agent_name],
                learning_rate=0.0003, batch_size=256, ent_coef=0.2,
                lstm_hidden_size=128, n_lstm_layers=1, dropout_rate=0.2
            )

    # Training logic
    print("Adjusted training steps for each agent:")
    for agent_name, steps in adjusted_timesteps.items():
        print(f"{agent_name}: {steps} steps ({TrainingModifiers[agent_name]*100}% of {total_timesteps})")

    if not any_models_exist:
        print("No saved models found. Training all agents from scratch...")
        for agent_name, agent in agents.items():
            print(f"Training new {agent_name} with {adjusted_timesteps[agent_name]} steps...")
            if adjusted_timesteps[agent_name] > 0:
                agent.train()
            # Save VecNormalize stats for this agent
            envs[agent_name].save(f"vec_normalize_{agent_name.replace(' ', '_').lower()}.pkl")
    else:
        print("Some saved models found:")
        for name, exists in models_exist.items():
            print(f"{name}: {'Exists' if exists else 'Not Found'}")
        response = input("Continue training with existing models where available, or start all from scratch? (continue/scratch): ").strip().lower()
        if response in ['continue', 'c']:
            for agent_name, agent in agents.items():
                if models_exist[agent_name]:
                    print(f"Loading and continuing training for {agent_name}...")
                    agent.load(envs[agent_name])
                    agent.total_timesteps = adjusted_timesteps[agent_name]
                    if adjusted_timesteps[agent_name] > 0:
                        agent.train()
                else:
                    print(f"Training new {agent_name}...")
                    agent.total_timesteps = adjusted_timesteps[agent_name]
                    if adjusted_timesteps[agent_name] > 0:
                        agent.train()
                # Save VecNormalize stats for this agent
                envs[agent_name].save(f"vec_normalize_{agent_name.replace(' ', '_').lower()}.pkl")
        elif response in ['scratch', 's']:
            for agent_name, agent in agents.items():
                print(f"Training new {agent_name}...")
                agent.total_timesteps = adjusted_timesteps[agent_name]
                if adjusted_timesteps[agent_name] > 0:
                    agent.train()
                # Save VecNormalize stats for this agent
                envs[agent_name].save(f"vec_normalize_{agent_name.replace(' ', '_').lower()}.pkl")

    return envs['PPO Agent'], agents['PPO Agent'], agents['PPO-LSTM Agent'], agents['SAC Agent'], agents['SAC-LSTM Agent']


### 6. Test and Validate Functions
def evaluate_visualize_agents(env, agents, data, n_tests=500, phase="Training", learn_during_validation=False, learn_steps=100):
    metrics = {}
    for agent_name, agent in agents.items():
        print(f"Testing {agent_name}...")
        metrics[agent_name] = evaluate(env, agent, data, agent_name=agent_name, n_tests=n_tests, visualize=True,
                                       learn_during_validation=learn_during_validation, learn_steps=learn_steps)
        print(f"Done testing {agent_name}!")

    if len(agents) > 1:
        steps_list = [metrics[agent_name]['steps'] for agent_name in agents.keys()]
        net_worths = [metrics[agent_name]['net_worths'] for agent_name in agents.keys()]
        networth_graph(steps_list, net_worths, list(agents.keys()), phase=phase)
    return metrics


def evaluate(env, agent, stock_data, agent_name=None, n_tests=500, visualize=True, learn_during_validation=False, learn_steps=100):
    window_size = getattr(agent, 'window_size', env.envs[0].window_size)
    eval_env = DummyVecEnv([lambda: GymTrading(stock_data, window_size=window_size)])
    vec_normalize_path = f"vec_normalize_{agent_name.replace(' ', '_').lower()}.pkl"
    if not os.path.exists(vec_normalize_path):
        raise FileNotFoundError(f"VecNormalize stats not found at {vec_normalize_path}. Ensure training has been completed for {agent_name}.")
    eval_env = VecNormalize.load(vec_normalize_path, eval_env)
    eval_env.training = learn_during_validation
    eval_env.norm_reward = learn_during_validation

    metrics = {
        'steps': [],
        'balances': [],
        'net_worths': [],
        'shares_held': {ticker: [] for ticker in eval_env.envs[0].tickers},
        'prices': {ticker: [] for ticker in eval_env.envs[0].tickers},
        'buy_trades': {ticker: [] for ticker in eval_env.envs[0].tickers},
        'sell_trades': {ticker: [] for ticker in eval_env.envs[0].tickers}
    }
    obs = eval_env.reset()
    state = None
    episode_start = np.array([True])

    data_length = eval_env.envs[0].data_length
    total_steps = min(n_tests, data_length)

    print(f"Starting evaluation for {agent_name} for 1 cycle ({total_steps} steps), data_length={data_length}")

    i = 0
    while i < total_steps:
        metrics['steps'].append(i)

        if isinstance(agent, (PPO_LSTM_trading_model, SAC_LSTM_trading_model)):
            action, state = agent.predict(obs, state=state, episode_start=episode_start, deterministic=True)
            print(f"Step {i}, {agent_name}: Action = {action}")
        else:
            action = agent.predict(obs)
            print(f"Step {i}, {agent_name}: Action = {action}")

        obs, rewards, dones, infos = eval_env.step(action)
        
        # Append metrics before checking dones to capture the state before reset
        metrics['balances'].append(eval_env.get_attr('balance')[0])
        metrics['net_worths'].append(eval_env.get_attr('net_worth')[0])
        env_shares_held = eval_env.get_attr('shares_held')[0]
        last_actions = eval_env.envs[0].last_actions
        current_step = eval_env.get_attr('current_step')[0]
        data_step = (current_step % data_length) if data_length > 0 else 0
        for ticker in eval_env.envs[0].tickers:
            metrics['shares_held'][ticker].append(env_shares_held.get(ticker, 0))
            price = (stock_data[ticker].iloc[data_step]['Close'] if 0 <= data_step < len(stock_data[ticker])
                     else stock_data[ticker].iloc[-1]['Close'] if not stock_data[ticker].empty else 0)
            metrics['prices'][ticker].append(price)
            if last_actions[ticker] > 0:
                metrics['buy_trades'][ticker].append((i, last_actions[ticker]))
            elif last_actions[ticker] < 0:
                metrics['sell_trades'][ticker].append((i, abs(last_actions[ticker])))

        if visualize:
            eval_env.envs[0].display_state(mode='human', agent_name=agent_name)

        if dones:
            print(f"Reached end of cycle at step {i+1} (current_step={current_step}, max_steps={total_steps}). Ending evaluation.")
            break  # Exit immediately after recording the last step's metrics

        if isinstance(agent, (PPO_LSTM_trading_model, SAC_LSTM_trading_model)):
            episode_start = np.array([dones])

        if learn_during_validation and i % 50 == 0 and i > 0:
            print(f"Step {i}: Performing online learning for {learn_steps} steps...")
            agent.model.learn(total_timesteps=learn_steps, reset_num_timesteps=False)

        i += 1

    print(f"Completed evaluation for {agent_name} with {len(metrics['steps'])} steps (1 cycle)")
    # Debug: Print the last few net worths to verify
    print(f"Last 5 net worths for {agent_name}: {metrics['net_worths'][-5:]}")
    return metrics


def networth_graph(steps_list, net_worths_list, labels, phase="Training"):
    plt.figure(figsize=(12, 6))
    
    # Normalize net worths relative to the initial value (assumed to be 1000)
    initial_net_worth = 1000.0
    normalized_net_worths = [(np.array(net_worths) / initial_net_worth) for net_worths in net_worths_list]
    
    # Define a clean color palette
    colors = plt.cm.Set2.colors[:len(labels)]
    
    # Plot normalized net worths with agent-specific steps
    for i, (steps, norm_net_worths, label, color) in enumerate(zip(steps_list, normalized_net_worths, labels, colors)):
        plt.plot(steps, norm_net_worths, label=label, color=color, linewidth=1.5, alpha=0.9)
    
    # Customize the plot
    plt.title(f'Normalized Net Worth Over Time ({phase})', pad=15, fontsize=12, weight='bold')
    plt.xlabel('Steps', labelpad=10, fontsize=10)
    plt.ylabel('Normalized Net Worth (Relative to Initial $1000)', labelpad=10, fontsize=10)
    plt.legend(loc='best', fontsize=9, frameon=True, facecolor='white', edgecolor='gray', framealpha=0.9)
    plt.grid(True, linestyle='--', alpha=0.3, linewidth=0.5, color='gray')
    plt.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5, linewidth=0.5)
    
    # Set y-limit with padding
    max_value = max(max(norm) for norm in normalized_net_worths) * 1.1
    min_value = min(min(norm) for norm in normalized_net_worths) * 0.9
    plt.ylim(max(0, min_value), max_value)
    
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.tick_params(axis='both', which='major', labelsize=9)
    plt.tight_layout()
    plt.show()

### 7. Compare Agents' Performance
def tabulate_sharpe_ratio(agents_metrics, labels, phase="Training"):
    sharpe_ratios = []
    annualized_rois = []
    initial_net_worths = []
    final_net_worths = []
    min_net_worths = []
    max_net_worths = []
    initial_capital = 1000
    TRADING_DAYS_PER_YEAR = 251

    # Calculate metrics for each agent
    for metrics in agents_metrics.values():
        net_worths = np.array(metrics['net_worths'])
        steps = metrics['steps']

        if len(net_worths) > 4 and len(steps) > 0:
            t = len(net_worths) - 4 if len(net_worths) >= 2 else -1

            # Calculate Sharpe Ratio
            returns = (net_worths - initial_capital) / initial_capital
            mean_return = np.mean(returns)
            std_return = np.std(returns)
            sharpe_ratio = mean_return / std_return if std_return != 0 else 0

            # Calculate Annualized ROI (CAGR)
            final_net_worth = net_worths[t]
            initial_net_worth = net_worths[0]
            total_steps = len(steps)
            phase_years = total_steps / TRADING_DAYS_PER_YEAR

            if initial_net_worth == 0 or phase_years <= 0:
                annualized_roi = 0
            else:
                total_return = final_net_worth / initial_net_worth
                annualized_roi = (total_return ** (1 / phase_years)) - 1

            initial_nw = net_worths[0]
            final_nw = net_worths[t]
            min_nw = np.min(net_worths)
            max_nw = np.max(net_worths)
        else:
            sharpe_ratio = annualized_roi = initial_nw = final_nw = min_nw = max_nw = 0

        # Append calculated metrics to lists
        sharpe_ratios.append(sharpe_ratio)
        annualized_rois.append(annualized_roi)
        initial_net_worths.append(initial_nw)
        final_net_worths.append(final_nw)
        min_net_worths.append(min_nw)
        max_net_worths.append(max_nw)

    # Create and sort DataFrame
    df = pd.DataFrame({
        'Agent': labels,
        'Initial Net Worth': initial_net_worths,
        'Final Net Worth': final_net_worths,
        'Min Net Worth': min_net_worths,
        'Max Net Worth': max_net_worths,
        'ROI': annualized_rois,
        'Sharpe Ratio': sharpe_ratios
    })
    df_sorted = df.sort_values(by='Sharpe Ratio', ascending=False)

    # Format DataFrame for display
    df_sorted['Initial Net Worth'] = df_sorted['Initial Net Worth'].apply(lambda x: f"${x:.2f}")
    df_sorted['Final Net Worth'] = df_sorted['Final Net Worth'].apply(lambda x: f"${x:.2f}")
    df_sorted['Min Net Worth'] = df_sorted['Min Net Worth'].apply(lambda x: f"${x:.2f}")
    df_sorted['Max Net Worth'] = df_sorted['Max Net Worth'].apply(lambda x: f"${x:.2f}")
    df_sorted['ROI'] = (df_sorted['ROI'] * 100).round(2).astype(str) + '%'
    df_sorted['Sharpe Ratio'] = df_sorted['Sharpe Ratio'].round(4)

    # Create Plotly table
    headers = list(df_sorted.columns)
    values = [df_sorted[col].tolist() for col in df_sorted.columns]

    fig = go.Figure(data=[
        go.Table(
            header=dict(
                values=headers,
                fill_color='paleturquoise',
                align='left'
            ),
            cells=dict(
                values=values,
                fill_color='lavender',
                align='left'
            )
        )
    ])

    # Update layout and display
    fig.update_layout(
        title=f"Agent Performance Metrics - {phase}",
        width=800,
        height=600
    )
    fig.show()
    

### 8. Visualize Most Traded Ticker
def visualize_most_traded_ticker(metrics, tickers, agent_name):
    # Calculate total shares traded per ticker
    total_traded = {}
    for ticker in tickers:
        buy_shares = sum(shares for _, shares in metrics['buy_trades'][ticker])
        sell_shares = sum(shares for _, shares in metrics['sell_trades'][ticker])
        total_traded[ticker] = buy_shares + sell_shares
    
    # Identify the most traded ticker
    largest_ticker = max(total_traded, key=total_traded.get)
    print(f"{agent_name} - Most traded ticker: {largest_ticker} with {total_traded[largest_ticker]} shares traded")
    
    # Extract steps and prices
    steps = metrics['steps']
    prices = metrics['prices'][largest_ticker]
    
    # Combine buy and sell trades into a single list with step, action type, and shares
    all_trades = []
    for step, shares in metrics['buy_trades'][largest_ticker]:
        all_trades.append((step, 'Buy', shares))
    for step, shares in metrics['sell_trades'][largest_ticker]:
        all_trades.append((step, 'Sell', shares))
    
    # Sort trades by step for chronological order
    all_trades.sort(key=lambda x: x[0])  # Sort by step (first element of tuple)
    
    # Plotting (moved before the log)
    plt.figure(figsize=(14, 8))
    plt.plot(steps, prices, label=f'{largest_ticker} Price', color='blue')
    
    # Separate buy and sell steps for plotting
    buy_steps = [step for step, action, _ in all_trades if action == 'Buy']
    buy_prices = [prices[step] for step in buy_steps]
    sell_steps = [step for step, action, _ in all_trades if action == 'Sell']
    sell_prices = [prices[step] for step in sell_steps]
    
    plt.scatter(buy_steps, buy_prices, color='green', label='Buy', marker='^', s=100)
    plt.scatter(sell_steps, sell_prices, color='red', label='Sell', marker='v', s=100)
    plt.title(f'{largest_ticker} Price with Buy/Sell Points ({agent_name})')
    plt.xlabel('Steps')
    plt.ylabel('Price')
    plt.legend()
    plt.grid(True)
    plt.show()  # Display the graph immediately
    
    # Print the combined and sorted trade log (moved after the plot)
    print(f"\n--- Trade Log for {largest_ticker} ({agent_name}) ---")
    for step, action, shares in all_trades:
        price = prices[step]
        print(f"Step {step}: {action} {shares} shares of {largest_ticker} at {price:.2f}")
    print("--------------------\n")

### 9. Main Execution
def main():
    total_timesteps = My_TrainingSteps
    n_tests = My_TestSteps
    learn_steps = 100

    print("Starting training phase...")
    env, ppo_agent, ppo_lstm_agent, sac_agent, sac_lstm_agent = train_arena(training_data, total_timesteps)
    agents = {
        'PPO Agent': ppo_agent,
        'PPO-LSTM Agent': ppo_lstm_agent,
        'SAC Agent': sac_agent,
        'SAC-LSTM Agent': sac_lstm_agent
    }

    model_paths = {
        'PPO Agent': "ppo_model.zip",
        'PPO-LSTM Agent': "ppo_lstm_model.zip",
        'SAC Agent': "sac_model.zip",
        'SAC-LSTM Agent': "sac_lstm_model.zip"
    }
    if not all(os.path.exists(path) for path in model_paths.values()):
        print("Error: Not all required models exist. Aborting test and validation.")
        return

    # Training evaluation (no learning)
    print("Evaluating on training data...")
    agents_metrics = evaluate_visualize_agents(env, agents, training_data, n_tests=n_tests, phase="Training Evaluation")
    tabulate_sharpe_ratio(agents_metrics, list(agents.keys()), phase="Training Evaluation")

    # Validation phase: Create environments and evaluate all agents together
    window_sizes = {
        'PPO Agent': 1,
        'PPO-LSTM Agent': 5,
        'SAC Agent': 1,
        'SAC-LSTM Agent': 5
    }
    validation_envs = {}
    for agent_name in agents.keys():
        validation_env = DummyVecEnv([lambda: GymTrading(validation_data, window_size=window_sizes[agent_name])])
        vec_normalize_path = f"vec_normalize_{agent_name.replace(' ', '_').lower()}.pkl"
        if not os.path.exists(vec_normalize_path):
            raise FileNotFoundError(f"VecNormalize stats not found at {vec_normalize_path}. Ensure training has been completed for {agent_name}.")
        validation_env = VecNormalize.load(vec_normalize_path, validation_env)
        validation_env.training = True
        validation_env.norm_reward = True
        validation_envs[agent_name] = validation_env
        print(f"Validation env for {agent_name} observation shape: {validation_env.observation_space.shape}")

    print("Validation phase running with learning enabled (simulating production, in-memory only)")
    validation_agents_metrics = {}
    for agent_name, agent in agents.items():
        validation_agents_metrics.update(
            evaluate_visualize_agents(
                validation_envs[agent_name], {agent_name: agent}, validation_data, n_tests=n_tests,
                phase="Validation", learn_during_validation=True, learn_steps=learn_steps
            )
        )
    if len(validation_agents_metrics) > 1:
        steps_list = [validation_agents_metrics[agent_name]['steps'] for agent_name in agents.keys()]
        net_worths = [validation_agents_metrics[agent_name]['net_worths'] for agent_name in agents.keys()]
        networth_graph(steps_list, net_worths, list(agents.keys()), phase="Validation")
    tabulate_sharpe_ratio(validation_agents_metrics, list(agents.keys()), phase="Validation")

    # Reload original models for testing
    print("Reloading original models for testing (discarding validation learning updates)...")
    reload_envs = {}
    for agent_name in agents.keys():
        reload_env = DummyVecEnv([lambda: GymTrading(training_data, window_size=window_sizes[agent_name])])
        vec_normalize_path = f"vec_normalize_{agent_name.replace(' ', '_').lower()}.pkl"
        if not os.path.exists(vec_normalize_path):
            raise FileNotFoundError(f"VecNormalize stats not found at {vec_normalize_path}. Ensure training has been completed for {agent_name}.")
        reload_env = VecNormalize.load(vec_normalize_path, reload_env)
        reload_envs[agent_name] = reload_env
        agents[agent_name].load(reload_envs[agent_name])
        print(f"Reloaded {agent_name} from {model_paths[agent_name]}")

    # Test phase: Create environments and evaluate all agents together
    test_envs = {}
    for agent_name in agents.keys():
        test_env = DummyVecEnv([lambda: GymTrading(test_data, window_size=window_sizes[agent_name])])
        vec_normalize_path = f"vec_normalize_{agent_name.replace(' ', '_').lower()}.pkl"
        if not os.path.exists(vec_normalize_path):
            raise FileNotFoundError(f"VecNormalize stats not found at {vec_normalize_path}. Ensure training has been completed for {agent_name}.")
        test_env = VecNormalize.load(vec_normalize_path, test_env)
        test_env.training = False
        test_env.norm_reward = False
        test_envs[agent_name] = test_env
        print(f"Test env for {agent_name} observation shape: {test_env.observation_space.shape}")

    print(f"Testing with 1 cycle of test data, no learning")
    test_agents_metrics = {}
    for agent_name, agent in agents.items():
        test_agents_metrics.update(
            evaluate_visualize_agents(
                test_envs[agent_name], {agent_name: agent}, test_data, n_tests=n_tests, phase="Test"
            )
        )
    if len(test_agents_metrics) > 1:
        steps_list = [test_agents_metrics[agent_name]['steps'] for agent_name in agents.keys()]
        net_worths = [test_agents_metrics[agent_name]['net_worths'] for agent_name in agents.keys()]
        networth_graph(steps_list, net_worths, list(agents.keys()), phase="Test")
    tabulate_sharpe_ratio(test_agents_metrics, list(agents.keys()), phase="Test")

    for agent_name in agents.keys():
        visualize_most_traded_ticker(test_agents_metrics[agent_name], test_envs[agent_name].envs[0].tickers, agent_name)

if __name__ == "__main__":
    main()