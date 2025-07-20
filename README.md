### TFT Based Kraken AI Trading Bot
Repository: https://github.com/DrBlackross/TFT-based-Ai-for-Kraken
 The gist of this, using transformers is way easier than all the extra scripts required for my simple RSI in this script or my other one both running CPU only. The other RSI scripts are a straight math/time (math over time), using transformers they have been "technically" easier (with setup and walk away). But also with transformers they learn the markets past and look forward which is IMHO easier.  

This repository contains a Python-based cryptocurrency trading bot designed to make automated trading decisions on the Kraken exchange. It leverages a Transformer-based neural network (inspired by the Temporal Fusion Transformer - TFT) and various technical indicators to predict market movements and execute trades.
The bot supports both live trading (with real funds) and paper trading (simulated trading without risk) for Bitcoin (XBTUSDT) and Dogecoin (DOGEUSDT).

### Also i kept this script based on Kraken (CoinBase being what it is api wise)
 Want to see how to (kinda) implement CoinBase trading look at this script of mine (nightmare)...
 https://github.com/DrBlackross/transformer-base-Ai-kraken/blob/main/coinbase-trans-bot.py

### Features
- Transformer-based Model: Utilizes a custom PyTorch Transformer model for time-series forecasting and trade signal generation.
- Technical Indicator Integration: Incorporates a wide range of popular technical indicators (RSI, MACD, Bollinger Bands, ATR, etc.) as features for the model.
- ATR Multiplier Optimization: Automatically searches for the optimal Average True Range (ATR) multiplier during training to define trade thresholds.
- Paper Trading Mode: Allows you to simulate trading strategies without risking real capital.
- Live Trading Mode: Connects to the Kraken API for actual trading (use with caution!).
- TensorBoard Logging: Logs trading performance and model metrics for visualization and analysis.
- Configurable Trading Pairs: Easily switch between DOGEUSDT and XBTUSDT.
- Scalable Feature Engineering: Dynamically generates features based on a configurable INDICATOR_CONFIG.

### How It Works
The bot operates in two main phases:

##### 1. Training and Optimization:
- The script first fetches historical market data for the specified trading pair.
- It then calculates a comprehensive set of technical indicators and time-based features.
- A crucial step involves an ATR multiplier search. The bot trains multiple Transformer models, each using a different ATR multiplier to define "BUY," "HOLD," and "SELL" targets based on future price movements. The ATR multiplier helps in dynamically setting volatility-based thresholds for trade signals.
- The model and the data scalers (for normalizing input features) that yield the best validation loss during this search are saved as the "best" model.

##### 2. Live/Paper Trading:
- Once an optimal model is found (or loaded from a previous training run), the bot enters a continuous trading loop.
- At regular intervals (defined by DECISION_INTERVAL_SECONDS), it fetches the latest market data.
- It calculates the same set of technical indicators and prepares the data for the loaded Transformer model.
- The model predicts a "BUY," "HOLD," or "SELL" action.
- Additionally, a simple RSI-based override is implemented: if RSI is overbought, it forces a SELL; if oversold, it forces a BUY.
- Based on the predicted action and current balances, the bot executes a trade (or simulates it in paper trading mode).
- All trading activities, balances, and portfolio values are logged to the console and TensorBoard for monitoring.

### Setup for Linux (Python)
Follow these steps to get the trading bot up and running on your Linux system.

##### 1. Prerequisites
##### Python 3.8+: Ensure you have a compatible Python version installed.
		python3 --version

##### Git: For cloning the repository.
		sudo apt update
		sudo apt install git


##### 2. Clone the Repository
First, clone this repository to your local machine:

		git clone https://github.com/DrBlackross/TFT-based-Ai-for-Kraken
		cd TFT-based-Ai-for-Kraken

##### 3. Create a Virtual Environment (Recommended)

Its highly recommended to use a Python virtual environment to manage dependencies.

		python3 -m venv venv
		source venv/bin/activate

##### 4. Install Dependencies
Install all required Python packages using pip:

	pip install pandas pandas-ta krakenex pykrakenapi transformers torch scikit-learn joblib tensorboard
##### OR
	pip install -r ./requirements.txt
  
##### 5. Configure Kraken API Keys
For live trading, you need Kraken API keys. Never hardcode your API keys directly into the script. Instead, set them as environment variables.

You can add these to your ~/.bashrc or ~/.profile file:

		echo "export KRAKEN_API_KEY='your_kraken_api_key_here'" >> ~/.bashrc
		echo "export KRAKEN_API_SECRET='your_kraken_api_secret_here'" >> ~/.bashrc
		source ~/.bashrc # Apply changes

Replace 'your_kraken_api_key_here' and 'your_kraken_api_secret_here' with your actual Kraken API key and secret. Make sure your API key has the necessary permissions for query private funds, query private orders & trades, and trade if you plan to use live trading.

##### 6. User Configuration
Open the TFT-based-Ai-for-Kraken.py file and modify the USER CONFIGURABLE SETTINGS section according to your preferences.

		 ======================
		 USER CONFIGURABLE SETTINGS
		 ======================

		# --- Core Trading Configuration ---
		LIVE_TRADING = False  # Set to True for real trading, False for paper trading
		# KRAKEN_API_KEY and KRAKEN_API_SECRET are loaded from environment variables

		# Select trading pair (DOGEUSDT or XBTUSDT)
		TRADING_PAIR = 'DOGEUSDT'  # Options: 'DOGEUSDT' (Dogecoin) or 'XBTUSDT' (Bitcoin)

		# --- Trading Pair Specific Settings ---
		# These settings will automatically adjust based on TRADING_PAIR
		# You can fine-tune INITIAL_USDT_BALANCE, INITIAL_CRYPTO_BALANCE, MIN_TRADE_AMOUNT
		# and DEFAULT_ATR_MULTIPLIER here.

		# --- Model Path Configuration ---
		# These paths define where models and scalers are saved/loaded.
		# Generally, you don't need to change these.

		# --- Trading Parameters ---
		INTERVAL = 1  # Candlestick interval in minutes (e.g., 1, 5, 15)
		LOOKBACK_DAYS_TRAINING = 730  # Days of historical data for initial training (~2 years)
		LOOKBACK_DAYS_WINDOW = 2  # Days of data to keep in memory for live trading
		SEQUENCE_LENGTH = 24  # Number of time steps in each training sequence (e.g., 24 = 24 minutes)
		TRADE_PERCENTAGE = 0.9  # Percentage of balance to trade (0.9 = 90%)
		DECISION_INTERVAL_SECONDS = INTERVAL * 300 # Time between trading decisions (e.g., 5 minutes for 1m candles)
		SLEEP_TIME_SECONDS = DECISION_INTERVAL_SECONDS + 1 # Sleep time between iterations

		# --- Model Training Parameters ---
		NUM_TRAINING_RUNS = 3  # Number of training runs per ATR multiplier (reduce for faster testing)
		LEARNING_RATE = 1e-5
		PER_DEVICE_BATCH_SIZE = 8
		NUM_TRAIN_EPOCHS = 100 # Max epochs (reduce for faster testing)
		EVAL_STEPS = 20
		LOGGING_STEPS = 10
		SAVE_STEPS = 20
		SAVE_TOTAL_LIMIT = 2
		WEIGHT_DECAY = 0.01
		DROPOUT_RATE = 0.3
		EARLY_STOPPING_PATIENCE = 5
		EARLY_STOPPING_THRESHOLD = 0.001

		# --- ATR Multiplier Optimization ---
		ATR_MULTIPLIERS_TO_TEST = [0.03, 0.05] # Adjust these to test different volatility sensitivities

		# --- Technical Indicator Configuration ---
		# Enable/disable and configure various technical indicators.
		# Modifying these will automatically update the FEATURES_LIST.
		INDICATOR_CONFIG = {
			'BASE_FEATURES': ['open', 'high', 'low', 'close', 'volume'],
			'RSI': {'enabled': True, 'length': 3, 'overbought': 70, 'oversold': 35},
			'MACD': {'enabled': True, 'fast': 12, 'slow': 26, 'signal': 9},
			# ... and other indicators
		}

#### Key settings to consider:

- LIVE_TRADING: Set to True for real trading with your Kraken account, False for paper trading (simulated).
- TRADING_PAIR: Choose 'DOGEUSDT' or 'XBTUSDT'.
- INITIAL_USDT_BALANCE / INITIAL_CRYPTO_BALANCE: For paper trading, set your starting balances. For live trading, these are ignored as your actual Kraken balances will be used.
- MIN_TRADE_AMOUNT: The minimum value of a trade in USDT.
- INTERVAL: The candlestick interval in minutes (e.g., 1, 5, 15).
- LOOKBACK_DAYS_TRAINING: How many days of historical data to fetch for the initial model training.
- SEQUENCE_LENGTH: The number of past time steps the Transformer model considers for its prediction.
- NUM_TRAINING_RUNS: The number of times the model will be trained for each ATR multiplier. Reduce this for faster testing, increase for more robust optimization.
- NUM_TRAIN_EPOCHS: Maximum training epochs. Reduce for faster initial runs.
- ATR_MULTIPLIERS_TO_TEST: A list of ATR multipliers to test during the optimization phase. Smaller values lead to more aggressive trading.
- INDICATOR_CONFIG: Customize which technical indicators are used and their parameters.

##### 7. Run the Bot
Once configured, you can run the bot from your terminal:

		python3 TFT-based-Ai-for-Kraken.py

The bot will first attempt to load a previously trained model. If none is found or if there's an error loading, it will initiate the training and optimization process. This can take a significant amount of time depending on LOOKBACK_DAYS_TRAINING, NUM_TRAINING_RUNS, and NUM_TRAIN_EPOCHS.
After training (or loading), the bot will start its trading loop, printing its decisions and portfolio status to the console.

It will restart and relearn everytime it starts. It will not save the model just the trading logs (delete the trading logs if you want it to start new before restarting)

##### 8. Monitor with TensorBoard
To visualize the trading performance and model metrics, you can use TensorBoard. While the bot is running (or after it has generated some logs), open a new terminal (I use screen for both the script and tensorboard) and run:

		tensorboard --logdir=./trading_logs

Then, open your web browser and navigate to the address provided by TensorBoard (usually http://localhost:6006).

## Important Notes
(the yada-yada)

##### Risk Warning: 
Live trading involves real financial risk. Only use LIVE_TRADING = True if you fully understand the risks and are comfortable with the bot's behavior.
API Rate Limits: Be mindful of Kraken's API rate limits. The SLEEP_TIME_SECONDS is designed to help mitigate this, but excessive requests can lead to temporary bans.
Model Performance: Past performance is not indicative of future results. Cryptocurrency markets are highly volatile and unpredictable.

Customization: Feel free to experiment with the USER CONFIGURABLE SETTINGS and even the model architecture (CryptoTFT class) to optimize performance for your specific needs.
Contributing
Contributions are welcome! If you have suggestions for improvements, bug fixes, or new features, please open an issue or submit a pull request.
License
This project is open-source and available under the MIT License.

#### If it works or you think its cool, send me a few Dogecoin DNgPXztNRmj5qp5jdPP2ZKm1r4u6eQmaZJ
