import streamlit as st
import pandas as pd
import requests
import openai
import yfinance as yf
import os
from datetime import datetime
from dotenv import load_dotenv
import alpaca_trade_api as tradeapi
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# --- Configuration ---
# Make sure to set these as environment variables for security
load_dotenv()
PERPLEXITY_API_KEY = os.environ.get("PERPLEXITY_API_KEY")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
QUANTIQ_API_KEY = os.environ.get("QUANTIQ_API")
ALPACA_API_KEY = os.environ.get("ALPACA_API_KEY")
ALPACA_SECRET_KEY = os.environ.get("ALPACA_SECRET_KEY")
# Use paper trading by default, change to "false" in .env for live
ALPACA_PAPER = os.environ.get("ALPACA_PAPER", "true").lower() == "true"

# Set up OpenAI client
openai.api_key = OPENAI_API_KEY

PORTFOLIO_CSV = "portfolio.csv"

# --- Helper Functions ---

def get_small_cap_stocks():
    """
    Uses the Perplexity API to get a list of US small and micro-cap stocks.
    """
    if PERPLEXITY_API_KEY == "YOUR_PERPLEXITY_API_KEY" or not PERPLEXITY_API_KEY:
        st.error("Perplexity API key is not set. Cannot fetch stocks.")
        return []
        
    headers = {
        "Authorization": f"Bearer {PERPLEXITY_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": "sonar-pro",
        "messages": [
            {"role": "system", "content": "You are an AI assistant that provides lists of stock tickers."},
            {"role": "user", "content": "Please provide a list of 10 interesting US micro-cap or small-cap stock tickers. Just provide the tickers, separated by commas."},
        ],
    }
    try:
        response = requests.post("https://api.perplexity.ai/chat/completions", headers=headers, json=payload)
        response.raise_for_status()
        content = response.json()['choices'][0]['message']['content']
        # Clean up the response to get only valid tickers
        tickers = [ticker.strip().upper() for ticker in content.split(',') if ticker.strip()]
        return tickers
    except requests.exceptions.RequestException as e:
        st.error(f"Error fetching from Perplexity API: {e}")
        return []
    
    
def get_government_official_trades(ticker):
    """
    Uses the QuantiQ.live API to get trades done by House and Senate officials on the supplied ticker.
    """
    url = f"https://www.quantiq.live/api/get-congress-trades?simbol={ticker}"

    payload = f"apiKey={QUANTIQ_API_KEY}"
    headers = {
        'Content-Type': 'application/x-www-form-urlencoded'
    }

    response = requests.request("POST", url, headers=headers, data=payload)
    data = response.json()
    try:
        if 'data' in data and 'data' in data['data']:
            data['data']['data'].pop('history', None)
    except Exception as e:
        # Silently fail if history key doesn't exist
        pass

    return data

def plot_technicals(df, ticker):
    """
    Generates a professional financial chart with technical indicators using Plotly.
    """
    # 1. Create a figure with subplots
    # We need 4 rows: 1 for price, 1 for volume, 1 for RSI, 1 for MACD
    fig = make_subplots(
        rows=4, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        subplot_titles=(f'{ticker} Price', 'Volume', 'RSI', 'MACD'),
        row_heights=[0.5, 0.1, 0.2, 0.2] # Give more space to the main price chart
    )

    # 2. Add the Candlestick chart for price
    fig.add_trace(go.Candlestick(
        x=df.index,
        open=df['open'],
        high=df['high'],
        low=df['low'],
        close=df['close'],
        name="Price"
    ), row=1, col=1)

    # 3. Add Overlays to the price chart (Moving Averages & Bollinger Bands)
    fig.add_trace(go.Scatter(x=df.index, y=df['SMA_20'], mode='lines', name='SMA 20', line=dict(color='yellow', width=1)), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['EMA_50'], mode='lines', name='EMA 50', line=dict(color='purple', width=1)), row=1, col=1)
    # Bollinger Bands
    fig.add_trace(go.Scatter(x=df.index, y=df['BBU_20_2.0'], mode='lines', name='Upper BB', line=dict(color='cyan', width=1, dash='dash')), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['BBL_20_2.0'], mode='lines', name='Lower BB', line=dict(color='cyan', width=1, dash='dash'), fill='tonexty', fillcolor='rgba(0, 176, 246, 0.1)'), row=1, col=1)

    # 4. Add the Volume chart
    # Color bars red for down days and green for up days
    colors = ['green' if row['close'] >= row['open'] else 'red' for index, row in df.iterrows()]
    fig.add_trace(go.Bar(x=df.index, y=df['volume'], name='Volume', marker_color=colors), row=2, col=1)

    # 5. Add the RSI chart
    fig.add_trace(go.Scatter(x=df.index, y=df['RSI_14'], mode='lines', name='RSI', line=dict(color='orange', width=2)), row=3, col=1)
    # Add overbought/oversold lines for RSI
    fig.add_hline(y=70, line_dash="dash", line_color="red", row=3, col=1)
    fig.add_hline(y=30, line_dash="dash", line_color="green", row=3, col=1)

    # 6. Add the MACD chart
    fig.add_trace(go.Scatter(x=df.index, y=df['MACD_12_26_9'], mode='lines', name='MACD', line=dict(color='blue', width=2)), row=4, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['MACDs_12_26_9'], mode='lines', name='Signal', line=dict(color='red', width=1)), row=4, col=1)
    # Color histogram based on positive or negative values
    macd_colors = ['green' if val >= 0 else 'red' for val in df['MACDh_12_26_9']]
    fig.add_trace(go.Bar(x=df.index, y=df['MACDh_12_26_9'], name='Histogram', marker_color=macd_colors), row=4, col=1)

    # 7. Update the layout for a clean look
    fig.update_layout(
        title=f'{ticker} Technical Analysis',
        height=800,
        showlegend=True,
        xaxis_rangeslider_visible=False, # Hide the main range slider
        template='plotly_dark' # Use a dark theme
    )
    # Hide the range slider for all but the last subplot
    fig.update_xaxes(rangeslider_visible=False)

    return fig
    
def get_financials(ticker):
    """
    Fetches financial data for a given stock ticker using the Quantiq API.
    """
    url = f"https://www.quantiq.live/api/get-market-data/{ticker}"

    payload = f"apiKey={QUANTIQ_API_KEY}"
    headers = {
        'Content-Type': 'application/x-www-form-urlencoded'
    }

    response = requests.request("POST", url, headers=headers, data=payload)
    data = response.json()
    try:
        if 'data' in data and 'data' in data['data']:
            data['data']['data'].pop('history', None)
    except Exception as e:
        # Silently fail if history key doesn't exist
        pass

    return data

def get_technicals(ticker):
    """
    Fetches technical indicators for a given stock ticker using yfinance.
    """
    try:
        url = f"https://www.quantiq.live/api/technical-indicator?symbol={ticker}&timeframe=1Hour&period=20"
        payload = f"apiKey={QUANTIQ_API_KEY}"
        headers = {
            'Content-Type': 'application/x-www-form-urlencoded'
        }

        response = requests.request("POST", url, headers=headers, data=payload)
        data = response.json()
        
        if 'bars' not in data or not data['bars']:
            st.warning(f"No bar data returned for {ticker} from the API.")
            return pd.DataFrame()

        # 1. Convert the list of bars into a pandas DataFrame
        df = pd.DataFrame(data['bars'])

        # 2. Prepare the DataFrame for pandas-ta
        #    - Rename columns to the required lowercase format
        #    - Set the 'Timestamp' as a proper DatetimeIndex
        df.rename(columns={
            'ClosePrice': 'close',
            'HighPrice': 'high',
            'LowPrice': 'low',
            'OpenPrice': 'open',
            'Volume': 'volume',
            'Timestamp': 'timestamp'
        }, inplace=True)
        
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df.set_index('timestamp', inplace=True)
        
        # Ensure data is in chronological order (oldest first) for correct calculations
        df.sort_index(inplace=True)

        # 3. Define your technical analysis strategy
        #    You can easily add or remove indicators here
        my_strategy = ta.Strategy(
            name="Common Indicators",
            description="SMA, EMA, RSI, and Bollinger Bands",
            ta=[
                {"kind": "sma", "length": 20},
                {"kind": "ema", "length": 50},
                {"kind": "rsi"}, # Uses default length of 14
                {"kind": "bbands", "length": 20, "std": 2},
                {"kind": "macd"},
            ]
        )

        # 4. Calculate the indicators and append them to the DataFrame
        df.ta.strategy(my_strategy)
        
        return df


    except Exception as e:
        st.error(f"Error fetching technicals for {ticker}: {e}")
        return pd.DataFrame()

def get_stock_recommendation(ticker, financials):
    """
    Uses GPT-5 to get a buy, sell, or short-sell recommendation for a stock.
    Note: "gpt-5" is a placeholder for a future or custom model name.
    """
    try:
        response = openai.chat.completions.create(
            model="gpt-5", 
            messages=[
                {"role": "system", "content": "You are a financial analyst. Provide a 'BUY', 'SELL', or 'SHORT' recommendation for the given stock ticker and a brief, one-sentence justification. Start your response with one of the keywords: BUY, SELL, or SHORT."},
                {"role": "user", "content": f"Should I invest in {ticker}? The financials are as follows: {financials}. Provide a recommendation."}
            ]
        )
        return response.choices[0].message.content
    except Exception as e:
        st.error(f"Error getting recommendation from OpenAI: {e}")
        return "Error"
    
def hedge_portfolio():
    """
    Analyzes the current portfolio, gets a hedge proposal from GPT-5,
    and returns the proposal as a dictionary. It can handle various listed assets.
    """
    # 1. Load and validate the current portfolio.
    if not os.path.exists(PORTFOLIO_CSV):
        return {"error": "Your portfolio is empty. Add some stocks before hedging."}

    portfolio_df = pd.read_csv(PORTFOLIO_CSV)
    if portfolio_df.empty:
        return {"error": "Your portfolio is empty. Add some stocks before hedging."}

    # Calculate current holdings, considering only positive positions (longs).
    holdings = portfolio_df.groupby('ticker')['shares'].apply(
        lambda x: x[portfolio_df.loc[x.index, 'action'] != 'SELL'].sum() - x[portfolio_df.loc[x.index, 'action'] == 'SELL'].sum()
    ).to_dict()
    
    current_positions = {ticker: shares for ticker, shares in holdings.items() if shares > 0}

    if not current_positions:
        return {"error": "You have no open long positions to hedge."}
    
    holdings_str = ", ".join([f"{shares} shares of {ticker}" for ticker, shares in current_positions.items()])

    # 2. Construct prompt and call GPT-5 for a hedging strategy.
    try:
        prompt_content = f"""
        Given the following equity portfolio: {holdings_str}.
        Propose a single, specific, and listed asset to act as a hedge. This could be a commodity ETF (e.g., for gold GLD, oil USO), a cryptocurrency (e.g., BTC-USD), REITs, Bonds or a volatility-based product (e.g., VIXY).
        The goal is to find an asset that is likely to have a negative correlation with the provided portfolio, especially during market downturns.
        Return your answer in the following strict format, and nothing else:
        BUY: [TICKER], JUSTIFICATION: [Your brief one-sentence justification here.]
        """
        
        response = openai.chat.completions.create(
            model="gpt-5", # NOTE: "gpt-5" is a placeholder for a future/custom model.
            messages=[
                {"role": "system", "content": "You are an expert hedge fund analyst. You provide concise, actionable hedging strategies in a specific format."},
                {"role": "user", "content": prompt_content}
            ]
        )
        proposal = response.choices[0].message.content.strip()

        # 3. Parse the proposal from the AI's response.
        if "BUY:" in proposal.upper() and "JUSTIFICATION:" in proposal.upper():
            # Use split and strip for robust, case-insensitive parsing.
            parts = proposal.split("JUSTIFICATION:")
            ticker = parts[0].replace("BUY:", "").strip()
            justification = parts[1].strip()

            return {
                "ticker": ticker,
                "justification": justification
            }
        else:
            return {"error": f"Failed to parse the hedge proposal from the AI. Raw response: {proposal}"}

    except Exception as e:
        return {"error": f"An error occurred while communicating with OpenAI: {e}"}


def update_portfolio(ticker, action, shares, price):
    """
    Updates the portfolio CSV file with a new transaction.
    """
    new_trade = pd.DataFrame([{
        "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "ticker": ticker,
        "action": action.upper(),
        "shares": shares,
        "price": price
    }])
    if os.path.exists(PORTFOLIO_CSV):
        portfolio_df = pd.read_csv(PORTFOLIO_CSV)
        portfolio_df = pd.concat([portfolio_df, new_trade], ignore_index=True)
    else:
        portfolio_df = new_trade
    portfolio_df.to_csv(PORTFOLIO_CSV, index=False)
    
def book_trade_alpaca(api, ticker, shares, action):
    """
    Books a trade with a market order through the Alpaca API.
    Returns (success_boolean, message_or_order_object).
    """
    if not api:
        return False, "Alpaca API client is not initialized. Check your API keys."

    if action.upper() in ["BUY"]:
        side = 'buy'
    elif action.upper() in ["SELL", "SHORT"]:
        side = 'sell'
    else:
        return False, f"Invalid action: {action}"

    try:
        order = api.submit_order(
            symbol=ticker,
            qty=shares,
            side=side,
            type='market',
            time_in_force='day'  
        )
        return True, order
    except Exception as e:
        return False, str(e)

def get_current_price(ticker):
    """
    Gets the current price of an asset (stock, crypto, ETF, etc.) using yfinance.
    """
    try:
        stock = yf.Ticker(ticker)
        # Use 'bid' or 'regularMarketPrice' for more reliable real-time data
        info = stock.info
        price = info.get('bid') or info.get('regularMarketPrice')
        if price:
            return price
        # Fallback to previous close if real-time price is unavailable
        price = stock.history(period="1d")['Close'].iloc[-1]
        return price
    except Exception as e:
        st.warning(f"Could not fetch price for {ticker}: {e}")
        return None

# --- Streamlit App ---

st.set_page_config(page_title="AI Stock Picker", layout="wide")
st.title("AI-Powered Stock Picking Assistant")

if ALPACA_API_KEY and ALPACA_SECRET_KEY and "YOUR" not in ALPACA_API_KEY:
    try:
        base_url = "https://paper-api.alpaca.markets" if ALPACA_PAPER else "https://api.alpaca.markets"
        alpaca_api = tradeapi.REST(ALPACA_API_KEY, ALPACA_SECRET_KEY, base_url, api_version='v2')
        account = alpaca_api.get_account()
        st.sidebar.success(f"✅ Alpaca Connected ({'Paper' if ALPACA_PAPER else 'Live'})")
    except Exception as e:
        st.sidebar.error(f"⚠️ Alpaca connection failed: {e}")
else:
    st.sidebar.warning("🔑 Alpaca keys not found. Trade booking is disabled.")

# --- Initialize Session State ---
if "messages" not in st.session_state:
    st.session_state.messages = []
if "actionable_recommendation" not in st.session_state:
    st.session_state.actionable_recommendation = None

# --- Page Navigation ---
page = st.sidebar.radio("Navigate", ["Chat", "Portfolio Performance"])

if page == "Chat":
    st.header("Chat with your AI Analyst")

    # Display chat messages from history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # --- Actionable Button Logic ---
    # Display the button if there's an actionable recommendation in the session state
    if rec := st.session_state.actionable_recommendation:
        ticker = rec["ticker"]
        action = rec["action"]
        
        # Using a container to group the button and message
        with st.container():
            st.info(f"Click the button below to execute the trade for {ticker}. You can adjust the number of shares before executing.")
            shares_to_trade = st.number_input(
                label="Number of Shares",
                min_value=1,
                value=10,  # Sets a default value
                step=1,
                key=f"shares_{ticker}_{action}" # A unique key is crucial
            )
            if st.button(f"Execute {action} for {ticker}", key=f"execute_{ticker}_{action}"):
                with st.spinner(f"Executing {action} for {ticker}..."):
                    price = get_current_price(ticker)
                    shares_to_trade = 10
                    if price:
                        # For simplicity, assuming 100 shares per trade
                        update_portfolio(ticker, action, 100, price)
                        success_message = f"Trade executed: {action} 100 shares of {ticker} at ${price:.2f}."
                        success, message = book_trade_alpaca(alpaca_api, ticker, shares_to_trade, action)
                        if success:
                            order_details = message
                            # Get price for logging, actual fill price is in order_details from Alpaca
                            price_for_log = get_current_price(ticker)
                            if price_for_log:
                             # Log the trade to our local CSV *after* successful submission
                                success_msg = f"✅ **Alpaca trade submitted!** {action} {shares_to_trade} shares of {ticker}. Order ID: `{order_details.id}`."
                                st.success(success_msg)
                                st.session_state.messages.append({"role": "assistant", "content": success_msg})
                            else:
                                st.error("Alpaca order submitted, but failed to fetch price for local portfolio log.")
                    else:
                        # If Alpaca trade fails
                        error_msg = f"❌ **Alpaca trade failed:** {message}"
                        st.error(error_msg)
                        st.session_state.messages.append({"role": "assistant", "content": error_msg})
                
                # Clear the recommendation to remove the button and prevent re-execution
                st.session_state.actionable_recommendation = None
                st.rerun() # Rerun to update the UI immediately

    # --- Chat Input Logic ---
    if prompt := st.chat_input("What would you like to do? (e.g., 'find stocks', 'analyze AAPL', 'hedge portfolio', 'get technicals MSFT')"):
        # Add user message to chat history and display it
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Process the prompt and generate assistant response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response_content = ""
                # Clear any previous recommendation before processing a new one
                st.session_state.actionable_recommendation = None

                if "find stocks" in prompt.lower():
                    tickers = get_small_cap_stocks()
                    if tickers:
                        response_content = f"Here are some small-cap stocks I found: {', '.join(tickers)}"
                    else:
                        response_content = "Sorry, I couldn't fetch any stock tickers at the moment."
                elif "check house and senate trades" in prompt.lower():
                    ticker = prompt.split(" ")[-1].upper()
                    response = get_government_official_trades(ticker)
                    if response: 
                        response_content = f"Here are some recent trades by House and Senate officials on {ticker}:\n\n{response}"
                    else:
                        response_content = f"Sorry, I couldn't find any trades for {ticker}."
                
                elif "analyze" in prompt.lower():
                    ticker = prompt.split(" ")[-1].upper()
                    financials = get_financials(ticker)
                    recommendation = get_stock_recommendation(ticker, financials)
                    response_content = recommendation
                    
                    rec_upper = recommendation.upper()
                    if rec_upper.startswith("BUY") or rec_upper.startswith("SHORT"):
                        action = "BUY" if rec_upper.startswith("BUY") else "SHORT"
                        st.session_state.actionable_recommendation = {"ticker": ticker, "action": action}

                elif "get technicals" in prompt.lower():
                    ticker = prompt.split(" ")[-1].upper()
                    st.info(f"Fetching technical data for {ticker}...")
                    
                    technicals = get_technicals(ticker)
                    
                    if not technicals.empty:
                        fig = plot_technicals(technicals, ticker)
                        
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.warning(f"No technicals available for {ticker}")
                        
                elif "sell" in prompt.lower():
                    ticker_to_sell = prompt.split(" ")[-1].upper()
                    if os.path.exists(PORTFOLIO_CSV):
                        portfolio_df = pd.read_csv(PORTFOLIO_CSV)
                        if ticker_to_sell in portfolio_df['ticker'].values:
                            response_content = f"You have a position in {ticker_to_sell}. Do you want to sell?"
                            st.session_state.actionable_recommendation = {"ticker": ticker_to_sell, "action": "SELL"}
                        else:
                            response_content = f"You do not own {ticker_to_sell}."
                    else:
                        response_content = "Your portfolio is empty."
                
                elif "hedge" in prompt.lower():
                    hedge_result = hedge_portfolio()
                    if "error" in hedge_result:
                        response_content = hedge_result["error"]
                    else:
                        ticker = hedge_result['ticker']
                        justification = hedge_result['justification']
                        response_content = f"🛡️ **Hedge Proposal:** To hedge your portfolio, I recommend buying **{ticker}**. \n\n*Justification:* {justification}"
                        
                        # Set up the actionable recommendation so the BUY button appears
                        st.session_state.actionable_recommendation = {"ticker": ticker, "action": "BUY"}

                else:
                    response_content = "I can help you find stocks, analyze them, hedge your portfolio, or sell positions. What would you like to do?"

                st.markdown(response_content)
        
        # Add assistant's response to chat history
        st.session_state.messages.append({"role": "assistant", "content": response_content})
        st.rerun() # Rerun to display the new button if one was set

elif page == "Portfolio Performance":
    st.header("Portfolio Performance")

    if not os.path.exists(PORTFOLIO_CSV):
        st.warning("No portfolio data found. Make some trades in the 'Chat' page.")
    else:
        portfolio_df = pd.read_csv(PORTFOLIO_CSV)
        st.subheader("Trade History")
        st.dataframe(portfolio_df)

        # Calculate current holdings
        holdings = portfolio_df.groupby('ticker')['shares'].apply(
            lambda x: x[portfolio_df.loc[x.index, 'action'] != 'SELL'].sum() - x[portfolio_df.loc[x.index, 'action'] == 'SELL'].sum()
        ).to_dict()

        # Display current holdings and performance
        performance_data = []
        total_value = 0
        total_cost_basis_overall = 0
        total_pnl_overall = 0

        for ticker, shares in holdings.items():
            if shares > 0:
                current_price = get_current_price(ticker)
                if current_price:
                    value = shares * current_price
                    total_value += value
                    
                    # Improved Gain/Loss Calculation
                    buy_trades = portfolio_df[(portfolio_df['ticker'] == ticker) & (portfolio_df['action'] != 'SELL')]
                    sell_trades = portfolio_df[(portfolio_df['ticker'] == ticker) & (portfolio_df['action'] == 'SELL')]
                    
                    cost_basis = 0
                    if not buy_trades.empty:
                        total_cost_of_buys = (buy_trades['shares'] * buy_trades['price']).sum()
                        total_shares_bought = buy_trades['shares'].sum()
                        avg_buy_price = total_cost_of_buys / total_shares_bought if total_shares_bought > 0 else 0
                        cost_basis = shares * avg_buy_price
                        gain_loss = value - cost_basis
                        
                        total_cost_basis_overall += cost_basis
                        total_pnl_overall += gain_loss

                    performance_data.append({
                        "Ticker": ticker, "Shares": shares,
                        "Current Price": f"${current_price:,.2f}",
                        "Current Value": f"${value:,.2f}",
                        "Gain/Loss": f"${gain_loss:,.2f}"
                    })

        st.subheader("Current Holdings")
        if performance_data:
            performance_df = pd.DataFrame(performance_data)
            st.dataframe(performance_df)
        else:
            st.info("You currently have no open positions.")

        st.markdown("---") # Adds a horizontal line for visual separation
        
        # --- Overall Performance Metrics ---
        st.subheader("Overall Portfolio Performance")
        
        percentage_pnl = (total_pnl_overall / total_cost_basis_overall * 100) if total_cost_basis_overall > 0 else 0
        
        col1, col2, col3 = st.columns(3)
        col1.metric(
            label="Total Portfolio Value 💰",
            value=f"${total_value:,.2f}"
        )
        col2.metric(
            label="Total Gain / Loss 📈",
            value=f"${total_pnl_overall:,.2f}",
            delta=f"{percentage_pnl:.2f}%"
        )
        col3.metric(
            label="Total Cost Basis 🏦",
            value=f"${total_cost_basis_overall:,.2f}"
        )
        st.markdown("---") 

        # --- Charting ---
        all_tickers = portfolio_df['ticker'].unique().tolist()
        if all_tickers:
            start_date = pd.to_datetime(portfolio_df['date']).min()
            end_date = datetime.now()
            
            st.subheader("Individual Asset Price Evolution")
            with st.spinner("Loading historical price data for charts..."):
                try:
                    historical_data = yf.download(all_tickers, start=start_date, end=end_date, progress=False)
                    if not historical_data.empty:
                        adj_close_prices = historical_data['Close']
                        # Handle case for single ticker download (returns a Series)
                        if isinstance(adj_close_prices, pd.Series):
                            adj_close_prices = adj_close_prices.to_frame(name=all_tickers[0])
                        
                        st.line_chart(adj_close_prices.dropna(axis=1, how='all'))
                    else:
                        st.warning("Could not retrieve historical price data for charting.")
                except Exception as e:
                    st.error(f"An error occurred while fetching historical data for charts: {e}")