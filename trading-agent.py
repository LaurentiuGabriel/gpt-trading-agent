import streamlit as st
import pandas as pd
import requests
import openai
import yfinance as yf
import os
from datetime import datetime
from dotenv import load_dotenv
import alpaca_trade_api as tradeapi

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
            st.info(f"Click the button below to execute the trade for {ticker}.")
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
    if prompt := st.chat_input("What would you like to do? (e.g., 'find stocks', 'analyze AAPL', 'hedge portfolio')"):
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

                elif "analyze" in prompt.lower():
                    ticker = prompt.split(" ")[-1].upper()
                    financials = get_financials(ticker)
                    recommendation = get_stock_recommendation(ticker, financials)
                    response_content = recommendation
                    
                    rec_upper = recommendation.upper()
                    if rec_upper.startswith("BUY") or rec_upper.startswith("SHORT"):
                        action = "BUY" if rec_upper.startswith("BUY") else "SHORT"
                        st.session_state.actionable_recommendation = {"ticker": ticker, "action": action}

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