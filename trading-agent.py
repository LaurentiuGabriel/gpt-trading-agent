# stock_picker_app.py

import streamlit as st
import pandas as pd
import requests
import openai
import yfinance as yf
import os
from datetime import datetime
from dotenv import load_dotenv

# --- Configuration ---
# Make sure to set these as environment variables for security
load_dotenv()
PERPLEXITY_API_KEY = os.environ.get("PERPLEXITY_API_KEY")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
QUANTIQ_API_KEY = os.environ.get("QUANTIQ_API")

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
        "model": "sonar-small-online",
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

    payload = 'apiKey=9e4a6403556810dfb25afa5211f7be21d8a8c954d10d8734086b28b79609c3f7'
    headers = {
        'Content-Type': 'application/x-www-form-urlencoded'
    }

    response = requests.request("POST", url, headers=headers, data=payload)
    data = response.json()
    try:
        if 'data' in data and 'data' in data['data']:
            data['data']['data'].pop('history', None)
    except Exception as e:
        print(f"Error removing 'history': {e}")

    return data

def get_stock_recommendation(ticker, financials):
    """
    Uses GPT-4o to get a buy, sell, or short-sell recommendation for a stock.
    """
    try:
        response = openai.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a financial analyst. Provide a 'BUY', 'SELL', or 'SHORT' recommendation for the given stock ticker and a brief, one-sentence justification. Start your response with one of the keywords: BUY, SELL, or SHORT."},
                {"role": "user", "content": f"Should I invest in {ticker}? The financials are as follows: {financials}. Provide a recommendation."}
            ]
        )
        return response.choices[0].message.content
    except Exception as e:
        st.error(f"Error getting recommendation from OpenAI: {e}")
        return "Error"

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

def get_current_price(ticker):
    """
    Gets the current price of a stock using yfinance.
    """
    try:
        stock = yf.Ticker(ticker)
        price = stock.history(period="1d")['Close'].iloc[-1]
        return price
    except Exception:
        # Fallback for when regular market is closed
        try:
            info = stock.info
            return info.get('preMarketPrice') or info.get('regularMarketPrice')
        except Exception as e:
            st.warning(f"Could not fetch price for {ticker}: {e}")
            return None

# --- Streamlit App ---

st.set_page_config(page_title="AI Stock Picker", layout="wide")
st.title("AI-Powered Stock Picking Assistant")

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
                    if price:
                        # For simplicity, assuming 100 shares per trade
                        update_portfolio(ticker, action, 100, price)
                        success_message = f"Trade executed: {action} 100 shares of {ticker} at ${price:.2f}."
                        st.success(success_message)
                        st.session_state.messages.append({"role": "assistant", "content": success_message})
                    else:
                        st.error("Could not execute trade due to price fetch error.")
                
                # Clear the recommendation to remove the button and prevent re-execution
                st.session_state.actionable_recommendation = None
                st.rerun() # Rerun to update the UI immediately

    # --- Chat Input Logic ---
    if prompt := st.chat_input("What would you like to do? (e.g., 'find stocks', 'analyze AAPL', 'sell AAPL')"):
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
                
                else:
                    response_content = "I can help you find stocks, analyze them, or sell positions. What would you like to do?"

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
        for ticker, shares in holdings.items():
            if shares > 0:
                current_price = get_current_price(ticker)
                if current_price:
                    value = shares * current_price
                    total_value += value
                    
                    # Improved Gain/Loss Calculation
                    buy_trades = portfolio_df[(portfolio_df['ticker'] == ticker) & (portfolio_df['action'] == 'BUY')]
                    gain_loss = 0
                    if not buy_trades.empty:
                        total_cost_of_buys = (buy_trades['shares'] * buy_trades['price']).sum()
                        total_shares_bought = buy_trades['shares'].sum()
                        avg_buy_price = total_cost_of_buys / total_shares_bought
                        current_cost_basis = shares * avg_buy_price
                        gain_loss = value - current_cost_basis

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
            st.metric("Total Portfolio Value", f"${total_value:,.2f}")
        else:
            st.info("You currently have no open positions.")

        # --- Charting ---
        st.subheader("Portfolio Value Over Time")
        chart_data = portfolio_df.copy()
        chart_data['date'] = pd.to_datetime(chart_data['date'])
        chart_data['value_change'] = chart_data.apply(lambda row: row['shares'] * row['price'] if row['action'] != 'SELL' else -row['shares'] * row['price'], axis=1)
        
        daily_value = chart_data.set_index('date').resample('D')['value_change'].sum().cumsum()
        st.area_chart(daily_value)