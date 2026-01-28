import os
import streamlit as st
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import yfinance as yf
from datetime import datetime, timedelta
from dotenv import load_dotenv


import re  # New import for ticker detection
from streamlit.runtime.scriptrunner import add_script_run_ctx # New import for thread fixing

# MODERN LANGCHAIN IMPORTS (v1.0+)
from langchain.agents import create_agent
#from langchain.agents import create_react_agent
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain.tools import tool


# COMMUNITY TOOLS & VECTOR STORE
from langchain_community.vectorstores import Chroma
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_community.utilities import DuckDuckGoSearchAPIWrapper
from langchain_text_splitters import RecursiveCharacterTextSplitter

# 1. Load Keys
load_dotenv()
GAPi = os.getenv('GROQ_API_KEY')

# 2. Setup Embeddings (Correct modern class)
embedding = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')

# 3. Setting up LLM
llm = ChatGroq(
        model_name = 'llama-3.3-70b-versatile',
        groq_api_key = GAPi,
        verbose = False, #tells what model is doing behind the scenes
        temperature = 0
    )

# 4. TOOLS


## 2. get price
price_df=pd.DataFrame({})
@tool
def get_stock_prices(ticker: str, end_date: str, duration_days: int = 25) -> str:
    """
    Fetches historical stock prices. 
    ticker: Stock symbol (e.g., 'RELIANCE.NS').
    end_date: Target date 'YYYY-MM-DD'.
    duration_days: How many days of history to fetch (default 25). 
                   Use 45 for 1 month, 200 for 6 months, etc.
    """
    add_script_run_ctx() # <--- ADD THIS LINE HERE
    try:
        # 1. Calculate the dynamic lookback
        # We fetch slightly more days (duration_days * 1.5) to account for weekends/holidays
        end_dt = datetime.strptime(end_date, '%Y-%m-%d')
        lookback = int(duration_days * 1.5) 
        start_dt = end_dt - timedelta(days=lookback)
        
        df = yf.download(ticker, 
                        start=start_dt.strftime('%Y-%m-%d'), 
                        end=(end_dt + timedelta(days=1)).strftime('%Y-%m-%d'))
        
        if df.empty:
            return f"No price data found for {ticker}. Dont guess the price, check the ticker and date once again in the user prompt"

        # 2. Cleanup Multi-Index (yfinance 2026 standard)
        df.columns = [col[0] if isinstance(col, tuple) else col for col in df.columns]
        df = df.reset_index()
        df['Date'] = df['Date'].dt.strftime('%Y-%m-%d')

        # 3. Store in session state for the Plotter tool
        st.session_state.price_df = df 
        
        # 4. Return the exact amount requested to the Agent
        return df[['Date', 'Close', 'Volume']].tail(duration_days).to_markdown(index=False)
        
    except Exception as e:
        return f"Error fetching data: {str(e)}"


## 3. get news
@tool
def get_news(ticker: str, start_date: str, end_date: str) -> list:
    """
    Finds stock news for a specific ticker within a date range.
    Args:
        ticker: The stock symbol (e.g., 'AAPL').
        target_date: The end of the date range (YYYY-MM-DD).
        start_date: The beginning of the date range (YYYY-MM-DD).
    Returns:
        A list of dictionaries containing 'snippet', 'title', and 'link'.
    """
    add_script_run_ctx() # <--- ADD THIS LINE HERE
    # 1. IMPROVED QUERY: Use 'site:' to target high-quality financial news
    # And 'intitle:' to ensure the ticker is the main subject
    search_query = f"{ticker} stock news site:moneycontrol.com OR site:economictimes.indiatimes.com OR site:reuters.com"
    
    # 2. DATE FILTER (Advanced): DuckDuckGo supports a custom range syntax in the query 
    # but the API Wrapper works best when we use the internal 'df' logic
    # Here we refine the query string to include the timeframe directly
    time_query = f"{search_query} after:{start_date} before:{end_date}"

    wrapper = DuckDuckGoSearchAPIWrapper(region="in-en", max_results=10)
    
    # 3. Use 'results' to get metadata like snippets and sources
    search_results = wrapper.results(time_query, max_results=10)
    
    if not search_results:
        return f"The street is silent on {ticker} between {start_date} and {end_date}, Lala."

    output = []
    for res in search_results:
        title = res.get('title', 'Market Update')
        url = res.get('link', '')
        snippet = res.get('snippet', '')
        
        # 4. CLEANER FORMATTING: Include a small snippet so the Agent can 'read' the news
        if url:
            # We use a clean bullet format that Harshad Mehta can summarize
            output.append(f"🔹 [{title[:60]}...]({url})")
    return "\n\n".join(output)

## 4. database
@tool() #groq is giving random error 400
def Db(news_text: str, query: str) -> str:
    """
    Analyzes news text to find specific answers. 
    Use this to 'read' between the lines of the news links.
    """
    add_script_run_ctx() # <--- ADD THIS LINE HERE
    if not news_text or "The street is silent" in news_text:
        return "NO RELEVANT NEWS"

    # 1. SMART CLEANING
    # Since get_news now returns a single Markdown string, we split by double newline
    texts = [t.strip() for t in news_text.split("\n\n") if t.strip()]
    
    # 2. FAST IN-MEMORY STORAGE
    # We use an ephemeral (temporary) collection that resets only when the tool starts
    # This prevents old stock data from bleeding into new queries
    try:
        chunk_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        docs = chunk_splitter.create_documents(texts)

        # 3. EPHEMERAL VECTOR DB
        # We add a unique 'collection_name' to ensure we aren't appending to old sessions
        vectorstore = Chroma.from_documents(
            documents=docs, 
            embedding=embedding,
            collection_name=f"temp_news_{datetime.now().strftime('%H%M%S')}"
        )

        # 4. MMR SEARCH (Maximal Marginal Relevance)
        # Standard similarity search often returns 5 versions of the SAME news.
        # MMR ensures 'Diversity'—it picks 5 DIFFERENT facts.
        results = vectorstore.max_marginal_relevance_search(query, k=4)

        if not results:
            return "NO RELEVANT NEWS"

        formatted_results = []
        for i, doc in enumerate(results):
            formatted_results.append(f"INSIGHT {i+1}: {doc.page_content}")

        return "\n\n".join(formatted_results)

    except Exception as e:
        print(f"VDB Error: {e}")
        return "Lala, the database is jammed. Use your own judgment."


## 5. plot
def plot_stock(price_df, ticker):
    """Creates the Neon Price Line Chart"""
    fig = px.line(
        price_df, x='Date', y='Close',
        title=f"{ticker} - Big Bull Neon View",
        template="plotly_dark"
    )
    fig.update_traces(line=dict(color='#00FFCC', width=3))
    fig.update_layout(
        paper_bgcolor="#000000", plot_bgcolor="#000000",
        font=dict(color="white"), title_font=dict(size=22)
    )
    return fig

def plot_volume(price_df, ticker):
    """Creates the Neon Volume Bar Chart"""
    fig = px.bar(
        price_df, x='Date', y='Volume',
        title=f"{ticker} - Market Volume",
        template="plotly_dark"
    )
    fig.update_traces(marker_color='#FF00FF', opacity=0.8)
    fig.update_layout(
        paper_bgcolor="#000000", plot_bgcolor="#000000",
        font=dict(color="white"), yaxis_title="Shares Traded"
    )
    return fig

@tool
def plotter(ticker: str) -> str:
    """
    Creates neon charts for price and volume. 
    MUST be called after 'get_stock_prices'.
    """
    add_script_run_ctx() # <--- ADD THIS LINE HERE
    # 1. Validation Check
    if 'price_df' not in st.session_state or st.session_state.price_df is None:
        return "Lala, I can't draw on an empty ledger! Get the stock prices first."
    
    df = st.session_state.price_df
    actual_days = len(df)
    
    # 2. Generate both Price and Volume charts
    fig_price = plot_stock(df, ticker)
    fig_vol = plot_volume(df, ticker)
    
    # 3. Store in pending_plots list for the UI to pick up
    # This ensures both charts are rendered side-by-side or one after another
    st.session_state.pending_plots = [fig_price, fig_vol]
    
    return f"Charts for {ticker} are ready. I've plotted {actual_days} trading days for you."


# 5. Agent


today = datetime.now().strftime('%Y-%m-%d')

system_prompt = f'''**ROLE: HARSHAD MEHTA (THE BIG BULL)**
You are the legendary 'Big Bull' of the stock market. You are high-energy, high-conviction, and speak with the authority of someone who owns the Bombay Stock Exchange. Your motto: "Risk hai toh Ishq hai!"

**OPERATIONAL PROTOCOLS (STRICT):**
1. STATELESS EXECUTION: Every query is a fresh trade. Do not use data from previous tickers. If the user asks about 'Tesla' after 'Gold', forget Gold entirely.
2. TOOL-FIRST REASONING: 
   - Never guess a price, RSI, or MACD. 
   - You MUST call `get_stock_prices` for every new ticker mentioned.
   - If the tool returns an error or empty JSON, state: "Lala, the ledger is empty for this stock. No data, no trade."
3. NO SYNTAX SPECULATION: Do not attempt to format tool calls with XML tags like <function>. Use the native tool-calling capability of the model.
4. ZERO TOLERANCE FOR HALLUCINATION: If you don't have the data in the current observation, do not make it up. "Jhoot market mein nahi chalta, Lala."

**TECHNICAL ANALYSIS GUIDELINES:**
- Use a 25-day and 50-day Moving Average strategy.
- RSI interpretation: >70 is Overbought (Careful!), <30 is Oversold (Opportunity!).
- MACD: Look for crossovers to confirm the trend.

**RESPONSE STRUCTURE:**
- The Big Bull Headline: A punchy, street-style summary of the stock's current vibe.
- The Technical Game: Bullet points of hard numbers (RSI, MA, MACD) fetched from the tool.
- Market Sentiment: Summary of news buzz.
- The Bottom Line: Clear "LONG" or "WAIT" recommendation with Target and Stop-Loss.
- Signature: Always end with "Risk hai toh Ishq hai!" or "Asli maza toh tab hai jab sab dar rahe ho!"

    TODAY'S DATE: {today}
         '''
tools = [ get_stock_prices, plotter, get_news, Db]

#*******************************

def extract_ticker(text):
    """Regex to identify the stock symbol in the user's query."""
    match = re.search(r'\b[A-Z]{1,10}(=F|\.NS)?\b', text.upper())
    return match.group(0) if match else None

def smart_invoke(agent_executor, user_query, system_prompt):
    """The Wrapper that handles Memory, Thread Context, and Groq 400 Errors."""
    # A. TICKER-SWITCH LOGIC
    current_ticker = extract_ticker(user_query)
    last_ticker = st.session_state.get('last_ticker')

    if current_ticker and last_ticker and current_ticker != last_ticker:
        st.session_state.messages = [] # Wipe history if it's a new stock
        st.session_state.last_ticker = current_ticker
    elif current_ticker:
        st.session_state.last_ticker = current_ticker

    # B. SMART HISTORY (Last 5 messages)
    history = st.session_state.messages[-5:] if st.session_state.messages else []
    
    # C. THREAD CONTEXT INJECTION
    add_script_run_ctx() 

    try:
        return agent_executor.invoke({
            "messages": [SystemMessage(content=system_prompt)] + history + [HumanMessage(content=user_query)]
        })
    except Exception as e:
        # D. GROQ 400 FALLBACK
        if "400" in str(e) or "failed_generation" in str(e):
            st.warning("🔄 Fixer-bot: Correcting Groq syntax...")
            correction = f"{user_query}\n\n(SYSTEM: USE NATIVE TOOLS ONLY. NO XML TAGS.)"
            return agent_executor.invoke({
                "messages": [SystemMessage(content=system_prompt), HumanMessage(content=correction)]
            })
        raise e

#*******************************
@st.cache_resource # Keeps the agent in memory
def init_agent(_llm):
    
    # Use an f-string to inject the date directly into the system prompt
    llm_with_tools = _llm.bind_tools(tools)

    # 2. Use the modern create_agent (built on LangGraph)
    
    agent_executor = create_agent(
        model=llm_with_tools, 
        tools=tools, 
        system_prompt=system_prompt
    )
    
    return agent_executor

agent_executor = init_agent(llm)
# 6. Steamlit UI

st.set_page_config(page_title="Big Bull Market Desk", page_icon="💰")
st.title("💰 GROWMORE")
st.markdown("*\"Lala, stock market ek gehra kuan hai...\"*")

# 1. Initialize chat history with LangChain message objects if empty
if 'messages' not in st.session_state:
    st.session_state.messages = []

# 2. Display history
for message in st.session_state.messages:
    # Check if the message is from Human or AI to set the avatar
    role = 'user' if isinstance(message, HumanMessage) else 'assistant'
    with st.chat_message(role):
        st.markdown(message.content)
        if hasattr(message, 'additional_kwargs') and 'charts' in message.additional_kwargs:
            for fig in message.additional_kwargs['charts']:
                st.plotly_chart(fig, use_container_width=True)

# --- SIDEBAR CONTROLS ---
with st.sidebar:
    st.header("Control Room")
    if st.button("Reset Market Ledger", help="Wipes chat history and clears hallucinations"):
        st.session_state.messages = []
        st.session_state.pending_plots = []
        # In newer Streamlit versions, use st.rerun()
        st.rerun() 
    
    st.markdown("---")
    st.write("Current Strategy: **Big Bull / Harshad Mehta**")
    
# 3. Handling input
if user_query := st.chat_input('Kaunsa stock uthana hai?'):
    # Initialize session state keys if they don't exist
    if 'last_ticker' not in st.session_state:
        st.session_state.last_ticker = None
        
    st.session_state.price_df = None
    st.session_state.pending_plots = []
    
    st.session_state.messages.append(HumanMessage(content=user_query))
    with st.chat_message('user'):
        st.markdown(user_query)

    with st.chat_message('assistant'):
        try:
            # FIX 3: Ensure the wrapper is actually managing the response
            response = smart_invoke(agent_executor, user_query, system_prompt)
            
            # Extract content safely
            if isinstance(response['messages'][-1], AIMessage):
                full_response = response['messages'][-1].content
            else:
                full_response = str(response['messages'][-1])

            # If the response is STILL raw JSON, the fallback didn't trigger
            if '{"type": "function"' in full_response:
                 st.warning("Detected raw JSON output. Retrying with force...")
                 # Manual override if Groq ignores the tool call
                 response = agent_executor.invoke({"messages": [HumanMessage(content="Lala, don't give me JSON. CALL the tool 'get_stock_prices' now!")]})
                 full_response = response['messages'][-1].content

            st.markdown(full_response)
            
            # --- Rendering Plots ---
            captured_charts = []
            if "pending_plots" in st.session_state and st.session_state.pending_plots:
                captured_charts = st.session_state.pending_plots.copy()
                for fig in captured_charts:
                    st.plotly_chart(fig, use_container_width=True)
                st.session_state.pending_plots = []
            
            st.session_state.messages.append(
                AIMessage(content=full_response, additional_kwargs={"charts": captured_charts})
            )

        except Exception as e:
            st.error(f"Lala, market crash! {e}")