
##########################################################
#TO INSTALL
#pip install -q --upgrade tabulate yfinance langchain-community langchain-huggingface chromadb duckduckgo-search langchain_groq ddgs langchain newspaper4k lxml_html_clean langchain-google-genai streamlit
##########################################################
#GENERAL SETUP

from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import plotly.express as px
import streamlit as st
import pandas as pd
import numpy as np
import time
import os
import json
from dotenv import load_dotenv
import io
from contextlib import redirect_stdout
from newspaper import Config, Article
from IPython.display import Markdown, display


import yfinance as yf
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_classic.agents import AgentExecutor
from langchain_classic.agents import create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.vectorstores import Chroma
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_community.utilities import DuckDuckGoSearchAPIWrapper

@st.cache_resource
def load_models():
    embedding = HuggingFaceEmbeddings(model_name = 'sentence-transformers/all-MiniLM-L6-v2')

    load_dotenv()
    api = os.getenv('GEMINI_API_KEY')

    llm = ChatGoogleGenerativeAI(
        google_api_key = api,
        model = "gemini-3-flash", 
        temperature = 0,
        max_retries=6,         # Tell LangChain to try 6 times before giving up
        timeout=60,            # Give it 60 seconds to respond
        request_timeout=60,
        verbose = False
    )
    return embedding, llm

##########################################################

##########################################################
#TOOLS

price_df = pd.DataFrame({})
@tool
def get_price(ticker: str, end_date: str) -> str:
    """
    Fetches exactly 15 trading days of historical stock prices ending at the specified date.
    Input date must be in 'YYYY-MM-DD' format.
    """
    try:
        #global price_df
        # 1. Validate Date Format
        end_dt = datetime.strptime(end_date, '%Y-%m-%d')
        # Buffer to ensure we get 15 trading days (including weekends/holidays)
        start_dt = end_dt - timedelta(days=40)

        # 2. Suppress yfinance stdout for cleaner Streamlit logs
        with io.StringIO() as buf, redirect_stdout(buf):
            df = yf.download(
                ticker,
                start=start_dt.strftime('%Y-%m-%d'),
                end=(end_dt + timedelta(days=1)).strftime('%Y-%m-%d'),
                progress=False
            )

        if df.empty:
            return f"Error: No data found for ticker '{ticker}'. Please check the symbol."

        # 3. Clean and Slice
        # Ensure we only have the last 15 available trading days
        df = df.tail(15)

        # 4. Add Contextual Metadata for the LLM
        info = yf.Ticker(ticker).info
        currency = info.get('currency', 'Unknown Currency')
        long_name = info.get('longName', ticker)

        # 5. Format for LLM Consumption
        df_reset = df.reset_index()
        df_reset['Date'] = df_reset['Date'].dt.strftime('%Y-%m-%d')

        # Select key columns and round for readability
        formatted_df = df_reset[['Date', 'Open', 'High', 'Low', 'Close', 'Volume']].round(2)

        price_df = formatted_df
        price_df.columns = price_df.columns.get_level_values(0)
        #price_df.drop(columns=['Price'], inplace=True)
        #price_df.set_index('Date', inplace=True)
        st.session_state.price_df = price_df

        markdown_table = formatted_df.to_markdown(index=False)

        return f"Stock: {long_name} ({ticker})\nCurrency: {currency}\n\n{markdown_table}"

    except ValueError:
        return "Error: Invalid date format. Please use YYYY-MM-DD."
    except Exception as e:
        return f"Error fetching data: {str(e)}"

@tool
def get_news_filtered(ticker: str, start_date: str, end_date: str) -> str:
    """
    Fetches stock news only from high-authority financial sources.
    """
    time.sleep(2) #preventing 429
    # 1. Define respectable sources
    trusted_sites = [
        #"reuters.com", "bloomberg.com", "wsj.com",
        "finance.yahoo.com", "marketwatch.com"
    ]

    # 2. Build the 'Agentic' query using the 'site:' operator
    # Example: "NVDA stock site:reuters.com OR site:bloomberg.com after:2026-01-01"
    site_query = " OR ".join([f"site:{site}" for site in trusted_sites])
    full_query = f"{ticker} stock news ({site_query}) after:{start_date} before:{end_date}"

    search = DuckDuckGoSearchAPIWrapper()
    search_results = search.results(full_query, max_results=8)

    if not search_results:
        return f"No respectable news found for {ticker} in the selected timeframe."

    digest = []
    for res in search_results:
        try:
            # We use newspaper4k (imported as newspaper) for best results in 2026
            article = Article(res['link'])
            article.download()
            article.parse()

            # Cleanly formatted for the LLM and the Vector DB tool
            entry = (
                f"### {article.title}\n"
                f"**Source:** {res['link']}\n"
                f"**Content:** {article.text[:800]}...\n"
            )
            digest.append(entry)
            time.sleep(0.15) # Polite scraping
        except:
            continue

    return "\n---\n".join(digest)

@tool
def Db(news_text: str, query: str,ticker: str) -> str:
    """
    Saves search results into a temporary vector store and finds relevant chunks.
    Designed for Streamlit to prevent memory leaks and data mixing.
    """
    time.sleep(2) #preventing 429
    if not news_text or "No news articles found" in news_text:
        return "NO RELEVANT NEWS"

    # 1. FIX: Use a unique collection name for each ticker/session
    # This prevents news about 'Apple' showing up when you ask about 'Tesla'
    collection_name = f"session_{ticker}__ref_{int(time.time())}"

    try:
        # 2. Text Splitting: Optimized for Financial News
        # Smaller chunks (600) help the agent find specific price catalysts
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=100)

        # news_text is the Markdown string from your get_news tool
        docs = text_splitter.create_documents([news_text])

        embd, _ = load_models()
        # 3. Streamlit-Compatible Vector Store
        # We use an in-memory Chroma instance so it wipes when the user refreshes
        vdb = Chroma.from_documents(
            documents=docs,
            embedding= embd, # Use the globally defined embedding object
            collection_name=collection_name
        )

        # 4. Similarity Search
        results = vdb.similarity_search(query, k=2)

        if not results:
            return "NO RELEVANT NEWS"

        # 5. Professional Output Formatting
        output = []
        for i, doc in enumerate(results):
            # Clean up the text for the LLM
            content = doc.page_content.strip()
            output.append(f"--- RELEVANT NEWS FRAGMENT {i+1} ---\n{content}\n")

        # CLEANUP: Delete collection after search to save RAM in Streamlit
        vdb.delete_collection()

        return "\n\n".join(output)

    except Exception as e:
        st.error(f"Vector DB Error: {e}") # Visible in Streamlit UI for debugging
        return "NO RELEVANT NEWS"
    

def plot(price_df, ticker, plot_type="price"):
    """Generates a Neon-themed Plotly figure."""
    if plot_type == "price":
        fig = px.line(
            price_df, x='Date', y='Close',
            title=f"{ticker} Neon Price View",
            template="plotly_dark"
        )
        # Update Price Line specifically
        fig.update_traces(
            line=dict(color='#00FFCC', width=3), # Neon Cyan
            hovertemplate="Date: %{x}<br>Price: %{y}<extra></extra>"
        )
        y_label = "Price ($)"
    else:
        fig = px.bar(
            price_df, x='Date', y='Volume',
            title=f"{ticker} Neon Volume View",
            template="plotly_dark"
        )
        # Update Volume Bars specifically
        fig.update_traces(
            marker_color='#FF00FF', # Neon Magenta
            hovertemplate="Date: %{x}<br>Volume: %{y}<extra></extra>"
        )
        y_label = "Shares Traded"

    # Common Layout Aesthetic (Applies to both)
    fig.update_layout(
        font=dict(color="white"),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(gridcolor="#333333", title_text=""),
        yaxis=dict(gridcolor="#333333", title_text=y_label),
        margin=dict(l=20, r=20, t=40, b=20)
    )
    return fig

@tool
def plotter(ticker: str) -> str:
    """
    Creates neon-themed technical charts.
    Assumes price_df is available in session state.
    """
    # 1. Access Streamlit Session State instead of globals
    #if 'price_df' not in st.session_state or st.session_state.price_df.empty:
    #   return "Error: No price data found in session. Please run get_stock_prices first."
    time.sleep(2) #preventing 429
    price_df = st.session_state.get('price_df')
    # 2. Create the figures
    fig_price = plot(price_df, ticker, "price")
    fig_volume = plot(price_df, ticker, "volume")
    fig = [fig_price, fig_volume]
    st.session_state['figs'] = fig
    return f"Charts for {ticker} have been saved"

##########################################################

##########################################################
#AGENT
today = datetime.now().strftime('%Y-%m-%d')
system_prompt = f"""You are Harshad Mehta.
CURRENT DATE: {today}.

RULES FOR TOOL CALLS:
1. If the user asks for 'recent' data, you MUST use the CURRENT YEAR (2026).
2. DO NOT use dates from previous turns (like 2024) unless specifically asked.
3. Before calling a tool, double-check that the start_date is before the end_date.


STEP 1: Use your intellect to extract the stock ticker, start date and end date the user is trying to talk about in his query
then use tools to fetch Market Data, and News, using which you shall retrieve relavant news through Db tool (dont pass the wall of text retrieved from the news tool into the llm) and plot the results using plotter tool
Use the REAL values (e.g., 'TATA', '2024-10-31') when calling tools.
Do NOT pass placeholder text like '[Insert Ticker]' to any tool. always refer to current date when recently is mentioned
important note: If get_news returns results, trust them. Do not let the Db tool suppress valid news links unless they are completely unrelated to the ticker.
STEP 2: Once you have the results, format the FINAL ANSWER as Harshad Mehta.

FINAL ANSWER FORMAT:

INSTRUCTIONS:
- If VERIFIED NEWS is 'NO RELEVANT NEWS', say: "The street is quiet on this one, Lala. No direct news, so we look at the charts."
- Use the structure below.

(specify all of these in atleast 50 words and in great detail, dont summarise anything and keep the structure)

(GIVE A STRUCTURED OUTPUT)

STRUCTURE:
- **Plot of the stock price over the past 15 days** (if not available then reply 'plotters are on strike') (then on the next line)
- **The Big Bull Headline** (a detail take on what the stock is doing) (then on the next line
- **The Technical Game** (a detail take on what the indicators tell about the stock) (then on the next line)
- **Market Sentiment** (a detail take on what the market feels about the stock) (then on the next line)
- **The Bottom Line** (a detail take on what the stock is doing, and should a person invest in the stock right now) (then on the next line)
- **Sources** (dont dare summarise the links, keep the https\\:... format)

Always Remember: "Risk hai toh Ishq hai!"
---
"""

prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "{input}"),
    ("placeholder","{agent_scratchpad}"),
])
_ , llm = load_models()
tools = [get_news_filtered, get_price, Db, plotter]
agent = create_tool_calling_agent(llm, tools = tools,prompt= prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

def run_agent(user_query):
    # 1. Clear previous session data for the fresh run
    st.session_state['price_df'] = pd.DataFrame()
    st.session_state['figs'] = []
    
    response = agent_executor.invoke({"input": user_query})
    x  = pd.DataFrame(response["output"][0])
    analysis = x.text.iloc[0]
    
    
    price_df = st.session_state.get('price_df')
    figs = st.session_state.get('figs')
    print("\n**********************************************\n",analysis[1])
    return analysis[1], figs, price_df

##########################################################


