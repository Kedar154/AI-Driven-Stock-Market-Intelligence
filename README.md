# PS1A : **Develop a forecasting engine that learns from historical market data to predict future trends for a specific period**

# Microsoft Stock Forecasting: Hybrid ARIMA-GARCH (Iteration 5)

## 1. Project Overview & Evolution

This project implements a hybrid forecasting engine that merges linear time-series modeling with conditional heteroskedasticity.

- **The Iterative Path:** As the **fifth iteration**, this version introduces critical fixes for data gaps (holidays/weekends) and optimizes the hand-off between the mean (ARIMA) and volatility (GARCH) components.
    
- **Objective:** To generate a 7-day "static" prediction of Microsoft stock prices using a rolling-window training approach.
    

## 2. Advanced Data Preprocessing

Financial data is notoriously "noisy" and non-stationary. This iteration uses a rigorous preprocessing pipeline:

- **Log Transformation:** Raw prices are converted to **Log Prices** and **Log Returns** ($100 \times \text{diff of log prices}$). This transformation linearizes exponential growth and stabilizes variance over time.
    
- **Business Day Alignment:** The dataset is reindexed to a standard Business Day ('B') frequency.
    
- **Linear Interpolation:** To ensure the rolling forecast loop does not break due to missing values (e.g., public holidays), missing prices are filled using linear interpolation between known points.
    
- **Recalculation:** Log returns are recalculated _after_ interpolation to maintain a mathematically consistent series.
    

## 3. Model Diagnostics: The Role of ACF & PACF

Selecting hyperparameters $(p, d, q)$ for ARIMA is guided by visual and statistical diagnostics:

- **ACF (Autocorrelation Function):** Measures the correlation of the series with its own past. In this notebook, it is used to identify the **Moving Average (q)** order. A tapering ACF suggests an AR process, while a sharp cutoff suggests a specific MA order.
    
- **PACF (Partial Autocorrelation Function):** Measures the direct correlation between a value and its lag, stripping away the influence of intervening observations. This is the primary tool for identifying the **Autoregressive (p)** order.
    

## 4. The Hybrid ARIMA-GARCH Engine

A single model rarely captures the complexity of stock markets. This iteration uses a two-stage approach:

### Stage 1: ARIMA (Linear Mean)

The ARIMA model predicts the "expected" return by looking at past returns and past prediction errors. It handles the linear structure of the data.

### Stage 2: GARCH (Volatility Modeling)

Standard models assume "homoskedasticity" (constant variance), but stocks exhibit "volatility clustering". The GARCH model is applied to the residuals (errors) of the ARIMA model.

- **GARCH Hyperparameter (p):** Lagged variance terms. It captures how long-lasting a volatility shock is in the market.
    
- **GARCH Hyperparameter (q):** Lagged squared residuals. It captures the immediate impact of recent "shocks" or news.
    

## 5. Implementation: The 7-Day Rolling Forecast

Accuracy in financial modeling degrades rapidly with time. To counter this, the notebook uses a **Rolling Forecast** strategy:

1. **Step-by-Step:** The model predicts exactly one day ahead ($T+1$).
    
2. **Window Update:** This predicted value is then treated as "known" data, appended to the training set, and the model is re-fitted.
    
3. **Iteration:** This process repeats 7 times, creating a chain of 1-day-ahead forecasts that form the final 7-day prediction.
    

## 6. Visual Output & Dashboard

The final stage of the notebook converts predicted log returns back into standard dollar prices. These are visualized in a Plotly-based dashboard that compares historical trends against the projected 7-day path, providing a clear visual representation of the model's expected market movement.

## 7. The finished **Webapp** : 

'https://ai-driven-stock-market-intelligence-msft-15418.streamlit.app/'



# PS1B:  **Create a natural language interface that allows users to query the dataset and understand the context behind the numbers.**


# The Big Bull: Agentic RAG with Llama-3.3-70b

This project implements an **Agentic Retrieval-Augmented Generation (RAG)** system designed for aggressive stock market analysis. Using the reasoning power of `llama-3.3-70b-versatile`, the agent autonomously orchestrates technical data fetching, semantic news filtering, and neon-themed visualization.

## Core Intelligence

- **LLM:** `llama-3.3-70b-versatile` (via Groq/LangChain).
    
- **Persona:** **Harshad Mehta (The Big Bull)**. The model is system-prompted to deliver findings with the charisma, terminology, and "risk-taking" attitude of the 1992 market icon.
    
- **Orchestration:** Built using the **LangChain Tool-Calling Agent** framework, allowing the model to select tools based on logical dependencies (e.g., getting dates before prices).
    

---

## 🛠️ The Agentic Toolbelt & Mechanism

The agent logic follows a strict sequential-to-recursive workflow:

### 1. Intent & Date Resolution (`extract_date_ticker`)

- **Role:** The entry point for any natural language query.
    
- **Mechanism:** The agent passes the user query to this tool to extract the **Ticker** (e.g., `NVDA`) and resolve relative dates (e.g., "last week") into an absolute **Time Frame**.
    
- **Internal Logic:** It calculates a 5-day lookback window automatically to provide context for the start of the analysis.
    

### 2. Market Data Ingestion (`get_prices`)

- **Role:** Technical data retrieval.
    
- **Action:** Using the ticker and time frame from Step 1, it downloads **OHLCV** (Open, High, Low, Close, Volume) data via `yfinance`.
    
- **Mechanism:** This data is stored in a global state (`price_df`) and formatted into a Markdown table for the LLM's immediate technical analysis.
    

### 3. Visual Execution (`plotter`)

- **Role:** Graphical representation.
    
- **Dependency:** Automatically triggered once `get_prices` populates the data.
    
- **Mechanism:** Renders **Neon-Themed Plotly** charts (Cyan for Price, Magenta for Volume) directly in the UI.
    

### 4. Semantic RAG Pipeline (`get_news` + `Db`)

- **Role:** News filtering and ranking.
    
- **Mechanism:** 1. **Fetch:** `get_news` pulls the latest headlines from DuckDuckGo. 2. **Store:** These findings are sent to the `Db` (Vector Database). 3. **Rank:** `Db` uses semantic embeddings to rank the news chunks against the user's specific query. 4. **Top-K Retrieval:** Only the most relevant, "insider-level" snippets are fed back to the LLM.
    

---

## 🔄 The "Mehta" Workflow

1. **Query Analysis:** "Lala, what's happening with NVIDIA lately?"
    
2. **Extraction:** Agent extracts `NVDA` and the recent date range.
    
3. **Technical Build:** Agent fetches prices and immediately triggers the neon plots.
    
4. **Sentiment Search:** Agent scrapes web news, embeds it into the **ChromaDB**, and pulls the most relevant "market rumors" or headlines.
    
5. **Synthesis:** `Llama-3.3` processes the price table + relevant news findings and generates a report using the **Harshad Mehta** structure:
    
    - **The Big Bull Headline**
        
    - **The Technical Game** (Deep dive into the 15-day charts)
        
    - **Market Sentiment** (Ranked findings from the DB)
        
    - **The Bottom Line** (Investment verdict)
        
    - **Sources** (Verbatim search links)
        

---

## 🚀 Key Technical Specifications

- **Model:** Meta Llama-3.3-70b-Versatile
    
- **Search:** DuckDuckGo Search API
    
- **Database:** ChromaDB (Vector Storage)
    
- **Visualization:** Plotly Express (Dark/Neon Template)
    
- **Closing Catchphrase:** _"Risk hai toh Ishq hai!"_