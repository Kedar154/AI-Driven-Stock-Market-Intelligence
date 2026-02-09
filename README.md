# Microsoft Stock Forecasting: Hybrid ARIMA-GARCH (Iteration 5)

## 📊 Overview

**Develop a forecasting engine that learns from historical market data to predict future trends.**

This project implements a hybrid **ARIMA(2,0,2) + GARCH(2,2)** model with walk-forward validation to predict Microsoft (MSFT) stock prices while accounting for volatility clustering.

---

## 🚀Introduction

This fifth iteration builds upon concepts of stationarity, log-returns, and rolling forecasts.

- **Model Selection:** Transitioned from Tesla (high volatility) to Microsoft for more stable modeling.
    
- **Architecture:** Hybrid ARIMA-GARCH chosen after testing SARIMA, ARCH, and AutoARIMA.
    
- **Resources:** [Interactive Colab Notebook](https://github.com/Kedar154/AI-Driven-Stock-Market-Intelligence/blob/main/ARIMA_GARCH_RF(5).ipynb)
    

---

## 🛠 Data Pipeline

### 1. Processing Steps

|**Step**|**Action**|**Purpose**|
|---|---|---|
|**Ingestion**|Load Kaggle MSFT Dataset|Historical baseline.|
|**Log Transform**|`np.log(prices)`|Stabilize variance.|
|**Differencing**|`log_return`|Stabilize mean (Achieve Stationarity).|
|**Interpolation**|Linear Interpolation|Fill holiday/missing gaps without "jumps".|

### 2. Validation

- **Stationarity:** Verified via **ADF Test** (Mean) and visual inspection (Variance).
    
- **Hyperparameters:** Determined $p=2, q=2$ via ACF/PACF analysis.
    
- **Seasonality:** Fourier Transform analysis confirmed no dominant seasonal frequencies beyond standard business cycles.
    

---

## 🧠 The Forecasting Engine

### ARIMA: The Directional Path

ARIMA predicts the next log-return ($r_t$) by analyzing past values and adjustment errors:

$$r_t = \text{Drift} + (\text{Past Returns} \times \text{Weights}) + (\text{Past Errors} \times \text{Weights})$$

### GARCH: The Volatility HUD

GARCH models the "wobble" or risk zone ($\sigma_t^2$) using volatility clustering:

$$\sigma_t^2 = \omega + \alpha(\epsilon_{t-1}^2) + \beta(\sigma_{t-1}^2)$$

### The Hybrid Transformation

1. **Forecast:** Get Expected Return ($\mu$) and Expected Risk ($\sigma$).
    
2. **Confidence:** Calculate 95% bounds ($1.96 \times \sigma$).
    
3. **Inversion:** Convert log-returns back to USD using $e^x$.
    

_Figure 1: Final Rolling Dashboard showing Historical vs. Forecasted prices with 95% GARCH Volatility Bands._

---

## 📈 Model Summary & Diagnostics

### Performance Metrics

> [!IMPORTANT]
> 
> **Predictive Core Metrics**
> 
> - **RMSE:** $1.81 (Standard Error)
>     
> - **MAE:** $1.52 (Mean Absolute Error)
>     
> - **MAPE:** 0.92% (Mean Absolute Percentage Error)
>     

#### **ARIMA Model: Trend & Noise Diagnostics**

This table evaluates how well the model captured the directional price patterns from 1,258 observations.

|**Diagnostic Metric**|**Value**|**Dashboard Significance**|
|---|---|---|
|**Ljung-Box / Prob(Q)**|`0.63`|**Pass.** Since $> 0.05$, the residuals are "White Noise," meaning no hidden patterns were missed.|
|**Heteroskedasticity**|`0.02`|**Pass.** Since $< 0.05$, it proves volatility is non-constant, justifying the need for the GARCH layer.|
|**Jarque-Bera**|`0.00`|Confirms "Fat Tails"; MSFT is prone to more extreme price jumps than a normal distribution predicts.|
|**Kurtosis**|`9.36`|Identifies that while MSFT is often calm, it is susceptible to sudden, violent price spikes.|
|**AIC / BIC**|`4407 / 4438`|Efficiency scores; lower values indicate the model is lean and not overfitted.|

---

#### **GARCH Model: Volatility & Stability**

This table defines the "Grey Band" behavior and ensures the risk calculations remain stable over time.

|**Stability Metric**|**Value**|**Technical Meaning**|
|---|---|---|
|**Stability Test**|$\alpha + \beta = 0.77$|The model is stable; the sum of reaction and persistence is less than 1.0.|
|**Mean Reversion**|**Verified**|Because the Stability Test is $< 1.0$, the "Grey Band" will settle back to average after a spike.|
|**Avg. Return ($\mu$)**|`0.0335`|The average daily return of residuals. A P-value of 0.381 shows the trend was correctly handled by ARIMA.|
|**Log-Likelihood**|`-2133.87`|Measures how well the GARCH math fits the "shakiness" of the stock.|
## 🌐 Deployment

### Streamlit Integration

The model is serialized via `.pkl` and hosted on a high-performance **Streamlit** instance for better inference.

**🔗 [Access Dashboard](https://ai-driven-stock-market-intelligence-msft-15418.streamlit.app/)**

### Dashboard Features

- **Feature Extraction:** numerical analysis of historical data using `.describe()`.
    
- **Technical Visuals:** Interactive PACF/ACF plots and the Neon Error Analysis we generated.



# PS1B — Natural Language Financial Intelligence System

_A Retrieval-Augmented Market Analysis Engine with Explainable Outputs_

---

## Project at a Glance

- **Core Goal:** Enable users to query stock market data in natural language and receive **context-aware, explainable financial insights**
    
- **Architecture:** Fixed-pipeline **RAG system** powered by `gemma-3-12b-it`, semantic search, and real-time market data
    
- **Output Style:** Interactive charts + structured analysis delivered via a **Harshad Mehta (“THE BIG BULL”) persona**

- - **Resources:** [Interactive Colab Notebook](https://colab.research.google.com/drive/1gGF30ymiIBLicGfdGj0_1vZ8-9uIjPuZ?authuser=1#scrollTo=Od6-WPSAoEGI)
    

---
## System Overview

This project implements a **natural language interface for financial datasets**, allowing users to ask questions like:

> _“Tell me about Tesla prices last week”_

The system:

- Retrieves historical OHLCV price data
    
- Scrapes and semantically filters relevant financial news
    
- Generates technical analysis and sentiment commentary
    
- Returns **interactive Plotly charts** and a **structured narrative explanation**
    

---

## Model & Architecture Decisions

### Large Language Model

- **Primary LLM:** `gemma-3-12b-it`
    
    - Selected for:
        
        - High **TPM**
            
        - Sufficient **context window**
            
        - Stable **RPD limits**
            
- **Embedding Model:**  
    `sentence-transformers/all-MiniLM-L6-v2`
    

#### Models Evaluated (and Rejected)

| Model                     | Issue Encountered        |
| ------------------------- | ------------------------ |
| `llama-3.3-70b-versatile` | Frequent **429 errors**  |
| `gemini-2.5-pro`          | TPM & RPD limits         |
| `gemini-2.5-flash`        | Rate-limit instability   |
| `gemini-2.0-flash`        | TPM & RPD limits         |
| `gemma-3-1b-it`           | Context window too small |

**Final Verdict:**  
`gemma-3-12b-it` provided the best balance between **capacity, stability, and reasoning depth**.

---

## Helper Functions

The system uses **four data-retrieval helpers** and **one visualization helper**.

---

### `get_ticker(query)`

- Extracts:
    
    - Stock ticker
        
    - Start date
        
    - End date
        
- Uses LLM-based parsing for robustness against unstructured user input
    
- Cleans and trims JSON artifacts from the LLM output
    
- Makes extracted values **globally accessible** for downstream functions
    

---

### `get_price(ticker, end_date)`

- Downloads OHLCV data via **YFinance**
    
- Performs:
    
    - Data cleaning
        
    - Index normalization
        
- Returns a **clean Pandas DataFrame**
    

---

### `get_news(ticker, start_date, end_date)`

- Scrapes articles from:
    
    - `finance.yahoo.com`
        
    - `marketwatch.com`
        
- Techniques used:
    
    - Fake browser configuration to bypass bot detection
        
    - `newspaper4k` for article extraction
        
    - Sleep timer to simulate human browsing
        
- Extracts:
    
    - Title
        
    - Link
        
    - Full article text
        

---

### `to_Db(news, query, ticker)`

- Performs semantic indexing and retrieval:
    
    - Chunks news into **600-word segments**
        
    - **100-word overlap**
        
- Vectorizes content and query
    
- Searches **ChromaDB** for nearest neighbors
    
- Returns **top 3 most relevant news snippets**
    
- Enables **context-aware sentiment analysis**
    

---

### `plotter(ticker, price_df)`

- Generates interactive **Plotly** charts:
    
    - Price vs Time
        
    - Volume vs Time
        
- Returns figures as a list for UI rendering
    

---

## Main Orchestration Pipeline (HM1992)

`HM1992` is the **central controller function**.

1. Invoke `get_ticker` to extract:
    
    - Ticker
        
    - Start date
        
    - End date
        
2. Sleep for **2 seconds** to avoid rate-limit errors
    
3. Fetch price data → `price_df`
    
4. Extract technical indicators:
    
    - RSI
        
    - Moving Averages
        
    - Volume signals
        
5. Shrink the dataframe to stay within LLM context limits
    
6. Fetch news → pass directly to `to_Db`
    
7. Generate interactive charts via `plotter`
    
8. Construct a **persona-driven augmented prompt**
    
9. Invoke LLM and return:
    
    - Structured analysis
        
    - Charts
        

---

### LLM Prompt Structure

The model is instructed to respond as  
**THE BIG BULL — Harshad Mehta**

```
🚀 THE BIG BULL HEADLINE
📊 THE TECHNICAL GAME
🗣️ MARKET SENTIMENT
🔍 THE BOTTOM LINE
💰 SHOULD YOU INVEST?
ℹ️ Sources
💀 DISCLAIMER
```

The response includes:

- Technical analysis
    
- News-driven sentiment
    
- Explicit Buy / Hold / Avoid verdict
    
- Mandatory closing phrase:
    
    > “Risk hai toh Ishq hai!”
    

---

## Why Simple RAG (Not Agents)

- Agentic RAG is best when:
    
    - Multiple decision paths exist
        
    - Tool selection is dynamic (e.g., travel booking bots)
        
- This project has:
    
    - A **fixed, deterministic pipeline**
        
    - No branching tool logic
        

> **Analogy:**  
> Buying a gaming PC just to use MS Word.

**Conclusion:**  
Simple RAG is **optimal, efficient, and justifiable** for this use case.

---

## Example Query & Output

**User Query:**

> _Tell me about Tesla prices last week_

---

### Generated Visualizations

![[Tesla Vol.png]]
![[Tesla Prices.png]]


---

## Price Trend

|**#**|**Date**|**Open**|**High**|**Low**|**Close**|**Volume**|
|--:|---|--:|--:|--:|--:|--:|
|**0**|2026-01-12|441.23|454.30|438.00|448.96|61,649,600|
|**1**|2026-01-13|450.20|451.81|443.95|447.20|53,719,200|
|**2**|2026-01-14|442.81|443.91|434.22|439.20|57,259,500|
|**3**|2026-01-15|441.13|445.36|437.65|438.57|49,465,800|
|**4**|2026-01-16|439.50|447.25|435.26|437.50|60,220,600|
|**5**|2026-01-20|429.36|430.73|417.44|419.25|63,187,300|
|**6**|2026-01-21|421.66|438.20|419.62|431.44|68,124,000|
|**7**|2026-01-22|435.16|449.50|432.63|449.36|71,546,700|
|**8**|2026-01-23|447.43|452.43|444.04|449.06|56,771,400|
|**9**|2026-01-26|445.00|445.04|434.28|435.20|49,397,400|
|**10**|2026-01-27|437.41|437.52|430.69|430.90|37,733,100|
|**11**|2026-01-28|431.91|438.26|430.10|431.46|54,857,400|
|**12**|2026-01-29|437.80|440.23|414.62|416.56|81,686,100|
|**13**|2026-01-30|425.35|439.88|422.70|430.41|82,626,100|
|**14**|2026-02-02|421.29|427.15|414.50|421.81|58,739,500|

**🚀 THE BIG BULL HEADLINE**

Tesla Takes a Hit: Robotaxis Can't Offset Earnings Blow! 

**📊 THE TECHNICAL GAME**

Look at this, Lala! The stock has been wrestling with itself. We saw a strong push on January 27th, volume hitting a healthy 37 lakh shares, but then it faltered. The next day, the volume exploded to over 54 lakh shares, showing serious interest, but the close barely held on. Then came the big drop on January 29th – a massive 81 lakh shares changing hands as the price plummeted. That was a panic sell, Bhai! The subsequent bounce on January 30th, with 82 lakh shares traded, suggests some support, but the close at 430.41 is concerning. Now, on February 2nd, we're seeing a further dip, volume at 58 lakh shares, and the price struggling. The RSI is clearly running a fever, indicating overbought conditions previously, and now a potential weakness. The moving averages are tangled, showing indecision. This isn't a breakout, Lala; it's a trap! The Operator is testing the waters, seeing who’s still holding on. 

**🗣️ MARKET SENTIMENT** 

The Operator has thrown a curveball, Bhai. Tesla’s Q3 earnings missed expectations – a 40% drop in operating profit! The news is out, and the market is reacting sharply. The stock is down over 5%, and the Bulls are licking their wounds. But Musk, that clever chap, is trying to distract everyone with these Robotaxi updates. He’s talking about no safety drivers in Austin by year-end and tests in multiple cities. It’s a shiny object, Lala, meant to keep the small investors interested while the big players reassess. The street is buzzing with chatter about the Robotaxi timeline, but the earnings miss is the elephant in the room. This is a classic case of trying to sell hope while the fundamentals are shaky. Dalal Street loves a good story, but it also respects the bottom line. 

**🔍 THE BOTTOM LINE** 

The real Khel here, Lala, is about perception versus reality. Tesla’s growth story is facing headwinds – post-EV tax credit environment, slowing demand, and now disappointing earnings. Musk’s Robotaxi vision is exciting, but it’s years away from generating significant revenue. The market is realizing this, and the stock is paying the price. The volume spikes are telling a story of panic and uncertainty, not of sustained bullish momentum. The Operator is trying to control the narrative, but the numbers don't lie. 

**💰 SHOULD YOU INVEST?**

Avoid, Bhai. Absolutely avoid. This isn't the time to be chasing a dream. The Lakshman Rekha is around 475, and we're getting dangerously close. Let the dust settle, let the market digest this news, and let someone else be the fall guy. There are better opportunities out there, Lala. Don't get caught in this Khel. 

**ℹ️ Sources**

[https://finance.yahoo.com/quote/TSLA/?fr=sycsrp_catchall](https://finance.yahoo.com/quote/TSLA/?fr=sycsrp_catchall) [https://finance.yahoo.com/news/tesla-q3-earnings-miss-analyst-expectations-stock-drops-despite-robotaxi-expansion-plans-133818343.html?fr=sycsrp_catchall](https://finance.yahoo.com/news/tesla-q3-earnings-miss-analyst-expectations-stock-drops-despite-robotaxi-expansion-plans-133818343.html?fr=sycsrp_catchall) 


**💀 DISCLAIMER** 

Lala, market mein sabse bada risk toh vishwaas hota hai. Ab yeh system ek 18 saal ke bacche ne banaya hai. Agar tum iske kehne par apni poori zindagi ki kamai daon par laga rahe ho... toh maanna padega, tum mujhse bhi bade khiladi nikle! Par yaad rakhna, yahan bina soche koodne waale aksar doob jaate hain. 
AS ALWAYS: Risk hai to Ishq hai
