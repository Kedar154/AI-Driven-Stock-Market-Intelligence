# Loading Dataset
# pip install opendatasets
#pip install statsmodels
# pip install arch

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from arch import arch_model
from statsmodels.tsa.arima.model import ARIMA
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from statsmodels.tsa.stattools import acf, pacf
import plotly.graph_objects as go
import plotly.express as px
import pickle #loading model
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error, mean_absolute_error
import streamlit as st


##############################################
# LOADING THE DATA


data = pd.read_csv('microsoft-stock-time-series-analysis/Microsoft_Stock.csv')
stock = pd.DataFrame()
#using close and date
stock['prices'] = data['Close']
stock['date'] = pd.to_datetime(data['Date']).dt.date
stock.set_index('date', inplace=True)
stock.sort_index(inplace=True)
# finding log_prices and log_return
stock['log_prices'] = np.log(stock['prices'])
stock.dropna(inplace = True)
# 2. Reindex to Business Days ('B')
# This adds the missing weekend/holiday rows as NaN
stock = stock.asfreq('B')

# 3. Fill the gaps
# Option A: Your idea (Arithmetic mean of neighbors)
stock['prices'] = stock['prices'].interpolate(method='linear')
stock['log_prices'] = stock['log_prices'].interpolate(method='linear')
# Option B: Financial standard (Carry last price forward)
# stock['prices'] = stock['prices'].ffill()

# 4. RE-CALCULATE Log Returns after filling
# You MUST do this because NaN prices create NaN returns
stock['log_return'] = 100 * np.log(stock['prices'] / stock['prices'].shift(1))
stock = stock.dropna() # Remove the very first row created by the shift

print(stock.head())


##############################################
st.title('MSFT STOCK PREDICTION')
st.subheader('Data from Apr-2015 to Apr-2021')
st.write(data.describe())
st.subheader('Extracted Features')
st.write(stock.describe())

##############################################

##############################################
# --- Match Theme Palette ---
GREEN_NEON = "#39FF14"
STREAMLIT_DARK = "#0e1117"  # The background color in your screenshot
TEXT_GREY = "#808495"       # Subdued grey for labels/grid
PURE_WHITE = "#FFFFFF"

# 1. Create Figure
fig = px.line(
    stock, 
    x=stock.index, 
    y='prices',
    # Title matched to your screenshot style (Simple white bold)
    title='<b>MSFT STOCK PRICE</b>',
    template='plotly_dark'
)

# 2. Style the Main Price Line
fig.update_traces(
    line=dict(color=GREEN_NEON, width=2.5), # Slightly thinner for a cleaner look
    name='Actual Price'
)

# 3. Style Axis, Legend, and Labels to match the screenshot
fig.update_layout(
    paper_bgcolor=STREAMLIT_DARK,
    plot_bgcolor=STREAMLIT_DARK,
    
    # Title Styling
    title_font=dict(size=26, family="Sans-serif", color=PURE_WHITE),
    
    # Legend Styling
    legend=dict(
        font=dict(color=PURE_WHITE),
        bgcolor='rgba(0,0,0,0)', # Transparent background
        orientation="h",        # Horizontal legend looks cleaner in web apps
        yanchor="bottom",
        y=1.02,
        xanchor="right",
        x=1
    ),
    
    # Axis Styling (Clean white text, subtle grid)
    xaxis=dict(
        title=dict(text="TIMELINE", font=dict(color=PURE_WHITE, size=14)),
        tickfont=dict(color=TEXT_GREY),
        gridcolor='#262730',   # Very subtle grid line
        showgrid=True,
        linecolor='#31333F'    # Border color
    ),
    yaxis=dict(
        title=dict(text="PRICE VALUE", font=dict(color=PURE_WHITE, size=14)),
        tickfont=dict(color=TEXT_GREY),
        gridcolor='#262730',
        showgrid=True,
        linecolor='#31333F'
    ),
    margin=dict(t=80) # Space for the title
)

# 4. Remove the mirror lines for a flatter, modern look
fig.update_xaxes(showline=True, linewidth=1, linecolor='#31333F', mirror=False)
fig.update_yaxes(showline=True, linewidth=1, linecolor='#31333F', mirror=False)

# Display in Streamlit
st.plotly_chart(fig, use_container_width=True)


##############################################

##############################################

#EDA:

#PACF + ACF
acf_values = acf(stock.log_return, nlags=40)

pacf_values = pacf(stock.log_return, nlags=40)

lags = np.arange(len(acf_values))



# Calculate confidence interval bounds

conf_int_upper = 1.96 / np.sqrt(len(stock.log_return))

conf_int_lower = -1.96 / np.sqrt(len(stock.log_return))



# X values for shading - extending across all lags

x_shade = list(lags) + list(lags[::-1])

y_shade_upper = [conf_int_upper] * len(lags)

y_shade_lower = [conf_int_lower] * len(lags)

y_shade_combined = y_shade_upper + y_shade_lower[::-1] # upper then lower reversed

# --- Theme Constants ---
STREAMLIT_DARK = "#0e1117"
TEXT_GREY = "#808495"
GRID_GREY = "#262730"
BORDER_GREY = "#31333F"
PURE_WHITE = "#FFFFFF"
CYAN_NEON = "#00FFFF"
GREEN_NEON = "#39FF14"

# ... (ACF/PACF calculation code remains same) ...

# Create subplots
fig = make_subplots(rows=2, cols=1,
                    vertical_spacing=0.15,
                    subplot_titles=('<b>Autocorrelation (ACF) - MSFT Log Returns</b>',
                                    '<b>Partial Autocorrelation (PACF) - MSFT Log Returns</b>'))

# 1. ACF Plot
# Shaded Region (Matched to Screenshot Table Grey)
fig.add_trace(go.Scatter(
    x=x_shade, y=y_shade_combined, fill='toself',
    fillcolor='rgba(49, 51, 63, 0.4)', # Match Streamlit widget hover color
    line=dict(width=0), name='95% Confidence Interval', showlegend=True
), row=1, col=1)

fig.add_trace(go.Bar(x=lags, y=acf_values, name='ACF', marker_color=CYAN_NEON), row=1, col=1)

# Subtle Grey Dash instead of bright red
fig.add_trace(go.Scatter(x=[lags[0], lags[-1]], y=[conf_int_upper, conf_int_upper],
                         mode='lines', line=dict(color=TEXT_GREY, width=1, dash='dot'), showlegend=False), row=1, col=1)
fig.add_trace(go.Scatter(x=[lags[0], lags[-1]], y=[conf_int_lower, conf_int_lower],
                         mode='lines', line=dict(color=TEXT_GREY, width=1, dash='dot'), showlegend=False), row=1, col=1)

# 2. PACF Plot
fig.add_trace(go.Scatter(
    x=x_shade, y=y_shade_combined, fill='toself',
    fillcolor='rgba(49, 51, 63, 0.4)', line=dict(width=0), showlegend=False
), row=2, col=1)

fig.add_trace(go.Bar(x=lags, y=pacf_values, name='PACF', marker_color=GREEN_NEON), row=2, col=1)

fig.add_trace(go.Scatter(x=[lags[0], lags[-1]], y=[conf_int_upper, conf_int_upper],
                         mode='lines', line=dict(color=TEXT_GREY, width=1, dash='dot'), showlegend=False), row=2, col=1)
fig.add_trace(go.Scatter(x=[lags[0], lags[-1]], y=[conf_int_lower, conf_int_lower],
                         mode='lines', line=dict(color=TEXT_GREY, width=1, dash='dot'), showlegend=False), row=2, col=1)

# --- Layout Styling to match Image ---
fig.update_layout(
    template='plotly_dark',
    paper_bgcolor=STREAMLIT_DARK,
    plot_bgcolor=STREAMLIT_DARK,
    height=800,
    title_font=dict(size=22, family="Sans-serif", color=PURE_WHITE),
    legend=dict(font=dict(color=PURE_WHITE), bgcolor='rgba(0,0,0,0)'),
    margin=dict(t=100, b=50, l=50, r=50)
)
fig.update_layout(
    template='plotly_dark',
    paper_bgcolor=STREAMLIT_DARK,
    plot_bgcolor=STREAMLIT_DARK,
    height=800,
    # FIX: Explicitly set title to empty string if you want it blank, 
    # or move the subheader text here.
    title_text="", 
    title_font=dict(size=22, family="Sans-serif", color=PURE_WHITE),
    legend=dict(font=dict(color=PURE_WHITE), bgcolor='rgba(0,0,0,0)'),
    margin=dict(t=50, b=50, l=50, r=50) # Reduced top margin (t) since title is gone
)
# Style Axes
for i in [1, 2]:
    fig.update_xaxes(
        title_text='Lags',
        title_font=dict(color=PURE_WHITE),
        tickfont=dict(color=TEXT_GREY),
        gridcolor=GRID_GREY,
        linecolor=BORDER_GREY,
        row=i, col=1
    )
    fig.update_yaxes(
        title_font=dict(color=PURE_WHITE),
        tickfont=dict(color=TEXT_GREY),
        gridcolor=GRID_GREY,
        linecolor=BORDER_GREY,
        zerolinecolor=TEXT_GREY, # Highlight the zero line
        row=i, col=1
    )

st.subheader('PACF AND ACF PLOTS')
st.plotly_chart(fig, use_container_width=True)


##############################################

##############################################

# MODEL

## TTS
tr = stock.iloc[:int(0.8*len(stock))]
val = stock.iloc[int(0.8*len(stock)):int(.85*len(stock))]
test = stock.iloc[int(.85*len(stock)):]

##ARIMA + GARCH + RF
forecast_days = 7
input_df = tr.copy()
output_df = val.head(forecast_days).copy() # Forces a strict 7-day window

# Initialize lists for the rolling process
history_returns = input_df['log_return'].tolist()
current_price = input_df['prices'].iloc[-1]

rolling_preds = []
upper_bounds = []
lower_bounds = []


# Load the ARIMA model
with open('ARIMA_MSFT_2_0_2.pkl', 'rb') as pkl_file:
    loaded_arima_model = pickle.load(pkl_file)

print("ARIMA model loaded successfully.")
with open('GARCH_MSFT_1_0_1.pkl', 'rb') as pkl_file:
    loaded_garch_model = pickle.load(pkl_file)

print("GARCH model loaded successfully.")
# You can now use loaded_arima_model for forecasting or analysis
# For example, to print a summary of the loaded model:
# print(loaded_arima_model.summary())



#prediction


# Assuming loaded_arima_model and loaded_garch_model are already loaded
# and history_returns contains the data the models were originally trained on.


for t in range(len(output_df)):
    # 1. Update ARIMA using the result object's .apply() method
    # history_returns must be a list or Series of the returns updated each loop
    arima_updated = loaded_arima_model.apply(history_returns)
    
    # Get the 1-step forecast for the mean
    arima_forecast = arima_updated.get_forecast(steps=1)
    mu_ret = arima_forecast.predicted_mean[0]
    
    # 2. Extract residuals to pass into GARCH
    current_resid = arima_updated.resid 

    # --- THE GARCH FIX SECTION ---
    
    # Step A: Create the "Skeleton" model using the new residuals
    # This variable MUST be defined before you can use it
    model_definition = arch_model(current_resid, vol='Garch', p=1, q=1, dist='normal', rescale=False)
    
    # Step B: Inject your PRETRAINED parameters into this skeleton
    # loaded_garch_model is your saved ARCHModelResult object
    garch_fixed_result = model_definition.fix(loaded_garch_model.params)
    
    # Step C: Forecast Volatility
    g_forecast = garch_fixed_result.forecast(horizon=1, reindex=False)
    sigma = np.sqrt(g_forecast.variance.values[-1, 0])

    # 3. CONVERT TO PRICE
    pred_p = current_price * np.exp(mu_ret / 100)
    up_p = current_price * np.exp((mu_ret + 1.96 * sigma) / 100)
    lo_p = current_price * np.exp((mu_ret - 1.96 * sigma) / 100)

    # Store results
    rolling_preds.append(pred_p)
    upper_bounds.append(up_p)
    lower_bounds.append(lo_p)

    # 4. UPDATE FOR NEXT DAY
    actual_return = output_df['log_return'].iloc[t]
    current_price = output_df['prices'].iloc[t]
    history_returns.append(actual_return)
    
    print(f"Day {t+1} Forecast: {pred_p:.2f} | Actual: {current_price:.2f}")
    
    
#metrics 

rmse = np.sqrt(mean_squared_error(output_df['prices'], rolling_preds))
mape = mean_absolute_percentage_error(output_df['prices'], rolling_preds)
mae = mean_absolute_error(output_df['prices'], rolling_preds)

print(f"\n--- MODEL PERFORMANCE ---")
print(f"RMSE: ${rmse:.2f}")
print(f"MAE: ${mae:.2f}")
print(f"MAPE: {mape:.2%}") # Lower is better


##############################################

##############################################

#plotting


# 1. Prepare Historical Data (Last 30 days of training)
hist_slice = input_df.tail(30)

# 2. Add the "Bridge" point (last point of history) to the forecast arrays
# This ensures the red line and grey band touch the blue historical line
plot_idx = [hist_slice.index[-1]] + list(output_df.index)
plot_preds = [hist_slice['prices'].iloc[-1]] + rolling_preds
plot_upper = [hist_slice['prices'].iloc[-1]] + upper_bounds
plot_lower = [hist_slice['prices'].iloc[-1]] + lower_bounds


fig = go.Figure()

# A. Plot Historical Training Data (Dashed Blue)
fig.add_trace(go.Scatter(
    x=hist_slice.index, y=hist_slice['prices'],
    mode='lines', name='Historical (30d)',
    line=dict(color='cyan', width=2)
))

# B. Plot Actual Prices for the 7-day period (Markers)
fig.add_trace(go.Scatter(
    x=output_df.index, y=output_df['prices'],
    mode='markers+lines', name='Actual (Val)',
    marker=dict(color='lime', size=8),
    line=dict(color='lime', width=1)
))

# C. Plot Rolling Forecast (Solid Red)
fig.add_trace(go.Scatter(
    x=plot_idx, y=plot_preds,
    mode='lines', name='7-Day Rolling Forecast',
    line=dict(color='red', width=2, dash='dash')
))

# D. Plot GARCH Confidence Interval (Shaded Grey)
fig.add_trace(go.Scatter(
    x=plot_idx + plot_idx[::-1], # x then x reversed
    y=plot_upper + plot_lower[::-1], # upper then lower reversed
    fill='toself',
    fillcolor='rgba(128, 128, 128, 0.2)',
    line=dict(color='rgba(255,255,255,0)'),
    hoverinfo="skip",
    showlegend=True,
    name='95% GARCH Band'
))

# E. Add the "Error Popup" Annotation
fig.add_annotation(
    text=f"<b>Predictive Core Metrics</b><br>RMSE: ${rmse:.2f}<br>MAE: ${mae:.2f}<br>MAPE: {mape:.2%}",
    xref="paper", yref="paper",
    x=0.02, y=0.98, showarrow=False,
    align="left",
    bgcolor="rgba(128, 128, 128, 0.2)",
    bordercolor="black", borderwidth=1, borderpad=10
)

# F. Final Layout Polish
fig.update_layout(
    template='plotly_dark',
    title='MSFT Predictive Core: ARIMA-GARCH Rolling Dashboard',
    xaxis_title='Date',
    yaxis_title='Stock Price ($)',
    hovermode='x unified',
    legend=dict(yanchor="bottom", y=0.01, xanchor="right", x=0.99)
)

st.subheader('PREDICTION OF MODEL')
st.plotly_chart(fig, use_container_width=True)


