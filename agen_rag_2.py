import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import time
# 1. Import your real logic from the other file (assuming it's named logic.py)
from logic import run_agent, load_models 

# --- INITIAL CONFIG ---
st.set_page_config(page_title="Big Bull", layout="wide")

# --- SESSION STATE MANAGEMENT ---
if "messages" not in st.session_state:
    st.session_state.messages = []
# These keys are updated by your tools (get_price and plotter)
if "price_df" not in st.session_state:
    st.session_state.price_df = pd.DataFrame()
if "figs" not in st.session_state:
    st.session_state.figs = []

# --- UI DISPLAY ---
st.title("⚡ Growmore")

# Display loop
for idx, message in enumerate(st.session_state.messages):
    with st.chat_message(message["role"]):
        content = message["content"]
        
        if isinstance(content, dict):
            # 1. Top: Figures
            if content["figs"]:
                col1, col2 = st.columns(2)
                with col1:
                    st.plotly_chart(content["figs"][0], use_container_width=True, key=f"chart1_{idx}")
                if len(content["figs"]) > 1:
                    with col2:
                        st.plotly_chart(content["figs"][1], use_container_width=True, key=f"chart2_{idx}")
            
            # 2. Middle: Expander
            with st.expander("View Raw Data"):
                if content["df"] is None or content["df"].empty:
                    st.write("Plotters are on strike")
                else:
                    st.dataframe(content["df"], use_container_width=True, key=f"df_{idx}")
            
            # 3. Bottom: Analysis
            st.markdown(content["text"])
            st.info("Risk hai toh Ishq hai!")
        else:
            st.write(content)

# --- CHAT INPUT ---
if prompt := st.chat_input("Kaunsa stock uthana hai..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    with st.chat_message("assistant"):
        with st.spinner('The Big Bull is analyzing...'):
            try:
                # 2. RUN REAL AGENT
                # This call updates st.session_state['price_df'] and ['figs'] via tools
                analysis, figs, price_df = run_agent(prompt)
                print("\n**********************************************\n",analysis)
                # 3. Store structured data for history
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": {
                        "text": analysis,
                        "figs": figs, # Retrieved from session state after agent finishes
                        "df": price_df # Retrieved from session state after agent finishes
                    }
                })
                
                # 5-message memory limit
                if len(st.session_state.messages) > 5:
                    st.session_state.messages = st.session_state.messages[-5:]
                
                st.rerun()
                
            except Exception as e:
                st.error(f"The market crashed, Lala! Error: {e}")