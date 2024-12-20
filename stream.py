import streamlit as st
import pandas as pd
from backend import BlackScholes, BionomialLattice, ticker

# PAGE CONFIG
st.set_page_config(
    page_title="Option Pricing",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="auto",
    menu_items={'Report a Bug': "https://www.linkedin.com/in/veer-vohra/"})

# SIDEBAR
with st.sidebar:
    linkedin_url = "https://www.linkedin.com/in/veer-vohra/"
    st.markdown(f'<a href="{linkedin_url}" target="_blank" style="text-decoration: none; color: inherit;"> <img src="https://cdn-icons-png.flaticon.com/512/174/174857.png" width="25" height="25" style="vertical-align: middle; margin-right: 10px;">`Veer Vohra`</a>', unsafe_allow_html=True)
    spot = st.number_input("Spot Price (S)", value=100.00)
    strike = st.number_input("Strike Price (K)", value=100.00)
    time_to_maturity = st.number_input("Time to Maturity (t) / days", value=1)
    volatility = st.number_input("Volatility (σ)", value=0.2000)
    interest_rate = st.number_input("Risk-Free Interest Rate (r)", value=0.0500)
    div_yield = st.number_input("Dividend Yield (q)", value=0.0000)
    numsteps = st.number_input("Number of Steps (Binomial Lattice)", value=5)
    symb = st.text_input("Ticker", value='AAPL')

# MAIN PAGE
st.title("Option Pricing Models")
st.info("Adjust the variables on the left to explore how they affect an option's value and greeks.")

# MODEL PRICES
bl_model = BionomialLattice(S=spot, K=strike, r=interest_rate, t=time_to_maturity, vol=volatility, steps=numsteps)
bl_call_price, bl_put_price = bl_model.calculate_prices()
bs_model = BlackScholes(S=spot, K=strike, r=interest_rate, t=time_to_maturity, vol=volatility, q=div_yield)
c_price, p_price, c_delta, p_delta, gamma, vega, c_theta, p_theta, c_rho, p_rho = bs_model.run()
col1, col2 = st.columns(2)
with col1:
    st.subheader("Binomial Lattice")
with col2:
    st.subheader("Black-Scholes")
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("CALL", round(bl_call_price, 3))
with col2:
    st.metric("PUT", round(bl_put_price, 3))
with col3:
    st.metric("CALL", round(c_price, 3))
with col4:
    st.metric("PUT", round(p_price, 3))

# GREEKS TABLE
df = pd.DataFrame(
    {
        "CallDelta": [c_delta],
        "CallTheta": [c_theta],
        "CallRho": [c_rho],
        "Gamma": [gamma],
        "Vega": [vega],
        "PutDelta": [p_delta],
        "PutTheta": [p_theta],
        "PutRho": [p_rho],
    }
)
st.dataframe(
    df,
    hide_index=True,
    use_container_width=True,
)

# REAL CONTRACTS; TABLE AND GRAPHS
try:
    st.info("Compare these real option contracts to the pricing models above.")
    df, spot, graphs = ticker(symb, interest_rate, div_yield)
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Ticker", symb)
    with col2:
        st.metric("Spot Price", round(spot, 3))
    st.dataframe(
        df,
        hide_index=True,
        use_container_width=True,
    )
    st.pyplot(graphs)

except:
    st.error("Invalid Ticker Symbol or Connection Error", icon="🚨")
