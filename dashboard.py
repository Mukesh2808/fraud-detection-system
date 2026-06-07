import streamlit as st
import requests
import pandas as pd
import random

st.set_page_config(page_title='Fraud Detector', page_icon='🔍', layout='wide')
st.title('🔍 Real-Time Fraud Detection')
st.markdown('Enter transaction details on the left and click Analyze.')

# Sidebar for inputs
st.sidebar.header('Transaction Details')
amount = st.sidebar.number_input('Transaction Amount (₹)', 0.0, 50000.0, 250.0)
time_val = st.sidebar.number_input('Time (seconds since first tx)', 0.0, 200000.0, 50000.0)

st.sidebar.markdown('---')
st.sidebar.markdown('**V1–V10 (anonymized features)**')
v_vals = []
for i in range(1, 29):
    v_vals.append(st.sidebar.slider(f'V{i}', -5.0, 5.0, 0.0, key=f'v{i}'))

# Load sample transactions for demo
if st.sidebar.button('🎲 Load Random Real Transaction'):
    df = pd.read_csv('creditcard.csv')
    row = df.sample(1).iloc[0]
    st.session_state.loaded = row.tolist()
    st.rerun()

features = [time_val] + v_vals + [amount]

if st.button('🔎 Analyze Transaction', type='primary'):
    with st.spinner('Analyzing...'):
        try:
            response = requests.post(
                'http://localhost:8000/predict',
                json={'features': features},
                timeout=30
            )
            result = response.json()

            col1, col2 = st.columns(2)

            with col1:
                prob_pct = result['fraud_probability'] * 100
                if result['is_fraud']:
                    st.error(f'🚨 FRAUD DETECTED')
                else:
                    st.success(f'✅ LEGITIMATE TRANSACTION')
                st.metric('Fraud Probability', f'{prob_pct:.1f}%')
                st.metric('Risk Level', result['risk_level'])

            with col2:
                st.subheader('Why this decision?')
                for factor in result['top_factors']:
                    direction = '↑ Toward fraud' if factor['impact'] > 0 else '↓ Toward safe'
                    st.write(f"**{factor['feature']}**: {direction} (impact: {factor['impact']})")

        except Exception as e:
            st.error(f'Error: {e}. Is your API running?')
