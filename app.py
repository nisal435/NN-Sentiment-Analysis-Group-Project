import streamlit as st
import requests
import pandas as pd
import plotly.express as px
from datetime import datetime
from collections import Counter

API_URL = "http://127.0.0.1:8000/predict/"

# Initialize session state
if 'sentiment_data' not in st.session_state:
    st.session_state.sentiment_data = []

# Watermark
st.markdown("""
    <style>
        .reportview-container { position: relative; }
        .watermark {
            position: absolute; top: 0; right: 0;
            width: 200px; opacity: 0.9; z-index: 1;
        }
    </style>
""", unsafe_allow_html=True)

st.markdown("""
    <div class="watermark">
        <img src="https://cdn.iconscout.com/icon/premium/png-256-thumb/emotion-analysis-1579748-1335674.png" />
    </div>
""", unsafe_allow_html=True)

# Title and Input
st.title("🌟 Mood Analyzer")
st.markdown("<br>", unsafe_allow_html=True)
st.subheader("Discover the sentiment behind your words!")
st.markdown("<br>", unsafe_allow_html=True)

user_input = st.text_area("Describe your experience:")

if st.button("Analyze Sentiment"):
    if user_input:
        try:
            with st.spinner("Analyzing sentiment..."):
                response = requests.post(API_URL, json={"text": user_input})
            if response.status_code == 200:
                result = response.json()
                timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                entry = {
                    "text": user_input,
                    "label": result["label"],
                    "score": result["score"],
                    "timestamp": timestamp
                }
                st.session_state.sentiment_data.append(entry)
                st.success(f"**Sentiment:** {result['label']}")
                st.info(f"**Confidence Score:** {result['score']:.4f}")
            else:
                st.error(f"Backend Error: {response.status_code}")
        except requests.exceptions.RequestException as e:
            st.error(f"Connection Error: {e}")
    else:
        st.warning("Please enter some text!")

# Sentiment Analysis History and Charts
if st.session_state.sentiment_data:
    st.markdown("<hr>", unsafe_allow_html=True)
    st.subheader("Sentiment Analysis History")
    df = pd.DataFrame(st.session_state.sentiment_data)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    st.dataframe(df)

    # Line Chart
    fig1 = px.line(df, x="timestamp", y="score", title="Sentiment Score Trend", markers=True)
    fig1.update_layout(
        xaxis_title='Time',
        yaxis_title='Score',
        plot_bgcolor='#F5F5F5',
        paper_bgcolor='#FFFFFF',
        font=dict(color='#000000')
    )
    st.plotly_chart(fig1)

    # Pie Chart
    label_counts = Counter(df["label"])
    fig2 = px.pie(
        names=list(label_counts.keys()),
        values=list(label_counts.values()),
        title="Sentiment Distribution",
        color_discrete_map={
            'Positive': '#28A745',
            'Negative': '#D9534F',
            'Neutral': '#888888'
        }
    )
    fig2.update_layout(
        plot_bgcolor='#F5F5F5',
        paper_bgcolor='#FFFFFF',
        font=dict(color='#000000')
    )
    st.plotly_chart(fig2)

    # Histogram
    fig3 = px.histogram(df, x="score", nbins=20, title="Sentiment Score Distribution")
    fig3.update_layout(
        plot_bgcolor='#F5F5F5',
        paper_bgcolor='#FFFFFF',
        font=dict(color='#000000'),
        xaxis_title='Score',
        yaxis_title='Count'
    )
    st.plotly_chart(fig3)

    # Box Plot – Score by Sentiment
    st.subheader("Sentiment Score Spread by Label")
    df['label'] = df['label'].str.capitalize()  # Normalize label casing
    df = df.dropna(subset=['score'])  # Drop missing scores

    fig_box = px.box(
        df,
        x="label",
        y="score",
        color="label",
        title="Sentiment Score Spread",
        color_discrete_map={
            'Positive': '#486cf0',  
            'Negative': '#D9534F',  
            'Neutral': '#888888'     
        }
    )
    fig_box.update_layout(
        xaxis_title='Sentiment',
        yaxis_title='Confidence Score',
        plot_bgcolor='#F5F5F5',
        paper_bgcolor='#FFFFFF',
        font=dict(color='#000000')
    )
    st.plotly_chart(fig_box)

else:
    st.info("No sentiment data available yet.")
