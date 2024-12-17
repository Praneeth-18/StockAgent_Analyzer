# main.py
import streamlit as st
import requests
from bs4 import BeautifulSoup
import yfinance as yf
import pandas as pd
import plotly.graph_objs as go
import io

# Custom theme and styling
st.set_page_config(page_title="StockAgent Analyzer", layout="wide")

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        background: linear-gradient(45deg, #1e3c72, #2a5298);
        padding: 20px;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 30px;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown("<div class='main-header'>StockAgent Analyzer</div>", unsafe_allow_html=True)

def fetch_market_data(symbol, exchange):
    url = f"https://www.google.com/finance/quote/{symbol}:{exchange}"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
    }
    
    try:
        response = requests.get(url, headers=headers)
        if response.status_code == 200:
            soup = BeautifulSoup(response.content, "html.parser")
            data = {
                "Asset Name": soup.find("div", class_="zzDege").text if soup.find("div", class_="zzDege") else "N/A",
                "Current Value": soup.find("div", class_="YMlKec fxKbKc").text if soup.find("div", class_="YMlKec fxKbKc") else "N/A",
                "Previous Session": soup.find("div", class_="P6K39c").text if soup.find("div", class_="P6K39c") else "N/A"
            }
            return data
        return None
    except Exception as e:
        st.error(f"Error fetching data: {str(e)}")
        return None

# Sidebar configuration
with st.sidebar:
    st.markdown("### Market Controls")
    st.markdown("---")
    
    analysis_type = st.radio(
        "Select Analysis Type",
        ["Real-time Analysis", "Historical Analysis"]
    )

if analysis_type == "Real-time Analysis":
    col1, col2 = st.columns([2, 1])
    
    with col2:
        symbol = st.text_input("Asset Symbol", placeholder="e.g., AAPL")
        market = st.selectbox("Select Market", 
                            ["NYSE", "NASDAQ", "LSE", "TSE", "NSE", "BSE"])
        
    if st.button("Analyze Asset", use_container_width=True):
        data = fetch_market_data(symbol, market)
        if data:
            with col1:
                for key, value in data.items():
                    st.markdown(f"""
                    <div class='metric-card'>
                        <h3>{key}</h3>
                        <h2>{value}</h2>
                    </div>
                    """, unsafe_allow_html=True)

else:
    st.markdown("### Historical Performance Analysis")
        
    col1, col2, col3 = st.columns([1, 1, 1])
    with col1:
        symbol = st.text_input("Asset Symbol", placeholder="e.g., AAPL")
    with col2:
        timeframe = st.selectbox("Timeframe", 
                                ["15m", "30m", "1h", "4h", "1d", "1wk"])
    with col3:
        period = st.selectbox("Analysis Period",
                            ["1mo", "3mo", "6mo", "1y", "2y", "5y"])

    if st.button("Generate Analysis", use_container_width=True):
        try:
            data = yf.download(symbol, interval=timeframe, period=period)
            
            if not data.empty:
                # Create advanced chart
                fig = go.Figure()
                
                # Candlestick chart
                fig.add_trace(go.Candlestick(
                    x=data.index,
                    open=data['Open'],
                    high=data['High'],
                    low=data['Low'],
                    close=data['Close'],
                    name='Price Action',
                    increasing_line_color='#26a69a',
                    decreasing_line_color='#ef5350'
                ))
                
                # Add EMA-21
                ema21 = data['Close'].ewm(span=21).mean()
                fig.add_trace(go.Scatter(
                    x=data.index,
                    y=ema21,
                    name='EMA-21',
                    line=dict(color='#f48fb1')
                ))
                
                fig.update_layout(
                    title=f'{symbol} Analysis',
                    yaxis_title='Price',
                    template='plotly_dark',
                    height=600
                )
                
                # Display the chart
                st.plotly_chart(fig, use_container_width=True)
                
                # Create a download button for CSV
                csv_buffer = io.StringIO()
                data.to_csv(csv_buffer)  # Convert DataFrame to CSV format
                csv_data = csv_buffer.getvalue()
                
                st.download_button(
                    label="Download Data as CSV",
                    data=csv_data,
                    file_name=f"{symbol}_historical_data.csv",
                    mime="text/csv"
                )
                
                # Key metrics
                metrics_col1, metrics_col2, metrics_col3 = st.columns(3)
                with metrics_col1:
                    st.metric("Period High", f"${data['High'].max():.2f}")
                with metrics_col2:
                    st.metric("Period Low", f"${data['Low'].min():.2f}")
                with metrics_col3:
                    st.metric("Trading Volume", f"{data['Volume'].mean():,.0f}")
                
        except Exception as e:
            st.error(f"Error analyzing data: {str(e)}")


# market_monitor.py
import streamlit as st
import multiprocessing as mp
import time
import requests
from bs4 import BeautifulSoup
import socket

def run_monitor(queue, symbol, market, port):
    from uagents import Agent, Context

    monitor = Agent(
        name="MarketMonitor",
        port=port,
        seed="market_monitor_seed",
        endpoint=[f"http://127.0.0.1:{port}/submit"],
    )

    @monitor.on_interval(period=45.0)
    async def monitor_price(ctx: Context):
        url = f"https://www.google.com/finance/quote/{symbol}:{market}"
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
        }
        
        try:
            response = requests.get(url, headers=headers)
            if response.status_code == 200:
                soup = BeautifulSoup(response.content, "html.parser")
                price_element = soup.find("div", class_="YMlKec fxKbKc")
                current_price = price_element.text if price_element else "N/A"
                queue.put({"price": current_price, "timestamp": time.strftime("%H:%M:%S")})
        except Exception as e:
            queue.put({"error": str(e)})

    monitor.run()

def find_available_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('', 0))
        return s.getsockname()[1]

def main():
    st.markdown("<h1 style='text-align: center;'>Market Monitor</h1>", unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        symbol = st.text_input("Asset Symbol")
        market = st.selectbox("Market", ["NYSE", "NASDAQ", "LSE", "TSE", "NSE", "BSE"])
    
    with col2:
        st.markdown("### Monitor Settings")
        st.info("Updates every 45 seconds")
    
    if st.button("Start Monitoring"):
        queue = mp.Queue()
        port = find_available_port()
        
        process = mp.Process(target=run_monitor, args=(queue, symbol, market, port))
        process.start()
        
        placeholder = st.empty()
        
        try:
            while True:
                if not queue.empty():
                    data = queue.get()
                    if "error" in data:
                        st.error(f"Monitor Error: {data['error']}")
                    else:
                        placeholder.markdown(f"""
                        <div style='background-color: #f0f2f6; padding: 20px; border-radius: 10px;'>
                            <h2>{symbol} ({market})</h2>
                            <h3>Current Value: {data['price']}</h3>
                            <p>Last Update: {data['timestamp']}</p>
                        </div>
                        """, unsafe_allow_html=True)
                time.sleep(0.1)
        except KeyboardInterrupt:
            process.terminate()
            process.join()

if __name__ == "__main__":
    main()

# analysis_ai.py
import os
from dotenv import load_dotenv
import streamlit as st
import pandas as pd
import openai

# Load environment variables from .env file
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
openai.api_key = api_key

def get_ai_insights(query, market_data):
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "user", "content": query},
                {"role": "user", "content": f"Market Data Analysis Request:\n\n{market_data}"}
            ]
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Analysis Error: {str(e)}"

st.markdown("<h1 style='text-align: center;'>AI Market Insights</h1>", unsafe_allow_html=True)
st.markdown("### Upload Market Data for Analysis")

uploaded_file = st.file_uploader("Upload Market Data (CSV)", type="csv")

if uploaded_file:
    try:
        df = pd.read_csv(uploaded_file)
        
        with st.expander("Preview Data"):
            st.dataframe(df)
        
        data_str = df.to_string(index=False)
        
        query = st.text_input("What would you like to know about this data?",
                            placeholder="e.g., 'Analyze trends', 'Identify patterns', 'Suggest strategies'")
        
        if st.button("Generate Insights"):
            with st.spinner("Analyzing market data..."):
                insights = get_ai_insights(query, data_str)
                st.markdown("### AI Analysis")
                st.markdown(insights)
                
    except Exception as e:
        st.error(f"Error processing file: {str(e)}")