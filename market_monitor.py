import streamlit as st
import multiprocessing as mp
import time
import requests
from bs4 import BeautifulSoup
import socket

# Function to run the agent process
def run_agent(queue, stock_symbol, exchange, port):
    from uagents import Agent, Context

    agent = Agent(
        name="Seek_bot",
        port=port,
        seed="stock_recovery_phase",
        endpoint=[f"http://127.0.0.1:{port}/submit"],
    )

    @agent.on_interval(period=30.0)
    async def fetch_stock_price(ctx: Context):
        url = f"https://www.google.com/finance/quote/{stock_symbol}:{exchange}"
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/90.0.4430.93 Safari/537.36"
        }
        
        response = requests.get(url, headers=headers)
        if response.status_code == 200:
            soup = BeautifulSoup(response.content, "html.parser")
            try:
                current_price = soup.find("div", class_="YMlKec fxKbKc").text
            except AttributeError:
                current_price = "N/A"
            queue.put(current_price)
        else:
            queue.put("Failed to retrieve data")

    agent.run()

# Helper function to find a free port
def find_free_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('', 0))
        return s.getsockname()[1]

# Streamlit UI
def main():
    st.title("Live Stock Price Fetcher Agent")
    st.sidebar.title("Stock Settings")

    stock_symbol = st.sidebar.text_input("Enter Stock Symbol")
    exchange = st.sidebar.selectbox("Select Exchange", ["NASDAQ", "NSE", "BSE"])

    if st.sidebar.button("Fetch"):
        # Multiprocessing Queue for concurrency
        queue = mp.Queue()
        
        # Find a free port
        port = find_free_port()
        
        # Start the agent process
        agent_process = mp.Process(target=run_agent, args=(queue, stock_symbol, exchange, port))
        agent_process.start()

        # Display live stock prices
        st.subheader(f"Live Stock Price for {stock_symbol} on {exchange}")
        placeholder = st.empty()

        try:
            while True:
                if not queue.empty():
                    current_price = queue.get()
                    placeholder.write(f"Current Price: {current_price}")
                # Sleep to reduce CPU usage
                time.sleep(0.1)
        except KeyboardInterrupt:
            agent_process.terminate()
            agent_process.join()

# Run the Streamlit app
if __name__ == "__main__":
    main()
