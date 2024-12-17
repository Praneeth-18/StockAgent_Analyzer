import os
from dotenv import load_dotenv
import streamlit as st
import pandas as pd
import yfinance as yf
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
import plotly.graph_objs as go

# Load environment variables
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

class MarketAnalyzer:
    def __init__(self):
        self.setup_llm()
        self.initialize_vector_store()
        
    def setup_llm(self):
        self.llm = ChatOpenAI(
            model_name="gpt-3.5-turbo",
            temperature=0.3
        )
        self.embeddings = OpenAIEmbeddings()
        
    def initialize_vector_store(self):
        if os.path.exists("market_vectorstore"):
            self.vector_store = FAISS.load_local("market_vectorstore", self.embeddings)
        else:
            self.vector_store = FAISS.from_texts(["initialization"], self.embeddings)
            
    def analyze_data(self, data_str: str, metadata: dict) -> dict:
        # Split text into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        chunks = text_splitter.split_text(data_str)
        
        # Add to vector store with metadata
        self.vector_store.add_texts(
            chunks,
            metadatas=[metadata for _ in chunks]
        )
        
        # Create QA chain
        qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.vector_store.as_retriever()
        )
        
        # Get analysis
        analysis = qa_chain.run(
            """Analyze this market data and provide:
            1. Main price trends
            2. Key support and resistance levels
            3. Volume analysis
            4. Notable patterns or signals
            Be specific with numbers and dates."""
        )
        
        return {
            "status": "success",
            "analysis": analysis
        }
    
    def get_answer(self, question: str) -> str:
        qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.vector_store.as_retriever()
        )
        return qa_chain.run(question)

def create_candlestick_chart(data, symbol):
    fig = go.Figure()
    
    # Candlestick chart
    fig.add_trace(go.Candlestick(
        x=data.index,
        open=data['Open'],
        high=data['High'],
        low=data['Low'],
        close=data['Close'],
        name='Price'
    ))
    
    # Add EMA
    ema = data['Close'].ewm(span=20).mean()
    fig.add_trace(go.Scatter(
        x=data.index,
        y=ema,
        name='20-day EMA',
        line=dict(color='orange')
    ))
    
    fig.update_layout(
        title=f'{symbol} Price Chart',
        yaxis_title='Price',
        template='plotly_dark'
    )
    
    return fig

# Streamlit UI
st.set_page_config(page_title="Market Analysis", layout="wide")
st.title("AI Market Analysis")

# Initialize analyzer
analyzer = MarketAnalyzer()

# Input fields
col1, col2, col3 = st.columns([1, 1, 1])
with col1:
    symbol = st.text_input("Enter Stock Symbol", placeholder="e.g., AAPL")
with col2:
    interval = st.selectbox("Time Interval", 
                          ['1d', '5d', '1wk', '1mo', '3mo'])
with col3:
    period = st.selectbox("Analysis Period",
                         ['1mo', '3mo', '6mo', '1y', '2y', '5y'])

if st.button("Analyze Stock"):
    if not symbol:
        st.warning("Please enter a stock symbol")
    else:
        with st.spinner("Fetching and analyzing data..."):
            try:
                # Get data
                data = yf.download(symbol, interval=interval, period=period)
                
                if data.empty:
                    st.error("No data found for this symbol")
                else:
                    # Display chart
                    st.plotly_chart(create_candlestick_chart(data, symbol), use_container_width=True)
                    
                    # Show data preview
                    with st.expander("View Raw Data"):
                        st.dataframe(data)
                    
                    # Get AI analysis
                    data_str = data.to_string()
                    metadata = {
                        "symbol": symbol,
                        "interval": interval,
                        "period": period
                    }
                    
                    result = analyzer.analyze_data(data_str, metadata)
                    
                    if result["status"] == "success":
                        st.subheader("AI Analysis")
                        st.write(result["analysis"])
                        
                        # Allow follow-up questions
                        question = st.text_input(
                            "Ask a specific question about the data",
                            placeholder="e.g., What's the average volume? What are the key resistance levels?"
                        )
                        
                        if question:
                            with st.spinner("Getting answer..."):
                                answer = analyzer.get_answer(question)
                                st.write("### Answer")
                                st.write(answer)
                                
            except Exception as e:
                st.error(f"Error: {str(e)}")