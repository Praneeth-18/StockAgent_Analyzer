# StockAgent Analyzer: A Multi-Faceted Data Analysis Platform

**Description:**

StockAgent is a versatile data analysis platform that provides users with real-time data streams, historical analysis, and AI-driven insights. The platform enables users to visualize trends, explore patterns, and leverage advanced AI capabilities for data interpretation, all within an intuitive web-based application. A key feature is the use of autonomous agents for continuous data monitoring and updates.

**Features:**

*   **Comprehensive Data Visualization**: Provides interactive charts and graphs for a variety of datasets, enabling users to explore trends, patterns, and key metrics using customizable visualizations.
*   **AI-Driven Insights**: Leverages AI to analyze data, draw inferences, identify trends, and provide predictive insights, supporting informed decision-making.
*   **Real-Time Data Monitoring**: Employs autonomous agents to fetch and update data streams continuously, ensuring that users have access to the most current information.

**Setup and Run:**

Follow these steps to set up and run StockAgent:

1.  **Clone/Download:** Download or clone the entire project repository to your local machine.

2.  **Create a Virtual Environment:** Navigate to the project directory in your terminal and create a virtual environment using:

    ```bash
    python3.11 -m venv venv
    ```

3.  **Activate the Virtual Environment:** Activate the newly created virtual environment:

    ```bash
    source venv/bin/activate
    ```
    (On Windows, use `venv\Scripts\activate`)

4.  **Install Dependencies:** Install all the necessary Python packages by running the following command:

    ```bash
    pip install -r requirements.txt
    ```

5. **Install Required Packages:** If you do not have a requirements.txt, you can directly install the packages using the following command:
    ```bash
    pip install streamlit pandas yfinance langchain openai==0.28 python-dotenv plotly
    ```
6.  **Get an API Key:** To use the AI analysis features, you need an OpenAI API key. Get an API key from [OpenAI](https://platform.openai.com/api-keys) and paste it into the `.env` file, specifically in the line:

    ```
    OPENAI_API_KEY=your_api_key_here
    ```
    (Make sure the .env file is in the same directory as the `main.py` file.)

7.  **Run the Application:** After completing the setup steps, run the application by executing the main file:

    ```bash
    streamlit run main.py
    ```

    This will launch the application in your web browser.
