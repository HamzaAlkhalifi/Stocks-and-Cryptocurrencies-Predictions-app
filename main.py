import streamlit as st
import pandas as pd
import time
import datetime

from layouts.FirstPage.Yahoo1 import Yahoo1
from layouts.FirstPage.Yahoo1 import Yahoo1_1
from layouts.FirstPage.Upload1 import Upload1
from layouts.FirstPage.Upload1 import Upload1_1
from layouts.SecondPage.Yahoo2 import Yahoo2
from layouts.SecondPage.Yahoo2 import Yahoo2_1
from layouts.SecondPage.Upload2 import Upload2
from layouts.SecondPage.Upload2 import Upload2_1
from layouts.ThirdPage.Chart import Chart
from layouts.ThirdPage.Chart import Chart_

st.set_page_config("Stocks & Cryptocurrencies Predictions")
st.title("Stocks & Cryptocurrencies Predictions")
st.caption("⚠ Disclaimer:\n This app is a **Machine Learning** project, and these models read historical data based on -{High, Close, Open, Low}- only.\
            \nSo it is not an investiment advice even if it gives a high accuracy, \
           it dose not mean that the price will reach the predicted point")

Page = st.sidebar.selectbox("Select Page", ["Highest Price Prediction", "Lowest Price Prediction", "Candlestick"])

# First Page
#-----------#
if Page == "Highest Price Prediction":
    st.header(f"{Page}",divider="red")
    select = st.selectbox('Data Download Method', ['Yahoo','Upload'])

    # First Slection
    if select == 'Yahoo':
        Model = st.selectbox("Select Model", ["LSTM", "FeedForward"])
        col1, col2 = st.columns(2)

        with col1:
            ticker = st.text_input("Cryptocurracny / Stock Symbol:","",placeholder="Example: BTC-USD")
        with col2:
            days = st.number_input("How Many Days to Predict:", 1, 365)

        st.caption('Note: \n1. Must write the Symbol as it is in Yahoo \n2. Click the ***Prediction*** button once only, clicking twice will cause a bad result')

        if st.button('Predict', use_container_width=True):
            if Model == 'LSTM':
                with st.spinner('Loading, please wait...',):
                    time.sleep(5)  # Simulate a long process (5 seconds)
                    Yahoo1(ticker, days)

            elif Model == "FeedForward":
                with st.spinner('Loading, please wait...',):
                    time.sleep(5)  # Simulate a long process (5 seconds)
                    Yahoo1_1(ticker, days)
            
    #-------------------------------------------------#

    # Second Slection
    elif select == 'Upload':
        uploaded_file = st.file_uploader('Upload CSV File', type='csv')
        col1, col2 = st.columns(2)
        with col1:
            Model = st.selectbox("Select Model", ["LSTM", "FeedForward"])
        with col2:
            days = st.number_input("How Many Days to Predict:", 1, 30)

        st.caption('Note:')
        st.caption('1.Must be in this order:')
        st.dataframe(pd.DataFrame({'Date':['2024-1-25'],'Close':[308],'High':[314],'Low':[294],'Open':[301],'Volume':[21056800]}))
        st.caption('2. Click the ***Prediction*** button once only, clicking twice will cause a bad result')

        if uploaded_file is not None:
            if st.button('Predict', use_container_width=True):
                if Model == 'LSTM':
                    with st.spinner('Loading, please wait...',):
                        time.sleep(5)  # Simulate a long process (5 seconds)
                        Upload1(uploaded_file, days)
                elif Model == "FeedForward":
                    with st.spinner('Loading, please wait...',):
                        time.sleep(5)  # Simulate a long process (5 seconds)
                        Upload1_1(uploaded_file, days)
#=======================================================================================#

# Second Page
#-------------#
elif Page ==  "Lowest Price Prediction":
    st.header(f"{Page}",divider="red")
    select = st.selectbox('Data Download Method', ['Yahoo','Upload'])

    # First Slection
    if select == 'Yahoo':
        Model = st.selectbox("Select Model", ["LSTM", "FeedForward"])
        col1, col2 = st.columns(2)
        
        with col1:
            ticker = st.text_input("Cryptocurracny / Stock Symbol:","",placeholder="Example: BTC-USD")
        with col2:
            days = st.number_input("How Many Days to Predict:", 1, 365)

        st.caption('Note: \n1. Must write the Symbol as it is in Yahoo \n2. Click the ***Prediction*** button once only, clicking twice will cause a bad result')
        if st.button('Predict', use_container_width=True):
            if Model == 'LSTM':
                with st.spinner('Loading, please wait...',):
                    time.sleep(5)  # Simulate a long process (5 seconds)
                    Yahoo2(ticker, days)

            elif Model == "FeedForward":
                with st.spinner('Loading, please wait...',):
                    time.sleep(5)  # Simulate a long process (5 seconds)
                    Yahoo2_1(ticker, days)
    #------------------------------------------------------------#

    # Second Slection
    elif select == 'Upload':
        uploaded_file = st.file_uploader('Upload CSV File', type='csv')
        col1, col2 = st.columns(2)
        with col1:
            Model = st.selectbox("Select Model", ["LSTM", "FeedForward"])
        with col2:
            days = st.number_input("How Many Days to Predict:", 1, 30)

        st.caption('Note:')
        st.caption('1.Must be in this order:')
        st.dataframe(pd.DataFrame({'Date':['2024-1-25'],'Close':[308],'High':[314],'Low':[294],'Open':[301],'Volume':[21056800]}))
        st.caption('2. Click the ***Prediction*** button once only, clicking twice will cause a bad result')

        if uploaded_file is not None:
            if st.button('Predict', use_container_width=True):
                if Model == 'LSTM':
                    with st.spinner('Loading, please wait...',):
                        time.sleep(5)  # Simulate a long process (5 seconds)
                        Upload2(uploaded_file, days)
                elif Model == "FeedForward":
                    with st.spinner('Loading, please wait...',):
                        time.sleep(5)  # Simulate a long process (5 seconds)
                        Upload2_1(uploaded_file, days)
#===========================================================================================#

# Third Page
#-----------#
elif Page ==  "Candlestick":
    st.header(f"{Page}",divider="red")
    col1, col2, col3 = st.columns(3)
    col4, col5 = st.columns(2)
    try:
        with col1:
            # Symobl
            ticker = st.text_input("Cryptocurracny / Stock Symbol:","BTC-USD")

        with col2:
            # Today date
            today = datetime.datetime.today()

            # Max date for the start date
            max_date = today - datetime.timedelta(days=40)

            startDate = st.date_input(
                "Start Date:", max_value= max_date,
                format="YYYY-MM-DD"
                )
            
        with col3:
            # Min date for end date
            minE_date = startDate + datetime.timedelta(days=40)

            endDate = st.date_input(
                "End Date:", 
                min_value=minE_date,
                max_value= today,
                format="YYYY-MM-DD"
                )
        
        with col4:
            SLine = st.date_input(
                'Trend Line Start:',
                min_value=startDate,
                max_value=pd.to_datetime((endDate - datetime.timedelta(days=2))),
                format="YYYY-MM-DD"
                )

        with col5:
            ELine = st.date_input(
                'Trend Line End:',
                min_value=pd.to_datetime((SLine + datetime.timedelta(days=1))),
                max_value=endDate,
                format="YYYY-MM-DD"
                )
        
        if st.checkbox("Generate Support & Resistance Lines"):   
            col6, col7 = st.columns(2)

            with col6:
                # Support & Resistance Line Start
                sLine = st.date_input(
                    'Support & Resistance Line Start:',
                    min_value=startDate,
                    max_value=pd.to_datetime((endDate - datetime.timedelta(days=30))),
                    format="YYYY-MM-DD"
                    )
                
            with col7:
                # Support & Resistance Line End
                eLine = st.date_input(
                    'Support & Resistance Line End:',
                    min_value=pd.to_datetime((sLine + datetime.timedelta(days=30))),
                    max_value=endDate,
                    format="YYYY-MM-DD"
                    )
            
            if st.button("Make The Chart",use_container_width=True):
                Chart(ticker, startDate, endDate, sLine, eLine, SLine, ELine)
                
        else:
            if st.button("Make The Chart",use_container_width=True):
                Chart_(ticker, startDate, endDate, SLine, ELine)

    except:
        st.warning("Invalid input")
#========================================================================================================#

# Footer
#-------#
st.divider()
footer = """
<div style="text-align: center;">
    <a href="https://www.linkedin.com/in/hamzah-sultan-/" target="https://www.linkedin.com/in/hamzah-sultan-/" style="margin-right: 15px; font-size: 16px;text-decoration: none">LinkedIn</a>
    <a href="https://github.com/HamzaAlkhalifi" target="https://github.com/HamzaAlkhalifi" style="font-size: 16px;text-decoration: none;">GitHub</a>
</div>
"""

# Render the footer at the bottom of the page
st.markdown(footer, unsafe_allow_html=True)