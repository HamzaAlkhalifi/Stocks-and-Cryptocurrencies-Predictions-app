import streamlit as st
import pandas as pd
import numpy as np
import ta.trend
import yfinance as yf
import ta
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.signal import argrelextrema
import datetime

def Chart(ticker, startDate, endDate, sLine, eLine, SLine, ELine):
    if ticker != '' :
        try:
            # Downloading data
            #=================#
            data = yf.download(
                ticker,
                start=(startDate - datetime.timedelta(days=100)),
                end=(endDate),
                multi_level_index= False
                )

            # Calculate Technical Indicators
            #================================#

            # 1. Moving Average Convergence Divergence (MACD)
            data['macd'] = ta.trend.macd(data['Close'])
            data['macd_signal'] = ta.trend.macd_signal(data['Close'])
            data['macd_diff'] = ta.trend.macd_diff(data['Close'])
            
            # 2. Relative Strength Index (RSI)
            data['rsi_14'] = ta.momentum.rsi(data['Close'], window=14)
            data['rsi_26'] = ta.momentum.rsi(data['Close'], window=26)

            # 3. Bollinger Bands
            data['bb_high'] = ta.volatility.bollinger_hband(data['Close'])
            data['bb_low'] = ta.volatility.bollinger_lband(data['Close'])
            data['bb_middle'] = ta.volatility.bollinger_mavg(data['Close'])

            # 4. Exponential Moving Average (EMA)
            data['ema_12'] = ta.trend.ema_indicator(data['Close'], window=12)
            data['ema_26'] = ta.trend.ema_indicator(data['Close'], window=26)

            # 5. Simple Moving Average (SMA)
            data['sma_20'] = ta.trend.sma_indicator(data['Close'], window=20)
            data['sma_50'] = ta.trend.sma_indicator(data['Close'], window=50)
            data['sma_100'] = ta.trend.sma_indicator(data['Close'], window=100)

            # 6. Average True Range (ATR)
            data['atr_14'] = ta.volatility.average_true_range(data['High'], data['Low'], data['Close'], window=14)
            data['atr_26'] = ta.volatility.average_true_range(data['High'], data['Low'], data['Close'], window=26)
            
            # 7. Stochastic Oscillator (Stoch)
            data['stoch_k'] = ta.momentum.stoch(data['High'], data['Low'], data['Close'])
            data['stoch_d'] = ta.momentum.stoch_signal(data['High'], data['Low'], data['Close'])

            # 8. On-Balance Volume (OBV)
            data['obv'] = ta.volume.on_balance_volume(data['Close'], data['Volume'])

            # 9. Chaikin Money Flow (CMF)
            data['cmf'] = ta.volume.chaikin_money_flow(data['High'], data['Low'], data['Close'], data['Volume'])

            # 10. Commodity Channel Index (CCI)
            data['cci'] = ta.trend.cci(data['High'], data['Low'], data['Close'])

            # 11. Change on Close
            data['Return'] = data['Close'].pct_change()
            data.dropna(inplace=True)

            #-------------------------------------------------------------------------------------------------------------------------------------------------
            # Finding the trend lline based on Open & Close

            Trend = data.copy()
            Trend = Trend[(Trend.index >= pd.to_datetime(SLine)) & (Trend.index <= pd.to_datetime(ELine))]
            Trend["O-C_mean"] = (Trend['Open'] + Trend['Close'])/2
            TrendC = Trend.dropna()

            if len(TrendC) > 1:
                trend_x = np.arange(len(TrendC))
                trend_y = TrendC['O-C_mean'].values
                slope, intercept = np.polyfit(trend_x, trend_y, 1)
                Trend_line = slope * trend_x + intercept

                std_dev = np.std(trend_y)
                upper_bound = Trend_line + std_dev
                lower_bound = Trend_line - std_dev

                # Adding wave around trend line
                amplitude = std_dev  # Amplitude of the wave
                frequency = 0.05  # Frequency of oscillation
                phase_shift = 0  # Phase shift of the sine wave
                wave = amplitude * np.sin(frequency * trend_x + phase_shift) + np.random.normal(0, 2, len(trend_x))
                waveL = Trend_line + wave
            #-----------------------------------------------------------------------------------------

            # Generating support & resistance lines
            #---------------------------------------#

            # Finding local minima (support) and maxima (resistance)

            window = 5  # Defining the window for detecting peaks/troughs
            data['minima'] = data['Low'].iloc[argrelextrema(data['Low'].values, np.less_equal, order=window)[0]]
            data['maxima'] = data['High'].iloc[argrelextrema(data['High'].values, np.greater_equal, order=window)[0]]
            df = data.copy()
            df = df[(df.index >= pd.to_datetime(sLine)) & (df.index <= pd.to_datetime(eLine))]

            # Fit lines to the minima and maxima (support and resistance lines)
            support_points = df.dropna(subset=['minima'])
            resistance_points = df.dropna(subset=['maxima'])
            
            # Fit line to the support points (local minima)
            if len(support_points) > 1:
                support_x = np.arange(len(support_points))
                support_y = support_points['minima'].values
                slope1, intercept1 = np.polyfit(support_x, support_y, 1)

            # Fit line to the resistance points (local maxima)
            if len(resistance_points) > 1:
                resistance_x = np.arange(len(resistance_points))
                resistance_y = resistance_points['maxima'].values
                slope2, intercept2 = np.polyfit(resistance_x, resistance_y, 1)
            #-----------------------------------------------------------------------------------------------------------------------------------------
            # Plot
            #=======#
            viz = make_subplots(
                rows=2, cols=1, shared_xaxes=True,
                vertical_spacing=0.2,row_heights=[0.7,0.3]
                )

            # Row 1
            candle = go.Candlestick(
                x=data.index, 
                high=data['High'], 
                low=data['Low'],
                open=data['Open'], 
                close=data['Close'],
                name="Candlestick"
            )
            trend_line = go.Scatter(
                x=Trend.index ,
                y=Trend_line, 
                name='Trend Line',
                line={"color": "white"}
            )

            strend_line = go.Scatter(
                x=Trend.index ,
                y=waveL, 
                name='Trend Oscillating Wave',
                line={"color": "grey"}
            )

            utrend_line = go.Scatter(
                x=Trend.index ,
                y=upper_bound, 
                name='Upper Bound Trend Line',
                line={"color": "white", 'dash':'dash'}
            )
            ltrend_line = go.Scatter(
                x=Trend.index ,
                y=lower_bound, 
                name='Lower Bound Trend Line',
                line={"color": "white", 'dash':'dash'}
            )
            
            supPoint = go.Scatter(
                x=df.index ,
                y=df['minima'], 
                mode='markers', 
                name='Support Point',
                marker=dict(
                    color='red', 
                    symbol='circle', 
                    size=8
                )
            )
            resPoint = go.Scatter(
                x=df.index ,
                y=df['maxima'], 
                mode='markers', 
                name='Resistance Point',
                marker=dict(
                    color='orange', 
                    symbol='x', 
                    size=8
                )
            )

            #----------------------
            bb_high = go.Scatter(
                x=data.index, 
                y=data["bb_high"], 
                name="BB High", 
                line={"color": "blue"}
            )
            bb_middle = go.Scatter(
                x=data.index, 
                y=data["bb_middle"], 
                name="BB Middle", 
                line={"color": "orange"}
            )
            bb_low = go.Scatter(
                x=data.index, 
                y=data["bb_low"], 
                name="BB Low", 
                line={"color": "blue"}
            )
            #----------------------
            ema_12 = go.Scatter(
                x=data.index, 
                y=data["ema_12"], 
                name="EMA12", 
                line={"color": "lightblue"}
            )
            ema_26 = go.Scatter(
                x=data.index, 
                y=data["ema_26"], 
                name="EMA26", 
                line={"color": "darkblue"}
            )
            #-------------------------
            sma_20 = go.Scatter(
                x=data.index, 
                y=data["sma_20"], 
                name="SMA20", 
                line={"color": "yellow"}
            )
            sma_50 = go.Scatter(
                x=data.index, 
                y=data["sma_50"], 
                name="SMA50", 
                line={"color": "red"}
            )
            sma_100 = go.Scatter(
                x=data.index, 
                y=data["sma_100"], 
                name="SMA100", 
                line={"color": "darkred"}
            )
            #-----------------------------
            
            # Adding traces for row 1
            viz.add_trace(candle, row=1, col=1)
            viz.add_trace(trend_line, row=1, col=1)
            viz.add_trace(strend_line, row=1, col=1)
            viz.add_trace(utrend_line, row=1, col=1)
            viz.add_trace(ltrend_line, row=1, col=1)

            viz.add_trace(go.Scatter(
                x=support_points.index, y=slope1 * support_x + intercept1, 
                mode='lines', name='Support Line', line=dict(dash='dash', color='orange')), 
                row=1, col=1
            )
            viz.add_trace(go.Scatter(
                x=resistance_points.index, y=slope2 * resistance_x + intercept2, 
                mode='lines', name='Resistance Line', line=dict(dash='dash', color='red')), 
                row=1, col=1
            )

            
            viz.add_trace(supPoint, row=1, col=1)
            viz.add_trace(resPoint, row=1, col=1)

            viz.add_trace(bb_high, row=1, col=1)
            viz.add_trace(bb_middle, row=1, col=1)
            viz.add_trace(bb_low, row=1, col=1)

            viz.add_trace(ema_12, row=1, col=1)
            viz.add_trace(ema_26, row=1, col=1)

            viz.add_trace(sma_20, row=1, col=1)
            viz.add_trace(sma_50, row=1, col=1)
            viz.add_trace(sma_100, row=1, col=1)

            

            # Row 2
            macd = go.Scatter(
                x=data.index, 
                y=data["macd"], 
                name="MACD", 
                line={"color": "orange"}
            )
            macd_signal = go.Scatter(
                x=data.index, 
                y=data["macd_signal"], 
                name="MACD Signal", 
                line={"color": "blue"}
            )
            macd_diff = go.Scatter(
                x=data.index, 
                y=data["macd_diff"], 
                name="MACD Diff", 
                line={"color": "rosybrown"}
            )
            #-------------------------
            rsi_14 = go.Scatter(
                x=data.index, 
                y=data["rsi_14"], 
                name="RSI14", 
                line={"color": "purple"}
            )
            rsi_26 = go.Scatter(
                x=data.index, 
                y=data["rsi_26"], 
                name="RSI26", 
                line={"color": "yellow"}
            )
            #------------------------------
            stoch_k = go.Scatter(
                x=data.index, 
                y=data["stoch_k"], 
                name="Stoch K", 
                line={"color": "white"}
            )
            stoch_d = go.Scatter(
                x=data.index, 
                y=data["stoch_d"], 
                name="Stoch D", 
                line={"color": "gold"}
            )
            #---------------------------
            atr_14 = go.Scatter(
                x=data.index, 
                y=data["atr_14"], 
                name="ATR14", 
                line={"color": "goldenrod"}
            )
            atr_26 = go.Scatter(
                x=data.index, 
                y=data["atr_26"], 
                name="ATR26", 
                line={"color": "lightcoral"}
            )
            #--------------------------------
            obv = go.Scatter(
                x=data.index, 
                y=data["obv"], 
                name="OBV", 
                line={"color": "maroon"}
            )
            cmf = go.Scatter(
                x=data.index, 
                y=data["cmf"], 
                name="CMF", 
                line={"color": "green"}
            )
            cci = go.Scatter(
                x=data.index, 
                y=data["cci"], 
                name="CCI", 
                line={"color": "red"}
            )

            # Adding traces for row 2
            viz.add_trace(macd, row=2, col=1)
            viz.add_trace(macd_signal, row=2, col=1)
            viz.add_trace(macd_diff, row=2, col=1)

            viz.add_trace(rsi_14, row=2, col=1)
            viz.add_trace(rsi_26, row=2, col=1)

            viz.add_trace(stoch_k, row=2, col=1)
            viz.add_trace(stoch_d, row=2, col=1)

            viz.add_trace(atr_14, row=2, col=1)
            viz.add_trace(atr_26, row=2, col=1)

            viz.add_trace(obv, row=2, col=1)
            viz.add_trace(cmf, row=2, col=1)
            viz.add_trace(cci, row=2, col=1)


            # Updating layout
            viz.update_layout(
                template="plotly_dark",
                height=800,
                title=f"{ticker} Price",
                xaxis_title='Date',
                yaxis_title='Price',
                xaxis_rangeslider_visible=False,
                hovermode='x unified'
            )

            st.caption("1. Double click to disable all lines except the one clicked")
            st.caption("2. One click to disable or enable a lines")
            st.caption("3. There might be some missing days")
            st.plotly_chart(viz, use_container_width=True)
        except:
            st.warning("Error,\nReasons might be: \n1. Change date inputs  \n2. The symbol you provide dose not has enough data \n3. Change Support & Resistance dates")

    else:
        st.warning("No data provided")


def Chart_(ticker, startDate, endDate, SLine, ELine):
    if ticker != '' :
        try:
            # Downloading data
            #=================#
            data = yf.download(
                ticker,
                start=(startDate - datetime.timedelta(days=100)),
                end=endDate,
                multi_level_index= False
                )


            # Calculate Technical Indicators
            #================================#

            # 1. Moving Average Convergence Divergence (MACD)
            data['macd'] = ta.trend.macd(data['Close'])
            data['macd_signal'] = ta.trend.macd_signal(data['Close'])
            data['macd_diff'] = ta.trend.macd_diff(data['Close'])
            
            # 2. Relative Strength Index (RSI)
            data['rsi_14'] = ta.momentum.rsi(data['Close'], window=14)
            data['rsi_26'] = ta.momentum.rsi(data['Close'], window=26)

            # 3. Bollinger Bands
            data['bb_high'] = ta.volatility.bollinger_hband(data['Close'])
            data['bb_low'] = ta.volatility.bollinger_lband(data['Close'])
            data['bb_middle'] = ta.volatility.bollinger_mavg(data['Close'])

            # 4. Exponential Moving Average (EMA)
            data['ema_12'] = ta.trend.ema_indicator(data['Close'], window=12)
            data['ema_26'] = ta.trend.ema_indicator(data['Close'], window=26)

            # 5. Simple Moving Average (SMA)
            data['sma_20'] = ta.trend.sma_indicator(data['Close'], window=20)
            data['sma_50'] = ta.trend.sma_indicator(data['Close'], window=50)
            data['sma_100'] = ta.trend.sma_indicator(data['Close'], window=100)

            # 6. Average True Range (ATR)
            data['atr_14'] = ta.volatility.average_true_range(data['High'], data['Low'], data['Close'], window=14)
            data['atr_26'] = ta.volatility.average_true_range(data['High'], data['Low'], data['Close'], window=26)
            
            # 7. Stochastic Oscillator (Stoch)
            data['stoch_k'] = ta.momentum.stoch(data['High'], data['Low'], data['Close'])
            data['stoch_d'] = ta.momentum.stoch_signal(data['High'], data['Low'], data['Close'])

            # 8. On-Balance Volume (OBV)
            data['obv'] = ta.volume.on_balance_volume(data['Close'], data['Volume'])

            # 9. Chaikin Money Flow (CMF)
            data['cmf'] = ta.volume.chaikin_money_flow(data['High'], data['Low'], data['Close'], data['Volume'])

            # 10. Commodity Channel Index (CCI)
            data['cci'] = ta.trend.cci(data['High'], data['Low'], data['Close'])

            # 11. Change on Close
            data['Return'] = data['Close'].pct_change()
            data.dropna(inplace=True)

            #-----------------------------------------------------------------------------------------------------------------------------------------

            # Finding the trend lline based on Open & Close

            Trend = data.copy()
            Trend = Trend[(Trend.index >= pd.to_datetime(SLine)) & (Trend.index <= pd.to_datetime(ELine))]
            Trend["O-C_mean"] = (Trend['Open'] + Trend['Close'])/2
            TrendC = Trend.dropna()

            if len(TrendC) > 1:
                trend_x = np.arange(len(TrendC))
                trend_y = TrendC['O-C_mean'].values
                slope, intercept = np.polyfit(trend_x, trend_y, 1)
                Trend_line = slope * trend_x + intercept

                std_dev = np.std(trend_y)
                upper_bound = Trend_line + std_dev
                lower_bound = Trend_line - std_dev

                # Adding wave around trend line
                amplitude = std_dev  # Amplitude of the wave
                frequency = 0.05  # Frequency of oscillation
                phase_shift = 0  # Phase shift of the sine wave
                wave = amplitude * np.sin(frequency * trend_x + phase_shift) + np.random.normal(0, 2, len(trend_x))
                waveL = Trend_line + wave

            #-------------------------------------------------------------------------------------------------------------------------------------------------

            
            
            # Plot
            #=======#
            viz = make_subplots(
                rows=2, cols=1, shared_xaxes=True,
                vertical_spacing=0.2,row_heights=[0.7,0.3]
                )

            # Row 1
            candle = go.Candlestick(
                x=data.index, 
                high=data['High'], 
                low=data['Low'],
                open=data['Open'], 
                close=data['Close'],
                name="Candlestick"
            )
            trend_line = go.Scatter(
                x=Trend.index ,
                y=Trend_line, 
                name='Trend Line',
                line={"color": "white"}
            )
            strend_line = go.Scatter(
                x=Trend.index ,
                y=waveL, 
                name='Trend Oscillating Wave',
                line={"color": "grey"}
            )

            utrend_line = go.Scatter(
                x=Trend.index ,
                y=upper_bound, 
                name='Upper Bound Trend Line',
                line={"color": "white", 'dash':'dash'}
            )
            ltrend_line = go.Scatter(
                x=Trend.index ,
                y=lower_bound, 
                name='Lower Bound Trend Line',
                line={"color": "white", 'dash':'dash'}
            )
            #----------------------
            bb_high = go.Scatter(
                x=data.index, 
                y=data["bb_high"], 
                name="BB High", 
                line={"color": "blue"}
            )
            bb_middle = go.Scatter(
                x=data.index, 
                y=data["bb_middle"], 
                name="BB Middle", 
                line={"color": "orange"}
            )
            bb_low = go.Scatter(
                x=data.index, 
                y=data["bb_low"], 
                name="BB Low", 
                line={"color": "blue"}
            )
            #----------------------
            ema_12 = go.Scatter(
                x=data.index, 
                y=data["ema_12"], 
                name="EMA12", 
                line={"color": "lightblue"}
            )
            ema_26 = go.Scatter(
                x=data.index, 
                y=data["ema_26"], 
                name="EMA26", 
                line={"color": "darkblue"}
            )
            #-------------------------
            sma_20 = go.Scatter(
                x=data.index, 
                y=data["sma_20"], 
                name="SMA20", 
                line={"color": "yellow"}
            )
            sma_50 = go.Scatter(
                x=data.index, 
                y=data["sma_50"], 
                name="SMA50", 
                line={"color": "red"}
            )
            sma_100 = go.Scatter(
                x=data.index, 
                y=data["sma_100"], 
                name="SMA100", 
                line={"color": "darkred"}
            )
            #-----------------------------
            
            # Adding traces for row 1
            viz.add_trace(candle, row=1, col=1)
            viz.add_trace(trend_line, row=1, col=1)
            viz.add_trace(strend_line, row=1, col=1)
            viz.add_trace(utrend_line, row=1, col=1)
            viz.add_trace(ltrend_line, row=1, col=1)

            viz.add_trace(bb_high, row=1, col=1)
            viz.add_trace(bb_middle, row=1, col=1)
            viz.add_trace(bb_low, row=1, col=1)

            viz.add_trace(ema_12, row=1, col=1)
            viz.add_trace(ema_26, row=1, col=1)

            viz.add_trace(sma_20, row=1, col=1)
            viz.add_trace(sma_50, row=1, col=1)
            viz.add_trace(sma_100, row=1, col=1)

            

            # Row 2
            macd = go.Scatter(
                x=data.index, 
                y=data["macd"], 
                name="MACD", 
                line={"color": "orange"}
            )
            macd_signal = go.Scatter(
                x=data.index, 
                y=data["macd_signal"], 
                name="MACD Signal", 
                line={"color": "blue"}
            )
            macd_diff = go.Scatter(
                x=data.index, 
                y=data["macd_diff"], 
                name="MACD Diff", 
                line={"color": "rosybrown"}
            )
            #-------------------------
            rsi_14 = go.Scatter(
                x=data.index, 
                y=data["rsi_14"], 
                name="RSI14", 
                line={"color": "purple"}
            )
            rsi_26 = go.Scatter(
                x=data.index, 
                y=data["rsi_26"], 
                name="RSI26", 
                line={"color": "yellow"}
            )
            #------------------------------
            stoch_k = go.Scatter(
                x=data.index, 
                y=data["stoch_k"], 
                name="Stoch K", 
                line={"color": "white"}
            )
            stoch_d = go.Scatter(
                x=data.index, 
                y=data["stoch_d"], 
                name="Stoch D", 
                line={"color": "gold"}
            )
            #---------------------------
            atr_14 = go.Scatter(
                x=data.index, 
                y=data["atr_14"], 
                name="ATR14", 
                line={"color": "goldenrod"}
            )
            atr_26 = go.Scatter(
                x=data.index, 
                y=data["atr_26"], 
                name="ATR26", 
                line={"color": "lightcoral"}
            )
            #--------------------------------
            obv = go.Scatter(
                x=data.index, 
                y=data["obv"], 
                name="OBV", 
                line={"color": "maroon"}
            )
            cmf = go.Scatter(
                x=data.index, 
                y=data["cmf"], 
                name="CMF", 
                line={"color": "green"}
            )
            cci = go.Scatter(
                x=data.index, 
                y=data["cci"], 
                name="CCI", 
                line={"color": "red"}
            )

            # Adding traces for row 2
            viz.add_trace(macd, row=2, col=1)
            viz.add_trace(macd_signal, row=2, col=1)
            viz.add_trace(macd_diff, row=2, col=1)

            viz.add_trace(rsi_14, row=2, col=1)
            viz.add_trace(rsi_26, row=2, col=1)

            viz.add_trace(stoch_k, row=2, col=1)
            viz.add_trace(stoch_d, row=2, col=1)

            viz.add_trace(atr_14, row=2, col=1)
            viz.add_trace(atr_26, row=2, col=1)

            viz.add_trace(obv, row=2, col=1)
            viz.add_trace(cmf, row=2, col=1)
            viz.add_trace(cci, row=2, col=1)


            # Updating layout
            viz.update_layout(
                template="plotly_dark",
                height=800,
                title=f"{ticker} Price",
                xaxis_title='Date',
                yaxis_title='Price',
                xaxis_rangeslider_visible=False,
                hovermode='x unified'
            )

            st.caption("1. Double click to disable all lines except the one clicked")
            st.caption("2. one click to disable or enable a lines")
            st.caption("3. There might be some missing days")
            st.plotly_chart(viz, use_container_width=True)

        except:
            st.warning("Error,\nReasons might be: \n1. Change date inputs  \n2. The symbol you provide dose not has enough data")

    else:
        st.warning("No data provided")