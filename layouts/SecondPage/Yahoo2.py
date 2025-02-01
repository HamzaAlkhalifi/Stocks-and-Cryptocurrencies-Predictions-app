import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import ta
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset
import yfinance as yf


def Yahoo2(ticker, days):

    if ticker != '' :
        try:
            # Downloading data
            #=================#
            data = yf.download(ticker,  multi_level_index= False)

            if data.shape[0] > 1000:
                data = data[-(1000 + days):-1]
            print(data.shape[0])


            # Data preprocessing
            #===================#

            # 1. EMA (Exponential Moving Average)
            data['EMA14'] = ta.trend.EMAIndicator(data['Low'], window=14).ema_indicator()
            data['EMA28'] = ta.trend.EMAIndicator(data['Low'], window=28).ema_indicator()

            # 2. RSI (Relative Strength Index)
            data['RSI_14'] = ta.momentum.RSIIndicator(data['Low'], window=14).rsi()
            data['RSI_28'] = ta.momentum.RSIIndicator(data['Low'], window=28).rsi()

            # 3. MACD (Moving Average Convergence Divergence)
            data['MACD'] = ta.trend.MACD(data['Low']).macd()
            data['MACD Signal'] = ta.trend.MACD(data['Low']).macd_signal()

            # 4. Bollinger Bands
            data['BB Upper'] = ta.volatility.bollinger_hband(data['Low'])
            data['BB Middle'] =  ta.volatility.bollinger_mavg(data['Low'])
            data['BB Lower'] = ta.volatility.bollinger_lband(data['Low'])

            # 5. volume-based feature
            data['Volume_change'] = data['Volume'].pct_change()  

            # 6. Lagged Features
            data['Prev_Low'] = data['Low'].shift(1)
            data['Prev_Low2'] = data['Low'].shift(2)
            data['Prev_Low3'] = data['Low'].shift(3)

            # 7. On-Balance Volume (OBV)
            data['obv'] = ta.volume.on_balance_volume(data['Close'], data['Volume'])

            # 8. Chaikin Money Flow (CMF)
            data['cmf'] = ta.volume.chaikin_money_flow(data['High'], data['Low'], data['Close'], data['Volume'])

            # 9. Commodity Channel Index (CCI)
            data['cci'] = ta.trend.cci(data['High'], data['Low'], data['Close'])
            
            data = data[[
                'Open',
                'Close',
                'Prev_Low3', 'Prev_Low2', 'Prev_Low', 'Low',
                'High',
                'EMA14', 'EMA28', 'BB Upper', 'BB Middle', 'BB Lower',
                'Volume', 'Volume_change','MACD', 'MACD Signal',
                'RSI_14', 'RSI_28', 'obv', 'cmf', 'cci'
            ]]


            data.dropna(inplace=True)
            df = data.copy()

            #Trageting the next Days
            df["TargetLow"] = df["Low"].shift(-days)
            df.dropna(inplace = True)
            
#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
            for iter in range(2):

                # Features to predict 
                x = df.drop(['TargetLow'],axis=1).values
                y = df["TargetLow"].values.reshape(-1,1) # Reshape to 2D for scaling


                # Normalizing the features using MinMaxScaler
                scaler_x = MinMaxScaler()
                x_scaled = scaler_x.fit_transform(x)

                scaler_y = MinMaxScaler()
                y_scaled = scaler_y.fit_transform(y)  


                # Splitting data to training & testing Data
                x_train, x_test, y_train, y_test = train_test_split(x_scaled, y_scaled, 
                                                                    test_size=0.007, 
                                                                    random_state=42,
                                                                    shuffle=False
                                                                )

                print(f"{x_train.shape}, {x_test.shape}, {y_train.shape}, {y_test.shape}")
                
                #Preprocessing for LSTM Model only
                def create_sequences(x, y, n_timesteps):
                    x_seq, y_seq = [], []
                    for i in range(len(x) - n_timesteps):
                        x_seq.append(x[i:i+n_timesteps])  # Collect sequences of timesteps
                        y_seq.append(y[i + n_timesteps])  # The target value for the next step
                    return np.array(x_seq), np.array(y_seq)

                n_timesteps = 3

                x_train, y_train = create_sequences(x_train, y_train, n_timesteps)
                x_test, y_test = create_sequences(x_test, y_test, n_timesteps)

                print(f"{x_train.shape}, {x_test.shape}, {y_train.shape}, {y_test.shape}")


                # Convert data type to torch tensors
                torch.manual_seed(1)
                x_train = torch.tensor(x_train, dtype=torch.float32)
                x_test = torch.tensor(x_test, dtype=torch.float32)
                y_train = torch.tensor(y_train, dtype=torch.float32)
                y_test = torch.tensor(y_test, dtype=torch.float32)

                train_dataset = TensorDataset(x_train, y_train)
                test_dataset = TensorDataset(x_test, y_test)

                train_loader = DataLoader(train_dataset, batch_size=32, shuffle= True)
                test_loader = DataLoader(test_dataset, batch_size=32, shuffle= False)
                print(f"{x_train.shape}, {x_test.shape}, {y_train.shape}, {y_test.shape}")

                
                # Building Model
                #===============#
                class LSTM(nn.Module):
                    def __init__(self, input_size, num_layers, output_size):
                        super(LSTM, self).__init__()

                        # Hidden layer sizes
                        self.first_h=int(input_size**2)
                        if self.first_h > 1000:
                            self.first_h = 1000
                        self.second_h=int(self.first_h/2)
                        
                        # LSTM layer 
                        self.lstm = nn.LSTM(input_size, 
                                            self.first_h, 
                                            num_layers, 
                                            batch_first=True,
                                        )
                                
                        # Fully connected layers
                        self.fc1 = nn.Linear(self.first_h, self.second_h)
                        self.fc2 = nn.Linear(self.second_h, self.first_h)
                        self.fc3 = nn.Linear(self.first_h, self.second_h)
                        self.fc4 = nn.Linear(self.second_h, input_size)
                        self.fc5 = nn.Linear(input_size, output_size)

                        # Activation function
                        self.relu = nn.ReLU()

                    
                    def forward(self, x):

                        # Forward pass through LSTM layer 
                        out, _ = self.lstm(x)
                        
                        out = self.fc1(out[:, -1, :])
                        out = self.relu(out)
                        out = self.fc2(out)
                        out = self.relu(out)
                        out = self.fc3(out)
                        out = self.relu(out)
                        out = self.fc4(out)
                        out = self.relu(out)

                        # Final output layer (no activation for regression task)
                        out = self.fc5(out)
                        
                        return out
                    
                
                # Create model, criterion and optimizer objects
                model = LSTM(x_train.shape[2], 1, 1) #LSTM Model
                print(model)
                criterion = nn.MSELoss()
                optimizer = optim.Adam(model.parameters(), lr=0.001)


                # Training
                #================================================#

                epochs = 200
                epochsi = []
                epochsi = [x + 1 for x in epochsi]
                train_losses = []
                test_losses = []
                try:
                    for epoch in range(epochs):
                        # Training phase
                        model.train()
                        running_loss = 0.0
                        for X_batch, y_batch in train_loader:
                            optimizer.zero_grad()
                            outputs = model(X_batch)
                            loss = criterion(outputs, y_batch)
                            loss.backward()
                            optimizer.step()
                            running_loss += loss.item()
                    
                        train_loss = running_loss / len(train_loader)
                        train_losses.append(train_loss)
                        
                        # Evaluation phase on test set
                        model.eval()
                        test_loss = 0.0
                        with torch.no_grad():
                            for X_batch, y_batch in test_loader:
                                test_outputs = model(X_batch)
                                loss = criterion(test_outputs, y_batch)
                                test_loss += loss.item()
                    
                        test_loss /= len(test_loader)
                        test_losses.append(test_loss)
                        epochsi.append(int(epoch))
                        print(f'Epoch [{epoch + 1}/{epochs}], Train Loss: {train_loss:.5f}, Test Loss: {test_loss:.5f}')
                        
                        if iter == 1:
                            if test_loss == least_test_loss:
                                break
                        
                except KeyboardInterrupt as k:
                    print(k)
                finally:
                    epochsi = [x + 1 for x in epochsi]
                    print(f"The Training has Finished")
                    least_test_loss = min(test_losses)
                    print(min(test_losses))
#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

            with st.container(border=True):
                # Testing to get accuracy 
                with torch.no_grad():
                    model.eval()
                    yhat = model(x_test)
                    
                st.subheader("Training Result")
                st.metric('Model Accuracy:',value=f'{(r2_score(y_test, yhat)):.2f}',)

                # Inverse transform the predictions to get the original Close price
                predictions = scaler_y.inverse_transform(yhat.detach().numpy())
                TLow = scaler_y.inverse_transform(y_test.detach().numpy())
 
                # Plotting differances
                fig, ax = plt.subplots()
                ax.plot(df.index[-x_test.shape[0]:], TLow, "-o", label=f'Low after {days} days', alpha= 0.5)
                ax.plot(df.index[-x_test.shape[0]:], predictions, "-or", label=f'Predicted Low after {days} days', alpha= 0.5)
                ax.plot(df.index[-x_test.shape[0]:], df.Low[-x_test.shape[0]:], "-og", label='Low today', alpha= 0.5)
                ax.legend()
                plt.xticks(rotation=90)
                st.pyplot(fig)


                # Prepering Data for the Next Days
                #====================================#
                Next2 = []

                for i in range(1,days + 1):
                    nextDay = data.index[-1] + pd.Timedelta(days=i)
                    Next2.append(nextDay)


                # Taking data for plot clarity and good understanding 
                data_plot =data[data.index >= "2024-1-1"]

                # Taking last days to predict the next two days
                z = data.iloc[-(days + 3): ].values
                z_scaled=scaler_x.transform(z)

                def create_sequences_Z(z, n_timesteps):
                    z_seq = []
                    for i in range(len(z) - n_timesteps):
                        z_seq.append(z[i:i+n_timesteps])  # Collect sequences of timesteps
                    return np.array(z_seq)

                n_timesteps = 3

                z_forPred = create_sequences_Z(z_scaled, n_timesteps)
                z_forPred = torch.tensor(z_forPred, dtype=torch.float32)



                # Prediction
                #============#
                st.header("Prediction",divider=True)
                with torch.no_grad():
                    model.eval()
                    Prediction = model(z_forPred)
                Prediction = scaler_y.inverse_transform(Prediction.detach().numpy())


                pre_df = pd.DataFrame({"Date": Next2, "Prediction":Prediction.reshape(-1)})
                with st.expander(f"### Prediction of the next {days} Days:"):
                    for i in range(days):
                        st.write(f"Date: {pre_df.Date[i]}, Predicted Low: {pre_df.Prediction[i]:,.2f}")

                viz = make_subplots(rows=1, cols=1, shared_xaxes=True)

                # Candlestick
                candle = go.Candlestick(
                    x=data_plot.index, 
                    high=data_plot['High'], 
                    low=data_plot['Low'],
                    open=data_plot['Open'], 
                    close=data_plot['Close'],
                    name="Candlestick"
                )

                line_pred = go.Scatter(
                    x=pre_df["Date"], 
                    y=pre_df["Prediction"], 
                    name="Prediction", 
                    line={"color": "red"}
                )

                # Adding traces
                viz.add_trace(candle, row=1, col=1)
                viz.add_trace(line_pred, row=1, col=1)

                # Updating layout
                viz.update_layout(
                    template="plotly_dark",
                    height=400,
                    width=1000,
                    title=f"{ticker} Prediction",
                    xaxis_title='Date',
                    yaxis_title='Price',
                    xaxis_rangeslider_visible=False
                )

                st.plotly_chart(viz)
        except:
            st.warning("Invalid Value")


    else:
        st.warning("No data provided")


def Yahoo2_1(ticker, days):

    if ticker != '' :
        try:
            # Downloading data
            #=================#
            data = yf.download(ticker,  multi_level_index= False)

            if data.shape[0] > 1000:
                data = data[-(1000 + days):-1]
            print(data.shape[0])


            # Data preprocessing
            #===================#

            # 1. EMA (Exponential Moving Average)
            data['EMA14'] = ta.trend.EMAIndicator(data['Low'], window=14).ema_indicator()
            data['EMA28'] = ta.trend.EMAIndicator(data['Low'], window=28).ema_indicator()

            # 2. Bollinger Bands
            data['BB Upper'] = ta.volatility.bollinger_hband(data['Low'])
            data['BB Middle'] =  ta.volatility.bollinger_mavg(data['Low'])
            data['BB Lower'] = ta.volatility.bollinger_lband(data['Low']) 

            # 3. Lagged Features
            data['Prev_Low'] = data['Low'].shift(1)
            data['Prev_Low2'] = data['Low'].shift(2)
            data['Prev_Low3'] = data['Low'].shift(3)

            # 4. On-Balance Volume (OBV)
            data['obv'] = ta.volume.on_balance_volume(data['Close'], data['Volume'])

            # 5. Chaikin Money Flow (CMF)
            data['cmf'] = ta.volume.chaikin_money_flow(data['High'], data['Low'], data['Close'], data['Volume'])

            # 6. Commodity Channel Index (CCI)
            data['cci'] = ta.trend.cci(data['High'], data['Low'], data['Close'])

            data = data[[
                'Open',
                'Close',
                'Prev_Low3', 'Prev_Low2', 'Prev_Low', 'Low',
                'High',
                'EMA14', 'EMA28', 'BB Upper', 'BB Middle', 'BB Lower',
                'Volume', 'obv', 'cmf', 'cci'
            ]]


            data.dropna(inplace=True)
            df = data.copy()

            #Trageting the next Days
            df["TargetLow"] = df["Low"].shift(-days)
            df.dropna(inplace = True)
            
#----------------------------------------------------------------------------------------------------------------------------------------------------------------------
            for iter in range(2):

                # Features to predict 
                x = df.drop(['TargetLow'],axis=1).values
                y = df["TargetLow"].values.reshape(-1,1) # Reshape to 2D for scaling


                # Normalizing the features using MinMaxScaler
                scaler_x = MinMaxScaler()
                x_scaled = scaler_x.fit_transform(x)

                scaler_y = MinMaxScaler()
                y_scaled = scaler_y.fit_transform(y)  


                # Splitting data to training & testing Data
                x_train, x_test, y_train, y_test = train_test_split(x_scaled, y_scaled, 
                                                                    test_size=0.007, 
                                                                    random_state=42,
                                                                    shuffle=False
                                                                )

                print(f"{x_train.shape}, {x_test.shape}, {y_train.shape}, {y_test.shape}")
                

                # Convert data type to torch tensors
                torch.manual_seed(1)
                x_train = torch.tensor(x_train, dtype=torch.float32)
                x_test = torch.tensor(x_test, dtype=torch.float32)
                y_train = torch.tensor(y_train, dtype=torch.float32)
                y_test = torch.tensor(y_test, dtype=torch.float32)

                train_dataset = TensorDataset(x_train, y_train)
                test_dataset = TensorDataset(x_test, y_test)

                train_loader = DataLoader(train_dataset, batch_size=32, shuffle= True)
                test_loader = DataLoader(test_dataset, batch_size=32, shuffle= False)
                print(f"{x_train.shape}, {x_test.shape}, {y_train.shape}, {y_test.shape}")

                
                # Building Model
                #===============#
                class FeedForward(nn.Module):
                    def __init__(self, input_size,output_size):
                        super(FeedForward, self).__init__()      

                        # Hidden layer sizes
                        self.first_h=int(input_size**2) 
                        if self.first_h > 1000:
                            self.first_h = 1000
                        self.second_h=int(self.first_h/2)

                        # Fully connected layers
                        self.fc1 = nn.Linear(input_size, self.first_h)
                        self.fc2 = nn.Linear(self.first_h, self.second_h)
                        self.fc3 = nn.Linear(self.second_h, self.first_h)
                        self.fc4 = nn.Linear(self.first_h, input_size)
                        self.fc5 = nn.Linear(input_size, output_size)


                        # Activation function
                        self.relu = nn.ReLU()
                    
                    def forward(self, x):
                        x = self.fc1(x)
                        x = self.relu(x)
                        x = self.fc2(x)
                        x = self.relu(x)
                        x = self.fc3(x)
                        x = self.relu(x)
                        x = self.fc4(x)
                        x = self.relu(x)
                        out = self.fc5(x)
                        
                        return out
                    
                
                # Create model, criterion and optimizer objects
                model = FeedForward(x_train.shape[1], 1) #FeedForward Model
                print(model)
                criterion = nn.MSELoss()
                optimizer = optim.Adam(model.parameters(), lr=0.001)


                # Training
                #================================================#

                epochs = 500
                epochsi = []
                epochsi = [x + 1 for x in epochsi]
                train_losses = []
                test_losses = []
                try:
                    for epoch in range(epochs):
                        # Training phase
                        model.train()
                        running_loss = 0.0
                        for X_batch, y_batch in train_loader:
                            optimizer.zero_grad()
                            outputs = model(X_batch)
                            loss = criterion(outputs, y_batch)
                            loss.backward()
                            optimizer.step()
                            running_loss += loss.item()
                    
                        train_loss = running_loss / len(train_loader)
                        train_losses.append(train_loss)
                        
                        # Evaluation phase on test set
                        model.eval()
                        test_loss = 0.0
                        with torch.no_grad():
                            for X_batch, y_batch in test_loader:
                                test_outputs = model(X_batch)
                                loss = criterion(test_outputs, y_batch)
                                test_loss += loss.item()
                    
                        test_loss /= len(test_loader)
                        test_losses.append(test_loss)
                        epochsi.append(int(epoch))
                        print(f'Epoch [{epoch + 1}/{epochs}], Train Loss: {train_loss:.5f}, Test Loss: {test_loss:.5f}')
                        
                        if iter == 1:
                            if test_loss == least_test_loss:
                                break
                        
                except KeyboardInterrupt as k:
                    print(k)
                finally:
                    epochsi = [x + 1 for x in epochsi]
                    print(f"The Training has Finished")
                    least_test_loss = min(test_losses)
                    print(min(test_losses))
#--------------------------------------------------------------------------------------------------------------------------------------------------------------------------

            with st.container(border=True):
                # Testing to get accuracy 
                with torch.no_grad():
                    model.eval()
                    yhat = model(x_test)
                    
                st.subheader("Training Result")
                st.metric('Model Accuracy:',value=f'{(r2_score(y_test, yhat)):.2f}',)

                # Inverse transform the predictions to get the original Close price
                predictions = scaler_y.inverse_transform(yhat.detach().numpy())
                TLow = scaler_y.inverse_transform(y_test.detach().numpy())

                # Plotting differances
                fig, ax = plt.subplots()
                ax.plot(df.index[-x_test.shape[0]:], TLow, "-o", label=f'Low after {days} days', alpha= 0.5)
                ax.plot(df.index[-x_test.shape[0]:], predictions, "-or", label=f'Predicted Low after {days} days', alpha= 0.5)
                ax.plot(df.index[-x_test.shape[0]:], df.Low[-x_test.shape[0]:], "-og", label='Low today', alpha= 0.5)
                ax.legend()
                plt.xticks(rotation=90)
                st.pyplot(fig)


                # Prepering Data for the Next Days
                #====================================#
                Next2 = []

                for i in range(1,days + 1):
                    nextDay = data.index[-1] + pd.Timedelta(days=i)
                    Next2.append(nextDay)


                # Taking data for plot clarity and good understanding 
                data_plot =data[data.index >= "2024-1-1"]

                # Taking last days to predict the next two days
                z = data.iloc[-days: ].values
                z_scaled = scaler_x.transform(z)

                
                z_forPred = torch.tensor(z_scaled, dtype=torch.float32)



                # Prediction
                #============#
                st.header("Prediction",divider=True)
                with torch.no_grad():
                    model.eval()
                    Prediction = model(z_forPred)
                Prediction = scaler_y.inverse_transform(Prediction.detach().numpy())


                pre_df = pd.DataFrame({"Date": Next2, "Prediction":Prediction.reshape(-1)})
                with st.expander(f"### Prediction of the next {days} Days:"):
                    for i in range(days):
                        st.write(f"Date: {pre_df.Date[i]}, Predicted Low: {pre_df.Prediction[i]:,.2f}")

                viz = make_subplots(rows=1, cols=1, shared_xaxes=True)

                # Candlestick
                candle = go.Candlestick(
                    x=data_plot.index, 
                    high=data_plot['High'], 
                    low=data_plot['Low'],
                    open=data_plot['Open'], 
                    close=data_plot['Close'],
                    name="Candlestick"
                )

                line_pred = go.Scatter(
                    x=pre_df["Date"], 
                    y=pre_df["Prediction"], 
                    name="Prediction", 
                    line={"color": "red"}
                )

                # Adding traces
                viz.add_trace(candle, row=1, col=1)
                viz.add_trace(line_pred, row=1, col=1)

                # Updating layout
                viz.update_layout(
                    template="plotly_dark",
                    height=400,
                    width=1000,
                    title=f"{ticker} Prediction",
                    xaxis_title='Date',
                    yaxis_title='Price',
                    xaxis_rangeslider_visible=False
                )

                st.plotly_chart(viz)
        except:
            st.warning("Invalid Value")


    else:
        st.warning("No data provided")