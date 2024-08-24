import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler

sns.set_style('darkgrid')

st.header('LSTM Deep Learning 6H Forecast Prototype Model')
st.write('All csv files on the NEM site for Aggregated price and demand data should work')
st.link_button('Grab a file to upload', 'https://aemo.com.au/energy-systems/electricity/national-electricity-market-nem/data-nem/aggregated-data')

st.write('Upload your Aggregated price and demand CSV to predict the future 6 hours of RRP')

data = st.file_uploader('Choose a file')
st.write('Give it a minute to analyse data to produce forecast.')
st.write('The orange prediction line shows you how well it predicts the actual data in blue.')
st.write('The more it overlaps the blue, the better the predictions and forecasts.')
if data is not None:
    # Read the CSV data
    x = pd.read_csv(data)

    # Preprocess data
    q1 = x.RRP.quantile(0.25)
    q3 = x.RRP.quantile(0.75)
    iqr = q3 - q1
    min_val = q1 - 1.5 * iqr
    max_val = q3 + 1.5 * iqr
    x.RRP = np.where((x.RRP > max_val) | (x.RRP < min_val), x.RRP.median(), x.RRP)
    x.RRP = np.where(x.RRP < 0, x.RRP.median(), x.RRP)

    x.SETTLEMENTDATE = pd.to_datetime(x.SETTLEMENTDATE, format='mixed')
    x['day_of_week'] = x.SETTLEMENTDATE.dt.dayofweek
    day = 3600 * 24
    x['seconds'] = x.SETTLEMENTDATE.dt.hour * 3600 + x.SETTLEMENTDATE.dt.minute * 60
    x['day_sin'] = np.sin(x.seconds * 2 * np.pi / day)
    x['day_cos'] = np.cos(x.seconds * 2 * np.pi / day)
    x['day'] = x.SETTLEMENTDATE.dt.day  # Corrected from 'Date' to 'SETTLEMENTDATE'
    xx = x[['RRP']]
    x = x[['TOTALDEMAND', 'RRP', 'day_sin', 'day_cos', 'day']]

    # Reshape `xx` to ensure it's a 2D array for MinMaxScaler
    xx = xx.values.reshape(-1, 1)

    def data_window(data, target, input_length, output_length):
        X, y = [], []
        target_columns = target.columns

        for i in range(len(data) - input_length - output_length + 1):
            # Past data and features
            past_data = data.iloc[i : i + input_length]
            past_features = past_data.drop(target_columns, axis=1).values

            # Correctly slicing the corresponding past target values
            past_target = past_data[target_columns].values

            # Concatenate past features and target
            X.append(np.concatenate([past_features, past_target], axis=1))

            # Future target values
            future_target = target.iloc[i + input_length : i + input_length + output_length].values
            y.append(future_target)

        return np.array(X), np.array(y)

    mmx = MinMaxScaler()
    mmxx = MinMaxScaler()

    # Scaling the features and target
    x_s = mmx.fit_transform(x)
    xx_s = mmxx.fit_transform(xx)  # Now correctly reshaped

    x_s_df = pd.DataFrame(x_s, columns=x.columns)
    xx_s_df = pd.DataFrame(xx_s, columns=['RRP'])

    X, y = data_window(x_s_df.copy(), xx_s_df.copy(), 144, 144)

    model = tf.keras.models.load_model('model6h.keras')
    p1 = model.predict(X)
    y00 = mmxx.inverse_transform([pd.DataFrame(y.reshape(-1, 144))[0]])
    p00 = mmxx.inverse_transform([pd.DataFrame(p1)[0]])
    fu1 = model.predict(X[-1].reshape(1, 144, 5))
    fut6 = mmxx.inverse_transform(fu1)

    # Create a figure and axis object
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(y00[0], label='Actual Values')
    ax.plot(p00[0], label='Model Predictions')
    ax.plot(range(len(p00[0]), len(p00[0]) + 144), fut6[0], label='Model Forecast')
    ax.legend()
    ax.set_ylabel('RRP')
    
    # Display the plot in Streamlit
    st.pyplot(fig)

    f6 = pd.DataFrame(fut6[0])
    f6.columns = ['6H Forecast']
    start_date = "2024/08/19 00:05:00"
    date_range = pd.date_range(start=start_date, periods=144, freq='5T')
    f6['Date'] = date_range
    st.write(f6)
