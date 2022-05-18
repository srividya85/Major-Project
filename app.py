import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
#import tensorflow as tf

from keras.models import load_model
import streamlit as st
import pandas_datareader as data
import plotly.graph_objs as go
from keras.preprocessing.sequence import TimeseriesGenerator

start='2015-01-01'
end = '2022-02-06'    

st.title('Stock Price Prediction')
user_input=st.text_input('Enter Stock Ticker','ONGC.NS')
df=data.DataReader(user_input,'yahoo',start,end)

#Describing the data
st.subheader('Data from 2015 - 2021')
st.write(df.describe())
#st.write(df.head())

df['Date'] = df.index
df['Date'] = pd.to_datetime(df['Date'])
                                                                     
trace = go.Scatter(
    x = df.Date,
    y = df.Close,
    mode = 'lines',
    name = 'Data'
)
layout = go.Layout(
    title = "",
    xaxis = {'title' : "Date"},
    yaxis = {'title' : "Close"}
)
fig = go.Figure(data=[trace], layout=layout)
#fig.show()             
st.plotly_chart(fig)              

close_data = df['Close'].values
close_data = close_data.reshape((-1,1))

split_percent = 0.80
split = int(split_percent*len(close_data))

close_train = close_data[:split]
close_test = close_data[split:]

date_train = df['Date'][:split]
date_test = df['Date'][split:]

look_back = 15

train_generator = TimeseriesGenerator(close_train, close_train, length=look_back, batch_size=20)     
test_generator = TimeseriesGenerator(close_test, close_test, length=look_back, batch_size=1)

model=load_model('keras_model.h5')


#import plotly.graph_objs as go
prediction = model.predict_generator(test_generator)

close_train = close_train.reshape((-1))
close_test = close_test.reshape((-1))
prediction = prediction.reshape((-1))

trace1 = go.Scatter(
    x = date_train,
    y = close_train,
    mode = 'lines',
    name = 'Data'
)
trace2 = go.Scatter(
    x = date_test,
    y = prediction,
    mode = 'lines',
    name = 'Prediction'
)
trace3 = go.Scatter(
    x = date_test,
    y = close_test,
    mode='lines',
    name = 'Ground Truth'
)
layout = go.Layout(
    title = "Stock",
    xaxis = {'title' : "Date"},
    yaxis = {'title' : "Close"}
)
fig = go.Figure(data=[trace1, trace2, trace3], layout=layout)
#fig.show()
st.plotly_chart(fig)  

close_data = close_data.reshape((-1))


def predict(num_prediction, model):
    prediction_list = close_data[-look_back:]
    
    for _ in range(num_prediction):
        x = prediction_list[-look_back:]
        x = x.reshape((1, look_back, 1))
        out = model.predict(x)[0][0]
        prediction_list = np.append(prediction_list, out)
    prediction_list = prediction_list[look_back-1:]
        
    return prediction_list
    
def predict_dates(num_prediction):
    last_date = df['Date'].values[-1]
    prediction_dates = pd.date_range(last_date, periods=num_prediction+1).tolist()
    return prediction_dates

num_prediction = 30
forecast = predict(num_prediction, model)
forecast_dates = predict_dates(num_prediction)

trace1 = go.Scatter(
    x = df['Date'].tolist(),
    y = close_data,
    mode = 'lines',
    name = 'Data'
)
trace2 = go.Scatter(
    x = forecast_dates,
    y = forecast,
    mode = 'lines',
    name = 'Prediction'
)
layout = go.Layout(
    title = "Future prices of Stock",
    xaxis = {'title' : "Date"},
    yaxis = {'title' : "Close"}
)

fig = go.Figure(data=[trace1, trace2], layout=layout)
#fig.show()
st.plotly_chart(fig)  
