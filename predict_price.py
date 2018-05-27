import pandas as pd
import time
import numpy as np
from scipy.stats.stats import pearsonr
import os
from PIL import Image
import io
import urllib
from PlotUtils import PlotUtils
from ModelUtils import ModelUtils
import datetime
# import the relevant Keras modules
from keras.models import load_model

coin_choices = {}
coin_choices['1'] = 'bitcoin'
coin_choices['2'] = 'ethereum'

def get_data():
    return

def predict_price(currency, start_date, end_date):
    window_len = 20
    model_path = "models/%s_model.h5" % currency

    market_info = pd.read_html("https://coinmarketcap.com/currencies/%s/historical-data/?start=%s&end=%s" % (currency, start_date, end_date))[0]

    # convert the date string to the correct date format
    market_info = market_info.assign(Date=pd.to_datetime(market_info['Date']))

    temp_list = []
    for i in range(0, len(market_info.columns)):
        if market_info.columns[i] == "Close**":
            temp_list.append("Close")
        elif market_info.columns[i] == "Open*":
            temp_list.append("Open")
        else:
            temp_list.append(market_info.columns[i])

    market_info.columns = temp_list
    # Feature Eng
    # print(market_info.columns)
    market_info.columns = [market_info.columns[0]] + [currency + '_' + i for i in market_info.columns[1:]]
    # print(market_info.columns)
    kwargs = { currency + '_day_diff': lambda x: (x[currency + '_Close'] - x[currency + '_Open']) / x[currency + '_Open']}
    market_info = market_info.assign(**kwargs)

    kwargs = { currency + '_close_off_high': lambda x: 2 * (x[currency + '_High'] - x[currency + '_Close']) / (x[currency + '_High'] - x[currency + '_Low']) - 1,
            currency + '_volatility': lambda x: (x[currency + '_High'] - x[currency + '_Low']) / (x[currency + '_Open'])}
    market_info = market_info.assign(**kwargs)
    model_data = market_info[['Date'] + [currency + "_" + metric for metric in ['Close', 'Volume', 'close_off_high', 'volatility', 'day_diff', 'Market Cap']]]

    # need to reverse the data frame so that subsequent rows represent later timepoints
    model_data = model_data.sort_values(by='Date')
    model_data = model_data.drop('Date', 1)

    norm_cols = [currency + "_" + metric for metric in ['Close', 'Volume', 'Market Cap']]

    LSTM_test_inputs = ModelUtils.buildLstmInput(model_data, norm_cols, window_len)


    LSTM_test_inputs = [np.array(LSTM_test_inputs) for LSTM_test_inputs in LSTM_test_inputs]
    LSTM_test_inputs = np.array(LSTM_test_inputs)

    if os.path.isfile(model_path):
        estimator = load_model(model_path, custom_objects={'r2_keras': ModelUtils.r2_keras})
        return (((np.transpose(estimator.predict(LSTM_test_inputs)) + 1) * model_data[currency + '_Close'].values[:-window_len])[0])[0]
    else:
        print("Please train the model for %s currency" % currency)
        return


if __name__ == "__main__":
    print("""
Choose the coin
===============
1. Bitcoin (BTC)
2. Ethereum (ETH)
          """)
    choice = input("Enter your choice: ")
    past_days = 1

    if choice in coin_choices:
        currency = coin_choices[choice]
    else:
        print("Please enter a correct choice. Program exiting!!!")
        sys.exit(1)

    start_date = (datetime.datetime.now() - datetime.timedelta(days=past_days + 20)).strftime("%Y%m%d")
    end_date = (datetime.datetime.now() - datetime.timedelta(days=past_days)).strftime("%Y%m%d")

    prediction = predict_price(currency, start_date, end_date)
    if prediction is not None:
        print("The predicted price for %s on %s is %s" % (currency, (datetime.datetime.now() - datetime.timedelta(days=past_days-1)).strftime("%d %b, %Y"), prediction))
