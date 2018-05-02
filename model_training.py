"""
This algorithm is a modification of :
    https://github.com/umbertogriffo/An-Experiment-On-Predicting-Cryptocurrency-Prices-With-LSTM

There are few limitations and unknowns in the above implementation,
which I am trying to resolve by creating my own minor modified algorithm.

This file contains the training part of the entire algorithm.
The training will be done on a specific set of date.
Later, an improvement planned is to include logic for daily training,
which enhances the performance as new data is fed into the model.

"""

import datetime
import sys
import io
import urllib

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from PIL import Image
import keras
from PlotUtils import PlotUtils
from ModelUtils import ModelUtils

coin_choices = {}
coin_choices['1'] = 'bitcoin'
coin_choices['2'] = 'ethereum'

def train_model(currency, from_date, to_date, model_path):
    # Splitting the data into 65-35 ratio. 65% point will be the split date.
    fd = datetime.datetime(int(from_date[0:4]), int(from_date[4:6]), int(from_date[6:]))
    td = datetime.datetime(int(to_date[0:4]), int(to_date[4:6]), int(to_date[6:]))
    delta = 0.65 * (td - fd)
    split_date = fd + datetime.timedelta(days = int(str(delta).split()[0]))

    # Our LSTM model will use previous data to predict the next day's closing price of eth.
    # We must decide how many previous days it will have access to
    window_len = 20
    eth_epochs = 100
    eth_batch_size = 32
    num_of_neurons_lv1 = 50
    num_of_neurons_lv2 = 25

    # Get Market info
    market_info = pd.read_html("https://coinmarketcap.com/currencies/%s/historical-data/?start=%s&end=%s" % (currency, from_date, to_date))[0]

    # convert the date string to the correct date format
    market_info = market_info.assign(Date=pd.to_datetime(market_info['Date']))
    market_info.columns = [market_info.columns[0]] + [currency + '_' + i for i in market_info.columns[1:]]

    kwargs = { currency + '_day_diff': lambda x: (x[currency + '_Close'] - x[currency + '_Open']) / x[currency + '_Open']}
    market_info = market_info.assign(**kwargs)

    kwargs = { currency + '_close_off_high': lambda x: 2 * (x[currency + '_High'] - x[currency + '_Close']) / (x[currency + '_High'] - x[currency + '_Low']) - 1,
            currency + '_volatility': lambda x: (x[currency + '_High'] - x[currency + '_Low']) / (x[currency + '_Open'])}
    market_info = market_info.assign(**kwargs)

    model_data = market_info[['Date'] + [currency + "_" + metric for metric in ['Close', 'Volume', 'close_off_high', 'volatility', 'day_diff', 'Market Cap']]]
    model_data = model_data.sort_values(by='Date')

    training_set, test_set = model_data[model_data['Date'] < split_date], model_data[model_data['Date'] >= split_date]

    # we don't need the date columns anymore
    training_set = training_set.drop('Date', 1)
    test_set = test_set.drop('Date', 1)

    norm_cols = [currency + "_" + metric for metric in ['Close', 'Volume', 'Market Cap']]

    LSTM_training_inputs = ModelUtils.buildLstmInput(training_set, norm_cols, window_len)
    LSTM_training_outputs = ModelUtils.buildLstmOutput(training_set, currency + '_Close', window_len)

    LSTM_test_inputs = ModelUtils.buildLstmInput(test_set, norm_cols, window_len)
    LSTM_test_outputs = ModelUtils.buildLstmOutput(test_set, currency + '_Close', window_len)

    LSTM_training_inputs = [np.array(LSTM_training_input) for LSTM_training_input in LSTM_training_inputs]
    LSTM_training_inputs = np.array(LSTM_training_inputs)

    LSTM_test_inputs = [np.array(LSTM_test_inputs) for LSTM_test_inputs in LSTM_test_inputs]
    LSTM_test_inputs = np.array(LSTM_test_inputs)

    # initialise model architecture
    eth_model = ModelUtils.build_model(LSTM_training_inputs, output_size=1, neurons_lv1=num_of_neurons_lv1, neurons_lv2=num_of_neurons_lv2)

    # train model on data
    eth_history = eth_model.fit(LSTM_training_inputs, LSTM_training_outputs,
                                epochs=eth_epochs, batch_size=eth_batch_size, verbose=2, shuffle=True,
                                validation_split=0.2,
                                callbacks=[keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=10, mode='min'),
                                           keras.callbacks.ModelCheckpoint(model_path, monitor='val_loss', save_best_only=True, mode='min', verbose=0)])

    # We've just built an LSTM model to predict tomorrow's Ethereum closing price.
    scores = eth_model.evaluate(LSTM_test_inputs, LSTM_test_outputs, verbose=1, batch_size=eth_batch_size)
    print('\nMSE: {}'.format(scores[1]))
    print('\nMAE: {}'.format(scores[2]))
    print('\nR^2: {}'.format(scores[3]))

    # Plot Error
    figErr, ax1 = plt.subplots(1, 1)
    ax1.plot(eth_history.epoch, eth_history.history['loss'])
    ax1.set_title('Training Error')
    if eth_model.loss == 'mae':
        ax1.set_ylabel('Mean Absolute Error (MAE)', fontsize=12)
    # just in case you decided to change the model loss calculation
    else:
        ax1.set_ylabel('Model Loss', fontsize=12)
    ax1.set_xlabel('# Epochs', fontsize=12)
    #plt.show()
    figErr.savefig("output/%s_error.png" % currency)

    #####################################
    # EVALUATE ON TEST DATA
    #####################################

    # Plot Performance
    fig, ax1 = plt.subplots(1, 1, figsize=(10, 10))
    ax1.set_xticks([datetime.date(2017, i + 1, 1) for i in range(12)])
    ax1.set_xticklabels([datetime.date(2017, i + 1, 1).strftime('%b %d %Y')  for i in range(12)])
    ax1.plot(model_data[model_data['Date'] >= split_date]['Date'][window_len:].astype(datetime.datetime),
             test_set[currency + '_Close'][window_len:], label='Actual')
    ax1.plot(model_data[model_data['Date'] >= split_date]['Date'][window_len:].astype(datetime.datetime),
             ((np.transpose(eth_model.predict(LSTM_test_inputs)) + 1) * test_set[currency + '_Close'].values[:-window_len])[0],
             label='Predicted')
    ax1.annotate('MAE: %.4f' % np.mean(np.abs((np.transpose(eth_model.predict(LSTM_test_inputs)) + 1) - \
                (test_set[currency + '_Close'].values[window_len:]) / (test_set[currency + '_Close'].values[:-window_len]))),
                 xy=(0.75, 0.9), xycoords='axes fraction',
                xytext=(0.75, 0.9), textcoords='axes fraction')
    ax1.set_title('Test Set: Single Timepoint Prediction', fontsize=13)
    ax1.set_ylabel('Ethereum Price ($)', fontsize=12)
    ax1.legend(bbox_to_anchor=(0.1, 1), loc=2, borderaxespad=0., prop={'size': 14})
    #plt.show()
    fig.savefig("output/%s_performanceTraining.png" % currency)


    return

if __name__ == "__main__":
    # This block is executed when the script is directly run from the terminal.
    # Else, the method call has to be made for an exclusive call to the logic.

    # This script trains the algorithm for the currency provided by the user,
    # the dates can be choosen by the user else a default date range will be taken.
    print("""
Choose the coin
===============
1. Bitcoin (BTC)
2. Ethereum (ETH)
          """)
    choice = input("Enter your choice: ")

    if choice in coin_choices:
        currency = coin_choices[choice]
    else:
        print("Please enter a correct choice. Program exiting!!!")
        sys.exit(1)

    from_date = '20170101'
    to_date = (datetime.datetime.utcnow() - datetime.timedelta(days=1)).strftime("%Y%m%d")
    train_model(currency, from_date, to_date, "models/"+currency+"_model.h5")
