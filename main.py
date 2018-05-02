import datetime
import sched
import time

import pandas as pd

import predict_price as pp
import bot_helper as bh
import model_training as mt


s = sched.scheduler(time.time, time.sleep)

def get_prev_day_price(currency):
    market_info = pd.read_html("https://coinmarketcap.com/currencies/%s/historical-data/?start=%s&end=%s" % (currency, (datetime.datetime.now() - datetime.timedelta(days=1)).strftime("%Y%m%d"), datetime.datetime.now().strftime("%Y%m%d")))[0]
    return market_info['Close'][0]

def price_notification():
    past_days = 1
    start_date = (datetime.datetime.now() - datetime.timedelta(days=past_days + 20)).strftime("%Y%m%d")
    end_date = (datetime.datetime.now() - datetime.timedelta(days=past_days)).strftime("%Y%m%d")

    for currency in ['bitcoin', 'ethereum']:
        prediction = float(pp.predict_price(currency, start_date, end_date))
        prev_price = float(get_prev_day_price(currency))
        percent_change = (prediction - prev_price) / prev_price * 100


        if prediction is not None:
            prediction = "%.2f" % prediction
            percent_change = "%.2f" % percent_change
            #output = "The predicted price for %s on %s is: %s" % (currency.upper(), (datetime.datetime.now() - datetime.timedelta(days=past_days-1)).strftime("%d %b, %Y"), prediction)
            seperator = "=" * (len((datetime.datetime.now() - datetime.timedelta(days=past_days-1)).strftime("%d %b, %Y")) + len(currency))
            output = "%s - %s\n%s\nPredicted Price = %s,\nA Change of %s %% from yesterday’s price of %s" % (currency.upper(), (datetime.datetime.now() - datetime.timedelta(days=past_days-1)).strftime("%d %b, %Y"), seperator, prediction, percent_change, prev_price)
            #output = "%s - %s\n%s\nPredicted Price = %s,\nChange of %s  from yesterday’s price of %s" % (currency.upper(), (datetime.datetime.now() - datetime.timedelta(days=past_days-1)).strftime("%d %b, %Y"), seperator, prediction, percent_change, prev_price)
            bh.send_message(output)
    print("Sending telegram updates complete!")

def day_job_runner():
    from_date = '20170101'
    to_date = (datetime.datetime.utcnow() - datetime.timedelta(days=1)).strftime("%Y%m%d")

    for currency in ['bitcoin', 'ethereum']:
        mt.train_model(currency, from_date, to_date, "models/"+currency+"_model.h5")

    price_notification()

    # Scheduling the next call for the methods.
    # current_date = datetime.datetime.now().strftime("%Y,%m,%d").split(',')
    # next_date = datetime.datetime(int(current_date[0]), int(current_date[1]), int(current_date[2]))
    # next_date += datetime.timedelta(days=1)
    current_date = datetime.datetime.now().strftime("%Y,%m,%d,%H,%M").split(',')
    next_date = datetime.datetime(int(current_date[0]), int(current_date[1]), int(current_date[2]), int(current_date[3]), int(current_date[4]))
    next_date += datetime.timedelta(minutes=1)
    next_timestamp = next_date.timestamp()

    s.enterabs(next_timestamp, 1, day_job_runner)
    s.run()

if __name__ == "__main__":
    day_job_runner()
