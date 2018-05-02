import datetime
import sched
import time

import predict_price as pp
import bot_helper as bh
import model_training as mt


s = sched.scheduler(time.time, time.sleep)

def price_notification():
    past_days = 1
    start_date = (datetime.datetime.now() - datetime.timedelta(days=past_days + 20)).strftime("%Y%m%d")
    end_date = (datetime.datetime.now() - datetime.timedelta(days=past_days)).strftime("%Y%m%d")

    for currency in ['bitcoin', 'ethereum']:
        prediction = pp.predict_price(currency, start_date, end_date)

        if prediction is not None:
            output = "The predicted price for %s on %s is: %s" % (currency.upper(), (datetime.datetime.now() - datetime.timedelta(days=past_days-1)).strftime("%d %b, %Y"), prediction)
            bh.send_message(output)
    print("Sending telegram updates complete!")

def day_job_runner():
    from_date = '20170101'
    to_date = (datetime.datetime.utcnow() - datetime.timedelta(days=1)).strftime("%Y%m%d")

    for currency in ['bitcoin', 'ethereum']:
        mt.train_model(currency, from_date, to_date, "models/"+currency+"_model.h5")

    price_notification()

    # Scheduling the next call for the methods.
    current_date = datetime.datetime.now().strftime("%Y,%m,%d").split(',')
    next_date = datetime.datetime(int(current_date[0]), int(current_date[1]), int(current_date[2]))
    next_date += datetime.timedelta(days=1)
    # current_date = datetime.datetime.now().strftime("%Y,%m,%d,%H,%M").split(',')
    # next_date = datetime.datetime(int(current_date[0]), int(current_date[1]), int(current_date[2]), int(current_date[3]), int(current_date[4]))
    # next_date += datetime.timedelta(minutes=1)
    next_timestamp = next_date.timestamp()

    s.enterabs(next_timestamp, 1, day_job_runner)
    s.run()

if __name__ == "__main__":
    day_job_runner()
