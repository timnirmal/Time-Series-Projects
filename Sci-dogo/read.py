import pandas as pd
import numpy as np
#import statsmodels as sm
import statsmodels.api as sm
from matplotlib import pyplot as plt, pyplot
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.arima_model import ARIMAResults
import statistics

hourly_sentiment_series = pd.read_csv('hourly_users_sentiment_subset.csv',
                                      index_col=0,
                                      parse_dates=True).squeeze("columns") # read in the hourly sentiment data


# Data View
"""
print("\nhourly_sentiment_series : ", hourly_sentiment_series)
print("\nhourly_sentiment_series.shape : ", hourly_sentiment_series.shape)
print("\nhourly_sentiment_series.index : ", hourly_sentiment_series.index)
print("\nhourly_sentiment_series.values : ", hourly_sentiment_series.values)
print("\nhourly_sentiment_series.dtype : ", hourly_sentiment_series.dtype)
print("\nhourly_sentiment_series.head() : ", hourly_sentiment_series.head())
print("\nhourly_sentiment_series.tail() : ", hourly_sentiment_series.tail())
"""

# Data Analysis
"""
print("\nhourly_sentiment_series.describe() : ", hourly_sentiment_series.describe())
print("\nhourly_sentiment_series.mean() : ", hourly_sentiment_series.mean())
print("\nhourly_sentiment_series.median() : ", hourly_sentiment_series.median())
print("\nhourly_sentiment_series.std() : ", hourly_sentiment_series.std())
print("\nhourly_sentiment_series.var() : ", hourly_sentiment_series.var())
print("\nhourly_sentiment_series.skew() : ", hourly_sentiment_series.skew())
print("\nhourly_sentiment_series.kurt() : ", hourly_sentiment_series.kurt())
print("\nhourly_sentiment_series.quantile(0.25) : ", hourly_sentiment_series.quantile(0.25))
"""

# Data Visualization
"""
pyplot.plot(hourly_sentiment_series)
pyplot.show()
"""
# As we can see, data is spread out over the whole range of time.
# So we need to differentiate between the data points. Best way is to use a moving average.
# Don't over differentiate the data. (more than 2)

# Differencing the data
#"""
hourly_sentiment_series_diff1 = hourly_sentiment_series.diff().fillna(hourly_sentiment_series)
print("\nhourly_sentiment_series_diff : ", hourly_sentiment_series_diff1)
pyplot.plot(hourly_sentiment_series_diff1)
#pyplot.show()

hourly_sentiment_series_diff2 = hourly_sentiment_series_diff1.diff().fillna(hourly_sentiment_series_diff1)
pyplot.plot(hourly_sentiment_series_diff2)
#pyplot.show()
#"""

plot_acf(hourly_sentiment_series_diff2)
#pyplot.show()

plot_pacf(hourly_sentiment_series_diff2, lags=8)
#pyplot.show()


ARMA1model_hourly_sentiment = sm.tsa.arima.ARIMA(hourly_sentiment_series, order=(5,2,1)).fit()

print(ARMA1model_hourly_sentiment.summary())

# Warnings : [1] Covariance matrix calculated using the outer product of gradients (complex-step).
"""The first one is actually more like a "note" than a "warning". It's just letting you know how the covariance 
matrix was computed. 

The second one is letting you know that parameter estimates may be unstable. Sometimes this is an indication of 
overfitting, but it can also arise from other things. This may indicate that you should try a simpler model (which 
then might forecast better), but it does not mean that you have to do that. """


# Predict the next 5 hours (5 time steps ahead),
# which is the test/holdout set
ARMA1predict_5hourly_sentiment = ARMA1model_hourly_sentiment.predict('2/6/2019  7:00:00 PM','2/6/2019  11:00:00 PM', typ='levels')
print('Forecast/preditions for 5 hours ahead ', ARMA1predict_5hourly_sentiment)

