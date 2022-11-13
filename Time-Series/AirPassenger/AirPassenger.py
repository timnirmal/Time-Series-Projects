import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
plt.style.use('dark_background')

#============================================ 1. Load Data
df = pd.read_csv('AirPassengers.csv')
#df.plot()
#plt.show()

df['Month'] = pd.to_datetime(df['Month'])
df.set_index('Month', inplace=True)
#df.plot()
#plt.show()

#============================================ 3. Stationary Test

# Check if data is stationary
from statsmodels.tsa.stattools import adfuller
adf, pvalue, used_lag, nobs, critical_values, icbest = adfuller(df['#Passengers'])
print('ADF Statistic: %f' % adf)
print('Used lag: %d' % used_lag)
print('p-value: %f' % pvalue)
print('Number of observations: %d' % nobs)
print('Critical values: %s' % critical_values)
print('Best-fit model: %s' % icbest)
# print("If the p-value is less than the critical value, then we reject the null hypothesis that the time series is
# stationary.")
print("If p-value is above 0.05, data is not stationary.")

"""
# Since data is not stationary, we need to transform the data to make it stationary.
# Transform data to log-scale
df_log = np.log(df['#Passengers'])
df_log.plot()
plt.show()
"""

# Since data is not stationary, SARIMA model is best suite.

#============================================ 3. Exploring Yealy and Monthly Data

df["year"] = [d.year for d in df.index]
df["month"] = [d.strftime("%b") for d in df.index]
year = df["year"].unique()
print(df.head())

# We can see if there is any pattern in the data year-wise and month-wise.

"""
df_year = df.groupby(["year"]).mean()
df_month = df.groupby(["month"]).mean()
print(df_year.head())
print(df_month.head())
# plot
fig, ax = plt.subplots(figsize=(12, 8))
#ax.plot(df_year.index, df_year['#Passengers'], label='Year')
ax.plot(df_month.index, df_month['#Passengers'], label='Month')
ax.legend(loc='best')
plt.show()
"""

#sns.set(style="darkgrid")
sns.boxplot(x="year", y="#Passengers", data=df) # Trend
plt.show()
sns.boxplot(x="month", y="#Passengers", data=df) # Seasonality
plt.show()


#============================================ 4. Extract plot Trends, Seasonality and Residuals

from statsmodels.tsa.seasonal import seasonal_decompose
decomposition = seasonal_decompose(df['#Passengers'], model='additive')
fig = decomposition.plot()
plt.show()

# additive time series model
# value = Base level + Trend + Seasonality + Residuals
# Multiplicative time series model
# value = Base level X Trend X Seasonality X Residuals

plt.figure(figsize=(12, 8))
plt.subplot(411)
plt.plot(df['#Passengers'], label='Original', color='yellow')
plt.legend(loc='best')
plt.subplot(412)
plt.plot(decomposition.trend, label='Trend', color='red')
plt.legend(loc='best')
plt.subplot(413)
plt.plot(decomposition.seasonal, label='Seasonality', color='green')
plt.legend(loc='best')
plt.subplot(414)
plt.plot(decomposition.resid, label='Residuals', color='blue')
plt.legend(loc='best')
plt.tight_layout()
plt.show()

#============================================ 5. Autocorrelation and Partial Autocorrelation

# Correlation is a measure of how much two variables vary together.
# Autocorrelation is a measure of how much a variable varies over time.
# Partial Autocorrelation is a measure of how much a variable varies over time with respect to other variables.

# Autocorrelation is correlated of series with its own lags.
# Any correlation above significant lnes are statistically significant.

from statsmodels.tsa.stattools import acf, pacf

acf_values = acf(df['#Passengers'], nlags=144)
pacf_values = pacf(df['#Passengers'], nlags=71, method='ols')
plt.plot(acf_values)
plt.show()
plt.plot(pacf_values)
plt.show()

# Obtain same with single line and more info
from pandas.plotting import autocorrelation_plot
autocorrelation_plot(df['#Passengers'])
plt.show()
# Show with 95% and 99% confidence interval