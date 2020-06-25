---
title: "Example Time Series Analysis"
date: 2020-06-25
tags: [data science, time series analysis, forecasting]
header:
  image:
excerpt: "Using a sales dataset to forecast sales of organic Hass avocados in the US"
mathjax: "true"
---

# Problem:

In order to create a supply chain plan that adequately fulfills the requirements of consumers we must create a forecast for the volume of organic Hass avocados that will be sold in retail channels across the US.

# Plan:

Using the dataset provided by the Hass Avocado Board, which contains weekly retail scan data for National retail volume from January 2015 - March 2018 we will use facebook's prophet library to create and validate a model that forecasts demand. 

# Data:


```python
# import necessary packages

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import warnings
warnings.filterwarnings('ignore')
from fbprophet import Prophet
plt.style.use('fivethirtyeight')
from fbprophet.diagnostics import cross_validation, performance_metrics
from fbprophet.plot import plot_cross_validation_metric
```


```python
# load in the dataset from the hass avocado board

df = pd.read_csv('avocado.csv')
df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Date</th>
      <th>AveragePrice</th>
      <th>Total Volume</th>
      <th>4046</th>
      <th>4225</th>
      <th>4770</th>
      <th>Total Bags</th>
      <th>Small Bags</th>
      <th>Large Bags</th>
      <th>XLarge Bags</th>
      <th>type</th>
      <th>year</th>
      <th>region</th>
      <th>Month</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2015-12-27</td>
      <td>1.33</td>
      <td>64237.0</td>
      <td>1036.74</td>
      <td>54454.85</td>
      <td>48.16</td>
      <td>8696.87</td>
      <td>8603.62</td>
      <td>93.25</td>
      <td>0.0</td>
      <td>conventional</td>
      <td>2015</td>
      <td>Albany</td>
      <td>2015-12-01</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2015-12-20</td>
      <td>1.35</td>
      <td>54877.0</td>
      <td>674.28</td>
      <td>44638.81</td>
      <td>58.33</td>
      <td>9505.56</td>
      <td>9408.07</td>
      <td>97.49</td>
      <td>0.0</td>
      <td>conventional</td>
      <td>2015</td>
      <td>Albany</td>
      <td>2015-12-01</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2015-12-13</td>
      <td>0.93</td>
      <td>118220.0</td>
      <td>794.70</td>
      <td>109149.67</td>
      <td>130.50</td>
      <td>8145.35</td>
      <td>8042.21</td>
      <td>103.14</td>
      <td>0.0</td>
      <td>conventional</td>
      <td>2015</td>
      <td>Albany</td>
      <td>2015-12-01</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2015-12-06</td>
      <td>1.08</td>
      <td>78992.0</td>
      <td>1132.00</td>
      <td>71976.41</td>
      <td>72.58</td>
      <td>5811.16</td>
      <td>5677.40</td>
      <td>133.76</td>
      <td>0.0</td>
      <td>conventional</td>
      <td>2015</td>
      <td>Albany</td>
      <td>2015-12-01</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2015-11-29</td>
      <td>1.28</td>
      <td>51040.0</td>
      <td>941.48</td>
      <td>43838.39</td>
      <td>75.78</td>
      <td>6183.95</td>
      <td>5986.26</td>
      <td>197.69</td>
      <td>0.0</td>
      <td>conventional</td>
      <td>2015</td>
      <td>Albany</td>
      <td>2015-11-01</td>
    </tr>
  </tbody>
</table>
</div>



# Analysis:


```python
# create a function that will print out a mean absolute percentage error based on your forecast and actuals 
def mean_absolute_percentage_error(y_actual, y_pred):
    y_actual, y_pred = np.array(y_actual), np.array(y_pred)
    return np.mean(np.abs((y_actual - y_pred) / y_actual))

# remove the totals section of the dataframe
time_df = df[df['region'] != 'TotalUS']
# filter for just the organic sales
time_df = time_df[time_df['type'] == 'organic']
# to do the time series analysis we only need date and volume so filter for just those
time_df = time_df[['Month', 'Total Volume']]
# group the data by month, as it is originally presented by week
time_df = time_df.groupby(['Month'])['Total Volume'].sum().reset_index()
time_df.rename(columns={'Month': 'ds', 'Total Volume': 'y'}, inplace=True)
time_df['ds'] = pd.to_datetime(time_df['ds'])
# instantiate the model
model = Prophet(yearly_seasonality=15, 
             n_changepoints=15,
             interval_width=0.95,   
             changepoint_range=0.85, 
             changepoint_prior_scale=0.03,
             seasonality_mode='multiplicative')
# fit the model to our dataframe
model.fit(time_df)
future = model.make_future_dataframe(periods=24, freq='MS')
forecast = model.predict(future)
fig = model.plot(forecast)
print(fig)
print(model.plot_components(forecast))
```

    INFO:fbprophet:Disabling weekly seasonality. Run prophet with weekly_seasonality=True to override this.
    INFO:fbprophet:Disabling daily seasonality. Run prophet with daily_seasonality=True to override this.
    

    Figure(720x432)
    Figure(648x432)
    


![png](output_6_2.png)



![png](output_6_3.png)


As one might expect based on media coverage and pop culture references, demand for avocados has grown quite a lot from 2015 to 2018. This model and graphic allows us to visualize that quite clearly. It also gives us the opportunity to clearly visualize some basic seasonality that we are seeing in the data. Looking at the above graphs, we can see that demand for organic Hass avocados usually peaks in the first half of the year. This can help inform our supply plan as it allows us to plan capacity and resources accordingly. 

Before we put too much stock in this model, let's do some validation to make sure that it is useful:


```python
''' I don't know much about the avocado industry but I am making the assumption that forecasting 120 days 
out will allow our supply chain team enough time to alter our supply plan to keep meeting the requirements of our customers.

Therefore I am using a horizon of 120 days in our validation to evaluate our model''' 

df_cv = cross_validation(model,horizon = '120 days')
mape_baseline = mean_absolute_percentage_error(df_cv.y, df_cv.yhat)
print("The MAPE of our model is: " + "{:.2%}".format(mape_baseline))
df_perf = performance_metrics(df_cv)
fig2 = plot_cross_validation_metric(df_cv, metric='mape')
print(fig2)
```

    INFO:fbprophet:Making 12 forecasts with cutoffs between 2016-01-11 00:00:00 and 2017-11-01 00:00:00
    INFO:fbprophet:n_changepoints greater than number of observations.Using 10.0.
    INFO:fbprophet:n_changepoints greater than number of observations.Using 11.0.
    INFO:fbprophet:n_changepoints greater than number of observations.Using 13.0.
    

    The MAPE of our model is: 15.79%
    Figure(720x432)
    


![png](output_8_2.png)


After tweaking the parameters of our model, we are able to achieve a mean absolute percentage error of 15.79%. This means that on average our forecasts have an absolute error value of about 15% or in other words our forecast is about 84% accurate. While not perfect, if we continue to tweak the parameters we may risk overfitting the model which may make our future predictions worse.  

While MAPE is often the standard to measure forecast accuracy it does have it's shortcomings. One of those shortcomings is that because it uses absolute values, it does not show if our model is biased (i.e. are we consistently over/under forecasting). One way we can check the bias of our model is to examine the distribution of the actual data and plot it against the distribution of our predictions:


```python
sns.distplot(df_cv.yhat, color="red", label="Predictions")
sns.distplot(df_cv.y, color="green", label="Actuals")
plt.legend() # add legend to plot (nust include 'label' arguments in above lines for this to work)
plt.show()
```


![png](output_10_0.png)


This plot shows that neither the actual sales nor our predictions are terribly skewed in one direction or the other. This is good news and shows that our model is not consistently predicting sales as too high or too low. 

If at this point we are satisfied with our predictions we can export to excel in order to use or distribute our forecasts:


```python
forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].to_excel('avocado volume predicitons.xlsx')
```

# Conclusions:

Using only historical sales we were able to create a forecast that is approximately 84% accurate when evaluating the model using simulated historical forecasts. I like implementing Prophet forecasts into my demand planning as it provides the opportunity for domain experts to intervene in the forecast and tweak parameters quite easily. Additionally, the Prophet library handles outliers and missing data quite well, which again gives the domain expert the opportunity to manually intervene in the forecast. 

Some areas of further exploration could include examining the relationship between price and volume to determine how elastic demand is for this industry, examining trends or seasonality across different cities to find areas where growth is happening faster or peaking at different times of the year than the rest of the country.
