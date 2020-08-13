---
title: "Demand Slope Analysis"
date: 2020-07-29
tags: [data science, price elasticity of demand, forecasting]
header:
  image:
excerpt: "Using least squares regression on an online sales dataset to determine the relationship between price and demand for shoes"
mathjax: "true"
---

# Problem:

Our company does not know the effect that changing unit price has on demand for shoes. This gap in knowledge makes it hard to calculate return on investment for discounts, forecast demand for sales, and optimze pricing to improve margin. 

# Plan:

We are going to use ordinary least squares linear regression analysis to determine if there is a strong relationship between price and demand for our weekly online sales data. If we deem the relationship strong, we will build a tool that lets users experiment with potential prices to see the effect that a price change will have on demand.

# Data:


```python
# import necessary packages

from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.compat import lzip
import seaborn as sns
from fbprophet import Prophet
import warnings
warnings.filterwarnings('ignore')
plt.style.use('seaborn-dark-palette')
```


```python
#load and clean dataset

df = pd.read_excel('Dummy Online Sales Data.xlsx')
df['SKU'] = df['SKU'].astype('str')
df.rename(columns={'Units Ordered': 'Quantity', 'Unit Price': 'Price', 'Page Views': 'PageViews'}, inplace=True)
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
      <th>SKU</th>
      <th>Product Description</th>
      <th>Sessions</th>
      <th>PageViews</th>
      <th>Quantity</th>
      <th>Week Of</th>
      <th>Sales</th>
      <th>Price</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>64846AP01-11</td>
      <td>Orange Running Shoes</td>
      <td>68</td>
      <td>88</td>
      <td>16</td>
      <td>2019-01-01</td>
      <td>1380.309333</td>
      <td>86.269333</td>
    </tr>
    <tr>
      <th>1</th>
      <td>64846AP01-11</td>
      <td>Orange Running Shoes</td>
      <td>72</td>
      <td>111</td>
      <td>14</td>
      <td>2019-01-08</td>
      <td>1227.040000</td>
      <td>87.645714</td>
    </tr>
    <tr>
      <th>2</th>
      <td>64846AP01-11</td>
      <td>Orange Running Shoes</td>
      <td>71</td>
      <td>100</td>
      <td>16</td>
      <td>2019-01-15</td>
      <td>1438.050000</td>
      <td>89.878125</td>
    </tr>
    <tr>
      <th>3</th>
      <td>64846AP01-11</td>
      <td>Orange Running Shoes</td>
      <td>69</td>
      <td>101</td>
      <td>16</td>
      <td>2019-01-22</td>
      <td>1430.682353</td>
      <td>89.417647</td>
    </tr>
    <tr>
      <th>4</th>
      <td>64846AP01-11</td>
      <td>Orange Running Shoes</td>
      <td>70</td>
      <td>115</td>
      <td>9</td>
      <td>2019-01-29</td>
      <td>909.900616</td>
      <td>101.100068</td>
    </tr>
  </tbody>
</table>
</div>



# Analysis:

Let's build a model to examine the relationship between quantity ordered and price and while we're at it let's print out some charts that help visualize the model's variances.


```python
price_model = ols("Quantity ~ Price", data=df).fit()
print(price_model.summary())
fig = plt.figure(figsize=(12,8))
fig = sm.graphics.plot_partregress_grid(price_model, fig=fig)
print(fig)
fig2 = plt.figure(figsize=(12, 8))
fig2 = sm.graphics.plot_ccpr_grid(price_model, fig=fig2)
print(fig2)
fig3 = plt.figure(figsize=(12,8))
fig3 = sm.graphics.plot_regress_exog(price_model, 'Price', fig=fig3)
print(fig3)
```

                                OLS Regression Results                            
    ==============================================================================
    Dep. Variable:               Quantity   R-squared:                       0.778
    Model:                            OLS   Adj. R-squared:                  0.773
    Method:                 Least Squares   F-statistic:                     178.3
    Date:                Tue, 28 Jul 2020   Prob (F-statistic):           2.82e-18
    Time:                        18:11:13   Log-Likelihood:                -130.16
    No. Observations:                  53   AIC:                             264.3
    Df Residuals:                      51   BIC:                             268.3
    Df Model:                           1                                         
    Covariance Type:            nonrobust                                         
    ==============================================================================
                     coef    std err          t      P>|t|      [0.025      0.975]
    ------------------------------------------------------------------------------
    Intercept     70.4590      4.220     16.698      0.000      61.988      78.930
    Price         -0.6041      0.045    -13.354      0.000      -0.695      -0.513
    ==============================================================================
    Omnibus:                       20.303   Durbin-Watson:                   2.275
    Prob(Omnibus):                  0.000   Jarque-Bera (JB):               29.010
    Skew:                           1.336   Prob(JB):                     5.02e-07
    Kurtosis:                       5.449   Cond. No.                         997.
    ==============================================================================
    
    


<img src="{{ site.url }}{{ site.baseurl }}/images/output_7_1.png">



<img src="{{ site.url }}{{ site.baseurl }}/images/output_7_2.png">



<img src="{{ site.url }}{{ site.baseurl }}/images/output_7_3.png">


These are some interesting results. Quite quickly we can see that there is a pretty high R-Squared value (0.778) indicating that price and units ordered are highly correlated. Additionally, looking at the p value and f statistic supports our belief that price does have an impact on the amount of units ordered. Looking at the charts above we can also see that the errors in the model appear to be homoscedastic which is good news because it means that there aren't patterns in our errors.

This is a lot of statistics but **the bottom line is that all of this points to us being able to trust this model**. 

To emphasize this point, let's take a look below at the regression (line) plotted against our actual observed values (dots).


```python
sns.pairplot(df, kind='reg', x_vars=['Price'], y_vars=['Quantity'], height=6, aspect=2)
```



<img src="{{ site.url }}{{ site.baseurl }}/images/output_9_1.png">


This helps us visualize how our model would perform in real world data and we can see that generally our model is pretty close to the observed values. 

This model looks promising so far, however, I know a question that I had when exploring the data was "how did trends and seasonality affect the results of the model?". 

Let's examine the time series data to give further context and check for stationarity. This is important because a steadily increasing average or a singular event (such as a celebrity endorsement for Orange Running Shoes) could be distorting the effect of price on quantity ordered.


```python
# build a time series forecast to visually examine trends and seasonality

time_df = df[['Week Of', 'Quantity']]
time_df.rename(columns = {'Week Of': 'ds', 'Quantity': 'y'}, inplace=True)
m = Prophet(changepoint_prior_scale=0.07, 
            changepoint_range=0.9, 
            yearly_seasonality=4, 
            seasonality_mode='additive')
m.fit(time_df)
future = m.make_future_dataframe(periods=16, freq='w')
forecast = m.predict(future)
fig = m.plot(forecast)
print(fig)
print(m.plot_components(forecast))
```
    


<img src="{{ site.url }}{{ site.baseurl }}/images/output_12_2.png">



<img src="{{ site.url }}{{ site.baseurl }}/images/output_12_3.png">


Looking at this graph we can pretty clearly see that our mean and variances are remaining pretty constant (the projected line is basically flat) so the dataset is most likely stationary but in the interest of double checking, let's run an Augmented Dickey-Fuller Test.


```python
from statsmodels.tsa.stattools import adfuller
def adf_test(y_values):
    print ('Results of Augmented Dickey-Fuller Test:')
    adftest = adfuller(y_values, autolag='AIC')
    adfoutput = pd.Series(adftest[0:4], index=['Test Statistic','p-value','Number of Lags Used','Number of Observations Used'])
    for key,value in adftest[4].items():
       adfoutput['Critical Value (%s)'%key] = value
    print(adfoutput)

adf_test(time_df['y'])
```

    Results of Augmented Dickey-Fuller Test:
    Test Statistic                -7.015316e+00
    p-value                        6.756415e-10
    Number of Lags Used            1.000000e+00
    Number of Observations Used    5.100000e+01
    Critical Value (1%)           -3.565624e+00
    Critical Value (5%)           -2.920142e+00
    Critical Value (10%)          -2.598015e+00
    dtype: float64
    

What we are looking for in this instance is for the test statistic to be less than the critical value because that means that 
the data is stationary. Since that criteria is met here we can feel confident that our data is stationary and thus trend and seasonality are not going to significantly affect our model.

**How can we translate this data into action for our company?**

This data can help inform how elastic our demand is and a big benefit is that we are now able to use the line formula to estimate how many shoes we would sell at a given price. Let's build an example tool below where we explore what a price increase to \$95 would mean for our demand.


```python
# PX is a hypothetical price that we want to experiment with. Users can play around with this number to explore what a price change would mean for demand.

PX = 95

QY = price_model.params['Price'] * PX + price_model.params['Intercept']
print('At a price of $' + str(PX) + ' we would expect to sell ' + '{:,.0f}'.format(float(QY)) + ' units per week')
```

    At a price of $95 we would expect to sell 13 units per week
    

# Conclusions:

Based on the above analysis, it appears that demand for the Orange Running Shoes is relatively elastic so there is some merit to the model that we built. This information will help us forecast demand during discounts and help us to optimize pricing based on company goals (e.g. improve margin, increase volume sales, maintain brand image etc.). 

### Limitations:
"all models are wrong, some are useful" 

This model was trained on a relatively narrow range of data (prices fell between ~ \$85-$115) so applying it to predict more extreme discounts may not bear fruit. Additionally, in the world of big data, 53 data points is not a large sample size. So, while I would be excited to see such positive results, my expectations should be tempered a little until we can run some experiments on our online site.
