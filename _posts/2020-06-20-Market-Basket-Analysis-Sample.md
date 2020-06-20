---
title: "Example Market Basket Analysis"
date: 2020-06-20
tags: [data science, association analysis, market basket analysis]
header:
  image:
excerpt: "Exploring a sample dataset to identify sets of items that customers purchase together"
mathjax: "true"
---
# Problem:

Our company does not understand our customer purchasing patterns. We do not know why or if our customers are purchasing products together. 

This lack of knowledge is affecting our ability to:

- run effiecient multi-buy bundle promotions;
- create a targeted recommender service for our online customers;
- customize targeted cross sell emails;
- optimize content placement on our online site. 

# Plan:

Use association analysis on a sample sales dataset to find relationships between the products that our customers frequently order together 

### Glossary

###### Support:
Support is an indication of how frequently the itemset appears in the dataset.  Oftentimes, you want to look for high support in order to minimize sample size bias. However, low support is still useful as it can help identify interesting relationships.

###### Confidence:
Confidence is an indication of how often the rule has been found to be true. A confidence of 0.2 would mean that in 20% of the cases where the antecedent was purchased, the purchase also included the consequent.

###### Lift:
Lift is the ratio of the observed support to that expected if the two rules were independent.
    - Lift of 1 means that the rules were independent of one another and thus no conclusions can be drawn from it. 
    - Lift >1 means that the items are dependent on each other and thus rules can potentially be drawn
    - Lift <1 means that the items are substitute to one another, so that the presence of one item has a negative effect on the presence of the other item
    
###### Conviction:
Conviction is the ratio of the expected frequency that X occurs without Y (that is to say, the frequency that the rule makes an incorrect prediction) if X and Y were independent divided by the observed frequency of incorrect predictions. Conviction of 1.2 would mean that the rule would be incorrect 20% more often if the association between X and Y was purely random


```python
# Import necessary packages

import pandas as pd
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules
```

# Data


```python
# Load in dataset

df = pd.read_excel('Sample - Superstore.xls')
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
      <th>Row ID</th>
      <th>Order ID</th>
      <th>Order Date</th>
      <th>Ship Date</th>
      <th>Ship Mode</th>
      <th>Customer ID</th>
      <th>Customer Name</th>
      <th>Segment</th>
      <th>Country</th>
      <th>City</th>
      <th>...</th>
      <th>Postal Code</th>
      <th>Region</th>
      <th>Product ID</th>
      <th>Category</th>
      <th>Sub-Category</th>
      <th>Product Name</th>
      <th>Sales</th>
      <th>Quantity</th>
      <th>Discount</th>
      <th>Profit</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>CA-2016-152156</td>
      <td>2016-11-08</td>
      <td>2016-11-11</td>
      <td>Second Class</td>
      <td>CG-12520</td>
      <td>Claire Gute</td>
      <td>Consumer</td>
      <td>United States</td>
      <td>Henderson</td>
      <td>...</td>
      <td>42420</td>
      <td>South</td>
      <td>FUR-BO-10001798</td>
      <td>Furniture</td>
      <td>Bookcases</td>
      <td>Bush Somerset Collection Bookcase</td>
      <td>261.9600</td>
      <td>2</td>
      <td>0.00</td>
      <td>41.9136</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>CA-2016-152156</td>
      <td>2016-11-08</td>
      <td>2016-11-11</td>
      <td>Second Class</td>
      <td>CG-12520</td>
      <td>Claire Gute</td>
      <td>Consumer</td>
      <td>United States</td>
      <td>Henderson</td>
      <td>...</td>
      <td>42420</td>
      <td>South</td>
      <td>FUR-CH-10000454</td>
      <td>Furniture</td>
      <td>Chairs</td>
      <td>Hon Deluxe Fabric Upholstered Stacking Chairs,...</td>
      <td>731.9400</td>
      <td>3</td>
      <td>0.00</td>
      <td>219.5820</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>CA-2016-138688</td>
      <td>2016-06-12</td>
      <td>2016-06-16</td>
      <td>Second Class</td>
      <td>DV-13045</td>
      <td>Darrin Van Huff</td>
      <td>Corporate</td>
      <td>United States</td>
      <td>Los Angeles</td>
      <td>...</td>
      <td>90036</td>
      <td>West</td>
      <td>OFF-LA-10000240</td>
      <td>Office Supplies</td>
      <td>Labels</td>
      <td>Self-Adhesive Address Labels for Typewriters b...</td>
      <td>14.6200</td>
      <td>2</td>
      <td>0.00</td>
      <td>6.8714</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>US-2015-108966</td>
      <td>2015-10-11</td>
      <td>2015-10-18</td>
      <td>Standard Class</td>
      <td>SO-20335</td>
      <td>Sean O'Donnell</td>
      <td>Consumer</td>
      <td>United States</td>
      <td>Fort Lauderdale</td>
      <td>...</td>
      <td>33311</td>
      <td>South</td>
      <td>FUR-TA-10000577</td>
      <td>Furniture</td>
      <td>Tables</td>
      <td>Bretford CR4500 Series Slim Rectangular Table</td>
      <td>957.5775</td>
      <td>5</td>
      <td>0.45</td>
      <td>-383.0310</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>US-2015-108966</td>
      <td>2015-10-11</td>
      <td>2015-10-18</td>
      <td>Standard Class</td>
      <td>SO-20335</td>
      <td>Sean O'Donnell</td>
      <td>Consumer</td>
      <td>United States</td>
      <td>Fort Lauderdale</td>
      <td>...</td>
      <td>33311</td>
      <td>South</td>
      <td>OFF-ST-10000760</td>
      <td>Office Supplies</td>
      <td>Storage</td>
      <td>Eldon Fold 'N Roll Cart System</td>
      <td>22.3680</td>
      <td>2</td>
      <td>0.20</td>
      <td>2.5164</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 21 columns</p>
</div>




```python
# Examining association between Consumer orders
''' Setup smaller dataframe containing only the columns we need, with each line showing how many units were ordered of 
each item on the corresponding sales order'''

consumerBasket  = (df[df['Segment'] == 'Consumer']
             .groupby(['Order ID', 'Product Name'])['Quantity']
             .sum().unstack().reset_index().fillna(0)
             .set_index('Order ID'))
```


```python
''' define a function that turns any value >0 to 1, this converts the items to a boolean in order to show whether they did/did not appear 
on a given sales order'''

def bool_qty(x):
    if x <= 0:
        return 0
    if x >= 1:
        return 1

consumerBasket_sets = consumerBasket.applymap(bool_qty)
```

# Analysis


```python
frequentItems = apriori(consumerBasket_sets, min_support=0.0005, use_colnames=True)
```


```python
rules = association_rules(frequentItems, metric='lift', min_threshold=1)

# filter for item combinations that have a lift greater than 6 and a confidence level higher than 50%
rules = rules[(rules['lift'] >= 6) & (rules['confidence'] >= 0.4)]

rules[['antecedents','consequents','antecedent support', 'consequent support','confidence', 'lift']]
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
      <th>antecedents</th>
      <th>consequents</th>
      <th>antecedent support</th>
      <th>consequent support</th>
      <th>confidence</th>
      <th>lift</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>(Acme Value Line Scissors)</td>
      <td>(Memorex Mini Travel Drive 64 GB USB 2.0 Flash...</td>
      <td>0.001933</td>
      <td>0.002320</td>
      <td>0.400000</td>
      <td>172.400000</td>
    </tr>
    <tr>
      <th>4</th>
      <td>(Avery Recycled Flexi-View Covers for Binding ...</td>
      <td>(Lumber Crayons)</td>
      <td>0.001160</td>
      <td>0.002320</td>
      <td>0.666667</td>
      <td>287.333333</td>
    </tr>
    <tr>
      <th>9</th>
      <td>(Hon Valutask Swivel Chairs)</td>
      <td>(GBC Plastic Binding Combs)</td>
      <td>0.001547</td>
      <td>0.002320</td>
      <td>0.500000</td>
      <td>215.500000</td>
    </tr>
    <tr>
      <th>10</th>
      <td>(Wilson Jones Active Use Binders)</td>
      <td>(GBC Recycled VeloBinder Covers)</td>
      <td>0.001933</td>
      <td>0.001933</td>
      <td>0.400000</td>
      <td>206.880000</td>
    </tr>
    <tr>
      <th>11</th>
      <td>(GBC Recycled VeloBinder Covers)</td>
      <td>(Wilson Jones Active Use Binders)</td>
      <td>0.001933</td>
      <td>0.001933</td>
      <td>0.400000</td>
      <td>206.880000</td>
    </tr>
    <tr>
      <th>12</th>
      <td>(O'Sullivan 4-Shelf Bookcase in Odessa Pine)</td>
      <td>(GBC Standard Recycled Report Covers, Clear Pl...</td>
      <td>0.001160</td>
      <td>0.002707</td>
      <td>0.666667</td>
      <td>246.285714</td>
    </tr>
    <tr>
      <th>14</th>
      <td>(GBC Standard Therm-A-Bind Covers)</td>
      <td>(Google Nexus 5)</td>
      <td>0.001547</td>
      <td>0.001160</td>
      <td>0.500000</td>
      <td>431.000000</td>
    </tr>
    <tr>
      <th>15</th>
      <td>(Google Nexus 5)</td>
      <td>(GBC Standard Therm-A-Bind Covers)</td>
      <td>0.001160</td>
      <td>0.001547</td>
      <td>0.666667</td>
      <td>431.000000</td>
    </tr>
    <tr>
      <th>17</th>
      <td>(Hon 4700 Series Mobuis Mid-Back Task Chairs w...</td>
      <td>(Staple envelope)</td>
      <td>0.001160</td>
      <td>0.011214</td>
      <td>0.666667</td>
      <td>59.448276</td>
    </tr>
    <tr>
      <th>18</th>
      <td>(Logitech VX Revolution Cordless Laser Mouse f...</td>
      <td>(Tennsco Industrial Shelving)</td>
      <td>0.001160</td>
      <td>0.002320</td>
      <td>0.666667</td>
      <td>287.333333</td>
    </tr>
    <tr>
      <th>20</th>
      <td>(Xerox 1971)</td>
      <td>(Staple envelope)</td>
      <td>0.001547</td>
      <td>0.011214</td>
      <td>0.500000</td>
      <td>44.586207</td>
    </tr>
    <tr>
      <th>23</th>
      <td>(Xerox 192)</td>
      <td>(X-Rack File for Hanging Folders)</td>
      <td>0.001547</td>
      <td>0.002320</td>
      <td>0.500000</td>
      <td>215.500000</td>
    </tr>
    <tr>
      <th>24</th>
      <td>(Xerox 1894)</td>
      <td>(Xerox 225)</td>
      <td>0.001547</td>
      <td>0.002320</td>
      <td>0.500000</td>
      <td>215.500000</td>
    </tr>
  </tbody>
</table>
</div>




```python
rules.to_excel('MarketBasketAnalysisSample.xlsx')
```

### How to interpret this dataset:

When customers buy "antecedent", they are "confidence" likely to also purchase "consequent". The support of these items show how often each item appears in the dataset.

# Conclusions:

We can see there are some examples of items that are likely to be purchased together (e.g. scissors and flashdrives, binders and binder covers). This information can help inform our decision making for the outlined problems above.

For example, future bundle promotions may include both Wilson Jones Active Use Binders and GBC Recycled VeloBinder Covers or we may want to consider suggesting flashdrives to customers who have already purchased scissors. 

### Next Steps:

Someone with expertise in the industry would probably have a lot of questions when looking at this dataset and they will easily be able to slice the data in different ways (e.g. look at purchasing patterns for specific cities, regions, categories) in order to draw more insights. 


```python

```
