
# Credit Card Market Segmentation Case Study

### Problem Statement

credit cards company just hired a data scientist to help them construct customers segments in order to customize, personalize ads and offers and understand their customers better. 


And in this repository i'll help solving this case study.


### Dataset Information

The dataset constructs of 8950 rows and 18 columns.

#### Features Description

- Cust_id : unique identifier of the customers
- Balance : Credit card Balance
- balance frequency : how frequent the balance is
- Purchases : the amount of purchases.
- Purchases frequency : how frequent the purchases is
- oneoff purchases : the purchases done one time.
- installments purchases : purchases done by installments.
- cash_advance : the amount of money paid in advance.
- ONEOFF_PURCHASES_FREQUENCY : the frequency of oneoff purchases.
- PURCHASES_INSTALLMENTS_FREQUENCY : the frequency of installments purchases.
- CASH_ADVANCE_FREQUENCY : the frequency of cash advance payments.
- CASH_ADVANCE_TRX : the number of cash advance transactions.
- PURCHASES_TRX : the number of purchases transactions.
- CREDIT_LIMIT : the maximum credit limit.
- PAYMENTS : the amount of payments.
- MINIMUM_PAYMENTS : the minimum payment allowed for the customer.
- PRC_full_Payments : Payments for professional regulation commission.
- Tenure : the credit card tenure in months.

#### Data Types
we have 17 numerical variables excluding cutomer id.


## Packages Used 
```python

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import spearmanr,pearsonr
from scipy.cluster.vq import whiten
from scipy.cluster.hierarchy import linkage,fcluster
pd.set_option("display.max_columns",None)
pd.set_option("display.max_rows",100)
%matplotlib inline
```

# Methods and techniques
- Data cleaning
- Data analysis
- Data Exploration
- Describtive Statistics
- Hypothesis Testing
- Cluster analysis
- Hierarchical Clustering
- Machine Learning


# Findings 

