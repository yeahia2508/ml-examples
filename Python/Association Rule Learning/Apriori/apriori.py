# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Data Preprocessing
dataset = pd.read_csv('/home/y34h1a/StudioProjects/Mechine Learning Examples/Data/Market_Basket_Optimisation.csv', header = None)
transactions = []

#This apyori library take input as a list so we have to create a list
for i in range(0, 7501):
    transactions.append([str(dataset.values[i,j]) for j in range(0, 20)])

# Training Apriori on the dataset
from apyori import apriori

#min_support 0.003 because (3*7/7501 = 0.003) here 3 is daily purchase of a product and devide by 7 because 7501 transaction collected in a week
#less confidence more rules will be found
#min_lift = 3 becuase if lift more than 3 then it's a good rule
#min_length  = 2 means minimum items in a rule
rules = apriori(transactions, min_support = 0.003, min_confidence = 0.2, min_lift = 3, min_length = 2)

# Visualising the results
#results is a list of list
results = list(rules)

# to see results you have to click on results dataset then click on value items from table