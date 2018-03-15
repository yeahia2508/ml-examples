# Data Preprocessing

# install.packages('arules')
library(arules)
#header is false because there is no header in dataset
dataset = read.csv('/Data/Market_Basket_Optimisation.csv', header = FALSE)

#remove duplicate item purchase
dataset = read.transactions('/Data/Market_Basket_Optimisation.csv', sep = ',', rm.duplicates = TRUE)

#get summary of dataset
summary(dataset)
itemFrequencyPlot(dataset, topN = 10)

# Training Apriori on the dataset

#min_support 0.003 because (3*7/7501 = 0.003) here 3 is daily purchase of a product and devide by 7 because 7501 transaction collected in a week
#less confidence more rules will be found
#try to test with different  min_support
rules = apriori(data = dataset, parameter = list(support = 0.003, confidence = 0.2))

# Visualising the results
#sorting rules by their lift
inspect(sort(rules, by = 'lift')[1:10])