# Data Preprocessing
# install.packages('arules')
library(arules)
dataset = read.csv('/Data/Market_Basket_Optimisation.csv')
dataset = read.transactions('/Data/Market_Basket_Optimisation.csv', sep = ',', rm.duplicates = TRUE)
summary(dataset)
itemFrequencyPlot(dataset, topN = 10)

# Training Eclat on the dataset
#min_support 0.003 because (3*7/7501 = 0.003) here 3 is daily purchase of a product and devide by 7 because 7501 transaction collected in a week
#eclat does not contain confidence
#minlen means min item in a set
rules = eclat(data = dataset, parameter = list(support = 0.003, minlen = 2))

# Visualising the results
inspect(sort(rules, by = 'support')[1:10])