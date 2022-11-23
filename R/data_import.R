### Packages and Data

# Dataset available at: https://github.com/danielemelotti/decision_tree_exploration

## Installing the necessary packages:
install.packages("rpart")
install.packages("rpart.plot")
install.packages("rattle")
install.packages("randomForest")
install.packages("ranger")
require(rpart)
require(rpart.plot)
require(rattle)
require(dplyr)
require(randomForest)
require(ranger)

# For the purpose of this exercise, we'll be using a dataset regarding housing prices in Miami.
# The dataset includes a total of 17 variables.

## Loading specific data and model_formula
download_data <- read.csv("miami-housing.csv")
str(download_data)

# Filtering for the variables of interest
fulldata <- download_data %>%
  select(-"PARCELNO")

str(fulldata)