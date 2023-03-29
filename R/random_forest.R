source("R/data_import.R")
source("R/functions.R")

## Implementing a random forest algorithm
# It is done to overcome the issues of bagging. In fact, the trees from bagging are not completely independent of each other since all the original predictors are considered at every split of every tree. For each tree, few strong predictors will be selected repeatedly, which will lead to very similar trees (especially in the top nodes), and to highly correlated predictions, which will have almost always the same structure.

# Random forest solves this issue in a 2-steps way::
# 1. Bootstrapping: similar to bagging, each tree is grown to a bootstrap resampled data set, which makes them different and somewhat decorrelates them.
# 2. Split-variable randomization: each time a split is to be performed, the search for the  split variable is limited to a random subset of m of the p variables. For regression trees,  typical default values are m = p/3, but this should be considered a tuning parameter. When m = p, the randomization amounts to using only step 1 and is the same as bagging. See https://uc-r.github.io/random_forests for reference.

# Advantages:
# - Typically have very good performance
# - Remarkably good “out-of-the box” - very little tuning required
# - Built-in validation set - don’t need to sacrifice data for extra validation
# - No pre-processing required
# - Robust to outliers

# Disadvantages:
# - Can become slow on large data sets
# - Although accurate, often cannot compete with advanced boosting algorithms
# - Less interpretable

# Building the model on the train set
rf_model <- randomForest(model_formula, data = train_set, importance = TRUE)

# calculating the RMSE_is
sqrt(mean(rf_model$mse))

# We can see how the MSE decreases as more and more trees are developed
plot(rf_model)

# It looks like about 150 trees are enough to achieve a stable error rate

# From this large number of trees, we can find the one with lowest MSE
which.min(rf_model$mse)
sqrt(rf_model$mse[which.min(rf_model$mse)])

# Producing predictions with the random forest model
test_pred <- predict(rf_model, newdata = test_set, type = "class")

# Calculating the RMSE_oos
rmse_oos(test_set$SALE_PRC, preds = test_pred)

## Tuning the random forest algorithm
# DESCRIPTION: randomForest parameters:
# - ntree:number of trees;
# - mtry: number of variables to randomly sample as candidates at each split; 
# - sampsize: the number of samples to train on. Typically, the value should lay between 60 and 80%;
# - nodesize: minimum number of samples within the terminal nodes. This parameters controls the complexity of the tree;
# - maxnodes: maximum number of terminal nodes. This parameter controls the complexity of the tree.

## Tuning with randomForest
# Fetching the names of features
features <- setdiff(names(train_set), "SALE_PRC")

set.seed(2012)
tuned_model <- tuneRF(
  x = train_set[features],
  y = train_set$Sale_Price,
  ntreeTry = 150,
  mtryStart = 2, # 6 features
  stepFactor = 1, # we increment by 3 features until improvement stops...
  improve = 0.01, #... improving by 1%
  trace = FALSE # do not show real-time progress
)

## Tuning with ranger
# Hyperparameter grid search
hyper_grid <- expand.grid(
  mtry       = seq(5, 10, by = 2),
  node_size  = seq(3, 9, by = 2),
  sample_size = c(.55, .632, .70, .80),
  OOB_RMSE   = 0
)

# Total number of combinations
nrow(hyper_grid)

for(i in seq_len(nrow(hyper_grid))) {
  
  # train model
  model <- ranger(
    formula = SALE_PRC ~ ., 
    data = train_set, 
    num.trees = 150,
    mtry = hyper_grid$mtry[i],
    min.node.size = hyper_grid$node_size[i],
    sample.fraction = hyper_grid$sample_size[i],
    seed = 123
  )
  
  # add OOB error to grid
  hyper_grid$OOB_RMSE[i] <- sqrt(model$prediction.error)
}

hyper_grid %>% 
  dplyr::arrange(OOB_RMSE) %>%
  head(10)
