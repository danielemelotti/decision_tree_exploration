source("R/data_import.R")
source("R/functions.R")

### Comparing the predictive accuracy of an OLS model and a Regression Tree by implementing Split-sample Cross Validation, Bootstrap, LOOCV, Bagging and Boosting.

# Performing a 75:25 split
set.seed(2012)
train_indices <- sample(seq_len(nrow(fulldata)), size = 0.75 * nrow(fulldata))

# Creating the train and test sets
train_set <- fulldata[train_indices, ]
test_set <- fulldata[-train_indices, ]

## Split-sample cross validation
## Estimating an ols model_formula on the train dataset
ols <- lm(model_formula, data = train_set)

## Building the tree
tree <- rpart(model_formula, data = train_set)

# Here is a way to find out the cp related to the lowest xerror without extensively looking through the table
c_par <- tree$cptable[which.min(tree$cptable[, "xerror"]), "CP"]
c_par

# We see that the lowest xerror is actually related to the very last split. Hence, there would be no real need to prune the tree.

## Pruning the tree according to c_par
prune_tree <- prune(tree, cp = 0.01)

# Playing with control parameters
# As we have seen, pruning the tree didn't really provide any change. In fact, rpart build trees according to some default control parameters, such as minsplit, minbucket, cp and more (see: https://stackoverflow.com/questions/13136683/is-rpart-automatic-pruning#:~:text=No%2C%20but%20the%20defaults%20for,definition%20of%20%22early%22). If we want to force tree growth, we can play with those parameters and change them. For instance, we can decrease cp.
tree_fun <- rpart(model_formula, data = train_set, control = rpart.control(cp = 0.007))

## Producing predictions

# Predicting the outcome variable on the test set for the ols and tree models
purchase_predicted_ols <- predict(ols, test_set)
purchase_predicted_tree <- predict(tree, test_set)

## Reporting prediction accuracy using Root Mean Squared Error (RMSE)

# In-sample trained model_formula RMSE, comparison of ols vs tree:
rmse_is_ols <- round(rmse_is(ols), 4)
rmse_is_tree <- round(rmse_is(tree), 4)

rmse_is_ols
rmse_is_tree

# Out-of-sample test data RMSE:
rmse_oos_ols <- rmse_oos(purchase_predicted_ols, test_set[, dv])
rmse_oos_tree <- rmse_oos(purchase_predicted_tree, test_set[, dv])

rmse_oos_ols
rmse_oos_tree

## Bootstrapping the split-sample CV RMSEs to see if results are consistent
sample_boot <- function(dataset, model_formula, yvar) {
  # Splitting the dataset in train and test sets
  train_indices <- sample(seq_len(nrow(fulldata)), size = 0.75 * nrow(fulldata))
  train_set <- fulldata[train_indices, ]
  test_set <- fulldata[-train_indices, ]

  # Building the tree
  tree <- rpart(model_formula, data = train_set)

  # Finding the lowest cp, pruning the tree
  c_par <- tree$cptable[which.min(tree$cptable[, "xerror"]), "CP"]
  prune_tree <- prune(tree, cp = c_par)

  # Making predictions
  purchase_predicted_ols <- predict(ols, test_set)
  purchase_predicted_tree <- predict(tree, test_set)

  # Calculating in-sample errors
  rmse_is_ols <- round(sqrt(mean(residuals(ols)^2)), 4)
  rmse_is_tree <- round(sqrt(mean(residuals(tree)^2)), 4)

  # Calculating out-of-sample errors
  rmse_oos_ols <- rmse_oos(purchase_predicted_ols, test_set[, yvar])
  rmse_oos_tree <- rmse_oos(purchase_predicted_tree, test_set[, yvar])

  # Storing the errors in a dataframe
  error_comparison <- c(rmse_is_ols, rmse_oos_ols, rmse_is_tree, rmse_oos_tree)
}

set.seed(2012)
boot_rmse <- replicate(25, sample_boot(fulldata, model_formula, dv))

# Comparing is and oos errors of ols, tree and pruned tree
boot_rmse_is_ols <- mean(boot_rmse[1, ])
boot_rmse_oos_ols <- mean(boot_rmse[2, ])
boot_rmse_is_tree <- mean(boot_rmse[3, ])
boot_rmse_oos_tree <- mean(boot_rmse[4, ])
boot_rmse_is_prune <- mean(boot_rmse[5, ])
boot_rmse_oos_prune <- mean(boot_rmse[6, ])

comparison <- matrix(c(boot_rmse_is_ols, boot_rmse_oos_ols, boot_rmse_is_tree, boot_rmse_oos_tree,
                       boot_rmse_is_prune, boot_rmse_oos_prune), ncol = 3, byrow = FALSE)

colnames(comparison) <- c("rmse_ols", "rmse_tree", "rmse_prune")
rownames(comparison) <- c("is", "oos")

comparison

## Implementing k-fold Cross-Validation
k_fold_rmse_ols <- k_fold_rmse(ols, fulldata, dv, k = 100, seed = 2012)
k_fold_rmse_tree <- k_fold_rmse(tree, fulldata, dv, k = 100, seed = 2012)

k_fold_rmse_ols # still oos < is ...
k_fold_rmse_tree 

# should compare datasets with same proportions

## Implementing Bagging
# Computing the is and oos RMSEs between ols and tree bagged models
# is
bag_is_rmse_ols <- bagged_learn(estimated_model = ols, dataset = train_set, seed = 2012) |>
  bagged_predict(new_data = train_set) |>
  rmse_oos(actuals = train_set[, dv])

bag_is_rmse_tree <- bagged_learn(estimated_model = tree, dataset = train_set, seed = 2012) |>
  bagged_predict(new_data = train_set) |>
  rmse_oos(actuals = train_set[, dv])

bag_is_rmse_ols
bag_is_rmse_tree

# oos
bag_oos_rmse_ols <- bagged_learn(estimated_model = ols, dataset = train_set, seed = 2012) |>
  bagged_predict(new_data = test_set) |>
  rmse_oos(actuals = test_set[, dv])

bag_oos_rmse_tree <- bagged_learn(estimated_model = tree, dataset = train_set, seed = 2012) |>
  bagged_predict(new_data = test_set) |>
  rmse_oos(actuals = test_set[, dv])

bag_oos_rmse_ols
bag_oos_rmse_tree

## Implementing Boosting
# Comparing the is and oos RMSEs between ols and tree boosted models
# is
boost_is_rmse_ols <- boost_learn(ols, train_set, dv) |>
  boost_predict(train_set) |> rmse_oos(actuals = train_set[, dv])

boost_is_rmse_tree <- boost_learn(tree, train_set, dv) |>
  boost_predict(train_set) |> rmse_oos(actuals = train_set[, dv])

boost_is_rmse_ols
boost_is_rmse_tree

# oos
boost_oos_rmse_ols <- boost_learn(ols, train_set, dv) |>
  boost_predict(test_set) |> rmse_oos(actuals = test_set[, dv])

boost_oos_rmse_tree <- boost_learn(tree, train_set, dv) |>
  boost_predict(test_set) |> rmse_oos(actuals = test_set[, dv])

boost_oos_rmse_ols
boost_oos_rmse_tree

## Simple Forest!
# is
sf_is_rmse_ols <- simple_forest(estimated_model = ols , dataset = train_set, b = 100, p =  xv_list$conditional, outcome = dv, seed = 2012) |>
  bagged_predict(new_data = train_set) |>
  rmse_oos(actuals = train_set[, dv])

sf_is_rmse_tree <- simple_forest(estimated_model = tree , dataset = train_set, b = 100, p =  xv_list$conditional, outcome = dv, seed = 2012) |>
  bagged_predict(new_data = train_set) |>
  rmse_oos(actuals = train_set[, dv])

sf_is_rmse_ols
sf_is_rmse_tree

# oos
sf_oos_rmse_ols <- simple_forest(estimated_model = ols , dataset = train_set, b = 100, p =  xv_list$conditional, outcome = dv, seed = 2012) |>
  bagged_predict(new_data = test_set) |>
  rmse_oos(actuals = test_set[, dv])

sf_oos_rmse_tree <- simple_forest(estimated_model = tree , dataset = train_set, b = 100, p =  xv_list$conditional, outcome = dv, seed = 2012) |>
  bagged_predict(new_data = test_set) |>
  rmse_oos(actuals = test_set[, dv])

sf_oos_rmse_ols
sf_oos_rmse_tree

# Complete error comparison
error_comparison <- matrix(c(rmse_is_ols, rmse_oos_ols, rmse_is_tree, rmse_oos_tree, NA,
                             k_fold_rmse_ols, NA, k_fold_rmse_tree, bag_is_rmse_ols,
                             bag_oos_rmse_ols, bag_is_rmse_tree, bag_oos_rmse_tree, boost_is_rmse_ols,
                             boost_oos_rmse_ols, boost_is_rmse_tree, boost_oos_rmse_tree, sf_is_rmse_ols,
                             sf_oos_rmse_ols, sf_is_rmse_tree, sf_oos_rmse_tree), ncol = 2, byrow = 
                             TRUE)

colnames(error_comparison) <- c("is", "oos")
rownames(error_comparison) <- c("split-sample_ols", "split-sample_tree", "k-fold_ols", "f-fold_tree",
                                    "bagging_ols", "bagging_tree", "boosting_ols", "boosting_tree", 
                                    "simple-forest_ols", "simple-forest_tree")

error_comparison

## Implementing k_fold with caret package
# Defining training control as cross-validation with k = 100
set.seed(2012)
train_control <- trainControl(method = "cv", number = 100, savePredictions = TRUE)

# ols
caret_ols <- train(model_formula, data = fulldata, method = "lm", trControl = train_control) ### fulldata

caret_rmse_ols <- caret_ols$results$RMSE
caret_rmse_ols

caret_ols_preds_is <- predict(caret_ols, train_set)
caret_ols_preds_oos <- predict(caret_ols, test_set)

caret_ols_rmse_is <- rmse_oos(actuals = train_set[, dv], preds = caret_ols_preds_is)
caret_ols_rmse_oos <- rmse_oos(actuals = test_set[, dv], preds = caret_ols_preds_oos)

caret_ols_rmse_is # still rmse_is > rmse_oos
caret_ols_rmse_oos

# tree
caret_tree <- train(model_formula, data = train_set, method = "rpart", trControl = train_control)
                    #tuneGrid = data.frame(
                    #  cp = c(0.0001, 0.001, 0.35, 0.65)))

#caret_rmse_tree <- caret_tree$results$RMSE
#caret_rmse_tree

caret_tree_preds_is <- predict(caret_tree, train_set)
caret_tree_preds_oos <- predict(caret_tree, test_set)

caret_tree_rmse_is <- rmse_oos(actuals = train_set[, dv], preds = caret_tree_preds_is)
caret_tree_rmse_oos <- rmse_oos(actuals = test_set[, dv], preds = caret_tree_preds_oos)

caret_tree_rmse_is # still rmse_is > rmse_oos
caret_tree_rmse_oos
