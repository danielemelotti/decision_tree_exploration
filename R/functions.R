# In-sample RMSE
rmse_is <- function(estimated_model) {
  sqrt(mean(residuals(estimated_model)^2))
}

# Out-of-sample RMSE
rmse_oos <- function(actuals, preds) {
  sqrt(mean((preds - actuals)^2))
}

# k-Fold Cross-Validation
fold_i_pe <- function(i, k, estimated_model, dataset, outcome) {
  folds <- cut(1:nrow(dataset), breaks = k, labels = FALSE)
  test_indices <- which(folds == i)
  test_set <- dataset[test_indices, ]
  train_set <- dataset[-test_indices, ]
  trained_model <- update(estimated_model, data = train_set)
  predictions <- predict(trained_model, test_set)
  dataset[test_indices, outcome] - predictions # predictive error
}

k_fold_rmse <- function(estimated_model, dataset, outcome, k=10) {
  shuffled_indicies <- sample(1:nrow(dataset))
  dataset <- dataset[shuffled_indicies,]
  
  fold_pred_errors <- sapply(1:k, \(kth) {
    fold_i_pe(kth, k, estimated_model, dataset, outcome)
  })
  
  pred_errors <- unlist(fold_pred_errors)
  rmse <- \(errs) sqrt(mean(errs^2))
  c(rmse_is = rmse(residuals(estimated_model)), rmse_oos = rmse(pred_errors))
}

# Bagging
bagged_learn <- function(estimated_model, dataset, b=100) {
  lapply(1:b, \(i) {
    data_i <- dataset[sample(nrow(dataset), replace = TRUE),]
    update(estimated_model, data=data_i) 
  })
}

bagged_predict <- function(bagged_models, new_data) {
  predictions <- lapply(bagged_models, \(m) predict(m, new_data))
  as.data.frame(predictions) |> apply(FUN=mean, MARGIN=1)
}

# Boosting
boost_learn <- function(estimated_model, dataset, outcome, n=100, rate=0.1) { 
  predictors <- dataset[,-which(names(dataset) == outcome)]
  
  res <- dataset[,outcome]
  models <- list()
  
  for (i in 1:n) {
    new_data <- cbind(res, predictors)
    colnames(new_data)[1] = outcome
    this_model <- update(estimated_model, data = new_data)
    res <- res - rate * predict(this_model, dataset)
    models[[i]] <- this_model
  }
  
  list(models=models, rate=rate)
}

boost_predict <- function(boosted_learning, new_data) {
  boosted_models <- boosted_learning$models
  rate <- boosted_learning$rate
  predictions <- lapply(boosted_models, \(this_model) {
    predict(this_model, new_data)
  })
  pred_frame <- as.data.frame(predictions) |> unname()
  apply(pred_frame, FUN = \(preds) rate * sum(preds), MARGIN=1)
}

# Double Bagging 
double_bagged_learn <- function(estimated_model, dataset, b=100, p, m, outcome) {
  m <- length(p)/3 # default m
  lapply(1:b, \(i) { # from bootstrap 1 to b
    data_i <- dataset[sample(nrow(dataset), replace = TRUE),] # shuffle rows with replacement
    
    # lapply(1:length(p), \(i) { # from variable 1 to p
    #   pred_i <- p[sample(length(p), size = round(m, 0), replace = FALSE)] # randomly extract m variables without replacement
    #   random_preds <- paste(outcome, "~", pred_i[1]) # start creating new model_formula according to extracted predictors
    #   for(i in 1:(length(random_preds) - 1)){ # repeat m times
    #     random_preds <- paste(random_preds,"+", test[i + 1]) # add a predictor to formula
    #   }
    #   new_pred_formula <- as.formula(random_preds)
    # })
    random_preds <- sample(p, size=m) |> paste(collapse = " + ")
    formula_str <- paste(dv, random_preds, sep = " ~ ")
    new_pred_formula <- as.formula(formula_str)
  
    # lm(new_pred_formula, data = data_i) # run models
    new_estimated_model <- update(ols, formula = new_pred_formula, data = data_i)
  })
}

double_bagged_predict <- function(bagged_models, new_data) {
  predictions <- lapply(bagged_models, \(m) predict(m, new_data))
  as.data.frame(predictions) |> apply(FUN=mean, MARGIN=1)
}