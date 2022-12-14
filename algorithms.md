
# Algorithms used in this project

1)  Split-Sample Cross-Validation
2)  k-Fold Cross-Validation
3)  Bagging
4)  Boosting
5)  Random Forest

## (1) Split-Sample Cross-Validation

- Split data in train and test sets
- Train the model on train set
- Compute predictions on the test set using the trained model
- Test predictions accuracy: compute and compare in-sample error with
  out-of-sample error

## (2) k-Fold Cross-Validation

- Shuffle dataset rows
- Split data in train and test sets k times
- Train the model from each generated train set
- Compute predictions on each test set using the trained models
- Test predictions accuracy - compute and compare in-sample error with
  out-of-sample error

## (3) Bagging

- Split data in train and test sets
- Create resampled n train sets (resample with replacement)
- Train the model from each generated train set
- Compute predictions on the test set using the trained models
- Average the predictions
- Test predictions accuracy - Compute and compare in-sample error with
  out-of-sample error

## (4) Boosting

### Learning:

- Split data in train and test sets
- Train the model on train set
- Extract fitted values
- Calculate residuals (in-sample error)
- Exchange the y variable of the model with the residuals
- Iterate through n rounds training a new model each time
  - Fit the model
  - Extract fitted values
  - Update the residuals with a learning rate
  - Store each trained model for later prediction

### Predicting:

- Iterate through the n stored models
  - Predict outcome for each model
  - Store predictions
- Sum predictions together with learning rate
  - Multiply prediciton by learning rate
  - Add prediction to previous rounds

The final outcome is a vector of predictions. Compute out-of-sample
error and compare with in-sample.

## (5) Random Forest

``` r
Given training data set
Select number of trees to build (ntrees)
for i = 1 to ntrees do
  |  Generate a bootstrap sample of the original data
  |  Grow a regression tree to the bootstrapped data
  |  for each split do
  |  | Select m variables at random from all p variables
  |  | Pick the best variable/split-point among the m
  |  | Split the node into two child nodes
  |  end
  | Use typical tree model stopping criteria to determine when a tree is complete (but do not prune)
end
```

Source: <https://uc-r.github.io/random_forests>
