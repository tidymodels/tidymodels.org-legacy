---
title: "Regression models two ways"
tags: [rsample, parsnip]
categories: [model fitting]
type: learn-subsection
weight: 1
description: | 
  Create and train different kinds of regression models with different computational engines.
---






## Introduction

To use the code in this article, you will need to install the following packages: glmnet, randomForest, ranger, and tidymodels.

We can create regression models with the tidymodels package [parsnip](https://parsnip.tidymodels.org/) to predict continuous or numeric quantities. Here, let's first fit a random forest model, which does _not_ require all numeric input (see discussion [here](https://bookdown.org/max/FES/categorical-trees.html)) and discuss how to use `fit()` and `fit_xy()`, as well as _data descriptors_. 

Second, let's fit a regularized linear regression model to demonstrate how to move between different types of models using parsnip. 

## The Ames housing data

We'll use the Ames housing data set to demonstrate how to create regression models using parsnip. First, set up the data set and create a simple training/test set split:


```r
library(tidymodels)

data(ames)

set.seed(4595)
data_split <- initial_split(ames, strata = "Sale_Price", prop = 0.75)

ames_train <- training(data_split)
ames_test  <- testing(data_split)
```

The use of the test set here is _only for illustration_; normally in a data analysis these data would be saved to the very end after many models have been evaluated. 

## Random forest

We'll start by fitting a random forest model to a small set of parameters. Let's create a model with the predictors `Longitude`, `Latitude`, `Lot_Area`, `Neighborhood`, and `Year_Sold`. A simple random forest model can be specified via:


```r
rf_defaults <- rand_forest(mode = "regression")
rf_defaults
#> Random Forest Model Specification (regression)
#> 
#> Computational engine: ranger
```

The model will be fit with the ranger package by default. Since we didn't add any extra arguments to `fit`, _many_ of the arguments will be set to their defaults from the function  `ranger::ranger()`. The help pages for the model function describe the default parameters and you can also use the `translate()` function to check out such details. 

The parsnip package provides two different interfaces to fit a model: 

- the formula interface (`fit()`), and
- the non-formula interface (`fit_xy()`).

Let's start with the non-formula interface:



```r
preds <- c("Longitude", "Latitude", "Lot_Area", "Neighborhood", "Year_Sold")

rf_xy_fit <- 
  rf_defaults %>%
  set_engine("ranger") %>%
  fit_xy(
    x = ames_train[, preds],
    y = log10(ames_train$Sale_Price)
  )

rf_xy_fit
#> parsnip model object
#> 
#> Ranger result
#> 
#> Call:
#>  ranger::ranger(x = maybe_data_frame(x), y = y, num.threads = 1,      verbose = FALSE, seed = sample.int(10^5, 1)) 
#> 
#> Type:                             Regression 
#> Number of trees:                  500 
#> Sample size:                      2197 
#> Number of independent variables:  5 
#> Mtry:                             2 
#> Target node size:                 5 
#> Variable importance mode:         none 
#> Splitrule:                        variance 
#> OOB prediction error (MSE):       0.0085 
#> R squared (OOB):                  0.724
```

The non-formula interface doesn't do anything to the predictors before passing them to the underlying model function. This particular model does _not_ require indicator variables (sometimes called "dummy variables") to be created prior to fitting the model. Note that the output shows "Number of independent variables:  5".

For regression models, we can use the basic `predict()` method, which returns a tibble with a column named `.pred`:


```r
test_results <- 
  ames_test %>%
  select(Sale_Price) %>%
  mutate(Sale_Price = log10(Sale_Price)) %>%
  bind_cols(
    predict(rf_xy_fit, new_data = ames_test[, preds])
  )
test_results %>% slice(1:5)
#> # A tibble: 5 × 2
#>   Sale_Price .pred
#>        <dbl> <dbl>
#> 1       5.39  5.25
#> 2       5.28  5.29
#> 3       5.23  5.26
#> 4       5.21  5.30
#> 5       5.60  5.51

# summarize performance
test_results %>% metrics(truth = Sale_Price, estimate = .pred) 
#> # A tibble: 3 × 3
#>   .metric .estimator .estimate
#>   <chr>   <chr>          <dbl>
#> 1 rmse    standard      0.0945
#> 2 rsq     standard      0.733 
#> 3 mae     standard      0.0629
```

Note that: 

 * If the model required indicator variables, we would have to create them manually prior to using `fit()` (perhaps using the recipes package).
 * We had to manually log the outcome prior to modeling. 

Now, for illustration, let's use the formula method using some new parameter values:


```r
rand_forest(mode = "regression", mtry = 3, trees = 1000) %>%
  set_engine("ranger") %>%
  fit(
    log10(Sale_Price) ~ Longitude + Latitude + Lot_Area + Neighborhood + Year_Sold,
    data = ames_train
  )
#> parsnip model object
#> 
#> Ranger result
#> 
#> Call:
#>  ranger::ranger(x = maybe_data_frame(x), y = y, mtry = min_cols(~3,      x), num.trees = ~1000, num.threads = 1, verbose = FALSE,      seed = sample.int(10^5, 1)) 
#> 
#> Type:                             Regression 
#> Number of trees:                  1000 
#> Sample size:                      2197 
#> Number of independent variables:  5 
#> Mtry:                             3 
#> Target node size:                 5 
#> Variable importance mode:         none 
#> Splitrule:                        variance 
#> OOB prediction error (MSE):       0.0084 
#> R squared (OOB):                  0.727
```
 
Suppose that we would like to use the randomForest package instead of ranger. To do so, the only part of the syntax that needs to change is the `set_engine()` argument:



```r
rand_forest(mode = "regression", mtry = 3, trees = 1000) %>%
  set_engine("randomForest") %>%
  fit(
    log10(Sale_Price) ~ Longitude + Latitude + Lot_Area + Neighborhood + Year_Sold,
    data = ames_train
  )
#> parsnip model object
#> 
#> 
#> Call:
#>  randomForest(x = maybe_data_frame(x), y = y, ntree = ~1000, mtry = min_cols(~3,      x)) 
#>                Type of random forest: regression
#>                      Number of trees: 1000
#> No. of variables tried at each split: 3
#> 
#>           Mean of squared residuals: 0.00847
#>                     % Var explained: 72.5
```

Look at the formula code that was printed out; one function uses the argument name `ntree` and the other uses `num.trees`. The parsnip models don't require you to know the specific names of the main arguments. 

Now suppose that we want to modify the value of `mtry` based on the number of predictors in the data. Usually, a good default value is `floor(sqrt(num_predictors))` but a pure bagging model requires an `mtry` value equal to the total number of parameters. There may be cases where you may not know how many predictors are going to be present when the model will be fit (perhaps due to the generation of indicator variables or a variable filter) so this might be difficult to know exactly ahead of time when you write your code. 

When the model it being fit by parsnip, [_data descriptors_](https://parsnip.tidymodels.org/reference/descriptors.html) are made available. These attempt to let you know what you will have available when the model is fit. When a model object is created (say using `rand_forest()`), the values of the arguments that you give it are _immediately evaluated_ unless you delay them. To delay the evaluation of any argument, you can used `rlang::expr()` to make an expression. 

Two relevant data descriptors for our example model are:

 * `.preds()`: the number of predictor _variables_ in the data set that are associated with the predictors **prior to dummy variable creation**.
 * `.cols()`: the number of predictor _columns_ after dummy variables (or other encodings) are created.

Since ranger won't create indicator values, `.preds()` would be appropriate for `mtry` for a bagging model. 

For example, let's use an expression with the `.preds()` descriptor to fit a bagging model: 


```r
rand_forest(mode = "regression", mtry = .preds(), trees = 1000) %>%
  set_engine("ranger") %>%
  fit(
    log10(Sale_Price) ~ Longitude + Latitude + Lot_Area + Neighborhood + Year_Sold,
    data = ames_train
  )
#> parsnip model object
#> 
#> Ranger result
#> 
#> Call:
#>  ranger::ranger(x = maybe_data_frame(x), y = y, mtry = min_cols(~.preds(),      x), num.trees = ~1000, num.threads = 1, verbose = FALSE,      seed = sample.int(10^5, 1)) 
#> 
#> Type:                             Regression 
#> Number of trees:                  1000 
#> Sample size:                      2197 
#> Number of independent variables:  5 
#> Mtry:                             5 
#> Target node size:                 5 
#> Variable importance mode:         none 
#> Splitrule:                        variance 
#> OOB prediction error (MSE):       0.00867 
#> R squared (OOB):                  0.718
```


## Regularized regression

A linear model might work for this data set as well. We can use the `linear_reg()` parsnip model. There are two engines that can perform regularization/penalization, the glmnet and sparklyr packages. Let's use the former here. The glmnet package only implements a non-formula method, but parsnip will allow either one to be used. 

When regularization is used, the predictors should first be centered and scaled before being passed to the model. The formula method won't do that automatically so we will need to do this ourselves. We'll use the [recipes](https://recipes.tidymodels.org/) package for these steps. 


```r
norm_recipe <- 
  recipe(
    Sale_Price ~ Longitude + Latitude + Lot_Area + Neighborhood + Year_Sold, 
    data = ames_train
  ) %>%
  step_other(Neighborhood) %>% 
  step_dummy(all_nominal()) %>%
  step_center(all_predictors()) %>%
  step_scale(all_predictors()) %>%
  step_log(Sale_Price, base = 10) %>% 
  # estimate the means and standard deviations
  prep(training = ames_train, retain = TRUE)

# Now let's fit the model using the processed version of the data

glmn_fit <- 
  linear_reg(penalty = 0.001, mixture = 0.5) %>% 
  set_engine("glmnet") %>%
  fit(Sale_Price ~ ., data = bake(norm_recipe, new_data = NULL))
glmn_fit
#> parsnip model object
#> 
#> 
#> Call:  glmnet::glmnet(x = maybe_matrix(x), y = y, family = "gaussian",      alpha = ~0.5) 
#> 
#>    Df %Dev Lambda
#> 1   0  0.0 0.1380
#> 2   1  2.0 0.1260
#> 3   1  3.7 0.1150
#> 4   1  5.3 0.1050
#> 5   2  7.1 0.0953
#> 6   3  9.6 0.0869
#> 7   4 12.6 0.0791
#> 8   5 15.4 0.0721
#> 9   5 17.9 0.0657
#> 10  7 20.8 0.0599
#> 11  7 23.5 0.0545
#> 12  7 25.8 0.0497
#> 13  8 28.2 0.0453
#> 14  8 30.3 0.0413
#> 15  8 32.1 0.0376
#> 16  8 33.7 0.0343
#> 17  8 35.0 0.0312
#> 18  8 36.1 0.0284
#> 19  8 37.0 0.0259
#> 20  9 37.9 0.0236
#> 21  9 38.6 0.0215
#> 22  9 39.3 0.0196
#> 23  9 39.8 0.0179
#> 24  9 40.3 0.0163
#> 25 10 40.7 0.0148
#> 26 11 41.1 0.0135
#> 27 11 41.4 0.0123
#> 28 11 41.6 0.0112
#> 29 11 41.9 0.0102
#> 30 12 42.1 0.0093
#> 31 12 42.3 0.0085
#> 32 12 42.4 0.0077
#> 33 12 42.6 0.0070
#> 34 12 42.7 0.0064
#> 35 12 42.8 0.0059
#> 36 12 42.8 0.0053
#> 37 12 42.9 0.0049
#> 38 12 43.0 0.0044
#> 39 12 43.0 0.0040
#> 40 12 43.0 0.0037
#> 41 12 43.1 0.0034
#> 42 12 43.1 0.0031
#> 43 12 43.1 0.0028
#> 44 12 43.1 0.0025
#> 45 12 43.1 0.0023
#> 46 12 43.2 0.0021
#> 47 12 43.2 0.0019
#> 48 12 43.2 0.0018
#> 49 12 43.2 0.0016
#> 50 12 43.2 0.0014
#> 51 12 43.2 0.0013
#> 52 12 43.2 0.0012
#> 53 12 43.2 0.0011
#> 54 12 43.2 0.0010
#> 55 12 43.2 0.0009
#> 56 12 43.2 0.0008
#> 57 12 43.2 0.0008
#> 58 12 43.2 0.0007
#> 59 12 43.2 0.0006
#> 60 12 43.2 0.0006
#> 61 12 43.2 0.0005
#> 62 12 43.2 0.0005
#> 63 12 43.2 0.0004
#> 64 12 43.2 0.0004
#> 65 12 43.2 0.0004
```

If `penalty` were not specified, all of the `lambda` values would be computed. 

To get the predictions for this specific value of `lambda` (aka `penalty`):


```r
# First, get the processed version of the test set predictors:
test_normalized <- bake(norm_recipe, new_data = ames_test, all_predictors())

test_results <- 
  test_results %>%
  rename(`random forest` = .pred) %>%
  bind_cols(
    predict(glmn_fit, new_data = test_normalized) %>%
      rename(glmnet = .pred)
  )
test_results
#> # A tibble: 733 × 3
#>    Sale_Price `random forest` glmnet
#>         <dbl>           <dbl>  <dbl>
#>  1       5.39            5.25   5.16
#>  2       5.28            5.29   5.27
#>  3       5.23            5.26   5.24
#>  4       5.21            5.30   5.24
#>  5       5.60            5.51   5.24
#>  6       5.32            5.29   5.26
#>  7       5.17            5.14   5.18
#>  8       5.06            5.13   5.17
#>  9       4.98            5.01   5.18
#> 10       5.11            5.14   5.19
#> # … with 723 more rows

test_results %>% metrics(truth = Sale_Price, estimate = glmnet) 
#> # A tibble: 3 × 3
#>   .metric .estimator .estimate
#>   <chr>   <chr>          <dbl>
#> 1 rmse    standard      0.142 
#> 2 rsq     standard      0.391 
#> 3 mae     standard      0.0979

test_results %>% 
  gather(model, prediction, -Sale_Price) %>% 
  ggplot(aes(x = prediction, y = Sale_Price)) + 
  geom_abline(col = "green", lty = 2) + 
  geom_point(alpha = .4) + 
  facet_wrap(~model) + 
  coord_fixed()
```

<img src="figs/glmn-pred-1.svg" width="672" />

This final plot compares the performance of the random forest and regularized regression models.

## Session information


```
#> ─ Session info ─────────────────────────────────────────────────────
#>  setting  value
#>  version  R version 4.2.1 (2022-06-23)
#>  os       macOS Big Sur ... 10.16
#>  system   x86_64, darwin17.0
#>  ui       X11
#>  language (EN)
#>  collate  en_US.UTF-8
#>  ctype    en_US.UTF-8
#>  tz       America/Los_Angeles
#>  date     2022-11-23
#>  pandoc   2.19.2 @ /Applications/RStudio.app/Contents/MacOS/quarto/bin/tools/ (via rmarkdown)
#> 
#> ─ Packages ─────────────────────────────────────────────────────────
#>  package      * version date (UTC) lib source
#>  broom        * 1.0.1   2022-08-29 [1] CRAN (R 4.2.0)
#>  dials        * 1.1.0   2022-11-04 [1] CRAN (R 4.2.0)
#>  dplyr        * 1.0.10  2022-09-01 [1] CRAN (R 4.2.0)
#>  ggplot2      * 3.4.0   2022-11-04 [1] CRAN (R 4.2.0)
#>  glmnet       * 4.1-4   2022-04-15 [1] CRAN (R 4.2.0)
#>  infer        * 1.0.3   2022-08-22 [1] CRAN (R 4.2.0)
#>  parsnip      * 1.0.3   2022-11-11 [1] CRAN (R 4.2.0)
#>  purrr        * 0.3.5   2022-10-06 [1] CRAN (R 4.2.0)
#>  randomForest * 4.7-1.1 2022-05-23 [1] CRAN (R 4.2.0)
#>  ranger       * 0.14.1  2022-06-18 [1] CRAN (R 4.2.0)
#>  recipes      * 1.0.3   2022-11-09 [1] CRAN (R 4.2.0)
#>  rlang          1.0.6   2022-09-24 [1] CRAN (R 4.2.0)
#>  rsample      * 1.1.0   2022-08-08 [1] CRAN (R 4.2.0)
#>  tibble       * 3.1.8   2022-07-22 [1] CRAN (R 4.2.0)
#>  tidymodels   * 1.0.0   2022-07-13 [1] CRAN (R 4.2.0)
#>  tune         * 1.0.1   2022-10-09 [1] CRAN (R 4.2.0)
#>  workflows    * 1.1.2   2022-11-16 [1] CRAN (R 4.2.0)
#>  yardstick    * 1.1.0   2022-09-07 [1] CRAN (R 4.2.0)
#> 
#>  [1] /Library/Frameworks/R.framework/Versions/4.2/Resources/library
#> 
#> ────────────────────────────────────────────────────────────────────
```
 
