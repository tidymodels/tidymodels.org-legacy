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

We can create regression models with the tidymodels package [parsnip](https://tidymodels.github.io/parsnip/) to predict continuous or numeric quantities. Here, let's first fit a random forest model, which does _not_ require all numeric input (see discussion [here](https://bookdown.org/max/FES/categorical-trees.html)) and discuss how to use `fit()` and `fit_xy()`, as well as _data descriptors_. 

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
#> Fit time:  944ms 
#> Ranger result
#> 
#> Call:
#>  ranger::ranger(x = maybe_data_frame(x), y = y, num.threads = 1,      verbose = FALSE, seed = sample.int(10^5, 1)) 
#> 
#> Type:                             Regression 
#> Number of trees:                  500 
#> Sample size:                      2199 
#> Number of independent variables:  5 
#> Mtry:                             2 
#> Target node size:                 5 
#> Variable importance mode:         none 
#> Splitrule:                        variance 
#> OOB prediction error (MSE):       0.00844 
#> R squared (OOB):                  0.736
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
#> # A tibble: 5 x 2
#>   Sale_Price .pred
#>        <dbl> <dbl>
#> 1       5.33  5.22
#> 2       5.02  5.21
#> 3       5.27  5.25
#> 4       5.60  5.51
#> 5       5.28  5.24

# summarize performance
test_results %>% metrics(truth = Sale_Price, estimate = .pred) 
#> # A tibble: 3 x 3
#>   .metric .estimator .estimate
#>   <chr>   <chr>          <dbl>
#> 1 rmse    standard      0.0914
#> 2 rsq     standard      0.717 
#> 3 mae     standard      0.0662
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
#> Fit time:  2.6s 
#> Ranger result
#> 
#> Call:
#>  ranger::ranger(x = maybe_data_frame(x), y = y, mtry = min_cols(~3,      x), num.trees = ~1000, num.threads = 1, verbose = FALSE,      seed = sample.int(10^5, 1)) 
#> 
#> Type:                             Regression 
#> Number of trees:                  1000 
#> Sample size:                      2199 
#> Number of independent variables:  5 
#> Mtry:                             3 
#> Target node size:                 5 
#> Variable importance mode:         none 
#> Splitrule:                        variance 
#> OOB prediction error (MSE):       0.00848 
#> R squared (OOB):                  0.735
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
#> Fit time:  7.6s 
#> 
#> Call:
#>  randomForest(x = maybe_data_frame(x), y = y, ntree = ~1000, mtry = min_cols(~3,      x)) 
#>                Type of random forest: regression
#>                      Number of trees: 1000
#> No. of variables tried at each split: 3
#> 
#>           Mean of squared residuals: 0.00847
#>                     % Var explained: 73.5
```

Look at the formula code that was printed out; one function uses the argument name `ntree` and the other uses `num.trees`. The parsnip models don't require you to know the specific names of the main arguments. 

Now suppose that we want to modify the value of `mtry` based on the number of predictors in the data. Usually, a good default value is `floor(sqrt(num_predictors))` but a pure bagging model requires an `mtry` value equal to the total number of parameters. There may be cases where you may not know how many predictors are going to be present when the model will be fit (perhaps due to the generation of indicator variables or a variable filter) so this might be difficult to know exactly ahead of time when you write your code. 

When the model it being fit by parsnip, [_data descriptors_](https://tidymodels.github.io/parsnip/reference/descriptors.html) are made available. These attempt to let you know what you will have available when the model is fit. When a model object is created (say using `rand_forest()`), the values of the arguments that you give it are _immediately evaluated_ unless you delay them. To delay the evaluation of any argument, you can used `rlang::expr()` to make an expression. 

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
#> Fit time:  3.5s 
#> Ranger result
#> 
#> Call:
#>  ranger::ranger(x = maybe_data_frame(x), y = y, mtry = min_cols(~.preds(),      x), num.trees = ~1000, num.threads = 1, verbose = FALSE,      seed = sample.int(10^5, 1)) 
#> 
#> Type:                             Regression 
#> Number of trees:                  1000 
#> Sample size:                      2199 
#> Number of independent variables:  5 
#> Mtry:                             5 
#> Target node size:                 5 
#> Variable importance mode:         none 
#> Splitrule:                        variance 
#> OOB prediction error (MSE):       0.00868 
#> R squared (OOB):                  0.728
```


## Regularized regression

A linear model might work for this data set as well. We can use the `linear_reg()` parsnip model. There are two engines that can perform regularization/penalization, the glmnet and sparklyr packages. Let's use the former here. The glmnet package only implements a non-formula method, but parsnip will allow either one to be used. 

When regularization is used, the predictors should first be centered and scaled before being passed to the model. The formula method won't do that automatically so we will need to do this ourselves. We'll use the [recipes](https://tidymodels.github.io/recipes/) package for these steps. 


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
#> Fit time:  8ms 
#> 
#> Call:  glmnet::glmnet(x = maybe_matrix(x), y = y, family = "gaussian",      alpha = ~0.5) 
#> 
#>    Df %Dev Lambda
#> 1   0  0.0 0.1370
#> 2   1  1.9 0.1250
#> 3   1  3.5 0.1140
#> 4   1  5.0 0.1040
#> 5   2  6.8 0.0946
#> 6   4  9.3 0.0862
#> 7   5 12.5 0.0785
#> 8   5 15.3 0.0716
#> 9   7 18.4 0.0652
#> 10  7 21.4 0.0594
#> 11  7 24.0 0.0541
#> 12  8 26.2 0.0493
#> 13  8 28.6 0.0449
#> 14  8 30.6 0.0409
#> 15  8 32.3 0.0373
#> 16  8 33.8 0.0340
#> 17  8 35.0 0.0310
#> 18  8 36.1 0.0282
#> 19  9 37.0 0.0257
#> 20  9 37.9 0.0234
#> 21  9 38.6 0.0213
#> 22  9 39.2 0.0195
#> 23  9 39.7 0.0177
#> 24  9 40.1 0.0161
#> 25  9 40.5 0.0147
#> 26  9 40.8 0.0134
#> 27 10 41.0 0.0122
#> 28 11 41.3 0.0111
#> 29 11 41.5 0.0101
#> 30 11 41.7 0.0092
#> 31 12 41.8 0.0084
#> 32 12 42.0 0.0077
#> 33 12 42.1 0.0070
#> 34 12 42.2 0.0064
#> 35 12 42.3 0.0058
#> 36 12 42.4 0.0053
#> 37 12 42.4 0.0048
#> 38 12 42.5 0.0044
#> 39 12 42.5 0.0040
#> 40 12 42.5 0.0036
#> 41 12 42.6 0.0033
#> 42 12 42.6 0.0030
#> 43 12 42.6 0.0028
#> 44 12 42.6 0.0025
#> 45 12 42.6 0.0023
#> 46 12 42.6 0.0021
#> 47 12 42.7 0.0019
#> 48 12 42.7 0.0017
#> 49 12 42.7 0.0016
#> 50 12 42.7 0.0014
#> 51 12 42.7 0.0013
#> 52 12 42.7 0.0012
#> 53 12 42.7 0.0011
#> 54 12 42.7 0.0010
#> 55 12 42.7 0.0009
#> 56 12 42.7 0.0008
#> 57 12 42.7 0.0008
#> 58 12 42.7 0.0007
#> 59 12 42.7 0.0006
#> 60 12 42.7 0.0006
#> 61 12 42.7 0.0005
#> 62 12 42.7 0.0005
#> 63 12 42.7 0.0004
#> 64 12 42.7 0.0004
#> 65 12 42.7 0.0004
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
#> # A tibble: 731 x 3
#>    Sale_Price `random forest` glmnet
#>         <dbl>           <dbl>  <dbl>
#>  1       5.33            5.22   5.27
#>  2       5.02            5.21   5.17
#>  3       5.27            5.25   5.23
#>  4       5.60            5.51   5.25
#>  5       5.28            5.24   5.25
#>  6       5.17            5.19   5.19
#>  7       5.02            4.97   5.19
#>  8       5.46            5.50   5.49
#>  9       5.44            5.46   5.48
#> 10       5.33            5.50   5.47
#> # … with 721 more rows

test_results %>% metrics(truth = Sale_Price, estimate = glmnet) 
#> # A tibble: 3 x 3
#>   .metric .estimator .estimate
#>   <chr>   <chr>          <dbl>
#> 1 rmse    standard      0.132 
#> 2 rsq     standard      0.410 
#> 3 mae     standard      0.0956

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
#> ─ Session info ───────────────────────────────────────────────────────────────
#>  setting  value                       
#>  version  R version 4.0.3 (2020-10-10)
#>  os       macOS Mojave 10.14.6        
#>  system   x86_64, darwin17.0          
#>  ui       X11                         
#>  language (EN)                        
#>  collate  en_US.UTF-8                 
#>  ctype    en_US.UTF-8                 
#>  tz       America/Denver              
#>  date     2020-12-17                  
#> 
#> ─ Packages ───────────────────────────────────────────────────────────────────
#>  package      * version date       lib source        
#>  broom        * 0.7.3   2020-12-16 [1] CRAN (R 4.0.3)
#>  dials        * 0.0.9   2020-09-16 [1] CRAN (R 4.0.2)
#>  dplyr        * 1.0.2   2020-08-18 [1] CRAN (R 4.0.2)
#>  ggplot2      * 3.3.2   2020-06-19 [1] CRAN (R 4.0.0)
#>  glmnet       * 4.0-2   2020-06-16 [1] CRAN (R 4.0.0)
#>  infer        * 0.5.3   2020-07-14 [1] CRAN (R 4.0.0)
#>  parsnip      * 0.1.4   2020-10-27 [1] CRAN (R 4.0.2)
#>  purrr        * 0.3.4   2020-04-17 [1] CRAN (R 4.0.0)
#>  randomForest * 4.6-14  2018-03-25 [1] CRAN (R 4.0.0)
#>  ranger       * 0.12.1  2020-01-10 [1] CRAN (R 4.0.0)
#>  recipes      * 0.1.15  2020-11-11 [1] CRAN (R 4.0.2)
#>  rlang          0.4.9   2020-11-26 [1] CRAN (R 4.0.2)
#>  rsample      * 0.0.8   2020-09-23 [1] CRAN (R 4.0.2)
#>  tibble       * 3.0.4   2020-10-12 [1] CRAN (R 4.0.2)
#>  tidymodels   * 0.1.2   2020-11-22 [1] CRAN (R 4.0.2)
#>  tune         * 0.1.2   2020-11-17 [1] CRAN (R 4.0.3)
#>  workflows    * 0.2.1   2020-10-08 [1] CRAN (R 4.0.2)
#>  yardstick    * 0.0.7   2020-07-13 [1] CRAN (R 4.0.2)
#> 
#> [1] /Library/Frameworks/R.framework/Versions/4.0/Resources/library
```
 
