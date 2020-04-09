---
title: "A predictive modeling case study"
weight: 5
tags: [parsnip, recipe, rsample, tune]
categories: [model fitting, tuning]
description: | 
  Develop, from beginning to end, a predictive model using best practices.
---





This article requires that you have the following packages installed: glmnet, ranger, readr, tidymodels, and vip.

# Introduction

The previous _Getting Started_ articles have been focused on single tasks related to modeling. This example is a broader case study of building a predictive model from beginning to end. It uses all of the previous topics.  

Our modeling goal here is to use a data set of hotel stays and predict which hotel stays include children (vs. do not include children or babies), based on the other characteristics of the stays such as which hotel the guests stay at, how much they pay, etc. The [paper that this data comes from](https://www.sciencedirect.com/science/article/pii/S2352340918315191) points out that the distribution of many of these variables (such as number of adults/children, room type, meals bought, country of origin of the guests, and so forth) is different for canceled vs. not canceled hotel bookings. This is mostly because more information is gathered when guests check in; the biggest contributor to these differences is not that people who cancel are different from people who do not.

To build our models, we filtered the data to only the bookings that did not cancel and will build a model to predict which non-canceled hotel stays include children and which do not.

# Data Spending

The [version](https://gist.github.com/topepo/05a74916c343e57a71c51d6bc32a21ce) of the data that we'll use can be accessed from the `tidymodels.org` site: 


```r
library(tidymodels)
library(readr)

hotels <- 
  read_csv('https://bit.ly/hotel_booking_data') %>%
  mutate_if(is.character, as.factor) 

dim(hotels)
#> [1] 50000    23
```

An important consideration for these data isthat children were only in 8.1% of the reservations. This type of severe class imbalance can often wreak havoc on an analysis. While there are several methods for combating this issue, the analyses shown below analyze the data as-is. 

For a data splitting strategy, 25% of the reservations were allocated to the test set using a stratified random sample:  


```r
set.seed(123)
splits <- initial_split(hotels, strata = children)

hotel_other <- training(splits)
hotel_test  <- testing(splits)
nrow(hotel_test)
#> [1] 12500
```

Rather than using multiple iterations of resampling, a single _validation set_ will be split apart from the 37,500 reservations that were not used for testing. In tidymodels, a validation set is treated as a single iteration of resampling. For this reason, the `validation_split()` function was used to allocate 20% of these to validation and 30,000 reservations to the training set:  


```r
set.seed(234)
val_set <- validation_split(hotel_other, strata = children, prop = 0.80)
val_set
#> # Validation Set Split (0.8/0.2)  using stratification 
#> # A tibble: 1 x 2
#>   splits             id        
#>   <named list>       <chr>     
#> 1 <split [30K/7.5K]> validation
```

The same functions from the tune package will be used as in previous articles but, in this case, the performance metrics will be computed on a single set of 7,500 reservations. This amount of data should provide enough precision to be a reliable indicator for how well each model predicts the outcome.  

# A first model: logistic regression

It makes sense to start with a simple model. Since the outcome is categorical, a logistic regression model would be a good first step. Specifically, let's use a glmnet model to potentially perform feature selection during model training. This method of estimating the logistic regression slope parameters uses a _penalty_ on the process so that less relevant predictors are driven towards a value of zero. One of the glmnet penalization methods, called the lasso method, can set the predictor slopes to absolute zero if a large enough penalty is used. 

To specify a penalized logistic regression model that uses a feature selection penalty:  


```r
lr_mod <- 
  logistic_reg(penalty = tune(), mixture = 1) %>% 
  set_engine("glmnet")
```

Setting `mixture` to a value of one means that the glmnet model will focus on potentially removing irrelevant predictors. 

To prepare the data for the model, the categorical predictors (e.g., `distribution_channel`, `hotel`, ...) should be converted to dummy variables. Additionally, it might make sense to create a set of date-based predictors that reflect important components related to the arrival date. First, `step_date()` creates predictors for the year, month, and day of the week. Secondly, `step_holiday()` generates a set of indicator variables for specific holidays. The addition of `step_zv()` means that no indicator variables that only contains a single unique value (e.g. all zeros) will be added to the model. This is important because, for penalized models, the the predictors should be centered and scaled. 

The recipe for these steps is: 


```r
holidays <- c("AllSouls", "AshWednesday", "ChristmasEve", "Easter", 
              "ChristmasDay", "GoodFriday", "NewYearsDay", "PalmSunday")

lr_recipe <- 
  recipe(children ~ ., data = hotel_other) %>% 
  step_date(arrival_date) %>% 
  step_holiday(arrival_date, holidays = holidays) %>% 
  step_rm(arrival_date) %>% 
  step_dummy(all_nominal(), -all_outcomes()) %>% 
  step_zv(all_predictors()) %>% 
  step_normalize(all_predictors())
```

The model and recipe can be bundled into a single `workflow()` object to make management of the R objects easier:


```r
lr_workflow <- 
  workflow() %>% 
  add_model(lr_mod) %>% 
  add_recipe(lr_recipe)
```

Finally a grid of penalty values are created and the model is tuned. The validation set predictions are saved (via the call to `control_grid()` below) so that diagnostic information can be available after the model fit. Also, the area under the ROC curve is used to quantify how well the model performs across a continuum of event thresholds (recall that the event rate is very low for these data). 



```r
lr_reg_grid <- expand.grid(penalty = 10^seq(-4, -1, length.out = 30))

tune_ctrl <- control_grid(save_pred = TRUE)
roc_only <- metric_set(roc_auc)

lr_res <- 
  lr_workflow %>% 
  tune_grid(val_set,
            grid = lr_reg_grid,
            control = tune_ctrl,
            metrics = roc_only)
#> ! validation: recipe: The `x` argument of `as_tibble.matrix()` must have column names ...
```

The resulting validation set metrics are computed and plotted against the amount of penalization: 


```r
lr_res %>% 
  collect_metrics() %>% 
  ggplot(aes(x = penalty, y = mean)) + 
  geom_point() + 
  geom_line() + 
  ylab("Area under the ROC Curve") +
  scale_x_log10()
```

<img src="figs/logistic-results-1.svg" width="576" />

Performance is generally better when very little penalization is used; this suggests that the majority of the predictors are important to the model. Note the steep drop in the area under the ROC curve that occurs when the amount of penalization is high; this happens because a large enough penalty will remove _all_ predictors from the model. 


Since there is a plateau of performance for small penalties, a value closer to the decline in performance is chosen as being best for this model: 


```r
lr_best <- 
  lr_res %>% 
  collect_metrics() %>% 
  arrange(desc(mean)) %>% 
  slice(8)
lr_best
#> # A tibble: 1 x 6
#>    penalty .metric .estimator  mean     n std_err
#>      <dbl> <chr>   <chr>      <dbl> <int>   <dbl>
#> 1 0.000530 roc_auc binary     0.871     1      NA
```

This value has effectively the same performance as the numerically best, but might eliminate more predictors. For this specific penalty value, the validation set ROC curve is:


```r
lr_auc <- 
  lr_res %>% 
  collect_predictions(parameters = lr_best) %>% 
  roc_curve(children, .pred_children) %>% 
  mutate(model = "Logistic Regression")

autoplot(lr_auc)
```

<img src="figs/logistic-roc-curve-1.svg" width="672" />

The level of performance generated by this logistic regression model is good but not groundbreaking. Perhaps the linear nature of the prediction equation is too limiting for this data set. 

As a next step, we might consider a highly non-linear model generated using tree-based methods. 

# Tree-based ensembles

One effective and low-maintenance modeling technique is a _random forest_ (also used in the [resampling article](/start/resampling/)). This model can be used with less preprocessing than the logistic regression; conversion to dummy variables and variable normalization are not required. As before, the date predictor is engineered so that the random forest model does not need to work hard to tease these potential patterns from the data.  


```r
rf_recipe <- 
  recipe(children ~ ., data = hotel_other) %>% 
  step_date(arrival_date) %>% 
  step_holiday(arrival_date) %>% 
  step_rm(arrival_date) 
```

The computations required for model tuning can usually be easily parallelized. However, when the models are resampled, the most efficient approach is to parallelize the resampling process. In this case study, a single validation set is being used so parallelization isn't an option using the tune package. Despite this, the ranger package can compute the individual random forest models in parallel. To do this, the number of cores to use should be specified. To determine this, the parallel package can be used to understand how much parallelization can be done on your specific computer: 


```r
cores <- parallel::detectCores()
cores
#> [1] 8
```

To declare that parallel processing should be used, the `num.threads` argument for `ranger::ranger()` can be passed when setting the computational engine: 


```r
rf_mod <- 
  rand_forest(mtry = tune(), min_n = tune(), trees = 1000) %>% 
  set_engine("ranger", num.threads = cores) %>% 
  set_mode("classification")

rf_workflow <- 
  workflow() %>% 
  add_model(rf_mod) %>% 
  add_recipe(rf_recipe)
```

Again, if any other resampling method were used, it is better to parallel process in the more usual way. 

To tune, a space-filling design with 25 candidate models is used: 


```r
set.seed(345)
rf_res <- 
  rf_workflow %>% 
  tune_grid(val_set,
            grid = 25,
            control = tune_ctrl,
            metrics = roc_only)
#> i Creating pre-processing data to finalize unknown parameter: mtry
#> Warning: The `x` argument of `as_tibble.matrix()` must have column names if `.name_repair` is omitted as of tibble 2.0.0.
#> Using compatibility `.name_repair`.
#> This warning is displayed once every 8 hours.
#> Call `lifecycle::last_warnings()` to see where this warning was generated.
```

The note about "finalizing the unknown parameter" is related to the size of the data set. Since `mtry` depends on the number of predictors in the data set, `tune_grid()` determines the upper bound for `mtry` once it receives the data. 

The results of the tuning process, when plotted, indicate that both `mtry` and the minimum number of data points required to keep splitting should be fairly small (on average). Note, however, that the range of the y-axis indicates that the model is very robust to the choice of these parameters. 


```r
autoplot(rf_res)
```

<img src="figs/rf-results-1.svg" width="672" />

If the model with the numerically best results are used, the final tuning parameter values would be:


```r
rf_best <- select_best(rf_res, metric = "roc_auc")
rf_best
#> # A tibble: 1 x 2
#>    mtry min_n
#>   <int> <int>
#> 1     8     7
```

As before, the validation set ROC curve can be produced and overlaid with the previous logistic regression model: 


```r
rf_auc <- 
  rf_res %>% 
  collect_predictions(parameters = rf_best) %>% 
  roc_curve(children, .pred_children) %>% 
  mutate(model = "Random Forest")

bind_rows(rf_auc, lr_auc) %>% 
  ggplot(aes(x = 1 - specificity, y = sensitivity, col = model)) + 
  geom_path() +
  geom_abline(lty = 2) + 
  coord_equal() + 
  scale_color_brewer(palette = "Set1")
```

<img src="figs/rf-roc-curve-1.svg" width="672" />

The random forest is uniformly better across event probability thresholds. 

If this model were chosen to be better than the other models, it can be used once again with `last_fit()` to fit the final model and then evaluate the test set. 

However, the model object is redefined so that the _variable importance_ scores are computed for this model. This should give some insight into which predictors are driving model performance.


```r
rf_mod <- 
  rand_forest(mtry = 5, min_n = 3, trees = 1000) %>% 
  set_engine("ranger", num.threads = cores, importance = 'impurity') %>% 
  set_mode("classification")

rf_workflow <- 
  workflow() %>% 
  add_model(rf_mod) %>% 
  add_recipe(rf_recipe)

rf_fit <- rf_workflow %>% last_fit(splits)
```

From this fitted workflow, the vip package can be used to visualize the results: 


```r
library(vip)

rf_fit$.workflow %>% 
  pluck(1) %>% 
  pull_workflow_fit() %>% 
  vip(num_features = 20) 
```

<img src="figs/rf-importance-1.svg" width="672" />

The most important predictors in whether a hotel stay had children or not were the daily cost for the room, the type of reservation, the time between the creation of the reservation and the arrival date, and the type of room that was reserved. 

# Test set results

How did this model do on the test set? Was the validation set a good estimate of future performance? 


```r
rf_fit %>% 
  collect_predictions() %>% 
  roc_auc(children, .pred_children)
#> # A tibble: 1 x 3
#>   .metric .estimator .estimate
#>   <chr>   <chr>          <dbl>
#> 1 roc_auc binary         0.930

rf_fit %>% 
  collect_predictions() %>% 
  roc_curve(children, .pred_children) %>% 
  autoplot()
```

<img src="figs/test-set-roc-curve-1.svg" width="672" />

Based on these results, the validation set and test set performance statistics are very close. 

# Session information


```
#> ─ Session info ───────────────────────────────────────────────────────────────
#>  setting  value                       
#>  version  R version 3.6.2 (2019-12-12)
#>  os       macOS Mojave 10.14.6        
#>  system   x86_64, darwin15.6.0        
#>  ui       X11                         
#>  language (EN)                        
#>  collate  en_US.UTF-8                 
#>  ctype    en_US.UTF-8                 
#>  tz       America/Denver              
#>  date     2020-04-09                  
#> 
#> ─ Packages ───────────────────────────────────────────────────────────────────
#>  package    * version     date       lib source                               
#>  broom      * 0.5.5       2020-02-29 [1] CRAN (R 3.6.0)                       
#>  dials      * 0.0.4.9000  2020-03-20 [1] local                                
#>  dplyr      * 0.8.5       2020-03-07 [1] CRAN (R 3.6.0)                       
#>  ggplot2    * 3.3.0       2020-03-05 [1] CRAN (R 3.6.0)                       
#>  infer      * 0.5.1       2019-11-19 [1] CRAN (R 3.6.0)                       
#>  parsnip    * 0.0.5.9001  2020-04-03 [1] Github (tidymodels/parsnip@0e83faf)  
#>  purrr      * 0.3.3       2019-10-18 [1] CRAN (R 3.6.0)                       
#>  ranger       0.12.1      2020-01-10 [1] CRAN (R 3.6.0)                       
#>  readr      * 1.3.1       2018-12-21 [1] CRAN (R 3.6.0)                       
#>  recipes    * 0.1.10.9000 2020-04-03 [1] local                                
#>  rlang        0.4.5.9000  2020-03-20 [1] Github (r-lib/rlang@a90b04b)         
#>  rsample    * 0.0.6       2020-03-31 [1] CRAN (R 3.6.2)                       
#>  tibble     * 3.0.0       2020-03-30 [1] CRAN (R 3.6.2)                       
#>  tidymodels * 0.1.0       2020-02-16 [1] CRAN (R 3.6.0)                       
#>  tune       * 0.1.0       2020-04-02 [1] CRAN (R 3.6.2)                       
#>  vip        * 0.2.1       2020-01-20 [1] CRAN (R 3.6.0)                       
#>  workflows  * 0.1.1.9000  2020-03-20 [1] Github (tidymodels/workflows@e995c18)
#>  yardstick  * 0.0.6       2020-03-17 [1] CRAN (R 3.6.0)                       
#> 
#> [1] /Library/Frameworks/R.framework/Versions/3.6/Resources/library
```


