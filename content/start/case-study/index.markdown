---
title: "A predictive modeling case study"
weight: 5
tags: [parsnip, recipes, rsample, workflows, tune]
categories: [model fitting, tuning]
description: | 
  Develop, from beginning to end, a predictive model using best practices.
---






## Introduction {#intro}

Each of the four previous [_Get Started_](/start/) articles has focused on a single task related to modeling. Along the way, we also introduced core packages in the tidymodels ecosystem and some of the key functions you'll need to start working with models. In this final case study, we will use all of the previous articles as a foundation to build a predictive model from beginning to end. 

To use code in this article,  you will need to install the following packages: glmnet, ranger, readr, tidymodels, and vip.


```r
library(tidymodels)  

# Helper packages
library(readr)       # for importing data
library(vip)         # for variable importance plots

# Modeling packages
library(glmnet)      # for penalized logistic regression model
library(ranger)      # for random forest model
```


## The Hotel Bookings Data {#data}

Let’s use hotel bookings data from [Antonio, Almeida, and Nunes (2019)](https://doi.org/10.1016/j.dib.2018.11.126) to predict which hotel stays included children and/or babies, based on the other characteristics of the stays such as which hotel the guests stay at, how much they pay, etc. This was also a [`#TidyTuesday`](https://github.com/rfordatascience/tidytuesday/tree/master/data/2020/2020-02-11) dataset with a [data dictionary](https://github.com/rfordatascience/tidytuesday/tree/master/data/2020/2020-02-11#data-dictionary) you may want to look over to learn more about the variables. We'll use a slightly [edited version of the dataset](https://gist.github.com/topepo/05a74916c343e57a71c51d6bc32a21ce) for this case study. 

To start, let's read our hotel data into R, which we'll do by providing [`readr::read_csv()`](https://readr.tidyverse.org/reference/read_delim.html) with a url where our CSV data is located ("<https://tidymodels.org/start/case-study/hotels.csv>"):


```r
library(tidymodels)
library(readr)

hotels <- 
  read_csv('https://bit.ly/hotel_booking_data') %>%
  mutate_if(is.character, as.factor) 

dim(hotels)
#> [1] 50000    23
```


In the original paper, the [authors](https://doi.org/10.1016/j.dib.2018.11.126) caution that the distribution of many of these variables (such as number of adults/children, room type, meals bought, country of origin of the guests, and so forth) is different for hotel bookings that were canceled versus not canceled. This makes sense because much of that information is gathered when guests check in for their stay, so canceled bookings are likely to be more variable and have more missing data than non-canceled bookings. Given this, it is unlikely that there are systematic differences between guests who cancel their bookings and those who do not in this dataset. To build our models, we have already filtered the data to include only the bookings that _did not cancel_. 


```r
glimpse(hotels)
#> Observations: 50,000
#> Variables: 23
#> $ hotel                          <fct> City_Hotel, City_Hotel, Resort_Hotel, …
#> $ lead_time                      <dbl> 217, 2, 95, 143, 136, 67, 47, 56, 80, …
#> $ stays_in_weekend_nights        <dbl> 1, 0, 2, 2, 1, 2, 0, 0, 0, 2, 1, 0, 1,…
#> $ stays_in_week_nights           <dbl> 3, 1, 5, 6, 4, 2, 2, 3, 4, 2, 2, 1, 2,…
#> $ adults                         <dbl> 2, 2, 2, 2, 2, 2, 2, 0, 2, 2, 2, 1, 2,…
#> $ children                       <fct> none, none, none, none, none, none, ch…
#> $ meal                           <fct> BB, BB, BB, HB, HB, SC, BB, BB, BB, BB…
#> $ country                        <fct> DEU, PRT, GBR, ROU, PRT, GBR, ESP, ESP…
#> $ market_segment                 <fct> Offline_TA/TO, Direct, Online_TA, Onli…
#> $ distribution_channel           <fct> TA/TO, Direct, TA/TO, TA/TO, Direct, T…
#> $ is_repeated_guest              <dbl> 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,…
#> $ previous_cancellations         <dbl> 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,…
#> $ previous_bookings_not_canceled <dbl> 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,…
#> $ reserved_room_type             <fct> A, D, A, A, F, A, C, B, D, A, A, D, A,…
#> $ assigned_room_type             <fct> A, K, A, A, F, A, C, A, D, A, D, D, A,…
#> $ booking_changes                <dbl> 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,…
#> $ deposit_type                   <fct> No_Deposit, No_Deposit, No_Deposit, No…
#> $ days_in_waiting_list           <dbl> 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,…
#> $ customer_type                  <fct> Transient-Party, Transient, Transient,…
#> $ average_daily_rate             <dbl> 80.75, 170.00, 8.00, 81.00, 157.60, 49…
#> $ required_car_parking_spaces    <fct> none, none, none, none, none, none, no…
#> $ total_of_special_requests      <dbl> 1, 3, 2, 1, 4, 1, 1, 1, 1, 1, 0, 1, 0,…
#> $ arrival_date                   <date> 2016-09-01, 2017-08-25, 2016-11-19, 2…
```

We will build a model to predict which actual hotel stays included children and/or babies, and which did not. Our outcome variable `children` is a factor variable with two levels:


```r
hotels %>% 
  count(children) %>% 
  mutate(prop = n/sum(n))
#> # A tibble: 2 x 3
#>   children     n   prop
#>   <fct>    <int>  <dbl>
#> 1 children  4038 0.0808
#> 2 none     45962 0.919
```

We can see that children were only in 8.1% of the reservations. This type of severe class imbalance can often wreak havoc on an analysis. While there are several methods for combating this issue using [recipes](/find/recipes/) (search for steps to `upsample` or `downsample`) or other more specialized packages like [themis](https://tidymodels.github.io/themis/), the analyses shown below analyze the data as-is. 

## Data Splitting {#data-split}

For a data splitting strategy, let's reserve 25% of the bookings to the test set. As in our [*Evaluate your model with resampling*](/start/resampling/#data-split) article, we know our outcome variable `children` is pretty imbalanced so we'll use a stratified random sample:  


```r
set.seed(123)
splits      <- initial_split(hotels, strata = children)

hotel_other <- training(splits)
hotel_test  <- testing(splits)

# training set proportions by children
hotel_other %>% 
  count(children) %>% 
  mutate(prop = n/sum(n))
#> # A tibble: 2 x 3
#>   children     n   prop
#>   <fct>    <int>  <dbl>
#> 1 children  3048 0.0813
#> 2 none     34452 0.919

# test set proportions by children
hotel_test  %>% 
  count(children) %>% 
  mutate(prop = n/sum(n))
#> # A tibble: 2 x 3
#>   children     n   prop
#>   <fct>    <int>  <dbl>
#> 1 children   990 0.0792
#> 2 none     11510 0.921
```

In our articles so far, we've relied on 10-fold cross-validation as the primary resampling method using [`rsample::vfold_cv()`](https://tidymodels.github.io/rsample/reference/vfold_cv.html). This has created 10 different resamples of the training set (which we further split into _analysis_ and _assessment_ sets), producing 10 different performance metrics that we then aggregated.

For this case study, rather than using multiple iterations of resampling, let's create a single resample called a _validation set_. In tidymodels, a validation set is treated as a single iteration of resampling. This will be a split from the 37,500 reservations that were not used for testing, which we called `hotel_other`. This split creates two new datasets: 

+ the set held out for the purpose of measuring performance, called the _validation set_, and 

+ the remaining data used to fit the model, called the _training set_. 

<img src="img/validation-split.svg" width="50%" style="display: block; margin: auto;" />

We'll use the `validation_split()` function to allocate 20% of the `hotel_other` bookings to the _validation set_ and 30,000 bookings to the _training set_. This means that our model performance metrics will be computed on a single set of 7,500 reservations. This is fairly large, so the amount of data should provide enough precision to be a reliable indicator for how well each model predicts the outcome with a single iteration of resampling.


```r
set.seed(234)
val_set <- validation_split(hotel_other, 
                            strata = children, 
                            prop = 0.80)
val_set
#> # Validation Set Split (0.8/0.2)  using stratification 
#> # A tibble: 1 x 2
#>   splits             id        
#>   <named list>       <chr>     
#> 1 <split [30K/7.5K]> validation
```

Note that this function, like `initial_split()` has the same `strata` argument to use stratified sampling to create the resample. This means that we'll have roughly the same proportions of bookings with and without children in our new validation and training sets, as compared to the original `hotel_other` proportions.

## A first model: penalized logistic regression {#first-model}

Since our outcome variable `children` is categorical, a logistic regression would be a good first model to start. Let's use a model to can perform feature selection during training. The [glmnet](https://cran.r-project.org/web/packages/glmnet/index.html) R package fits a generalized linear model via penalized maximum likelihood. This method of estimating the logistic regression slope parameters uses a _penalty_ on the process so that less relevant predictors are driven towards a value of zero. One of the glmnet penalization methods, called the [lasso method](https://en.wikipedia.org/wiki/Lasso_(statistics)), can set the predictor slopes to absolute zero if a large enough penalty is used. 

### Build the model

To specify a penalized logistic regression model that uses a feature selection penalty, let's use the parsnip package with the glmnet engine:  


```r
lr_mod <- 
  logistic_reg(penalty = tune(), mixture = 1) %>% 
  set_engine("glmnet")
```

We'll tag the `penalty` argument with `tune()` as a placeholder for now. This is a model hyperparameter that we will [tune](/start/tuning/) to find the best value for improving our predictions. Setting `mixture` to a value of one means that the glmnet model will focus on potentially removing irrelevant predictors. 

### Create the recipe 

Let's create a [recipe](/start/recipes/) to define the preprocessing steps we need to prepare our hotel bookings data for this model. All categorical predictors (e.g., `distribution_channel`, `hotel`, ...) should be converted to dummy variables. Additionally, it might make sense to create a set of date-based predictors that reflect important components related to the arrival date. We have already introduced a [number of useful recipe steps](/start/recipes/#features) for creating features from dates:

+ `step_date()` creates predictors for the year, month, and day of the week.

+ `step_holiday()` generates a set of indicator variables for specific holidays. Although we don't know where these two hotels are located, we do know that the countries for origin for most bookings are based in Europe.

+ `step_rm()` removes variables; here we'll use it to remove the original date variable since we no longer want it in the model.

+ `step_zv()` removes indicator variables that only contain a single unique value (e.g. all zeros). This is important because, for penalized models, the predictors should be centered and scaled.

+ `step_normalize()` centers and scales numeric variables.

Putting all these steps together into a recipe for a penalized logistic regression model, we have: 


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


### Create the workflow

As we introduced in [*Preprocess your data with recipes*](/start/recipes/#fit-workflow), let's bundle the model and recipe into a single `workflow()` object to make management of the R objects easier:


```r
lr_workflow <- 
  workflow() %>% 
  add_model(lr_mod) %>% 
  add_recipe(lr_recipe)
```

### Create the grid for tuning

Before we fit this model, we need to set up a grid of `penalty` values to tune. In our [*Tune model parameters*](/start/tuning/) article, we used [`dials::grid_regular()`](start/tuning/#tune-grid) to create the grid. We can also set the grid up manually using [`base::expand.grid()`](https://rdrr.io/r/base/expand.grid.html) to create a data frame with 30 candidate values:


```r
lr_reg_grid <- expand.grid(penalty = 10^seq(-4, -1, length.out = 30))

# show the 5 lowest and highest values
head(lr_reg_grid)
#>        penalty
#> 1 0.0001000000
#> 2 0.0001268961
#> 3 0.0001610262
#> 4 0.0002043360
#> 5 0.0002592944
#> 6 0.0003290345
tail(lr_reg_grid)
#>       penalty
#> 25 0.03039195
#> 26 0.03856620
#> 27 0.04893901
#> 28 0.06210169
#> 29 0.07880463
#> 30 0.10000000
```

### Train and tune the model

Let's use `tune::tune_grid()` to train these 30 penalized logistic regression models. We'll also save the validation set predictions (via the call to `control_grid()`) so that diagnostic information can be available after the model fit. The area under the ROC curve will be used to quantify how well the model performs across a continuum of event thresholds (recall that the event rate is very low for these data). 


```r
roc_only <- metric_set(roc_auc)

lr_res <- 
  lr_workflow %>% 
  tune_grid(val_set,
            grid = lr_reg_grid,
            control = control_grid(save_pred = TRUE),
            metrics = roc_only)
```

It might be easier to visualize the validation set metrics by plotting the area under the ROC curve against the range of penalty values: 


```r
lr_plot <- 
  lr_res %>% 
  collect_metrics() %>% 
  ggplot(aes(x = penalty, y = mean)) + 
  geom_point() + 
  geom_line() + 
  ylab("Area under the ROC Curve") +
  scale_x_log10(labels = scales::label_number())

lr_plot 
```

<img src="figs/logistic-results-1.svg" width="576" />

This plots shows us that model performance is generally better at the smaller penalty values. This suggests that the majority of the predictors are important to the model. We also see a steep drop in the area under the ROC curve towards the highest penalty values. This happens because a large enough penalty will remove _all_ predictors from the model, and not surprisingly predictive accuracy plummets with no predictors in the model (recall that an AUC = .5 means that the model does no better than chance at predicting the correct class).

Our model performance seems to plateau at the smaller penalty values, so going by the `roc_auc` metric alone could lead us to multiple options for the "best" value: 


```r
top_models <-
  lr_res %>% 
  show_best("roc_auc", n = 15) %>% 
  arrange(penalty) 
top_models
#> # A tibble: 15 x 6
#>     penalty .metric .estimator  mean     n std_err
#>       <dbl> <chr>   <chr>      <dbl> <int>   <dbl>
#>  1 0.0001   roc_auc binary     0.880     1      NA
#>  2 0.000127 roc_auc binary     0.881     1      NA
#>  3 0.000161 roc_auc binary     0.881     1      NA
#>  4 0.000204 roc_auc binary     0.881     1      NA
#>  5 0.000259 roc_auc binary     0.881     1      NA
#>  6 0.000329 roc_auc binary     0.881     1      NA
#>  7 0.000418 roc_auc binary     0.881     1      NA
#>  8 0.000530 roc_auc binary     0.881     1      NA
#>  9 0.000672 roc_auc binary     0.881     1      NA
#> 10 0.000853 roc_auc binary     0.881     1      NA
#> 11 0.00108  roc_auc binary     0.881     1      NA
#> 12 0.00137  roc_auc binary     0.881     1      NA
#> 13 0.00174  roc_auc binary     0.881     1      NA
#> 14 0.00221  roc_auc binary     0.880     1      NA
#> 15 0.00281  roc_auc binary     0.879     1      NA
```



We know that every candidate model in this tibble includes more predictor variables than the model in the row below it. If we used `select_best()`, it would return candidate model 8 with a penalty value of 0.00053, shown with the dotted line below. 

<img src="figs/lr-plot-lines-1.svg" width="576" />

But, we may want to choose a penalty value further along the x-axis, closer to where we start to see the decline in model performance. For example, candidate model 12 with a penalty value of 0.00137 has effectively the same performance as the numerically best model, but might eliminate more predictors. This penalty value is marked by the solid line above.

Let's select this value and visualize the validation set ROC curve:

```r
lr_best <- 
  lr_res %>% 
  collect_metrics() %>% 
  arrange(penalty) %>% 
  slice(12)
lr_best
#> # A tibble: 1 x 6
#>   penalty .metric .estimator  mean     n std_err
#>     <dbl> <chr>   <chr>      <dbl> <int>   <dbl>
#> 1 0.00137 roc_auc binary     0.881     1      NA
```



```r
lr_auc <- 
  lr_res %>% 
  collect_predictions(parameters = lr_best) %>% 
  roc_curve(children, .pred_children) %>% 
  mutate(model = "Logistic Regression")

autoplot(lr_auc)
```

<img src="figs/logistic-roc-curve-1.svg" width="672" />

The level of performance generated by this logistic regression model is good, but not groundbreaking. Perhaps the linear nature of the prediction equation is too limiting for this data set. As a next step, we might consider a highly non-linear model generated using a tree-based ensemble method. 

## A second model: tree-based ensemble {#second-model}

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
#> [1] 12
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

## Test set results

How did this model do on the test set? Was the validation set a good estimate of future performance? 


```r
rf_fit %>% 
  collect_predictions() %>% 
  roc_auc(children, .pred_children)
#> # A tibble: 1 x 3
#>   .metric .estimator .estimate
#>   <chr>   <chr>          <dbl>
#> 1 roc_auc binary         0.925

rf_fit %>% 
  collect_predictions() %>% 
  roc_curve(children, .pred_children) %>% 
  autoplot()
```

<img src="figs/test-set-roc-curve-1.svg" width="672" />

Based on these results, the validation set and test set performance statistics are very close. 

## Session information


```
#> ─ Session info ───────────────────────────────────────────────────────────────
#>  setting  value                       
#>  version  R version 3.6.1 (2019-07-05)
#>  os       macOS Catalina 10.15.3      
#>  system   x86_64, darwin15.6.0        
#>  ui       X11                         
#>  language (EN)                        
#>  collate  en_US.UTF-8                 
#>  ctype    en_US.UTF-8                 
#>  tz       America/Los_Angeles         
#>  date     2020-04-18                  
#> 
#> ─ Packages ───────────────────────────────────────────────────────────────────
#>  package    * version date       lib source        
#>  broom      * 0.5.5   2020-02-29 [1] CRAN (R 3.6.0)
#>  dials      * 0.0.4   2019-12-02 [1] CRAN (R 3.6.0)
#>  dplyr      * 0.8.5   2020-03-07 [1] CRAN (R 3.6.0)
#>  ggplot2    * 3.3.0   2020-03-05 [1] CRAN (R 3.6.0)
#>  infer      * 0.5.1   2019-11-19 [1] CRAN (R 3.6.0)
#>  parsnip    * 0.0.5   2020-01-07 [1] CRAN (R 3.6.0)
#>  purrr      * 0.3.3   2019-10-18 [1] CRAN (R 3.6.0)
#>  ranger       0.11.2  2019-03-07 [1] CRAN (R 3.6.0)
#>  readr      * 1.3.1   2018-12-21 [1] CRAN (R 3.6.0)
#>  recipes    * 0.1.10  2020-03-18 [1] CRAN (R 3.6.0)
#>  rlang        0.4.5   2020-03-01 [1] CRAN (R 3.6.0)
#>  rsample    * 0.0.6   2020-03-31 [1] CRAN (R 3.6.2)
#>  tibble     * 2.1.3   2019-06-06 [1] CRAN (R 3.6.0)
#>  tidymodels * 0.1.0   2020-02-16 [1] CRAN (R 3.6.0)
#>  tune       * 0.1.0   2020-04-02 [1] CRAN (R 3.6.2)
#>  vip        * 0.2.2   2020-04-06 [1] CRAN (R 3.6.2)
#>  workflows  * 0.1.1   2020-03-17 [1] CRAN (R 3.6.0)
#>  yardstick  * 0.0.5   2020-01-23 [1] CRAN (R 3.6.0)
#> 
#> [1] /Library/Frameworks/R.framework/Versions/3.6/Resources/library
```


