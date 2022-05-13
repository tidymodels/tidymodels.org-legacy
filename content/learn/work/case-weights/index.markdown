---
title: "Using case weights"
tags: [parsnip,tune,recipes,workflows]
categories: [model fitting]
type: learn-subsection
weight: 5
description: | 
  Create models that use coefficients, extract them from fitted models, and visualize them.
---


  



Post idea:
- Build overly simplistic model, only using the date variable
- Include weights as a transformation of days before 2016.

## Introduction

To use the code in this article, you will need to install the following packages: tidymodels.

This article demonstrates how to create and use importance weights in a predictive model. Using importance weights is a way to have our model care more about some observations than others.

## Example Data

To demonstrate we will use the Chocago data from the modeldata package.


```r
library(tidymodels)
data(Chicago)

Chicago <- Chicago %>%
  select(ridership, date, one_of(stations))
```

From `?Chicago`

> These data are from Kuhn and Johnson (2020) and contain an abbreviated training set for modeling the number of people (in thousands) who enter the Clark and Lake L station.

> The date column corresponds to the current date. The columns with station names (Austin through California) are a sample of the columns used in the original analysis (for file size reasons). These are 14 day lag variables (i.e. date - 14 days). There are columns related to weather and sports team schedules.

For simplicity we have limited our view to the date and station variables.

## Creating weights

This data set contains daily information from `r `min(Chicago$date)` to 2016-08-28, we will pretend that it is January 1st 2016 and we want to predict the ridership for the remainder of 2016 using the date variable and station variables as predictors. Without any weighting all the previous observations would have the same influence in the model, which may or may not be ideal since some observations appears a long time ago and not be as representative as of the future as more recent observations. 

One way to deal with this is to only use recent observations to fit the model, ensuring that the training data stays as close the testing data as possible. While a tempting idea, it would throw out a lot of informative data. Instead let us assign a weight to each observation, related to how long ago the observation was taken. This way we are not throwing away any observation completely.

We need to decide on a way to calculate these weights. The main thing constraint is that the weight cannot be negative, and it would be nice if today was weighted as 1. So we need a function that is 1 when `x=0` and decreasing otherwise. There are many kinds of functions like that, and we will be using this exponential decay function

$$ weight = base ^ x $$

where `base` is some constant and `x` is the number of days. To make sure that we select `base` we need to do some manual testing, starting with looking at how old the oldest observation is.


```r
difftime("2016-01-01", min(Chicago$date))
#> Time difference of 5457 days
```

Using this information we can visualize the weight curve, to see if we like the value of `base`.


```r
tibble_days <- tibble(days = 0:5457)

tibble_days %>%
  ggplot(aes(days)) +
  geom_function(fun = ~ 0.99 ^ .x)
```

<img src="figs/unnamed-chunk-3-1.svg" width="672" />

setting `base` to 0.99 appears to down weight too much. Any observation more than year old would be have no influence.

Let us try a few more values to find 


```r
map_dfr(
  c(0.99, 0.999, 0.9999),
  ~ tibble_days %>% mutate(base = factor(.x), value = .x ^ days)
) %>%
  ggplot(aes(days, value, group = base, color = base)) +
  geom_line()
```

<img src="figs/unnamed-chunk-4-1.svg" width="672" />

And from this we would pick something around 0.999 maybe a little less since it gives a better balance. Creating a small function to help us encode this weight. Notice that I'm setting any future weight to 1.


```r
weights_from_dates <- function(x, ref) {
  if_else(
    condition = x >= ref,
    true = 1,
    false = 0.999 ^ as.numeric(difftime(ref, x, units = "days"))
  )
}
```

We then modify `Chicago` to add a weight column, explicitly making it a importance weight with `importance_weight()`.


```r
Chicago <- Chicago %>%
  mutate(weight = weights_from_dates(date, "2016-01-01"),
         weight = importance_weights(weight))
```

This approach to creating importance weights from dates are not limited to cases where we have daily observations like we do in this data set. You are free to create weights like this, if you have gaps or repeated observations within the same day. Likewise you don't need to use days as the unit. seconds, weeks or years could be used as well.

## Modeling

We start by splitting up our data into a training and testing set based on the day `"2016-01-01"`. We added weights to the data set before splitting it so each set has weights.


```r
Chicago_train <- Chicago %>% filter(date < "2016-01-01")
Chicago_test <- Chicago %>% filter(date >= "2016-01-01")
```

Next we are going to create a recipe. The weights won't have any influence on the preprocessing since none of these operations are supervised and we are using importance weights.


```r
base_recipe <-
  recipe(ridership ~ ., data = Chicago_train) %>%
  # Create date features
  step_date(date) %>%
  step_holiday(date, keep_original_cols = FALSE) %>%
  # Remove any columns with a single unique value
  step_zv(all_predictors()) %>%
  # Normalize all the numerical features
  step_normalize(all_numeric_predictors()) %>%
  # Perform PCA to reduce the correlation bet the stations
  step_pca(all_numeric_predictors(), threshold = 0.95)
```

Next we need to build the rest of the workflow. We use a linear regression specification


```r
lm_spec <-
  linear_reg() %>%
  set_engine("lm")
```

and we add these together in the workflow. To activate the case weights we use the `add_case_weights()` function to specify the name of the case weights being used.


```r
lm_wflow <-
  workflow() %>% 
  add_case_weights(weight) %>%
  add_recipe(base_recipe) %>%
  add_model(lm_spec)

lm_wflow
#> ══ Workflow ══════════════════════════════════════════════════════════
#> Preprocessor: Recipe
#> Model: linear_reg()
#> 
#> ── Preprocessor ──────────────────────────────────────────────────────
#> 5 Recipe Steps
#> 
#> • step_date()
#> • step_holiday()
#> • step_zv()
#> • step_normalize()
#> • step_pca()
#> 
#> ── Case Weights ──────────────────────────────────────────────────────
#> weight
#> 
#> ── Model ─────────────────────────────────────────────────────────────
#> Linear Regression Model Specification (regression)
#> 
#> Computational engine: lm
```

With all that done we can fit the workflow like normal


```r
lm_fit <- fit(lm_wflow, data = Chicago_train)
lm_fit
#> ══ Workflow [trained] ════════════════════════════════════════════════
#> Preprocessor: Recipe
#> Model: linear_reg()
#> 
#> ── Preprocessor ──────────────────────────────────────────────────────
#> 5 Recipe Steps
#> 
#> • step_date()
#> • step_holiday()
#> • step_zv()
#> • step_normalize()
#> • step_pca()
#> 
#> ── Case Weights ──────────────────────────────────────────────────────
#> weight
#> 
#> ── Model ─────────────────────────────────────────────────────────────
#> 
#> Call:
#> stats::lm(formula = ..y ~ ., data = data, weights = weights)
#> 
#> Coefficients:
#>   (Intercept)    date_dowMon    date_dowTue    date_dowWed    date_dowThu  
#>       1.76260       13.30765       14.68903       14.62018       14.38231  
#>   date_dowFri    date_dowSat  date_monthFeb  date_monthMar  date_monthApr  
#>      13.69543        1.22823        0.36434        1.34823        1.40990  
#> date_monthMay  date_monthJun  date_monthJul  date_monthAug  date_monthSep  
#>       1.18819        2.59830        2.21972        2.40700        1.93206  
#> date_monthOct  date_monthNov  date_monthDec            PC1            PC2  
#>       2.65555        0.90901       -0.00475        0.07301       -1.59102  
#>           PC3            PC4            PC5  
#>       0.60839       -0.20531       -0.69601
```

## Session information


```
#> ─ Session info ─────────────────────────────────────────────────────
#>  setting  value
#>  version  R version 4.1.2 (2021-11-01)
#>  os       macOS Monterey 12.2.1
#>  system   aarch64, darwin20
#>  ui       X11
#>  language (EN)
#>  collate  en_US.UTF-8
#>  ctype    en_US.UTF-8
#>  tz       America/Los_Angeles
#>  date     2022-05-03
#>  pandoc   2.17.1.1 @ /Applications/RStudio.app/Contents/MacOS/quarto/bin/ (via rmarkdown)
#> 
#> ─ Packages ─────────────────────────────────────────────────────────
#>  package    * version    date (UTC) lib source
#>  broom      * 0.8.0      2022-04-13 [1] CRAN (R 4.1.2)
#>  dials      * 0.1.1      2022-04-06 [1] CRAN (R 4.1.2)
#>  dplyr      * 1.0.9      2022-04-28 [1] CRAN (R 4.1.2)
#>  ggplot2    * 3.3.5      2021-06-25 [1] CRAN (R 4.1.1)
#>  infer      * 1.0.0      2021-08-13 [1] CRAN (R 4.1.1)
#>  parsnip    * 0.2.1.9001 2022-05-02 [1] Github (tidymodels/parsnip@284252b)
#>  purrr      * 0.3.4      2020-04-17 [1] CRAN (R 4.1.0)
#>  recipes    * 0.2.0.9001 2022-05-02 [1] Github (tidymodels/recipes@ce1c657)
#>  rlang        1.0.2      2022-03-04 [1] CRAN (R 4.1.2)
#>  rsample    * 0.1.1      2021-11-08 [1] CRAN (R 4.1.2)
#>  tibble     * 3.1.6      2021-11-07 [1] CRAN (R 4.1.1)
#>  tidymodels * 0.2.0      2022-03-19 [1] CRAN (R 4.1.2)
#>  tune       * 0.2.0.9001 2022-05-02 [1] Github (tidymodels/tune@488aff3)
#>  workflows  * 0.2.6.9001 2022-05-02 [1] Github (tidymodels/workflows@9a9e231)
#>  yardstick  * 0.0.9.9000 2022-05-02 [1] Github (tidymodels/yardstick@e56b69f)
#> 
#>  [1] /Library/Frameworks/R.framework/Versions/4.1-arm64/Resources/library
#> 
#> ────────────────────────────────────────────────────────────────────
```
