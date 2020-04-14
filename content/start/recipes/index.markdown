---
title: "Preprocess your data with recipes"
weight: 2
tags: [recipes, parsnip, workflows, yardstick, broom]
categories: [pre-processing]
description: | 
  Prepare data for modeling with modular preprocessing steps.
---





This article requires that you have the following packages installed: nycflights13, skimr, and tidymodels.

# Introduction

After reviewing our [first steps with models](/start/models/), let's explore how to use the recipes package. 

Recipes are tools primarily used for data preprocessing prior to creating a model. Such preprocessing might consist of: 

 * converting qualitative predictors to indicator variables (also known as dummy variables),
 
 * transforming data to be on a different scale (e.g., taking the logarithm of a variable), 
 
 * transforming whole groups of predictors together,
 
and so on. This might sound an awful lot like a model formula, if you have used R's formula interface. Recipes can be used to do many of the same things, but they have a much wider range of possibilities. This guide shows how to use recipes for modeling. 

# The New York City flight data

In this example, let's use the nycflights13 data. This data set contains information on flights departing near New York City in 2013. We will try to predict if a plane arrives more than 30 minutes late. Let's start by loading the data and making a few changes to the variables:


```r
library(nycflights13)
library(tidymodels)

set.seed(123)
flight_data <- 
  flights %>% 
  mutate(
    # Convert the arrival delay to a factor
    arr_delay = ifelse(arr_delay >= 30, "late", "on_time"),
    arr_delay = factor(arr_delay),
    # We will use the date (not date-time) in the recipe below
    date = as.Date(time_hour)
  ) %>% 
  # Include the weather data
  inner_join(weather, by = c("origin", "time_hour")) %>% 
  # Only retain the specific columns we will use
  select(dep_time, flight, origin, dest, air_time, distance, 
         carrier, date, arr_delay, time_hour) %>% 
  # Exclude missing data
  na.omit() %>% 
  # For creating models, it is better to have qualitative columns
  # encoded as factors (instead of character strings)
  mutate_if(is.character, as.factor)
```

Using this estimate, about 16% of the flights in this data set had late arrivals. A summary of the variables are:


```r
skimr::skim(flight_data)
#> Skim summary statistics
#>  n obs: 325819 
#>  n variables: 10 
#> 
#> ── Variable type:Date ──────────────────────────────────────────────────────────────────────────────────
#>  variable missing complete      n        min        max     median n_unique
#>      date       0   325819 325819 2013-01-01 2013-12-30 2013-07-03      364
#> 
#> ── Variable type:factor ────────────────────────────────────────────────────────────────────────────────
#>   variable missing complete      n n_unique
#>  arr_delay       0   325819 325819        2
#>    carrier       0   325819 325819       16
#>       dest       0   325819 325819      104
#>     origin       0   325819 325819        3
#>                                      top_counts ordered
#>                  on_: 273279, lat: 52540, NA: 0   FALSE
#>      UA: 57489, B6: 53715, EV: 50868, DL: 47465   FALSE
#>  ATL: 16771, ORD: 16507, LAX: 15942, BOS: 14948   FALSE
#>    EWR: 116504, JFK: 108539, LGA: 100776, NA: 0   FALSE
#> 
#> ── Variable type:integer ───────────────────────────────────────────────────────────────────────────────
#>  variable missing complete      n    mean      sd p0 p25  p50  p75 p100
#>  dep_time       0   325819 325819 1348.15  487.92  1 907 1400 1743 2400
#>    flight       0   325819 325819 1943.54 1621.73  1 544 1471 3416 8500
#>      hist
#>  ▁▁▇▆▆▇▆▂
#>  ▇▅▂▃▂▁▁▁
#> 
#> ── Variable type:numeric ───────────────────────────────────────────────────────────────────────────────
#>  variable missing complete      n    mean     sd p0 p25 p50  p75 p100     hist
#>  air_time       0   325819 325819  150.63  93.66 20  82 129  191  695 ▇▇▂▃▁▁▁▁
#>  distance       0   325819 325819 1048.18 735.86 80 509 888 1389 4983 ▇▇▂▂▁▁▁▁
#> 
#> ── Variable type:POSIXct ───────────────────────────────────────────────────────────────────────────────
#>   variable missing complete      n        min        max     median n_unique
#>  time_hour       0   325819 325819 2013-01-01 2013-12-30 2013-07-03     6885
```

There are some interesting things to notice from this output. First, the flight number is a numeric value. In our analyses below, this column won't be used as a predictor but retained as an identification variable (along with `time_hour`) that can be used to troubleshoot poorly predicted data points.  

Second, there are 104 destination values contained in `dest` as well as 16 carriers. In the initial analysis, these will be converted to simple [dummy variables](https://bookdown.org/max/FES/creating-dummy-variables-for-unordered-categories.html). However, some of these values do not occur very frequently and this could complicate our analysis, as we discuss more below. 

To get started, let's split these data into two subsets. We'll keep most of the data (subset chosen randomly) in a _training set_. This subset of the data is used create the model. The remainder of the data are used as a _test set_ and this subset is only used to measure model performance^[This straightforward split of data into two subsets is fairly common. However, it is inadvisable to directly predict the training set one time after fitting the model. Instead, [resampling methods](https://bookdown.org/max/FES/resampling.html) can be used to estimate how well the model performs on the training set.].

To do this, we can use the [rsample](https://tidymodels.github.io/rsample/) package to create an object that contains the information on _how_ to split the data, and then two more rsample functions to create data frames for the training and testing sets: 


```r
# Fix the random numbers by setting the seed 
# This enables the analysis to be reproducible when random numbers are used 
set.seed(234)
# Put 3/4 of the data into the training set 
data_split <- initial_split(flight_data, prop = 3/4)

# Create data frames for the two sets:
train_data <- training(data_split)
test_data  <- testing(data_split)
```
 
# Create the predictor definitions

To get started, let's create a simple logistic regression model. Before creating the model fit object, we can use a recipe to create a few new predictors and conduct some preprocessing required by the model. 

To get started, an initial recipe declares the roles of the columns of the data: 


```r
flights_rec <- 
  recipe(arr_delay ~ ., data = train_data) %>% 
  update_role(flight, time_hour, new_role = "ID") 
```

The `recipe()` command enumerates the columns in the data, their type (e.g. categorical, numeric), and their role. For the latter, any variable on the left-hand side of the tilde (`~`) is considered the model outcome (`arr_delay` in this case) and the others are initially considered to be predictors. 

After defining the initial recipe, a series of data processing steps can be added. For example, `update_role()` tells the recipe that two of the columns are not predictors _or_ outcomes. The role of these columns is changed to `"ID"` (a role can have any character value). The purpose of changing this characteristic for these columns is that they can be retained in the data but not included in the model. This can be convenient when, after the model is fit, we want to investigate some poorly predicted value. These ID columns will be available and can be used to try to understand what went wrong.

To get the current set of variables and roles, use the `summary()` function: 


```r
summary(flights_rec)
#> # A tibble: 10 x 4
#>    variable  type    role      source  
#>    <chr>     <chr>   <chr>     <chr>   
#>  1 dep_time  numeric predictor original
#>  2 flight    numeric ID        original
#>  3 origin    nominal predictor original
#>  4 dest      nominal predictor original
#>  5 air_time  numeric predictor original
#>  6 distance  numeric predictor original
#>  7 carrier   nominal predictor original
#>  8 date      date    predictor original
#>  9 time_hour date    ID        original
#> 10 arr_delay nominal outcome   original
```

A value of "nominal" means that the column is either of type factor or character. 

Note that a recipe is always associated with the data set used to create the model; we used `data = train_data` when specifying it. A recipe is associated with a training set, as opposed to a test set.  

We can add many other operations to the recipe. Perhaps it is reasonable for the date of the flight to have an effect on the likelihood of a late arrival. A little bit of **feature engineering** might go a long way to improving our model. How should the date be encoded into the model?  The `date` column has an R `date` object so including that column as it exists now in the model will just convert it to a numeric format that is the number of days after a reference date: 


```r
days <- 
  flight_data %>% 
  distinct(date) %>% 
  slice(1:3) %>% 
  pull(date)

days
#> [1] "2013-01-01" "2013-01-02" "2013-01-03"
as.numeric(days)
#> [1] 15706 15707 15708
```

It's possible this is a good option for modeling; perhaps the model would benefit from a linear trend between the log-odds of a late arrival and the day number. However, it might be better to add model terms _derived_ from the date that have a better potential to be important to the model. For example, we could use the date to derive: 

 * the day of the week,
 
 * the month,
 
 * whether or not the date corresponds to a holiday, 
 
and so on. To do this, let's add two more steps to the recipe:



```r
flights_rec <- 
  recipe(arr_delay ~ ., data = train_data) %>% 
  update_role(flight, time_hour, new_role = "ID") %>% 
  step_date(date, features = c("dow", "month")) %>% 
  step_holiday(date, holidays = timeDate::listHolidays("US")) %>% 
  step_rm(date)
```

`step_date()` creates basic date-oriented features. In this case, two factor columns are added with the appropriate day of the week and the month. `step_holiday()` creates binary indicator variables detailing if the current date is a holiday or not. The argument value of `timeDate::listHolidays("US")` uses the timeDate package to list the 17 standard US holidays. Finally, since we longer want the date column itself in the model, `step_rm()` eliminates it from the data set. 

To train this model, each predictor will need to be in a numeric format^[When computing the model coefficients, the equations that logistic regression uses requires numbers (as opposed to factor variables). In other words, there may be a difference in how we store our data (in factors inside a data frame) and how the underlying equations require them (a purely numeric matrix). Luckily, it is easy to go between these two formats in R.]. For columns like `dest` and `origin`, which are currently factor columns, the standard method to convert them to be numeric is to create _dummy_ or _indicator_ variables. These are binary values for the levels of the factors. For example, since `origin` has values of `"EWR"`, `"JFK"`, and `"LGA"`, the standard dummy variable encoding will create _two_ numeric columns of the data that are 1 when the originating airport is `"JFK"` or `"LGA"` and zero otherwise, respectively^[Why two and not three? If you know the value of two of the columns, the third can be inferred. The standard in R is to leave out the dummy variable column for the first level of the factor which, in this case, corresponds to `"EWR"`.].

Unlike the standard model formula methods in R, a recipe **does not** automatically create dummy variables^[We don't make the assumption that dummy variables are going to be used. For one reason, [not all models require the predictors to be numeric](https://bookdown.org/max/FES/categorical-trees.html). For another reason, recipes can be used to prepare the data for other, non-model purposes that prefer factors (such as a table or plot).]. Instead, `step_dummy()` can be used for this purpose: 


```r
flights_rec <- 
  recipe(arr_delay ~ ., data = train_data) %>% 
  update_role(flight, time_hour, new_role = "ID") %>% 
  step_date(date, features = c("dow", "month")) %>% 
  step_holiday(date, holidays = timeDate::listHolidays("US")) %>% 
  step_rm(date) %>% 
  step_dummy(all_nominal(), -all_outcomes())
```

Note that the dplyr selectors don't have to use column _names_. Since a recipe knows the _role_ of each column, they can also be selected using this information. The selectors above translate to

> Create dummy variables for all of the factor or character columns _unless_ they are outcomes. 

At this stage in the recipe, this selects columns `origin`, `dest`, `carrier`, `date_dow`, and `date_month` (the latter two were created by `step_date()`). 

One final step is added to the recipe; since `carrier` and `dest` have some infrequently occurring values, it is possible that dummy variables might be created for values that don't exist in the training set. For example, there is one destination that is only in the test set: 


```r
setdiff(flight_data$dest, train_data$dest)
#> character(0)
```

When the recipe is applied to the training set, a column is made for  but it will contain all zeros. This is a "zero-variance predictor" that has no information within the column. While some R functions will not produce an error for such predictors, it usually causes warnings and other issues. `step_zv()` will remove columns from the data when the training set data have a single value, so it is added to the recipe: 


```r
flights_rec <- 
  recipe(arr_delay ~ ., data = train_data) %>% 
  update_role(flight, time_hour, new_role = "ID") %>% 
  step_date(date, features = c("dow", "month")) %>% 
  step_holiday(date, holidays = timeDate::listHolidays("US")) %>% 
  step_rm(date) %>% 
  step_dummy(all_nominal(), -all_outcomes()) %>% 
  step_zv(all_predictors())
```

Now that we've created this _specification_ of what should be done with the data, how do we use it? 

# Use a recipe with a model

Let's use straightforward logistic regression to model the flight data. We can build a model specification using the parsnip package: 


```r
lr_mod <- 
  logistic_reg() %>% 
  set_engine("glm")
```

How will we evaluate our model performance? Let's stick with the area under the [ROC curve](https://bookdown.org/max/FES/measuring-performance.html#class-metrics), computed using the yardstick package functions called `roc_curve()` and `roc_auc()`. 

During our modeling process, we will use our recipe in multiple steps. We will: 

 1. **Process the recipe using the training set**: This involves any estimation or calculations on these data. For our recipe, the training set would be used to determine what dummy variable columns should be created and which zero-variance predictors are slated for removal. 
 
 1. **Apply the recipe to the training set**: We create the final predictor set on the training set. 
 
 1. **Apply the recipe to the test set**: We create the final predictor set on the test set. Nothing is recomputed; the dummy variable and zero-variance results from the training set are applied to the test set. 
 
There are a few methods for doing this. One straightforward and simple approach is to use a _model workflow_ which pairs a model and recipe together.


```r
flights_wflow <- 
  workflow() %>% 
  add_model(lr_mod) %>% 
  add_recipe(flights_rec)
flights_wflow
#> ══ Workflow ════════════════════════════════════════════════════════════════════════════════════════════
#> Preprocessor: Recipe
#> Model: logistic_reg()
#> 
#> ── Preprocessor ────────────────────────────────────────────────────────────────────────────────────────
#> 5 Recipe Steps
#> 
#> ● step_date()
#> ● step_holiday()
#> ● step_rm()
#> ● step_dummy()
#> ● step_zv()
#> 
#> ── Model ───────────────────────────────────────────────────────────────────────────────────────────────
#> Logistic Regression Model Specification (classification)
#> 
#> Computational engine: glm
```

There are two nice properties of using a `workflow()`. First, the model and recipe are bundled so that you can more easily keep track of them. Second, there is a single function that can be used to prepare the recipe and create the model from the resulting predictors: 


```r
flights_fit <- fit(flights_wflow, data = train_data)
#> Warning: The `x` argument of `as_tibble.matrix()` must have column names if `.name_repair` is omitted as of tibble 2.0.0.
#> Using compatibility `.name_repair`.
#> This warning is displayed once every 8 hours.
#> Call `lifecycle::last_warnings()` to see where this warning was generated.
```
 
This object has the finalized recipe and model objects inside. To extract those objects, use the functions `pull_workflow_fit()` and `pull_workflow_recipe()`. For example, use the `broom::tidy()` function to get data on the model coefficients: 


```r
flights_fit %>% 
  pull_workflow_fit() %>% 
  tidy()
#> # A tibble: 158 x 5
#>    term                         estimate std.error statistic  p.value
#>    <chr>                           <dbl>     <dbl>     <dbl>    <dbl>
#>  1 (Intercept)                   3.06    2.72           1.12 2.62e- 1
#>  2 dep_time                     -0.00166 0.0000141   -118.   0.      
#>  3 air_time                     -0.0437  0.000562     -77.8  0.      
#>  4 distance                      0.00728 0.00150        4.85 1.21e- 6
#>  5 date_USChristmasDay           1.29    0.184          7.02 2.17e-12
#>  6 date_USColumbusDay            0.777   0.170          4.56 5.12e- 6
#>  7 date_USCPulaskisBirthday      0.718   0.135          5.34 9.50e- 8
#>  8 date_USDecorationMemorialDay  0.325   0.116          2.80 5.08e- 3
#>  9 date_USElectionDay            0.726   0.173          4.20 2.67e- 5
#> 10 date_USGoodFriday             1.19    0.160          7.39 1.48e-13
#> # … with 148 more rows
```

There is also a single interface for getting predictions on new data. The `predict()` method applies the recipe to the new data, then passes them to the fitted model. For the ROC curve, we need the predicted class probabilities (along with the true outcome column):  


```r
flights_pred <- 
  predict(flights_fit, test_data, type = "prob") %>% 
  bind_cols(test_data %>% select(arr_delay, time_hour, flight)) 

# The data look like: 
flights_pred %>% slice(1:5)
#> # A tibble: 5 x 5
#>   .pred_late .pred_on_time arr_delay time_hour           flight
#>        <dbl>         <dbl> <fct>     <dttm>               <int>
#> 1     0.0898         0.910 on_time   2013-01-01 05:00:00   1696
#> 2     0.0378         0.962 on_time   2013-01-01 06:00:00   1124
#> 3     0.163          0.837 late      2013-01-01 06:00:00    707
#> 4     0.0115         0.988 on_time   2013-01-01 06:00:00    709
#> 5     0.0236         0.976 on_time   2013-01-01 06:00:00   1019
```

We can create the ROC curve with these values, using `roc_curve()` and then piping to the `autoplot()` method: 


```r
flights_pred %>% 
  roc_curve(truth = arr_delay, .pred_late) %>% 
  autoplot()
```

<img src="figs/roc-plot-1.svg" width="672" />

Similarly, `roc_auc()` estimates the area under the curve: 


```r
flights_pred %>% roc_auc(truth = arr_delay, .pred_late)
#> # A tibble: 1 x 3
#>   .metric .estimator .estimate
#>   <chr>   <chr>          <dbl>
#> 1 roc_auc binary         0.765
```

Not too bad!


# Session information


```
#> ─ Session info ───────────────────────────────────────────────────────────────
#>  setting  value                       
#>  version  R version 3.6.1 (2019-07-05)
#>  os       macOS Mojave 10.14.6        
#>  system   x86_64, darwin15.6.0        
#>  ui       X11                         
#>  language (EN)                        
#>  collate  en_US.UTF-8                 
#>  ctype    en_US.UTF-8                 
#>  tz       America/New_York            
#>  date     2020-04-13                  
#> 
#> ─ Packages ───────────────────────────────────────────────────────────────────
#>  package      * version date       lib source        
#>  broom        * 0.5.4   2020-01-27 [1] CRAN (R 3.6.0)
#>  dials        * 0.0.6   2020-04-03 [1] CRAN (R 3.6.2)
#>  dplyr        * 0.8.5   2020-03-07 [1] CRAN (R 3.6.0)
#>  ggplot2      * 3.3.0   2020-03-05 [1] CRAN (R 3.6.0)
#>  infer        * 0.5.1   2019-11-19 [1] CRAN (R 3.6.0)
#>  nycflights13 * 1.0.1   2019-09-16 [1] CRAN (R 3.6.0)
#>  parsnip      * 0.1.0   2020-04-09 [1] CRAN (R 3.6.2)
#>  purrr        * 0.3.3   2019-10-18 [1] CRAN (R 3.6.0)
#>  recipes      * 0.1.10  2020-03-18 [1] CRAN (R 3.6.0)
#>  rlang          0.4.5   2020-03-01 [1] CRAN (R 3.6.0)
#>  rsample      * 0.0.6   2020-03-31 [1] CRAN (R 3.6.2)
#>  skimr        * 1.0.7   2019-06-20 [1] CRAN (R 3.6.0)
#>  tibble       * 3.0.0   2020-03-30 [1] CRAN (R 3.6.1)
#>  tidymodels   * 0.1.0   2020-02-16 [1] CRAN (R 3.6.0)
#>  tune         * 0.1.0   2020-04-02 [1] CRAN (R 3.6.2)
#>  workflows    * 0.1.0   2019-12-30 [1] CRAN (R 3.6.1)
#>  yardstick    * 0.0.5   2020-01-23 [1] CRAN (R 3.6.0)
#> 
#> [1] /Library/Frameworks/R.framework/Versions/3.6/Resources/library
```
