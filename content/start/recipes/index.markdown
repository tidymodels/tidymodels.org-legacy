---
title: "Preprocess your data with recipes"
weight: 2
tags: [recipes, parsnip, workflows, yardstick, broom]
categories: [pre-processing]
description: | 
  Prepare data for modeling with modular preprocessing steps.
---





# Introduction

In our [first article](/start/models/), we built models using the [parsnip package](https://tidymodels.github.io/parsnip/). Now, let's explore how to use another tidymodels package: [recipes](https://tidymodels.github.io/recipes/). Recipes are designed to preprocess data prior to building and fitting a model. Preprocessing might consist of: 

+ converting qualitative predictors to indicator variables (also known as dummy variables),
 
+ transforming data to be on a different scale (e.g., taking the logarithm of a variable), 
 
+ transforming whole groups of predictors together,

+ extracting key features from raw variables (e.g., getting the day of the week out of a date variable),
 
and so on. This might sound an awful lot like a model formula, if you have used R's formula interface. Recipes can be used to do many of the same things, but they have a much wider range of possibilities. This guide shows how to use recipes for modeling. 

This article requires that you have the following packages installed: nycflights13, skimr, and tidymodels.


```r
library(tidymodels) 

# Helper packages
library(nycflights13)    # for flight data
library(skimr)           # for variable summaries
```


# The New York City flight data



Let's use the [nycflights13 data](https://github.com/hadley/nycflights13) to try to predict whether a plane arrives more than 30 minutes late. This data set contains information on 325819 flights departing near New York City in 2013. Let's start by loading the data and making a few changes to the variables:


```r
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



We can see that about 16% of the flights in this data set arrived more than 30 minutes late. 


```r
flight_data %>% 
  count(arr_delay) %>% 
  mutate(prop = n/sum(n))
#> # A tibble: 2 x 3
#>   arr_delay      n  prop
#>   <fct>      <int> <dbl>
#> 1 late       52540 0.161
#> 2 on_time   273279 0.839
```


Before we start building our recipe, we want to highlight a few features of this data that will be important for both preprocessing and modeling.


```r
glimpse(flight_data)
#> Observations: 325,819
#> Variables: 10
#> $ dep_time  <int> 517, 533, 542, 544, 554, 554, 555, 557, 557, 558, 558, 558,…
#> $ flight    <int> 1545, 1714, 1141, 725, 461, 1696, 507, 5708, 79, 301, 49, 7…
#> $ origin    <fct> EWR, LGA, JFK, JFK, LGA, EWR, EWR, LGA, JFK, LGA, JFK, JFK,…
#> $ dest      <fct> IAH, IAH, MIA, BQN, ATL, ORD, FLL, IAD, MCO, ORD, PBI, TPA,…
#> $ air_time  <dbl> 227, 227, 160, 183, 116, 150, 158, 53, 140, 138, 149, 158, …
#> $ distance  <dbl> 1400, 1416, 1089, 1576, 762, 719, 1065, 229, 944, 733, 1028…
#> $ carrier   <fct> UA, UA, AA, B6, DL, UA, B6, EV, B6, AA, B6, B6, UA, UA, AA,…
#> $ date      <date> 2013-01-01, 2013-01-01, 2013-01-01, 2013-01-01, 2013-01-01…
#> $ arr_delay <fct> on_time, on_time, late, on_time, on_time, on_time, on_time,…
#> $ time_hour <dttm> 2013-01-01 05:00:00, 2013-01-01 05:00:00, 2013-01-01 05:00…
```

First, there are two variables that we don't want to use as predictors in our model, but that we would like to retain as identification variables that can be used to troubleshoot poorly predicted data points. These are `flight`, a numeric value, and `time_hour`, a date-time value.

Second, there are 104 flight destinations contained in `dest` and 16 distinct carriers. 


```r
flight_data %>% 
  skimr::skim(dest, carrier)
```


|                         |           |
|:------------------------|:----------|
|Name                     |Piped data |
|Number of rows           |325819     |
|Number of columns        |10         |
|_______________________  |           |
|Column type frequency:   |           |
|factor                   |2          |
|________________________ |           |
|Group variables          |None       |


**Variable type: factor**

|skim_variable | n_missing| complete_rate|ordered | n_unique|top_counts                                     |
|:-------------|---------:|-------------:|:-------|--------:|:----------------------------------------------|
|dest          |         0|             1|FALSE   |      104|ATL: 16771, ORD: 16507, LAX: 15942, BOS: 14948 |
|carrier       |         0|             1|FALSE   |       16|UA: 57489, B6: 53715, EV: 50868, DL: 47465     |


We'll be using a simple logistic regression model, which means that `dest` and `carrier` will be converted to [dummy variables](https://bookdown.org/max/FES/creating-dummy-variables-for-unordered-categories.html). However, some of these values do not occur very frequently and this could complicate our analysis, as we discuss more below. 

# Data splitting

To get started, let's split this single dataset into two: a _training_ set and a _testing_ set. We'll keep most of the rows in the original dataset (subset chosen randomly) in the _training_ set. The training data will be used to *fit* the model, and the _testing_ set will be used to measure model performance. 

To do this, we can use the [rsample](https://tidymodels.github.io/rsample/) package to create an object that contains the information on _how_ to split the data, and then two more rsample functions to create data frames for the training and testing sets: 


```r
# Fix the random numbers by setting the seed 
# This enables the analysis to be reproducible when random numbers are used 
set.seed(555)
# Put 3/4 of the data into the training set 
data_split <- initial_split(flight_data, prop = 3/4)

# Create data frames for the two sets:
train_data <- training(data_split)
test_data  <- testing(data_split)
```

In practice, our preferred approach is to use [resampling methods](https://bookdown.org/max/FES/resampling.html) with the training set first, before using the testing set. We'll work with this single split now, then return to more advanced resampling methods in the [next article](/start/resampling/).
 
# Create recipe and roles

To get started, let's create a recipe for a simple logistic regression model. Before training the model, we can use a recipe to create a few new predictors and conduct some preprocessing required by the model. 

Let's initiate a simple recipe to start: 


```r
flights_rec <- 
  recipe(arr_delay ~ ., data = train_data) 
```

The [`recipe()` function](https://tidymodels.github.io/recipes/reference/recipe.html) has two arguments:

+ A **formula**. Any variable on the left-hand side of the tilde (`~`) is considered the model outcome (here, `arr_delay`). On the right-hand side of the tilde are the predictors. Variables may be listed by name, or you can use the dot (`.`) to indicate all other variables as predictors.

+ The **data**. A recipe is associated with the data set used to create the model. This will typically be the _training_ set, so `data = train_data` here. Naming a data set doesn't actually change the data itself; it is only used to catalog the names of the variables and their types, like factors, integers, dates, etc.

Now let's add [roles](https://tidymodels.github.io/recipes/reference/roles.html) to this recipe. We can use the [`update_role()` function](https://tidymodels.github.io/recipes/reference/roles.html) to let recipes know that `flight` and `time_hour` are variables with a custom role that we called `"ID"` (a role can have any character value). Whereas our formula included all variables in the training set other than `arr_delay` as predictors, this tells the recipe to keep these two variables but not use them as either outcomes or predictors.


```r
flights_rec <- 
  recipe(arr_delay ~ ., data = train_data) %>% 
  update_role(flight, time_hour, new_role = "ID") 
```

The purpose of changing this characteristic for these columns is that they can be retained in the data but not included in the model. This can be convenient when, after the model is fit, we want to investigate some poorly predicted value. These ID columns will be available and can be used to try to understand what went wrong.

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



# Create features

After defining the outcome and predictor variables, we can start adding steps onto our recipe using the pipe operator. Perhaps it is reasonable for the date of the flight to have an effect on the likelihood of a late arrival. A little bit of **feature engineering** might go a long way to improving our model. How should the date be encoded into the model? The `date` column has an R `date` object so including that column "as is" will mean that the model will convert it to a numeric format equal to the number of days after a reference date: 


```r
flight_data %>% 
  distinct(date) %>% 
  mutate(numeric_date = as.numeric(date)) %>% 
  head()
#> # A tibble: 6 x 2
#>   date       numeric_date
#>   <date>            <dbl>
#> 1 2013-01-01        15706
#> 2 2013-01-02        15707
#> 3 2013-01-03        15708
#> 4 2013-01-04        15709
#> 5 2013-01-05        15710
#> 6 2013-01-06        15711
```

It's possible this is a good option for modeling; perhaps the model would benefit from a linear trend between the log-odds of a late arrival and the day number. However, it might be better to add model terms _derived_ from the date that have a better potential to be important to the model. For example, we could derive the following meaningful features from the single `date` variable: 

* the day of the week using `step_date(..., features = "dow"))`,
 
* the month using `step_date(..., features = "month"))`, and
 
* whether or not the date corresponds to a holiday using `step_holiday(...)`. 
 
Let's do all three of these by adding three new steps to our recipe:



```r
flights_rec <- 
  recipe(arr_delay ~ ., data = train_data) %>% 
  update_role(flight, time_hour, new_role = "ID") %>% 
  step_date(date, features = c("dow", "month")) %>% 
  step_holiday(date, holidays = timeDate::listHolidays("US")) %>% 
  step_rm(date)
```

With [`step_date()`](https://tidymodels.github.io/recipes/reference/step_date.html), we created two new factor columns with the appropriate day of the week and the month. With [`step_holiday()`](https://tidymodels.github.io/recipes/reference/step_holiday.html), we created binary indicator variables detailing if the current date is a holiday or not. The argument value of `timeDate::listHolidays("US")` uses the timeDate package to list the 17 standard US holidays. Finally, with [`step_rm()`](https://tidymodels.github.io/recipes/reference/step_rm.html), we remove the original `date` variable since we no longer want it in the model.

Next, we need to turn our attention to the variables types of our predictors. Because we plan to train a logistic regression model, we know that predictors will ultimately need to be numeric, as opposed to factor variables. In other words, there may be a difference in how we store our data (in factors inside a data frame), and how the underlying equations require them (a purely numeric matrix).

For factors like `dest` and `origin`, [standard practice](https://bookdown.org/max/FES/creating-dummy-variables-for-unordered-categories.html) is to convert them into _dummy_ or _indicator_ variables to make them numeric. These are binary values for each level of the factor. For example, since `origin` has values of `"EWR"`, `"JFK"`, and `"LGA"`, the standard dummy variable encoding will create _two_ numeric columns of the data that are 1 when the originating airport is `"JFK"` or `"LGA"` and zero otherwise, respectively.

Unlike the standard model formula methods in R, a recipe **does not** automatically create dummy variables. This is for two reasons. First, not all models require [numeric predictors](https://bookdown.org/max/FES/categorical-trees.html), so dummy variables may not always be necessary. Second, recipes can be used to prepare the data for other, non-model purposes that prefer factors (such as a table or plot). Instead, `step_dummy()` can be used for this purpose: 


```r
flights_rec <- 
  recipe(arr_delay ~ ., data = train_data) %>% 
  update_role(flight, time_hour, new_role = "ID") %>% 
  step_date(date, features = c("dow", "month")) %>% 
  step_holiday(date, holidays = timeDate::listHolidays("US")) %>% 
  step_rm(date) %>% 
  step_dummy(all_nominal(), -all_outcomes())
```

Here, we did something different: we used [dplyr selectors](https://tidymodels.github.io/recipes/reference/selections.html) to apply this recipe step to several variables, based on type. Nominal means that the variable is either a factor or a character. The second selector removes the outcome variable(s) from this recipe step, so effectively the last recipe step above translates to:

> Create dummy variables for all of the factor or character columns _unless_ they are outcomes. 

At this stage in the recipe, this step selects columns `origin`, `dest`, and `carrier`. It also includes `date_dow` and `date_month`, two variables created by the earlier `step_date()`. 

These selectors mean that you don't always have to apply recipe steps by naming individual variables. Since a recipe knows the _role_ of each column, they can also be selected using this information. 

We need one final step to add to our recipe. Since `carrier` and `dest` have some infrequently occurring values, it is possible that dummy variables might be created for values that don't exist in the training set. For example, there is one destination that is only in the test set: 


```r
test_data %>% 
  distinct(dest) %>% 
  anti_join(train_data)
#> Joining, by = "dest"
#> # A tibble: 1 x 1
#>   dest 
#>   <fct>
#> 1 LEX
```

When the recipe is applied to the training set, a column is made for LEX but it will contain all zeros. This is a "zero-variance predictor" that has no information within the column. While some R functions will not produce an error for such predictors, it usually causes warnings and other issues. `step_zv()` will remove columns from the data when the training set data have a single value, so it is added to the recipe: 


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

Let's use straightforward logistic regression to model the flight data. As we saw in the [previous article](/start/models/), we can build a model specification using the parsnip package: 


```r
lr_mod <- 
  logistic_reg() %>% 
  set_engine("glm")
```


We need to use our recipe across several steps in our modeling process. We will: 

1. **Process the recipe using the training set**: This involves any estimation or calculations on these data. For our recipe, the training set would be used to determine what dummy variable columns should be created and which zero-variance predictors are slated for removal. 
 
1. **Apply the recipe to the training set**: We create the final predictor set on the training set. 
 
1. **Apply the recipe to the test set**: We create the final predictor set on the test set. Nothing is recomputed; the dummy variable and zero-variance results from the training set are applied to the test set. 
 
There are a few methods for doing this. One straightforward and simple approach is to use a _model workflow_ which pairs a model and recipe together. Different recipes are often needed for different models, so when a model and recipe are bundled, you can more easily keep track of them. We'll use the [workflows package](https://tidymodels.github.io/workflows/) from tidymodels to bundle our parsnip model (`lr_mod`) with our recipe (`flights_rec`).


```r
flights_wflow <- 
  workflow() %>% 
  add_model(lr_mod) %>% 
  add_recipe(flights_rec)
flights_wflow
#> ══ Workflow ═══════════════════════════════════════════════════════════
#> Preprocessor: Recipe
#> Model: logistic_reg()
#> 
#> ── Preprocessor ───────────────────────────────────────────────────────
#> 5 Recipe Steps
#> 
#> ● step_date()
#> ● step_holiday()
#> ● step_rm()
#> ● step_dummy()
#> ● step_zv()
#> 
#> ── Model ──────────────────────────────────────────────────────────────
#> Logistic Regression Model Specification (classification)
#> 
#> Computational engine: glm
```

Now, there is a single function that can be used to prepare the recipe and create the model from the resulting predictors: 


```r
flights_fit <- fit(flights_wflow, data = train_data)
```
 
This object has the finalized recipe and fitted model objects inside. You may want to extract the model or recipe objects from the workflow. To do this, you can use the helper functions `pull_workflow_fit()` and `pull_workflow_recipe()`. For example, here we pull the fitted model object then use the `broom::tidy()` function to get a tidy tibble of model coefficients: 


```r
flights_fit %>% 
  pull_workflow_fit() %>% 
  tidy()
#> # A tibble: 157 x 5
#>    term                         estimate std.error statistic  p.value
#>    <chr>                           <dbl>     <dbl>     <dbl>    <dbl>
#>  1 (Intercept)                   3.91    2.73           1.43 1.51e- 1
#>  2 dep_time                     -0.00167 0.0000141   -118.   0.      
#>  3 air_time                     -0.0439  0.000561     -78.4  0.      
#>  4 distance                      0.00686 0.00150        4.57 4.84e- 6
#>  5 date_USChristmasDay           1.12    0.173          6.49 8.45e-11
#>  6 date_USColumbusDay            0.474   0.159          2.99 2.81e- 3
#>  7 date_USCPulaskisBirthday      0.864   0.139          6.21 5.47e-10
#>  8 date_USDecorationMemorialDay  0.279   0.110          2.53 1.15e- 2
#>  9 date_USElectionDay            0.696   0.169          4.12 3.82e- 5
#> 10 date_USGoodFriday             1.28    0.166          7.71 1.27e-14
#> # … with 147 more rows
```

# Evaluate model predictions

How will we evaluate our model performance? Let's use the area under the [ROC curve](https://bookdown.org/max/FES/measuring-performance.html#class-metrics) as our metric, computed using the yardstick package functions called `roc_curve()` and `roc_auc()`. 

Just as there was a single call to `fit()` for fitting our workflow, there is also a single call to `predict()` for getting predictions on new data. The `predict()` method applies the recipe to the new data, then passes them to the fitted model. To generate a ROC curve, we need the predicted class probabilities for `late` and `on_time`. 


```r
flights_pred <- 
  predict(flights_fit, test_data, type = "prob") %>% 
  bind_cols(test_data %>% select(arr_delay, time_hour, flight)) 

# The data look like: 
flights_pred %>% slice(1:5)
#> # A tibble: 5 x 5
#>   .pred_late .pred_on_time arr_delay time_hour           flight
#>        <dbl>         <dbl> <fct>     <dttm>               <int>
#> 1     0.0565         0.944 on_time   2013-01-01 05:00:00   1714
#> 2     0.0264         0.974 on_time   2013-01-01 06:00:00     79
#> 3     0.0481         0.952 on_time   2013-01-01 06:00:00    301
#> 4     0.0325         0.967 on_time   2013-01-01 06:00:00     49
#> 5     0.0711         0.929 on_time   2013-01-01 06:00:00   1187
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
flights_pred %>% 
  roc_auc(truth = arr_delay, .pred_late)
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
#>  os       macOS Catalina 10.15.3      
#>  system   x86_64, darwin15.6.0        
#>  ui       X11                         
#>  language (EN)                        
#>  collate  en_US.UTF-8                 
#>  ctype    en_US.UTF-8                 
#>  tz       America/Los_Angeles         
#>  date     2020-04-11                  
#> 
#> ─ Packages ───────────────────────────────────────────────────────────────────
#>  package      * version    date       lib source                               
#>  broom        * 0.5.5      2020-02-29 [1] CRAN (R 3.6.0)                       
#>  dials        * 0.0.4      2019-12-02 [1] CRAN (R 3.6.0)                       
#>  dplyr        * 0.8.5      2020-03-07 [1] CRAN (R 3.6.0)                       
#>  ggplot2      * 3.3.0.9000 2020-02-21 [1] Github (tidyverse/ggplot2@b434351)   
#>  infer        * 0.5.1      2019-11-19 [1] CRAN (R 3.6.0)                       
#>  nycflights13 * 1.0.1      2019-09-16 [1] CRAN (R 3.6.0)                       
#>  parsnip      * 0.0.5      2020-01-07 [1] CRAN (R 3.6.0)                       
#>  purrr        * 0.3.3      2019-10-18 [1] CRAN (R 3.6.0)                       
#>  recipes      * 0.1.9      2020-01-14 [1] Github (tidymodels/recipes@5e7c702)  
#>  rlang          0.4.5      2020-03-01 [1] CRAN (R 3.6.0)                       
#>  rsample      * 0.0.5.9000 2020-03-20 [1] Github (tidymodels/rsample@4fdbd6c)  
#>  skimr        * 2.0.2      2019-11-26 [1] CRAN (R 3.6.0)                       
#>  tibble       * 2.1.3      2019-06-06 [1] CRAN (R 3.6.0)                       
#>  tidymodels   * 0.1.0      2020-02-16 [1] CRAN (R 3.6.0)                       
#>  tune         * 0.0.1.9000 2020-03-17 [1] Github (tidymodels/tune@93f7b2e)     
#>  workflows    * 0.1.0.9000 2020-01-14 [1] Github (tidymodels/workflows@c89bc0c)
#>  yardstick    * 0.0.5      2020-01-23 [1] CRAN (R 3.6.0)                       
#> 
#> [1] /Library/Frameworks/R.framework/Versions/3.6/Resources/library
```
