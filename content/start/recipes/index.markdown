---
title: "Preprocess your data with recipes"
weight: 2
tags: [recipes, parsnip, workflows, yardstick, broom]
categories: [pre-processing]
description: | 
  Prepare data for modeling with modular preprocessing steps.
---
<script src="{{< blogdown/postref >}}index_files/kePrint/kePrint.js"></script>
<link href="{{< blogdown/postref >}}index_files/lightable/lightable.css" rel="stylesheet" />





## Introduction {#intro}

In our [*Build a Model*](/start/models/) article, we learned how to specify and train models with different engines using the [parsnip package](https://parsnip.tidymodels.org/). In this article, we'll explore another tidymodels package, [recipes](https://recipes.tidymodels.org/), which is designed to help you preprocess your data *before* training your model. Recipes are built as a series of preprocessing steps, such as: 

+ converting qualitative predictors to indicator variables (also known as dummy variables),
 
+ transforming data to be on a different scale (e.g., taking the logarithm of a variable), 
 
+ transforming whole groups of predictors together,

+ extracting key features from raw variables (e.g., getting the day of the week out of a date variable),
 
and so on. If you are familiar with R's formula interface, a lot of this might sound familiar and like what a formula already does. Recipes can be used to do many of the same things, but they have a much wider range of possibilities. This article shows how to use recipes for modeling. 

To use code in this article,  you will need to install the following packages: nycflights13, skimr, and tidymodels.


```r
library(tidymodels)      # for the recipes package, along with the rest of tidymodels

# Helper packages
library(nycflights13)    # for flight data
library(skimr)           # for variable summaries
```

{{< test-drive url="https://rstudio.cloud/project/2674862" >}}

## The New York City flight data {#data}



Let's use the [nycflights13 data](https://github.com/hadley/nycflights13) to predict whether a plane arrives more than 30 minutes late. This data set contains information on 325,819 flights departing near New York City in 2013. Let's start by loading the data and making a few changes to the variables:


```r
set.seed(123)

flight_data <- 
  flights %>% 
  mutate(
    # Convert the arrival delay to a factor
    arr_delay = ifelse(arr_delay >= 30, "late", "on_time"),
    arr_delay = factor(arr_delay),
    # We will use the date (not date-time) in the recipe below
    date = lubridate::as_date(time_hour)
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
#> # A tibble: 2 × 3
#>   arr_delay      n  prop
#>   <fct>      <int> <dbl>
#> 1 late       52540 0.161
#> 2 on_time   273279 0.839
```


Before we start building up our recipe, let's take a quick look at a few specific variables that will be important for both preprocessing and modeling.

First, notice that the variable we created called `arr_delay` is a factor variable; it is important that our outcome variable for training a logistic regression model is a factor.


```r
glimpse(flight_data)
#> Rows: 325,819
#> Columns: 10
#> $ dep_time  <int> 517, 533, 542, 544, 554, 554, 555, 557, 557, 558, 558, 558, …
#> $ flight    <int> 1545, 1714, 1141, 725, 461, 1696, 507, 5708, 79, 301, 49, 71…
#> $ origin    <fct> EWR, LGA, JFK, JFK, LGA, EWR, EWR, LGA, JFK, LGA, JFK, JFK, …
#> $ dest      <fct> IAH, IAH, MIA, BQN, ATL, ORD, FLL, IAD, MCO, ORD, PBI, TPA, …
#> $ air_time  <dbl> 227, 227, 160, 183, 116, 150, 158, 53, 140, 138, 149, 158, 3…
#> $ distance  <dbl> 1400, 1416, 1089, 1576, 762, 719, 1065, 229, 944, 733, 1028,…
#> $ carrier   <fct> UA, UA, AA, B6, DL, UA, B6, EV, B6, AA, B6, B6, UA, UA, AA, …
#> $ date      <date> 2013-01-01, 2013-01-01, 2013-01-01, 2013-01-01, 2013-01-01,…
#> $ arr_delay <fct> on_time, on_time, late, on_time, on_time, on_time, on_time, …
#> $ time_hour <dttm> 2013-01-01 05:00:00, 2013-01-01 05:00:00, 2013-01-01 05:00:…
```

Second, there are two variables that we don't want to use as predictors in our model, but that we would like to retain as identification variables that can be used to troubleshoot poorly predicted data points. These are `flight`, a numeric value, and `time_hour`, a date-time value.

Third, there are 104 flight destinations contained in `dest` and 16 distinct `carrier`s. 


```r
flight_data %>% 
  skimr::skim(dest, carrier) 
```


<table style='width: auto;'
      class='table table-condensed'>
<caption>Table 1: Data summary</caption>
<tbody>
  <tr>
   <td style="text-align:left;"> Name </td>
   <td style="text-align:left;"> Piped data </td>
  </tr>
  <tr>
   <td style="text-align:left;"> Number of rows </td>
   <td style="text-align:left;"> 325819 </td>
  </tr>
  <tr>
   <td style="text-align:left;"> Number of columns </td>
   <td style="text-align:left;"> 10 </td>
  </tr>
  <tr>
   <td style="text-align:left;"> _______________________ </td>
   <td style="text-align:left;">  </td>
  </tr>
  <tr>
   <td style="text-align:left;"> Column type frequency: </td>
   <td style="text-align:left;">  </td>
  </tr>
  <tr>
   <td style="text-align:left;"> factor </td>
   <td style="text-align:left;"> 2 </td>
  </tr>
  <tr>
   <td style="text-align:left;"> ________________________ </td>
   <td style="text-align:left;">  </td>
  </tr>
  <tr>
   <td style="text-align:left;"> Group variables </td>
   <td style="text-align:left;"> None </td>
  </tr>
</tbody>
</table>


**Variable type: factor**

<table>
 <thead>
  <tr>
   <th style="text-align:left;"> skim_variable </th>
   <th style="text-align:right;"> n_missing </th>
   <th style="text-align:right;"> complete_rate </th>
   <th style="text-align:left;"> ordered </th>
   <th style="text-align:right;"> n_unique </th>
   <th style="text-align:left;"> top_counts </th>
  </tr>
 </thead>
<tbody>
  <tr>
   <td style="text-align:left;"> dest </td>
   <td style="text-align:right;"> 0 </td>
   <td style="text-align:right;"> 1 </td>
   <td style="text-align:left;"> FALSE </td>
   <td style="text-align:right;"> 104 </td>
   <td style="text-align:left;"> ATL: 16771, ORD: 16507, LAX: 15942, BOS: 14948 </td>
  </tr>
  <tr>
   <td style="text-align:left;"> carrier </td>
   <td style="text-align:right;"> 0 </td>
   <td style="text-align:right;"> 1 </td>
   <td style="text-align:left;"> FALSE </td>
   <td style="text-align:right;"> 16 </td>
   <td style="text-align:left;"> UA: 57489, B6: 53715, EV: 50868, DL: 47465 </td>
  </tr>
</tbody>
</table>


Because we'll be using a simple logistic regression model, the variables `dest` and `carrier` will be converted to [dummy variables](https://bookdown.org/max/FES/creating-dummy-variables-for-unordered-categories.html). However, some of these values do not occur very frequently and this could complicate our analysis. We'll discuss specific steps later in this article that we can add to our recipe to address this issue before modeling. 

## Data splitting {#data-split}

To get started, let's split this single dataset into two: a _training_ set and a _testing_ set. We'll keep most of the rows in the original dataset (subset chosen randomly) in the _training_ set. The training data will be used to *fit* the model, and the _testing_ set will be used to measure model performance. 

To do this, we can use the [rsample](https://rsample.tidymodels.org/) package to create an object that contains the information on _how_ to split the data, and then two more rsample functions to create data frames for the training and testing sets: 


```r
# Fix the random numbers by setting the seed 
# This enables the analysis to be reproducible when random numbers are used 
set.seed(222)
# Put 3/4 of the data into the training set 
data_split <- initial_split(flight_data, prop = 3/4)

# Create data frames for the two sets:
train_data <- training(data_split)
test_data  <- testing(data_split)
```

 
## Create recipe and roles {#recipe}

To get started, let's create a recipe for a simple logistic regression model. Before training the model, we can use a recipe to create a few new predictors and conduct some preprocessing required by the model. 

Let's initiate a new recipe: 


```r
flights_rec <- 
  recipe(arr_delay ~ ., data = train_data) 
```

The [`recipe()` function](https://recipes.tidymodels.org/reference/recipe.html) as we used it here has two arguments:

+ A **formula**. Any variable on the left-hand side of the tilde (`~`) is considered the model outcome (here, `arr_delay`). On the right-hand side of the tilde are the predictors. Variables may be listed by name, or you can use the dot (`.`) to indicate all other variables as predictors.

+ The **data**. A recipe is associated with the data set used to create the model. This will typically be the _training_ set, so `data = train_data` here. Naming a data set doesn't actually change the data itself; it is only used to catalog the names of the variables and their types, like factors, integers, dates, etc.

Now we can add [roles](https://recipes.tidymodels.org/reference/roles.html) to this recipe. We can use the [`update_role()` function](https://recipes.tidymodels.org/reference/roles.html) to let recipes know that `flight` and `time_hour` are variables with a custom role that we called `"ID"` (a role can have any character value). Whereas our formula included all variables in the training set other than `arr_delay` as predictors, this tells the recipe to keep these two variables but not use them as either outcomes or predictors.


```r
flights_rec <- 
  recipe(arr_delay ~ ., data = train_data) %>% 
  update_role(flight, time_hour, new_role = "ID") 
```

This step of adding roles to a recipe is optional; the purpose of using it here is that those two variables can be retained in the data but not included in the model. This can be convenient when, after the model is fit, we want to investigate some poorly predicted value. These ID columns will be available and can be used to try to understand what went wrong.

To get the current set of variables and roles, use the `summary()` function: 


```r
summary(flights_rec)
#> # A tibble: 10 × 4
#>    variable  type      role      source  
#>    <chr>     <list>    <chr>     <chr>   
#>  1 dep_time  <chr [2]> predictor original
#>  2 flight    <chr [2]> ID        original
#>  3 origin    <chr [3]> predictor original
#>  4 dest      <chr [3]> predictor original
#>  5 air_time  <chr [2]> predictor original
#>  6 distance  <chr [2]> predictor original
#>  7 carrier   <chr [3]> predictor original
#>  8 date      <chr [1]> predictor original
#>  9 time_hour <chr [1]> ID        original
#> 10 arr_delay <chr [3]> outcome   original
```



## Create features {#features}

Now we can start adding steps onto our recipe using the pipe operator. Perhaps it is reasonable for the date of the flight to have an effect on the likelihood of a late arrival. A little bit of **feature engineering** might go a long way to improving our model. How should the date be encoded into the model? The `date` column has an R `date` object so including that column "as is" will mean that the model will convert it to a numeric format equal to the number of days after a reference date: 


```r
flight_data %>% 
  distinct(date) %>% 
  mutate(numeric_date = as.numeric(date)) 
#> # A tibble: 364 × 2
#>   date       numeric_date
#>   <date>            <dbl>
#> 1 2013-01-01        15706
#> 2 2013-01-02        15707
#> 3 2013-01-03        15708
#> 4 2013-01-04        15709
#> 5 2013-01-05        15710
#> # … with 359 more rows
```

It's possible that the numeric date variable is a good option for modeling; perhaps the model would benefit from a linear trend between the log-odds of a late arrival and the numeric date variable. However, it might be better to add model terms _derived_ from the date that have a better potential to be important to the model. For example, we could derive the following meaningful features from the single `date` variable: 

* the day of the week,
 
* the month, and
 
* whether or not the date corresponds to a holiday. 
 
Let's do all three of these by adding steps to our recipe:



```r
flights_rec <- 
  recipe(arr_delay ~ ., data = train_data) %>% 
  update_role(flight, time_hour, new_role = "ID") %>% 
  step_date(date, features = c("dow", "month")) %>%               
  step_holiday(date, 
               holidays = timeDate::listHolidays("US"), 
               keep_original_cols = FALSE)
```

What do each of these steps do?

* With [`step_date()`](https://recipes.tidymodels.org/reference/step_date.html), we created two new factor columns with the appropriate day of the week and the month. 

* With [`step_holiday()`](https://recipes.tidymodels.org/reference/step_holiday.html), we created a binary variable indicating whether the current date is a holiday or not. The argument value of `timeDate::listHolidays("US")` uses the [timeDate package](https://cran.r-project.org/web/packages/timeDate/index.html) to list the 17 standard US holidays.

* With `keep_original_cols = FALSE`, we remove the original `date` variable since we no longer want it in the model. Many recipe steps that create new variables have this argument.

Next, we'll turn our attention to the variable types of our predictors. Because we plan to train a logistic regression model, we know that predictors will ultimately need to be numeric, as opposed to nominal data like strings and factor variables. In other words, there may be a difference in how we store our data (in factors inside a data frame), and how the underlying equations require them (a purely numeric matrix).

For factors like `dest` and `origin`, [standard practice](https://bookdown.org/max/FES/creating-dummy-variables-for-unordered-categories.html) is to convert them into _dummy_ or _indicator_ variables to make them numeric. These are binary values for each level of the factor. For example, our `origin` variable has values of `"EWR"`, `"JFK"`, and `"LGA"`. The standard dummy variable encoding, shown below, will create _two_ numeric columns of the data that are 1 when the originating airport is `"JFK"` or `"LGA"` and zero otherwise, respectively. 



<table class="table" style="width: auto !important; margin-left: auto; margin-right: auto;">
 <thead>
  <tr>
   <th style="text-align:left;"> origin </th>
   <th style="text-align:right;"> origin_JFK </th>
   <th style="text-align:right;"> origin_LGA </th>
  </tr>
 </thead>
<tbody>
  <tr>
   <td style="text-align:left;"> JFK </td>
   <td style="text-align:right;"> 1 </td>
   <td style="text-align:right;"> 0 </td>
  </tr>
  <tr>
   <td style="text-align:left;"> EWR </td>
   <td style="text-align:right;"> 0 </td>
   <td style="text-align:right;"> 0 </td>
  </tr>
  <tr>
   <td style="text-align:left;"> LGA </td>
   <td style="text-align:right;"> 0 </td>
   <td style="text-align:right;"> 1 </td>
  </tr>
</tbody>
</table>


But, unlike the standard model formula methods in R, a recipe **does not** automatically create these dummy variables for you; you'll need to tell your recipe to add this step. This is for two reasons. First, many models do not require [numeric predictors](https://bookdown.org/max/FES/categorical-trees.html), so dummy variables may not always be preferred. Second, recipes can also be used for purposes outside of modeling, where non-dummy versions of the variables may work better. For example, you may want to make a table or a plot with a variable as a single factor. For those reasons, you need to explicitly tell recipes to create dummy variables using `step_dummy()`: 


```r
flights_rec <- 
  recipe(arr_delay ~ ., data = train_data) %>% 
  update_role(flight, time_hour, new_role = "ID") %>% 
  step_date(date, features = c("dow", "month")) %>%               
  step_holiday(date, 
               holidays = timeDate::listHolidays("US"), 
               keep_original_cols = FALSE) %>% 
  step_dummy(all_nominal_predictors())
```

Here, we did something different than before: instead of applying a step to an individual variable, we used [selectors](https://recipes.tidymodels.org/reference/selections.html) to apply this recipe step to several variables at once, `all_nominal_predictors()`. The [selector functions](https://recipes.tidymodels.org/reference/selections.html) can be combined to select intersections of variables.

At this stage in the recipe, this step selects the `origin`, `dest`, and `carrier` variables. It also includes two new variables, `date_dow` and `date_month`, that were created by the earlier `step_date()`. 

More generally, the recipe selectors mean that you don't always have to apply steps to individual variables one at a time. Since a recipe knows the _variable type_ and _role_ of each column, they can also be selected (or dropped) using this information. 

We need one final step to add to our recipe. Since `carrier` and `dest` have some infrequently occurring factor values, it is possible that dummy variables might be created for values that don't exist in the training set. For example, there is one destination that is only in the test set: 


```r
test_data %>% 
  distinct(dest) %>% 
  anti_join(train_data)
#> Joining, by = "dest"
#> # A tibble: 1 × 1
#>   dest 
#>   <fct>
#> 1 LEX
```

When the recipe is applied to the training set, a column is made for LEX because the factor levels come from `flight_data` (not the training set), but this column will contain all zeros. This is a "zero-variance predictor" that has no information within the column. While some R functions will not produce an error for such predictors, it usually causes warnings and other issues. `step_zv()` will remove columns from the data when the training set data have a single value, so it is added to the recipe *after* `step_dummy()`: 


```r
flights_rec <- 
  recipe(arr_delay ~ ., data = train_data) %>% 
  update_role(flight, time_hour, new_role = "ID") %>% 
  step_date(date, features = c("dow", "month")) %>%               
  step_holiday(date, 
               holidays = timeDate::listHolidays("US"), 
               keep_original_cols = FALSE) %>% 
  step_dummy(all_nominal_predictors()) %>% 
  step_zv(all_predictors())
```


Now we've created a _specification_ of what should be done with the data. How do we use the recipe we made? 

## Fit a model with a recipe {#fit-workflow}

Let's use logistic regression to model the flight data. As we saw in [*Build a Model*](/start/models/), we start by [building a model specification](/start/models/#build-model) using the parsnip package: 


```r
lr_mod <- 
  logistic_reg() %>% 
  set_engine("glm")
```


We will want to use our recipe across several steps as we train and test our model. We will: 

1. **Process the recipe using the training set**: This involves any estimation or calculations based on the training set. For our recipe, the training set will be used to determine which predictors should be converted to dummy variables and which predictors will have zero-variance in the training set, and should be slated for removal. 
 
1. **Apply the recipe to the training set**: We create the final predictor set on the training set. 
 
1. **Apply the recipe to the test set**: We create the final predictor set on the test set. Nothing is recomputed and no information from the test set is used here; the dummy variable and zero-variance results from the training set are applied to the test set. 
 
To simplify this process, we can use a _model workflow_, which pairs a model and recipe together. This is a straightforward approach because different recipes are often needed for different models, so when a model and recipe are bundled, it becomes easier to train and test _workflows_. We'll use the [workflows package](https://workflows.tidymodels.org/) from tidymodels to bundle our parsnip model (`lr_mod`) with our recipe (`flights_rec`).


```r
flights_wflow <- 
  workflow() %>% 
  add_model(lr_mod) %>% 
  add_recipe(flights_rec)

flights_wflow
#> ══ Workflow ══════════════════════════════════════════════════════════
#> Preprocessor: Recipe
#> Model: logistic_reg()
#> 
#> ── Preprocessor ──────────────────────────────────────────────────────
#> 4 Recipe Steps
#> 
#> • step_date()
#> • step_holiday()
#> • step_dummy()
#> • step_zv()
#> 
#> ── Model ─────────────────────────────────────────────────────────────
#> Logistic Regression Model Specification (classification)
#> 
#> Computational engine: glm
```

Now, there is a single function that can be used to prepare the recipe and train the model from the resulting predictors: 


```r
flights_fit <- 
  flights_wflow %>% 
  fit(data = train_data)
```
 
This object has the finalized recipe and fitted model objects inside. You may want to extract the model or recipe objects from the workflow. To do this, you can use the helper functions `extract_fit_parsnip()` and `extract_recipe()`. For example, here we pull the fitted model object then use the `broom::tidy()` function to get a tidy tibble of model coefficients: 


```r
flights_fit %>% 
  extract_fit_parsnip() %>% 
  tidy()
#> # A tibble: 157 × 5
#>   term                estimate std.error statistic  p.value
#>   <chr>                  <dbl>     <dbl>     <dbl>    <dbl>
#> 1 (Intercept)          7.28    2.73           2.67 7.64e- 3
#> 2 dep_time            -0.00166 0.0000141   -118.   0       
#> 3 air_time            -0.0440  0.000563     -78.2  0       
#> 4 distance             0.00507 0.00150        3.38 7.32e- 4
#> 5 date_USChristmasDay  1.33    0.177          7.49 6.93e-14
#> # … with 152 more rows
```

## Use a trained workflow to predict {#predict-workflow}

Our goal was to predict whether a plane arrives more than 30 minutes late. We have just:

1. Built the model (`lr_mod`),

1. Created a preprocessing recipe (`flights_rec`),

1. Bundled the model and recipe (`flights_wflow`), and 

1. Trained our workflow using a single call to `fit()`. 

The next step is to use the trained workflow (`flights_fit`) to predict with the unseen test data, which we will do with a single call to `predict()`. The `predict()` method applies the recipe to the new data, then passes them to the fitted model. 


```r
predict(flights_fit, test_data)
#> # A tibble: 81,455 × 1
#>   .pred_class
#>   <fct>      
#> 1 on_time    
#> 2 on_time    
#> 3 on_time    
#> 4 on_time    
#> 5 on_time    
#> # … with 81,450 more rows
```

Because our outcome variable here is a factor, the output from `predict()` returns the predicted class: `late` versus `on_time`. But, let's say we want the predicted class probabilities for each flight instead. To return those, we can specify `type = "prob"` when we use `predict()` or use `augment()` with the model plus test data to save them together:


```r
flights_aug <- 
  augment(flights_fit, test_data)

# The data look like: 
flights_aug %>%
  select(arr_delay, time_hour, flight, .pred_class, .pred_on_time)
#> # A tibble: 81,455 × 5
#>   arr_delay time_hour           flight .pred_class .pred_on_time
#>   <fct>     <dttm>               <int> <fct>               <dbl>
#> 1 on_time   2013-01-01 05:00:00   1545 on_time             0.945
#> 2 on_time   2013-01-01 05:00:00   1714 on_time             0.949
#> 3 on_time   2013-01-01 06:00:00    507 on_time             0.964
#> 4 on_time   2013-01-01 06:00:00   5708 on_time             0.961
#> 5 on_time   2013-01-01 06:00:00     71 on_time             0.962
#> # … with 81,450 more rows
```

Now that we have a tibble with our predicted class probabilities, how will we evaluate the performance of our workflow? We can see from these first few rows that our model predicted these 5 on time flights correctly because the values of `.pred_on_time` are *p* > .50. But we also know that we have 81,455 rows total to predict. We would like to calculate a metric that tells how well our model predicted late arrivals, compared to the true status of our outcome variable, `arr_delay`.

Let's use the area under the [ROC curve](https://bookdown.org/max/FES/measuring-performance.html#class-metrics) as our metric, computed using `roc_curve()` and `roc_auc()` from the [yardstick package](https://yardstick.tidymodels.org/). 

To generate a ROC curve, we need the predicted class probabilities for `late` and `on_time`, which we just calculated in the code chunk above. We can create the ROC curve with these values, using `roc_curve()` and then piping to the `autoplot()` method: 


```r
flights_aug %>% 
  roc_curve(truth = arr_delay, .pred_late) %>% 
  autoplot()
```

<img src="figs/roc-plot-1.svg" width="672" />

Similarly, `roc_auc()` estimates the area under the curve: 


```r
flights_aug %>% 
  roc_auc(truth = arr_delay, .pred_late)
#> # A tibble: 1 × 3
#>   .metric .estimator .estimate
#>   <chr>   <chr>          <dbl>
#> 1 roc_auc binary         0.764
```

Not too bad! We leave it to the reader to test out this workflow [*without*](https://workflows.tidymodels.org/reference/add_formula.html) this recipe. You can use `workflows::add_formula(arr_delay ~ .)` instead of `add_recipe()` (remember to remove the identification variables first!), and see whether our recipe improved our model's ability to predict late arrivals.




## Session information {#session-info}


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
#>  date     2022-12-02
#>  pandoc   2.19.2 @ /Applications/RStudio.app/Contents/MacOS/quarto/bin/tools/ (via rmarkdown)
#> 
#> ─ Packages ─────────────────────────────────────────────────────────
#>  package      * version date (UTC) lib source
#>  broom        * 1.0.1   2022-08-29 [1] CRAN (R 4.2.0)
#>  dials        * 1.1.0   2022-11-04 [1] CRAN (R 4.2.0)
#>  dplyr        * 1.0.10  2022-09-01 [1] CRAN (R 4.2.0)
#>  ggplot2      * 3.4.0   2022-11-04 [1] CRAN (R 4.2.0)
#>  infer        * 1.0.4   2022-12-02 [1] CRAN (R 4.2.1)
#>  nycflights13 * 1.0.2   2021-04-12 [1] CRAN (R 4.2.0)
#>  parsnip      * 1.0.3   2022-11-11 [1] CRAN (R 4.2.0)
#>  purrr        * 0.3.5   2022-10-06 [1] CRAN (R 4.2.0)
#>  recipes      * 1.0.3   2022-11-09 [1] CRAN (R 4.2.0)
#>  rlang          1.0.6   2022-09-24 [1] CRAN (R 4.2.0)
#>  rsample      * 1.1.0   2022-08-08 [1] CRAN (R 4.2.0)
#>  skimr        * 2.1.4   2022-04-15 [1] CRAN (R 4.2.0)
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
