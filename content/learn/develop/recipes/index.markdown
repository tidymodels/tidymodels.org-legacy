---
title: "Create your own recipe step function"
tags: [recipes]
categories: []
type: learn-subsection
weight: 1
description: | 
  Write a new recipe step for data preprocessing.
---





## Introduction

To use the code in this article, you will need to install the following packages: modeldata and tidymodels.

There are many existing recipe steps in packages like recipes, themis, textrecipes, and others. A full list of steps in CRAN packages [can be found here](/find/recipes/). However, you might need to define your own preprocessing operations; this article describes how to do that. If you are looking for good examples of existing steps, we suggest looking at the code for [centering](https://github.com/tidymodels/recipes/blob/master/R/center.R) or [PCA](https://github.com/tidymodels/recipes/blob/master/R/pca.R) to start. 

For check operations (e.g. `check_class()`), the process is very similar. Notes on this are available at the end of this article. 

The general process to follow is to:

1. Define a step constructor function.

2. Create the minimal S3 methods for `prep()`, `bake()`, and `print()`.  

3. Optionally add some extra methods to work with other tidymodels packages, such as `tunable()` and `tidy()`. 

As an example, we will create a step for converting data into percentiles. 

## A new step definition

Let's create a step that replaces the value of a variable with its percentile from the training set. The example data we'll use is from the modeldata package:


```r
library(modeldata)
data(biomass)
str(biomass)
#> 'data.frame':	536 obs. of  8 variables:
#>  $ sample  : chr  "Akhrot Shell" "Alabama Oak Wood Waste" "Alder" "Alfalfa" ...
#>  $ dataset : chr  "Training" "Training" "Training" "Training" ...
#>  $ carbon  : num  49.8 49.5 47.8 45.1 46.8 ...
#>  $ hydrogen: num  5.64 5.7 5.8 4.97 5.4 5.75 5.99 5.7 5.5 5.9 ...
#>  $ oxygen  : num  42.9 41.3 46.2 35.6 40.7 ...
#>  $ nitrogen: num  0.41 0.2 0.11 3.3 1 2.04 2.68 1.7 0.8 1.2 ...
#>  $ sulfur  : num  0 0 0.02 0.16 0.02 0.1 0.2 0.2 0 0.1 ...
#>  $ HHV     : num  20 19.2 18.3 18.2 18.4 ...

biomass_tr <- biomass[biomass$dataset == "Training",]
biomass_te <- biomass[biomass$dataset == "Testing",]
```

To illustrate the transformation with the `carbon` variable, note the training set distribution of this variable with a vertical line below for the first value of the test set. 


```r
library(ggplot2)
theme_set(theme_bw())
ggplot(biomass_tr, aes(x = carbon)) + 
  geom_histogram(binwidth = 5, col = "blue", fill = "blue", alpha = .5) + 
  geom_vline(xintercept = biomass_te$carbon[1], lty = 2)
```

<img src="figs/carbon_dist-1.svg" width="100%" />

Based on the training set, 42.1% of the data are less than a value of 46.35. There are some applications where it might be advantageous to represent the predictor values as percentiles rather than their original values. 

Our new step will do this computation for any numeric variables of interest. We will call this new recipe step `step_percentile()`. The code below is designed for illustration and not speed or best practices. We've left out a lot of error trapping that we would want in a real implementation.  

## Create the function

To start, there is a _user-facing_ function. Let's call that `step_percentile()`. This is just a simple wrapper around a _constructor function_, which defines the rules for any step object that defines a percentile transformation. We'll call this constructor `step_percentile_new()`. 

The function `step_percentile()` takes the same arguments as your function and simply adds it to a new recipe. The `...` signifies the variable selectors that can be used^[Towards the end of 2020, recipes will make the change to move to the tidyverse's new selection style that _does not_ use `...` to capture the selectors. See the [tidyverse principles page](https://principles.tidyverse.org/dots-data.html) for a discussion.].


```r
step_percentile <- function(
  recipe, 
  ..., 
  role = NA, 
  trained = FALSE, 
  ref_dist = NULL,
  options = list(probs = (0:100)/100, names = TRUE),
  skip = FALSE,
  id = rand_id("percentile")
  ) {

  ## The variable selectors are not immediately evaluated by using
  ##  the `quos()` function in `rlang`. `ellipse_check()` captures 
  ##  the values and also checks to make sure that they are not empty.  
  terms <- ellipse_check(...) 

  add_step(
    recipe, 
    step_percentile_new(
      terms = terms, 
      trained = trained,
      role = role, 
      ref_dist = ref_dist,
      options = options,
      skip = skip,
      id = id
    )
  )
}
```

You should always keep the first four arguments (`recipe` though `trained`) the same as listed above. Some notes:

 * the `role` argument is used when you either 1) create new variables and want their role to be pre-set or 2) replace the existing variables with new values. The latter is what we will be doing and using `role = NA` will leave the existing role intact. 
 * `trained` is set by the package when the estimation step has been run. You should default your function definition's argument to `FALSE`. 
 * `skip` is a logical. Whenever a recipe is prepped, each step is trained and then baked. However, there are some steps that should not be applied when a call to `bake()` is used. For example, if a step is applied to the variables with roles of "outcomes", these data would not be available for new samples. 
 * `id` is a character string that can be used to identify steps in package code. `rand_id()` will create an ID that has the prefix and a random character sequence. 

In order to calculate the percentile, the training data for the relevant columns need to be saved. This data will be saved in the `ref_dist` object. `approx()` would be used when you want to save a grid of pre-computed percentiles from the training set and use these to estimate the percentile for a new data point. 

We will use the `stats::quantile()` to compute the grid. However, we might also want to have control over the granularity of this grid, so the `options` argument will be used to define how that calculations is done. We can use the ellipses (aka `...`) so that any options passed to `step_percentile()` that are not one of its arguments will then be passed to `stats::quantile()`. We recommend making a separate list object with the options and use these inside the function. 

It is also important to consider if there are any _main arguments_ to the step. For example, for spline-related steps such as `step_ns()`, users typically want to adjust the argument for the degrees of freedom in the spline (e.g. `splines::ns(x, df)`). Rather letting users add `df` to the `options` argument: 

* Allow the important arguments to be main arguments to the step function. 

* Follow the tidymodels [conventions for naming arguments](https://tidymodels.github.io/model-implementation-principles/standardized-argument-names.html). Whenever possible, avoid jargon and keep common argument names.  

There are benefits to following these principles (as shown below). 

## Initialize a new object

Now, the constructor function can be created.

The function cascade is: 

```
step_percentile() calls recipes::add_step()
└──> recipes::add_step() calls step_percentile_new()
    └──> step_percentile_new() calls recipes::step()
```

`step()` is a general constructor for recipes that mainly makes sure that the resulting step object is a list with an appropriate S3 class structure. Using `subclass = "percentile"` will set the class of new objects to `"step_percentile()"`. 


```r
step_percentile_new <- 
  function(terms, role, trained, ref_dist, options, skip, id) {
    step(
      subclass = "percentile", 
      terms = terms,
      role = role,
      trained = trained,
      ref_dist = ref_dist,
      options = options,
      skip = skip,
      id = id
    )
  }
```

This constructor function should have no default argument values. Defaults should be set in the user-facing step object. 

## Define the procedure

You will need to create a new `prep()` method for your step's class. To do this, three arguments that the method should have are:

```r
function(x, training, info = NULL)
```

where

 * `x` will be the `step_percentile` object,
 * `training` will be a _tibble_ that has the training set data, and
 * `info` will also be a tibble that has information on the current set of data available. This information is updated as each step is evaluated by its specific `prep()` method so it may not have the variables from the original data. The columns in this tibble are `variable` (the variable name), `type` (currently either "numeric" or "nominal"), `role` (defining the variable's role), and `source` (either "original" or "derived" depending on where it originated).

You can define other options as well. 

The first thing that you might want to do in the `prep()` function is to translate the specification listed in the `terms` argument to column names in the current data. There is an internal function called `terms_select()` that can be used to obtain this. 


```r
prep.step_percentile <- function(x, training, info = NULL, ...) {
  col_names <- terms_select(terms = x$terms, info = info) 
}
```

After this function call, it is a good idea to check that the selected columns have the appropriate type (e.g. numeric for this example). See `recipes::check_type()` to do this for basic types. 

Once we have this, we can save the approximation grid. For the grid, we will use a helper function that enables us to run `rlang::exec()` to splice in any extra arguments contained in the `options` list to the call to `quantile()`: 


```r
get_train_pctl <- function(x, args = NULL) {
  res <- rlang::exec("quantile", x = x, !!!args)
  # Remove duplicate percentile values
  res[!duplicated(res)]
}

# For example:
get_train_pctl(biomass_tr$carbon, list(probs = 0:1))
#>   0% 100% 
#> 14.6 97.2
get_train_pctl(biomass_tr$carbon)
#>   0%  25%  50%  75% 100% 
#> 14.6 44.7 47.1 49.7 97.2
```

Now, the `prep()` method can be created: 


```r
prep.step_percentile <- function(x, training, info = NULL, ...) {
  col_names <- terms_select(terms = x$terms, info = info) 
  ## You can add error trapping for non-numeric data here and so on. 
  
  ## We'll use the names later so
  if (x$options$names == FALSE) {
    rlang::abort("`names` should be set to TRUE")
  }
  
  if (!any(names(x$options) == "probs")) {
    x$options$probs <- (0:100)/100
  } else {
    x$options$probs <- sort(unique(x$options$probs))
  }
  
  # Compute percentile grid
  ref_dist <- purrr::map(training[, col_names],  get_train_pctl, args = x$options)

  ## Use the constructor function to return the updated object. 
  ## Note that `trained` is now set to TRUE
  
  step_percentile_new(
    terms = x$terms, 
    trained = TRUE,
    role = x$role, 
    ref_dist = ref_dist,
    options = x$options,
    skip = x$skip,
    id = x$id
  )
}
```

We suggest favoring `rlang::abort()` and `rlang::warn()` over `stop()` and `warning()`. The former can be used for better traceback results.


## Create the `bake` method

Remember that the `prep()` function does not _apply_ the step to the data; it only estimates any required values such as `ref_dist`. We will need to create a new method for our `step_percentile()` class. The minimum arguments for this are

```r
function(object, new_data, ...)
```

where `object` is the updated step function that has been through the corresponding `prep()` code and `new_data` is a tibble of data to be processed. 

Here is the code to convert the new data to percentiles. The input data (`x` below) comes in as a numeric vector and the output is a vector of approximate percentiles: 


```r
pctl_by_approx <- function(x, ref) {
  # In case duplicates were removed, get the percentiles from
  # the names of the reference object
  grid <- as.numeric(gsub("%$", "", names(ref))) 
  approx(x = ref, y = grid, xout = x)$y/100
}
```

These computations are done column-wise using `purrr::map2_dfc()` to modify the new data in-place:


```r
bake.step_percentile <- function(object, new_data, ...) {
  ## For illustration (and not speed), we will loop through the affected variables
  ## and do the computations
  vars <- names(object$ref_dist)
  
  new_data[, vars] <-
    purrr::map2_dfc(new_data[, vars], object$ref_dist, pctl_by_approx)
  
  ## Always convert to tibbles on the way out
  tibble::as_tibble(new_data)
}
```

## Run the example

Let's use the example data to make sure that it works: 


```r
rec_obj <- 
  recipe(HHV ~ ., data = biomass_tr) %>%
  step_percentile(ends_with("gen")) %>%
  prep(training = biomass_tr)

biomass_te %>% select(ends_with("gen")) %>% slice(1:2)
#>   hydrogen oxygen nitrogen
#> 1     5.67   47.2     0.30
#> 2     5.50   48.1     2.85
bake(rec_obj, biomass_te %>% slice(1:2), ends_with("gen"))
#> # A tibble: 2 x 3
#>   hydrogen oxygen nitrogen
#>      <dbl>  <dbl>    <dbl>
#> 1     0.45  0.903    0.21 
#> 2     0.38  0.922    0.928

# Checking to get approximate result: 
mean(biomass_tr$hydrogen <= biomass_te$hydrogen[1])
#> [1] 0.452
mean(biomass_tr$oxygen   <= biomass_te$oxygen[1])
#> [1] 0.901
```

The plot below shows how the original hydrogen percentiles line up with the estimated values:


```r
hydrogen_values <- 
  bake(rec_obj, biomass_te, hydrogen) %>% 
  bind_cols(biomass_te %>% select(original = hydrogen))

ggplot(biomass_tr, aes(x = hydrogen)) + 
  # Plot the empirical distribution function of the 
  # hydrogen training set values as a black line
  stat_ecdf() + 
  # Overlay the estimated percentiles for the new data: 
  geom_point(data = hydrogen_values, 
             aes(x = original, y = hydrogen), 
             col = "red", alpha = .5, cex = 2) + 
  labs(x = "New Hydrogen Values", y = "Percentile Based on Training Set")
```

<img src="figs/cdf_plot-1.svg" width="672" />

These line up very nicely! 

## Custom check operations 

The process here is exactly the same as steps; the internal functions have a similar naming convention: 

 * `add_check()` instead of `add_step()`
 * `check()` instead of `step()`, and so on. 
 
It is strongly recommended that:
 
 1. The operations start with `check_` (i.e. `check_range()` and `check_range_new()`)
 1. The check uses `rlang::abort(paste0(...))` when the conditions are not met
 1. The original data are returned (unaltered) by the check when the conditions are satisfied. 

## Other step methods

There are a few other S3 methods that can be created for your step function. They are not required unless you plan on using your step in the broader tidymodels package set. 

### A print method

Printing `rec_obj` is a bit ugly; since there is no print method for `step_percentile()` it prints it as a list of (potentially large) objects. The recipes package contains a helper function called `printer()` that should work for most cases. It requires the the names of the selected columns that are resolved after `prep()` has been run as well as the original terms specification. For the former, our step object is structured so that the list object `ref_dist` has the names of the selected variables: 


```r
print.step_percentile <-
  function(x, width = max(20, options()$width - 35), ...) {
    cat("Percentile transformation on ", sep = "")
    printer(
      # Names before prep (could be selectors)
      untr_obj = x$terms,
      # Names after prep:
      tr_obj = names(x$ref_dist),
      # Has it been prepped? 
      trained = x$trained,
      # An estimate of how many characters to print on a line: 
      width = width
    )
    invisible(x)
  }

# Results before `prep()`:
recipe(HHV ~ ., data = biomass_tr) %>%
  step_percentile(ends_with("gen"))
#> Data Recipe
#> 
#> Inputs:
#> 
#>       role #variables
#>    outcome          1
#>  predictor          7
#> 
#> Operations:
#> 
#> Percentile transformation on ends_with("gen")

# Results after `prep()`: 
rec_obj
#> Data Recipe
#> 
#> Inputs:
#> 
#>       role #variables
#>    outcome          1
#>  predictor          7
#> 
#> Training data contained 456 data points and no missing data.
#> 
#> Operations:
#> 
#> Percentile transformation on hydrogen, oxygen, nitrogen [trained]
```
 
### A tidy method

The `broom::tidy()` method is a means to return information about the step in a usable format. For our step, it would be helpful to know the reference values. 

When the recipe has been prepped, those data are in the list `ref_dist`. A small function can be used to reformat that data into a tibble. It is customary to return the main values as `value`:


```r
format_pctl <- function(x) {
  tibble::tibble(
    value = unname(x),
    percentile = as.numeric(gsub("%$", "", names(x))) 
  )
}

# For example: 
pctl_step_object <- rec_obj$steps[[1]]
pctl_step_object
#> Percentile transformation on hydrogen, oxygen, nitrogen [trained]
format_pctl(pctl_step_object$ref_dist[["hydrogen"]])
#> # A tibble: 87 x 2
#>    value percentile
#>    <dbl>      <dbl>
#>  1 0.03           0
#>  2 0.934          1
#>  3 1.60           2
#>  4 2.07           3
#>  5 2.45           4
#>  6 2.74           5
#>  7 3.15           6
#>  8 3.49           7
#>  9 3.71           8
#> 10 3.99           9
#> # … with 77 more rows
```

The tidy method could return these values for each selected column. Before `prep()`, missing values can be used as placeholders. 


```r
tidy.step_percentile <- function(x, ...) {
  if (is_trained(x)) {
    res <- map_dfr(x$ref_dist, format_pctl, .id = "term")
  }
  else {
    term_names <- sel2char(x$terms)
    res <-
      tibble(
        terms = term_names,
        value = rlang::na_dbl,
        percentile = rlang::na_dbl
      )
  }
  # Always return the step id: 
  res$id <- x$id
  res
}

tidy(rec_obj, number = 1)
#> # A tibble: 274 x 4
#>    term     value percentile id              
#>    <chr>    <dbl>      <dbl> <chr>           
#>  1 hydrogen 0.03           0 percentile_cAjrt
#>  2 hydrogen 0.934          1 percentile_cAjrt
#>  3 hydrogen 1.60           2 percentile_cAjrt
#>  4 hydrogen 2.07           3 percentile_cAjrt
#>  5 hydrogen 2.45           4 percentile_cAjrt
#>  6 hydrogen 2.74           5 percentile_cAjrt
#>  7 hydrogen 3.15           6 percentile_cAjrt
#>  8 hydrogen 3.49           7 percentile_cAjrt
#>  9 hydrogen 3.71           8 percentile_cAjrt
#> 10 hydrogen 3.99           9 percentile_cAjrt
#> # … with 264 more rows
```

### Methods for tuning parameters

The tune package can be used to find reasonable values of step arguments by model tuning. There are some S3 methods that are useful to define for your step. The percentile example doesn't really have any tunable parameters, so we will demonstrate using `step_poly()`, which returns a polynomial expansion of selected columns. Its function definition has the arguments: 


```r
args(step_poly)
#> function (recipe, ..., role = "predictor", trained = FALSE, objects = NULL, 
#>     degree = 2, options = list(), skip = FALSE, id = rand_id("poly")) 
#> NULL
```

The argument `degree` is tunable.

To work with tune it is _helpful_ (but not required) to use an S3 method called `tunable()` to define which arguments should be tuned and how values of those arguments should be generated. 

`tunable()` takes the step object as its argument and returns a tibble with columns: 

* `name`: The name of the argument. 

* `call_info`: A list that describes how to call a function that returns a dials parameter object. 

* `source`: A character string that indicates where the tuning value comes from (i.e., a model, a recipe etc.). Here, it is just `"recipe"`. 

* `component`: A character string with more information about the source. For recipes, this is just the name of the step (e.g. `"step_poly"`). 

* `component_id`: A character string to indicate where a unique identifier is for the object. For recipes, this is just the `id` value of the step object.  

The main piece of information that requires some detail is `call_info`. This is a list column in the tibble. Each element of the list is a list that describes the package and function that can be used to create a dials parameter object. 

For example, for a nearest-neighbors `neighbors` parameter, this value is just: 


```r
info <- list(pkg = "dials", fun = "neighbors")

# FYI: how it is used under-the-hood: 
new_param_call <- rlang::call2(.fn = info$fun, .ns = info$pkg)
rlang::eval_tidy(new_param_call)
#> # Nearest Neighbors  (quantitative)
#> Range: [1, 10]
```

For `step_poly()`, a dials object is needed that returns an integer that is the number of new columns to create. It turns out that there are a few different types of tuning parameters related to degree: 

```r
> lsf.str("package:dials", pattern = "degree")
degree : function (range = c(1, 3), trans = NULL)  
degree_int : function (range = c(1L, 3L), trans = NULL)  
prod_degree : function (range = c(1L, 2L), trans = NULL)  
spline_degree : function (range = c(3L, 10L), trans = NULL)  
```

Looking at the `range` values, some return doubles and others return integers. For our problem, `degree_int()` would be a good choice. 

For `step_poly()` the `tunable()` S3 method could be: 


```r
tunable.step_poly <- function (x, ...) {
  tibble::tibble(
    name = c("degree"),
    call_info = list(list(pkg = "dials", fun = "degree_int")),
    source = "recipe",
    component = "step_poly",
    component_id = x$id
  )
}
```


## Session information


```
#> ─ Session info ───────────────────────────────────────────────────────────────
#>  setting  value                       
#>  version  R version 4.0.2 (2020-06-22)
#>  os       macOS Catalina 10.15.6      
#>  system   x86_64, darwin17.0          
#>  ui       X11                         
#>  language (EN)                        
#>  collate  en_US.UTF-8                 
#>  ctype    en_US.UTF-8                 
#>  tz       America/Denver              
#>  date     2020-07-21                  
#> 
#> ─ Packages ───────────────────────────────────────────────────────────────────
#>  package    * version date       lib source        
#>  broom      * 0.7.0   2020-07-09 [1] CRAN (R 4.0.0)
#>  dials      * 0.0.8   2020-07-08 [1] CRAN (R 4.0.0)
#>  dplyr      * 1.0.0   2020-05-29 [1] CRAN (R 4.0.0)
#>  ggplot2    * 3.3.2   2020-06-19 [1] CRAN (R 4.0.0)
#>  infer      * 0.5.3   2020-07-14 [1] CRAN (R 4.0.2)
#>  modeldata  * 0.0.2   2020-06-22 [1] CRAN (R 4.0.2)
#>  parsnip    * 0.1.2   2020-07-03 [1] CRAN (R 4.0.1)
#>  purrr      * 0.3.4   2020-04-17 [1] CRAN (R 4.0.0)
#>  recipes    * 0.1.13  2020-06-23 [1] CRAN (R 4.0.0)
#>  rlang        0.4.7   2020-07-09 [1] CRAN (R 4.0.2)
#>  rsample    * 0.0.7   2020-06-04 [1] CRAN (R 4.0.0)
#>  tibble     * 3.0.3   2020-07-10 [1] CRAN (R 4.0.2)
#>  tidymodels * 0.1.1   2020-07-14 [1] CRAN (R 4.0.2)
#>  tune       * 0.1.1   2020-07-08 [1] CRAN (R 4.0.0)
#>  workflows  * 0.1.2   2020-07-07 [1] CRAN (R 4.0.0)
#>  yardstick  * 0.0.7   2020-07-13 [1] CRAN (R 4.0.2)
#> 
#> [1] /Library/Frameworks/R.framework/Versions/4.0/Resources/library
```
 
 
