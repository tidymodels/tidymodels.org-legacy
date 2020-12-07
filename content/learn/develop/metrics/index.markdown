---
title: "Custom performance metrics"
tags: [yardstick]
categories: []
type: learn-subsection
weight: 3
description: | 
  Create a new performance metric and integrate it with yardstick functions.
---






## Introduction

To use the code in this article, you will need to install the following packages: rlang and tidymodels.

The [yardstick](https://tidymodels.github.io/yardstick/) package already includes a large number of metrics, but there's obviously a chance that you might have a custom metric that hasn't been implemented yet. In that case, you can use a few of the tools yardstick exposes to create custom metrics.

Why create custom metrics? With the infrastructure yardstick provides, you get:

- Standardization between your metric and other preexisting metrics
- Automatic error handling for types and lengths
- Automatic selection of binary / multiclass metric implementations
- Automatic `NA` handling
- Support for grouped data frames
- Support for use alongside other metrics in `metric_set()`

The implementation for metrics differ slightly depending on whether you are implementing a numeric, class, or class probability metric. Examples for numeric and classification metrics are given below. We would encourage you to look into the implementation of `roc_auc()` after reading this vignette if you want to work on a class probability metric.

## Numeric example: MSE

Mean squared error (sometimes MSE or from here on, `mse()`) is a numeric metric that measures the average of the squared errors. Numeric metrics are generally the simplest to create with yardstick, as they do not have multiclass implementations. The formula for `mse()` is:

$$ MSE = \frac{1}{N} \sum_{i=1}^{N} (truth_i - estimate_i) ^ 2 = mean( (truth - estimate) ^ 2) $$

All metrics should have a data frame version, and a vector version. The data frame version here will be named `mse()`, and the vector version will be `mse_vec()`. 

### Vector implementation

To start, create the vector version. Generally, all metrics have the  same arguments unless the metric requires an extra parameter (such as `beta` in `f_meas()`). To create the vector function, you need to do two things:

1) Create an internal implementation function, `mse_impl()`.
2) Pass on that implementation function to `metric_vec_template()`.

Below, `mse_impl()` contains the actual implementation of the metric, and takes `truth` and `estimate` as arguments along with any metric specific arguments.

The yardstick function `metric_vec_template()` accepts the implementation function along with the other arguments to `mse_vec()` and actually executes `mse_impl()`. Additionally, it has a `cls` argument to specify the allowed class type of `truth` and `estimate`. If the classes are the same, a single character class can be passed, and if they are different a character vector of length 2 can be supplied.

The `metric_vec_template()` helper handles the removal of `NA` values in your metric, so your implementation function does not have to worry about them. It performs type checking using `cls` and also checks that the `estimator` is valid, the second of which is covered in the classification example. This way, all you have to worry about is the core implementation.


```r
library(tidymodels)

mse_vec <- function(truth, estimate, na_rm = TRUE, ...) {
  
  mse_impl <- function(truth, estimate) {
    mean((truth - estimate) ^ 2)
  }
  
  metric_vec_template(
    metric_impl = mse_impl,
    truth = truth, 
    estimate = estimate,
    na_rm = na_rm,
    cls = "numeric",
    ...
  )
  
}
```

At this point, you've created the vector version of the mean squared error metric.


```r
data("solubility_test")

mse_vec(
  truth = solubility_test$solubility, 
  estimate = solubility_test$prediction
)
#> [1] 0.521
```

Intelligent error handling is immediately available.


```r
mse_vec(truth = "apple", estimate = 1)
#> Error: `truth` should be a numeric but a character was supplied.

mse_vec(truth = 1, estimate = factor("xyz"))
#> Error: `estimate` should be a numeric but a factor was supplied.
```

`NA` values are removed if `na_rm = TRUE` (the default). If `na_rm = FALSE` and any `NA` values are detected, then the metric automatically returns `NA`.


```r
# NA values removed
mse_vec(truth = c(NA, .5, .4), estimate = c(1, .6, .5))
#> [1] 0.01

# NA returned
mse_vec(truth = c(NA, .5, .4), estimate = c(1, .6, .5), na_rm = FALSE)
#> [1] NA
```

### Data frame implementation

The data frame version of the metric should be fairly simple. It is a generic function with a `data.frame` method that calls the yardstick helper, `metric_summarizer()`, and passes along the `mse_vec()` function to it along with versions of `truth` and `estimate` that have been wrapped in `rlang::enquo()` and then unquoted with `!!` so that non-standard evaluation can be supported.


```r
library(rlang)

mse <- function(data, ...) {
  UseMethod("mse")
}

mse <- new_numeric_metric(mse, direction = "minimize")

mse.data.frame <- function(data, truth, estimate, na_rm = TRUE, ...) {
  
  metric_summarizer(
    metric_nm = "mse",
    metric_fn = mse_vec,
    data = data,
    truth = !! enquo(truth),
    estimate = !! enquo(estimate), 
    na_rm = na_rm,
    ...
  )
  
}
```

And that's it. The yardstick package handles the rest with an internal call to `summarise()`.


```r
mse(solubility_test, truth = solubility, estimate = prediction)
#> # A tibble: 1 x 3
#>   .metric .estimator .estimate
#>   <chr>   <chr>          <dbl>
#> 1 mse     standard       0.521

# Error handling
mse(solubility_test, truth = solubility, estimate = factor("xyz"))
#> Error: Problem with `summarise()` input `.estimate`.
#> x `estimate` should be a numeric but a factor was supplied.
#> ℹ Input `.estimate` is `metric_fn(truth = solubility, estimate = factor("xyz"), na_rm = na_rm)`.
```

Let's test it out on a grouped data frame.


```r
library(dplyr)

set.seed(1234)
size <- 100
times <- 10

# create 10 resamples
solubility_resampled <- bind_rows(
  replicate(
    n = times,
    expr = sample_n(solubility_test, size, replace = TRUE),
    simplify = FALSE
  ),
  .id = "resample"
)

solubility_resampled %>%
  group_by(resample) %>%
  mse(solubility, prediction)
#> # A tibble: 10 x 4
#>    resample .metric .estimator .estimate
#>    <chr>    <chr>   <chr>          <dbl>
#>  1 1        mse     standard       0.512
#>  2 10       mse     standard       0.454
#>  3 2        mse     standard       0.513
#>  4 3        mse     standard       0.414
#>  5 4        mse     standard       0.543
#>  6 5        mse     standard       0.456
#>  7 6        mse     standard       0.652
#>  8 7        mse     standard       0.642
#>  9 8        mse     standard       0.404
#> 10 9        mse     standard       0.479
```

## Class example: miss rate

Miss rate is another name for the false negative rate, and is a classification metric in the same family as `sens()` and `spec()`. It follows the formula:

$$ miss\_rate = \frac{FN}{FN + TP} $$

This metric, like other classification metrics, is more easily computed when expressed as a confusion matrix. As you will see in the example, you can  achieve this with a call to `base::table(estimate, truth)` which correctly puts the "correct" result in the columns of the confusion matrix.

Classification metrics are more complicated than numeric ones because you have to think about extensions to the multiclass case. For now, let's start with  the binary case.

### Vector implementation

The vector implementation for classification metrics initially has the same setup  as numeric metrics, but has an additional argument, `estimator` that determines the type of estimator to use (binary or some kind of multiclass implementation or averaging). This argument is auto-selected for the user, so default it to  `NULL`. Additionally, pass it along to `metric_vec_template()` so that it can check the provided `estimator` against the classes of `truth` and `estimate` to see if they are allowed.


```r
# Logic for `event_level`
event_col <- function(xtab, event_level) {
  if (identical(event_level, "first")) {
    colnames(xtab)[[1]]
  } else {
    colnames(xtab)[[2]]
  }
}

miss_rate_vec <- function(truth, 
                          estimate, 
                          estimator = NULL, 
                          na_rm = TRUE, 
                          event_level = "first",
                          ...) {
  estimator <- finalize_estimator(truth, estimator)
  
  miss_rate_impl <- function(truth, estimate) {
    # Create 
    xtab <- table(estimate, truth)
    col <- event_col(xtab, event_level)
    col2 <- setdiff(colnames(xtab), col)
    
    tp <- xtab[col, col]
    fn <- xtab[col2, col]
    
    fn / (fn + tp)
  }
  
  metric_vec_template(
    metric_impl = miss_rate_impl,
    truth = truth,
    estimate = estimate,
    na_rm = na_rm,
    cls = "factor",
    estimator = estimator,
    ...
  )
}
```

Another change from the numeric metric is that a call to `finalize_estimator()` is made. This is the infrastructure that auto-selects the type of estimator to use.


```r
data("two_class_example")
miss_rate_vec(two_class_example$truth, two_class_example$predicted)
#> [1] 0.12
```

What happens if you try and pass in a multiclass result?


```r
data("hpc_cv")
fold1 <- filter(hpc_cv, Resample == "Fold01")
miss_rate_vec(fold1$obs, fold1$pred)
#>      F      M      L 
#> 0.0621 0.0000 0.0000
```

This isn't great, as currently multiclass `miss_rate()` isn't supported and it would have been better to throw an error if the `estimator` was not `"binary"`. Currently, `finalize_estimator()` uses its default implementation which selected `"macro"` as the `estimator` since `truth` was a factor with more than 2 classes. When we implement multiclass averaging, this is what you want, but if your metric only works with a binary implementation (or has other specialized multiclass versions), you might want to guard against this.

To fix this, a generic counterpart to `finalize_estimator()`, called `finalize_estimator_internal()`, exists that helps you restrict the input types. If you provide a method to `finalize_estimator_internal()` where the method name is the same as your metric name, and then set the `metric_class` argument in `finalize_estimator()` to be the same thing, you can control how the auto-selection of the `estimator` is handled.  

Don't worry about the `metric_dispatcher` argument. This is handled for you and just exists as a dummy argument to dispatch off of.

It is also good practice to call `validate_estimator()` which handles the case where a user passed in the estimator themselves. This validates that the supplied `estimator` is one of the allowed types and error otherwise.


```r
finalize_estimator_internal.miss_rate <- function(metric_dispatcher, x, estimator) {
  
  validate_estimator(estimator, estimator_override = "binary")
  if (!is.null(estimator)) {
    return(estimator)
  }
  
  lvls <- levels(x)
  if (length(lvls) > 2) {
    stop("A multiclass `truth` input was provided, but only `binary` is supported.")
  } 
  "binary"
}

miss_rate_vec <- function(truth, 
                          estimate, 
                          estimator = NULL, 
                          na_rm = TRUE, 
                          event_level = "first",
                          ...) {
  # calls finalize_estimator_internal() internally
  estimator <- finalize_estimator(truth, estimator, metric_class = "miss_rate")
  
  miss_rate_impl <- function(truth, estimate) {
    # Create 
    xtab <- table(estimate, truth)
    col <- event_col(xtab, event_level)
    col2 <- setdiff(colnames(xtab), col)
    
    tp <- xtab[col, col]
    fn <- xtab[col2, col]
    
    fn / (fn + tp)
    
  }
  
  metric_vec_template(
    metric_impl = miss_rate_impl,
    truth = truth,
    estimate = estimate,
    na_rm = na_rm,
    cls = "factor",
    estimator = estimator,
    ...
  )
}

# Error thrown by our custom handler
miss_rate_vec(fold1$obs, fold1$pred)
#> Error in finalize_estimator_internal.miss_rate(metric_dispatcher, x, estimator): A multiclass `truth` input was provided, but only `binary` is supported.

# Error thrown by validate_estimator()
miss_rate_vec(fold1$obs, fold1$pred, estimator = "macro")
#> Error: `estimator` must be one of: "binary". Not "macro".
```

### Supporting multiclass miss rate

Like many other classification metrics such as `precision()` or `recall()`, miss rate does not have a natural multiclass extension, but one can be created using methods such as macro, weighted macro, and micro averaging. If you have not, I encourage you to read `vignette("multiclass", "yardstick")` for more information about how these methods work.

Generally, they require more effort to get right than the binary case, especially if you want to have a performant version. Luckily, a somewhat standard template is used in yardstick and can be used here as well.

Let's first remove the "binary" restriction we created earlier.


```r
rm(finalize_estimator_internal.miss_rate)
```

The main changes below are:

- The binary implementation is moved to `miss_rate_binary()`.

- `miss_rate_estimator_impl()` is a helper function for switching between binary and multiclass implementations. It also applies the weighting required for multiclass estimators. It is called from `miss_rate_impl()` and also accepts the `estimator` argument using R's function scoping rules.

- `miss_rate_multiclass()` provides the implementation for the multiclass case. It calculates the true positive and false negative values as vectors with one value per class. For the macro case, it returns a vector of miss rate calculations, and for micro, it first sums the individual pieces and returns a single miss rate calculation. In the macro case, the vector is then weighted appropriately in `miss_rate_estimator_impl()` depending on whether or not it was macro or  weighted macro. 


```r
miss_rate_vec <- function(truth, 
                          estimate, 
                          estimator = NULL, 
                          na_rm = TRUE, 
                          event_level = "first",
                          ...) {
  # calls finalize_estimator_internal() internally
  estimator <- finalize_estimator(truth, estimator, metric_class = "miss_rate")
  
  miss_rate_impl <- function(truth, estimate) {
    xtab <- table(estimate, truth)
    # Rather than implement the actual method here, we rely on
    # an *_estimator_impl() function that can handle binary
    # and multiclass cases
    miss_rate_estimator_impl(xtab, estimator, event_level)
  }
  
  metric_vec_template(
    metric_impl = miss_rate_impl,
    truth = truth,
    estimate = estimate,
    na_rm = na_rm,
    cls = "factor",
    estimator = estimator,
    ...
  )
}


# This function switches between binary and multiclass implementations
miss_rate_estimator_impl <- function(data, estimator, event_level) {
  if(estimator == "binary") {
    miss_rate_binary(data, event_level)
  } else {
    # Encapsulates the macro, macro weighted, and micro cases
    wt <- get_weights(data, estimator)
    res <- miss_rate_multiclass(data, estimator)
    weighted.mean(res, wt)
  }
}


miss_rate_binary <- function(data, event_level) {
  col <- event_col(data, event_level)
  col2 <- setdiff(colnames(data), col)
  
  tp <- data[col, col]
  fn <- data[col2, col]
  
  fn / (fn + tp)
}

miss_rate_multiclass <- function(data, estimator) {
  
  # We need tp and fn for all classes individually
  # we can get this by taking advantage of the fact
  # that tp + fn = colSums(data)
  tp <- diag(data)
  tpfn <- colSums(data)
  fn <- tpfn - tp
  
  # If using a micro estimator, we sum the individual
  # pieces before performing the miss rate calculation
  if (estimator == "micro") {
    tp <- sum(tp)
    fn <- sum(fn)
  }
  
  # return the vector 
  tp / (tp + fn)
}
```

For the macro case, this separation of weighting from the core implementation might seem strange, but there is good reason for it. Some metrics are combinations of other metrics, and it is nice to be able to reuse code when calculating more complex metrics. For example, `f_meas()` is a combination of `recall()` and `precision()`. When calculating a macro averaged `f_meas()`, the weighting  must be applied 1 time, at the very end of the calculation. `recall_multiclass()` and `precision_multiclass()` are defined similarly to how `miss_rate_multiclass()` is defined and returns the unweighted vector of calculations. This means we can directly use this in `f_meas()`, and then weight everything once at the end of that calculation.

Let's try it out now:


```r
# two class
miss_rate_vec(two_class_example$truth, two_class_example$predicted)
#> [1] 0.12

# multiclass
miss_rate_vec(fold1$obs, fold1$pred)
#> [1] 0.548
```

#### Data frame implementation

Luckily, the data frame implementation is as simple as the numeric case, we just need to add an extra `estimator` argument and pass that through.


```r
miss_rate <- function(data, ...) {
  UseMethod("miss_rate")
}

miss_rate <- new_class_metric(miss_rate, direction = "minimize")

miss_rate.data.frame <- function(data, 
                                 truth, 
                                 estimate, 
                                 estimator = NULL, 
                                 na_rm = TRUE, 
                                 event_level = "first",
                                 ...) {
  metric_summarizer(
    metric_nm = "miss_rate",
    metric_fn = miss_rate_vec,
    data = data,
    truth = !! enquo(truth),
    estimate = !! enquo(estimate), 
    estimator = estimator,
    na_rm = na_rm,
    event_level = event_level,
    ...
  )
}
```


```r
# Macro weighted automatically selected
fold1 %>%
  miss_rate(obs, pred)
#> # A tibble: 1 x 3
#>   .metric   .estimator .estimate
#>   <chr>     <chr>          <dbl>
#> 1 miss_rate macro          0.548

# Switch to micro
fold1 %>%
  miss_rate(obs, pred, estimator = "micro")
#> # A tibble: 1 x 3
#>   .metric   .estimator .estimate
#>   <chr>     <chr>          <dbl>
#> 1 miss_rate micro          0.726

# Macro weighted by resample
hpc_cv %>%
  group_by(Resample) %>%
  miss_rate(obs, pred, estimator = "macro_weighted")
#> # A tibble: 10 x 4
#>    Resample .metric   .estimator     .estimate
#>    <chr>    <chr>     <chr>              <dbl>
#>  1 Fold01   miss_rate macro_weighted     0.726
#>  2 Fold02   miss_rate macro_weighted     0.712
#>  3 Fold03   miss_rate macro_weighted     0.758
#>  4 Fold04   miss_rate macro_weighted     0.712
#>  5 Fold05   miss_rate macro_weighted     0.712
#>  6 Fold06   miss_rate macro_weighted     0.697
#>  7 Fold07   miss_rate macro_weighted     0.675
#>  8 Fold08   miss_rate macro_weighted     0.721
#>  9 Fold09   miss_rate macro_weighted     0.673
#> 10 Fold10   miss_rate macro_weighted     0.699

# Error handling
miss_rate(hpc_cv, obs, VF)
#> Error: Problem with `summarise()` input `.estimate`.
#> x `estimate` should be a factor but a numeric was supplied.
#> ℹ Input `.estimate` is `metric_fn(truth = obs, estimate = VF, na_rm = na_rm, event_level = "first")`.
```

## Using custom metrics

The `metric_set()` function validates that all metric functions are of the same metric type by checking the class of the function. If any metrics are not of the right class, `metric_set()` fails. By using `new_numeric_metric()` and `new_class_metric()` in the above custom metrics, they work out of the box without any additional adjustments.


```r
numeric_mets <- metric_set(mse, rmse)

numeric_mets(solubility_test, solubility, prediction)
#> # A tibble: 2 x 3
#>   .metric .estimator .estimate
#>   <chr>   <chr>          <dbl>
#> 1 mse     standard       0.521
#> 2 rmse    standard       0.722
```


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
#>  date     2020-12-07                  
#> 
#> ─ Packages ───────────────────────────────────────────────────────────────────
#>  package    * version date       lib source        
#>  broom      * 0.7.2   2020-10-20 [1] CRAN (R 4.0.2)
#>  dials      * 0.0.9   2020-09-16 [1] CRAN (R 4.0.2)
#>  dplyr      * 1.0.2   2020-08-18 [1] CRAN (R 4.0.2)
#>  ggplot2    * 3.3.2   2020-06-19 [1] CRAN (R 4.0.0)
#>  infer      * 0.5.3   2020-07-14 [1] CRAN (R 4.0.0)
#>  parsnip    * 0.1.4   2020-10-27 [1] CRAN (R 4.0.2)
#>  purrr      * 0.3.4   2020-04-17 [1] CRAN (R 4.0.0)
#>  recipes    * 0.1.15  2020-11-11 [1] CRAN (R 4.0.2)
#>  rlang      * 0.4.9   2020-11-26 [1] CRAN (R 4.0.2)
#>  rsample    * 0.0.8   2020-09-23 [1] CRAN (R 4.0.2)
#>  tibble     * 3.0.4   2020-10-12 [1] CRAN (R 4.0.2)
#>  tidymodels * 0.1.2   2020-11-22 [1] CRAN (R 4.0.2)
#>  tune       * 0.1.2   2020-11-17 [1] CRAN (R 4.0.3)
#>  workflows  * 0.2.1   2020-10-08 [1] CRAN (R 4.0.2)
#>  yardstick  * 0.0.7   2020-07-13 [1] CRAN (R 4.0.2)
#> 
#> [1] /Library/Frameworks/R.framework/Versions/4.0/Resources/library
```
