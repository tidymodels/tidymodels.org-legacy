---
title: "Create your own broom tidier methods"
tags: [broom]
categories: []
type: learn-subsection
weight: 5
description: | 
  Write tidy(), glance(), and augment() methods for new model objects.
---





## Introduction

To use the code in this article, you will need to install the following packages: generics, tidymodels, tidyverse, and usethis.

The broom package provides tools to summarize key information about models in tidy `tibble()`s. The package provides three verbs, or "tidiers," to help make model objects easier to work with:

* `tidy()` summarizes information about model components
* `glance()` reports information about the entire model
* `augment()` adds information about observations to a dataset

Each of the three verbs above are _generic_, in that they do not define a procedure to tidy a given model object, but instead redirect to the relevant _method_ implemented to tidy a specific type of model object. The broom package provides methods for model objects from over 100 modeling packages along with nearly all of the model objects in the stats package that comes with base R. However, for maintainability purposes, the broom package authors now ask that requests for new methods be first directed to the parent package (i.e. the package that supplies the model object) rather than to broom. New methods will generally only be integrated into broom in the case that the requester has already asked the maintainers of the model-owning package to implement tidier methods in the parent package.

We'd like to make implementing external tidier methods as painless as possible. The general process for doing so is:

* re-export the tidier generics
* implement tidying methods
* document the new methods

In this article, we'll walk through each of the above steps in detail, giving examples and pointing out helpful functions when possible.

## Re-export the tidier generics

The first step is to re-export the generic functions for `tidy()`, `glance()`, and/or `augment()`. You could do so from `broom` itself, but we've provided an alternative, much lighter dependency called `generics`.

First you'll need to add the [generics](https://github.com/r-lib/generics) package to `Imports`. We recommend using the [usethis](https://github.com/r-lib/usethis) package for this:


```r
usethis::use_package("generics", "Imports")
```

Next, you'll need to re-export the appropriate tidying methods. If you plan to implement a `glance()` method, for example, you can re-export the `glance()` generic by adding the following somewhere inside the `/R` folder of your package:


```r
#' @importFrom generics glance
#' @export
generics::glance
```

Oftentimes it doesn't make sense to define one or more of these methods for a particular model. In this case, only implement the methods that do make sense.

{{% warning %}} Please do not define `tidy()`, `glance()`, or `augment()` generics in your package. This will result in namespace conflicts whenever your package is used along other packages that also export tidying methods. {{%/ warning %}}

## Implement tidying methods

You'll now need to implement specific tidying methods for each of the generics you've re-exported in the above step. For each of `tidy()`, `glance()`, and `augment()`, we'll walk through the big picture, an example, and helpful resources.

In this article, we'll use the base R dataset `trees`, giving the tree girth (in inches), height (in feet), and volume (in cubic feet), to fit an example linear model using the base R `lm()` function. 


```r
# load in the trees dataset
data(trees)

# take a look!
str(trees)
#> 'data.frame':	31 obs. of  3 variables:
#>  $ Girth : num  8.3 8.6 8.8 10.5 10.7 10.8 11 11 11.1 11.2 ...
#>  $ Height: num  70 65 63 72 81 83 66 75 80 75 ...
#>  $ Volume: num  10.3 10.3 10.2 16.4 18.8 19.7 15.6 18.2 22.6 19.9 ...

# fit the timber volume as a function of girth and height
trees_model <- lm(Volume ~ Girth + Height, data = trees)
```

Let's take a look at the `summary()` of our `trees_model` fit.


```r
summary(trees_model)
#> 
#> Call:
#> lm(formula = Volume ~ Girth + Height, data = trees)
#> 
#> Residuals:
#>    Min     1Q Median     3Q    Max 
#> -6.406 -2.649 -0.288  2.200  8.485 
#> 
#> Coefficients:
#>             Estimate Std. Error t value Pr(>|t|)    
#> (Intercept)  -57.988      8.638   -6.71  2.7e-07 ***
#> Girth          4.708      0.264   17.82  < 2e-16 ***
#> Height         0.339      0.130    2.61    0.014 *  
#> ---
#> Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
#> 
#> Residual standard error: 3.88 on 28 degrees of freedom
#> Multiple R-squared:  0.948,	Adjusted R-squared:  0.944 
#> F-statistic:  255 on 2 and 28 DF,  p-value: <2e-16
```

This output gives some summary statistics on the residuals (which would be described more fully in an `augment()` output), model coefficients (which, in this case, make up the `tidy()` output), and some model-level summarizations such as RSE, `\(R^2\)`, etc. (which make up the `glance()` output.)

### Implementing the `tidy()` method

The `tidy(x, ...)` method will return a tibble where each row contains information about a component of the model. The `x` input is a model object, and the dots (`...`) are an optional argument to supply additional information to any calls inside your method. New `tidy()` methods can take additional arguments, but _must_ include the `x` and `...` arguments to be compatible with the generic function. (For a glossary of currently acceptable additional arguments, see [the end of this article](#glossary).)  Examples of model components include regression coefficients (for regression models), clusters (for classification/clustering models), etc. These `tidy()` methods are useful for inspecting model details and creating custom model visualizations.

Returning to the example of our linear model on timber volume, we'd like to extract information on the model components. In this example, the components are the regression coefficients. After taking a look at the model object and its `summary()`, you might notice that you can extract the regression coefficients as follows:


```r
summary(trees_model)$coefficients
#>             Estimate Std. Error t value Pr(>|t|)
#> (Intercept)  -57.988      8.638   -6.71 2.75e-07
#> Girth          4.708      0.264   17.82 8.22e-17
#> Height         0.339      0.130    2.61 1.45e-02
```

This object contains the model coefficients as a table, where the information giving which coefficient is being described in each row is given in the row names. Converting to a tibble where the row names are contained in a column, you might write:


```r
trees_model_tidy <- summary(trees_model)$coefficients %>% 
  as_tibble(rownames = "term")

trees_model_tidy
#> # A tibble: 3 x 5
#>   term        Estimate `Std. Error` `t value` `Pr(>|t|)`
#>   <chr>          <dbl>        <dbl>     <dbl>      <dbl>
#> 1 (Intercept)  -58.0          8.64      -6.71   2.75e- 7
#> 2 Girth          4.71         0.264     17.8    8.22e-17
#> 3 Height         0.339        0.130      2.61   1.45e- 2
```

The broom package standardizes common column names used to describe coefficients. In this case, the column names are:


```r
colnames(trees_model_tidy) <- c("term", "estimate", "std.error", "statistic", "p.value")
```

A glossary giving the currently acceptable column names outputted by `tidy()` methods can be found [at the end of this article](#glossary). As a rule of thumb, column names resulting from `tidy()` methods should be all lowercase and contain only alphanumerics or periods (though there are plenty of exceptions).

Finally, it is common for `tidy()` methods to include an option to calculate confidence/credible intervals for each component based on the model, when possible. In this example, the `confint()` function can be used to calculate confidence intervals from a model object resulting from `lm()`:


```r
confint(trees_model)
#>                2.5 %  97.5 %
#> (Intercept) -75.6823 -40.293
#> Girth         4.1668   5.249
#> Height        0.0726   0.606
```

With these considerations in mind, a reasonable `tidy()` method for `lm()` might look something like:


```r
tidy.lm <- function(x, conf.int = FALSE, conf.level = 0.95, ...) {
  
  result <- summary(x)$coefficients %>%
    tibble::as_tibble(rownames = "term") %>%
    dplyr::rename(estimate = Estimate,
                  std.error = `Std. Error`,
                  statistic = `t value`,
                  p.value = `Pr(>|t|)`)
  
  if (conf.int) {
    ci <- confint(x, level = conf.level)
    result <- dplyr::left_join(result, ci, by = "term")
  }
  
  result
}
```

{{% note %}}  If you're interested, the actual `tidy.lm()` source can be found [here](https://github.com/tidymodels/broom/blob/master/R/stats-lm-tidiers.R)! It's not too different from the version above except for some argument checking and additional columns. {{%/ note %}}

With this method exported, then, if a user calls `tidy(fit)`, where `fit` is an output from `lm()`, the `tidy()` generic would "redirect" the call to the `tidy.lm()` function above.

Some things to keep in mind while writing your `tidy()` method:

* Sometimes a model will have several different types of components. For example, in mixed models, there is different information associated with fixed effects and random effects. Since this information doesn't have the same interpretation, it doesn't make sense to summarize the fixed and random effects in the same table. In cases like this you should add an argument that allows the user to specify which type of information they want. For example, you might implement an interface along the lines of:


```r
model <- mixed_model(...)
tidy(model, effects = "fixed")
tidy(model, effects = "random")
```

* How are missing values encoded in the model object and its `summary()`? Ensure that rows are included even when the associated model component is missing or rank deficient.
* Are there other measures specific to each component that could reasonably be expected to be included in their summarizations? Some common arguments to `tidy()` methods include:
  - `conf.int`: A logical indicating whether or not to calculate confidence/credible intervals. This should default to `FALSE`.
  - `conf.level`: The confidence level to use for the interval when `conf.int = TRUE`. Typically defaults to `.95`.
  - `exponentiate`: A logical indicating whether or not model terms should be presented on an exponential scale (typical for logistic regression).

### Implementing the `glance()` method

`glance()` returns a one-row tibble providing model-level summarizations (e.g. goodness of fit measures and related statistics). This is useful to check for model misspecification and to compare many models. Again, the `x` input is a model object, and the `...` is an optional argument to supply additional information to any calls inside your method. New `glance()` methods can also take additional arguments and _must_ include the `x` and `...` arguments. (For a glossary of currently acceptable additional arguments, see [the end of this article](#glossary).)

Returning to the `trees_model` example, we could pull out the `\(R^2\)` value with the following code:


```r
summary(trees_model)$r.squared
#> [1] 0.948
```

Similarly, for the adjusted `\(R^2\)`:


```r
summary(trees_model)$adj.r.squared
#> [1] 0.944
```

Unfortunately, for many model objects, the extraction of model-level information is largely a manual process. You will likely need to build a `tibble()` element-by-element by subsetting the `summary()` object repeatedly. The `with()` function, however, can help make this process a bit less tedious by evaluating expressions inside of the `summary(trees_model)` environment. To grab those those same two model elements from above using `with()`:


```r
with(summary(trees_model),
     tibble::tibble(r.squared = r.squared,
                    adj.r.squared = adj.r.squared))
#> # A tibble: 1 x 2
#>   r.squared adj.r.squared
#>       <dbl>         <dbl>
#> 1     0.948         0.944
```

A reasonable `glance()` method for `lm()`, then, might look something like:


```r
glance.lm <- function(x, ...) {
  with(
    summary(x),
    tibble::tibble(
      r.squared = r.squared,
      adj.r.squared = adj.r.squared,
      sigma = sigma,
      statistic = fstatistic["value"],
      p.value = pf(
        fstatistic["value"],
        fstatistic["numdf"],
        fstatistic["dendf"],
        lower.tail = FALSE
      ),
      df = fstatistic["numdf"],
      logLik = as.numeric(stats::logLik(x)),
      AIC = stats::AIC(x),
      BIC = stats::BIC(x),
      deviance = stats::deviance(x),
      df.residual = df.residual(x),
      nobs = stats::nobs(x)
    )
  )
}
```

{{% note %}} This is the actual definition of `glance.lm()` provided by broom! {{%/ note %}}

Some things to keep in mind while writing `glance()` methods:
* Output should not include the name of the modeling function or any arguments given to the modeling function.
* In some cases, you may wish to provide model-level diagnostics not returned by the original object. For example, the above `glance.lm()` calculates `AIC` and `BIC` from the model fit. If these are easy to compute, feel free to add them. However, tidier methods are generally not an appropriate place to implement complex or time consuming calculations.
* The `glance` method should always return the same columns in the same order when given an object of a given model class. If a summary metric (such as `AIC`) is not defined in certain circumstances, use `NA`.

### Implementing the `augment()` method

`augment()` methods add columns to a dataset containing information such as fitted values, residuals or cluster assignments. All columns added to a dataset have a `.` prefix to prevent existing columns from being overwritten. (Currently acceptable column names are given in [the glossary](#glossary).) The `x` and `...` arguments share their meaning with the two functions described above. `augment` methods also optionally accept a `data` argument that is a `data.frame` (or `tibble`) to add observation-level information to, returning a `tibble` object with the same number of rows as `data`. Many `augment()` methods also accept a `newdata` argument, following the same conventions as the `data` argument, except with the underlying assumption that the model has not "seen" the data yet. As a result, `newdata` arguments need not contain the response columns in `data`. Only one of `data` or `newdata` should be supplied. A full glossary of acceptable arguments to `augment()` methods can be found at [the end of this article](#glossary).

If a `data` argument is not specified, `augment()` should try to reconstruct the original data as much as possible from the model object. This may not always be possible, and often it will not be possible to recover columns not used by the model.

With this is mind, we can look back to our `trees_model` example. For one, the `model` element inside of the `trees_model` object will allow us to recover the original data:


```r
trees_model$model
#>    Volume Girth Height
#> 1    10.3   8.3     70
#> 2    10.3   8.6     65
#> 3    10.2   8.8     63
#> 4    16.4  10.5     72
#> 5    18.8  10.7     81
#> 6    19.7  10.8     83
#> 7    15.6  11.0     66
#> 8    18.2  11.0     75
#> 9    22.6  11.1     80
#> 10   19.9  11.2     75
#> 11   24.2  11.3     79
#> 12   21.0  11.4     76
#> 13   21.4  11.4     76
#> 14   21.3  11.7     69
#> 15   19.1  12.0     75
#> 16   22.2  12.9     74
#> 17   33.8  12.9     85
#> 18   27.4  13.3     86
#> 19   25.7  13.7     71
#> 20   24.9  13.8     64
#> 21   34.5  14.0     78
#> 22   31.7  14.2     80
#> 23   36.3  14.5     74
#> 24   38.3  16.0     72
#> 25   42.6  16.3     77
#> 26   55.4  17.3     81
#> 27   55.7  17.5     82
#> 28   58.3  17.9     80
#> 29   51.5  18.0     80
#> 30   51.0  18.0     80
#> 31   77.0  20.6     87
```

Similarly, the fitted values and residuals can be accessed with the following code:


```r
head(trees_model$fitted.values)
#>     1     2     3     4     5     6 
#>  4.84  4.55  4.82 15.87 19.87 21.02
head(trees_model$residuals)
#>      1      2      3      4      5      6 
#>  5.462  5.746  5.383  0.526 -1.069 -1.318
```

As with `glance()` methods, it's fine (and encouraged!) to include common metrics associated with observations if they are not computationally intensive to compute. A common metric associated with linear models, for example, is the standard error of fitted values:


```r
se.fit <- predict(trees_model, newdata = trees, se.fit = TRUE)$se.fit %>%
  unname()

head(se.fit)
#> [1] 1.321 1.489 1.633 0.944 1.348 1.532
```

Thus, a reasonable `augment()` method for `lm` might look something like this:


```r
augment.lm <- function(x, data = x$model, newdata = NULL, ...) {
  if (is.null(newdata)) {
    dplyr::bind_cols(tibble::as_tibble(data),
                     tibble::tibble(.fitted = x$fitted.values,
                                    .se.fit = predict(x, 
                                                      newdata = data, 
                                                      se.fit = TRUE)$se.fit,
                                   .resid =  x$residuals))
  } else {
    predictions <- predict(x, newdata = newdata, se.fit = TRUE)
    dplyr::bind_cols(tibble::as_tibble(newdata),
                     tibble::tibble(.fitted = predictions$fit,
                                    .se.fit = predictions$se.fit))
  }
}
```

Some other things to keep in mind while writing `augment()` methods:
* The `newdata` argument should default to `NULL`. Users should only ever specify one of `data` or `newdata`. Providing both `data` and `newdata` should result in an error. The `newdata` argument should accept both `data.frame`s and `tibble`s.
* Data given to the `data` argument must have both the original predictors and the original response. Data given to the `newdata` argument only needs to have the original predictors. This is important because there may be important information associated with training data that is not associated with test data. This means that the `original_data` object in `augment(model, data = original_data)` should provide `.fitted` and `.resid` columns (in most cases), whereas `test_data` in `augment(model, data = test_data)` only needs a `.fitted` column, even if the response is present in `test_data`.
* If the `data` or `newdata` is specified as a `data.frame` with rownames, `augment` should return them in a column called `.rownames`.
* For observations where no fitted values or summaries are available (where there's missing data, for example), return `NA`.
* *The `augment()` method should always return as many rows as were in `data` or `newdata`*, depending on which is supplied

{{% note %}} The recommended interface and functionality for `augment()` methods may change soon. {{%/ note %}}

## Document the new methods

The only remaining step is to integrate the new methods into the parent package! To do so, just drop the methods into a `.R` file inside of the `/R` folder and document them using roxygen2. If you're unfamiliar with the process of documenting objects, you can read more about it [here](http://r-pkgs.had.co.nz/man.html). Here's an example of how our `tidy.lm()` method might be documented:


```r
#' Tidy a(n) lm object
#'
#' @param x A `lm` object.
#' @param conf.int Logical indicating whether or not to include 
#'   a confidence interval in the tidied output. Defaults to FALSE.
#' @param conf.level The confidence level to use for the confidence 
#'   interval if conf.int = TRUE. Must be strictly greater than 0 
#'   and less than 1. Defaults to 0.95, which corresponds to a 
#'   95 percent confidence interval.
#' @param ... Unused, included for generic consistency only.
#' @return A tidy [tibble::tibble()] summarizing component-level
#'   information about the model
#'
#' @examples
#' # load the trees dataset
#' data(trees)
#' 
#' # fit a linear model on timber volume
#' trees_model <- lm(Volume ~ Girth + Height, data = trees)
#'
#' # summarize model coefficients in a tidy tibble!
#' tidy(trees_model)
#'
#' @export
tidy.lm <- function(x, conf.int = FALSE, conf.level = 0.95, ...) {

  # ... the rest of the function definition goes here!
```

Once you've documented each of your new methods and executed `devtools::document()`, you're done! Congrats on implementing your own broom tidier methods for a new model object!

## Glossaries: argument and column names {#glossary}



Tidier methods have a standardized set of acceptable argument and output column names. The currently acceptable argument names by tidier method are:

<table class="table" style="width: auto !important; margin-left: auto; margin-right: auto;">
 <thead>
  <tr>
   <th style="text-align:left;"> Method </th>
   <th style="text-align:left;"> Argument </th>
  </tr>
 </thead>
<tbody>
  <tr>
   <td style="text-align:left;"> tidy </td>
   <td style="text-align:left;"> alpha </td>
  </tr>
  <tr>
   <td style="text-align:left;"> tidy </td>
   <td style="text-align:left;"> boot_se </td>
  </tr>
  <tr>
   <td style="text-align:left;"> tidy </td>
   <td style="text-align:left;"> by_class </td>
  </tr>
  <tr>
   <td style="text-align:left;"> tidy </td>
   <td style="text-align:left;"> col.names </td>
  </tr>
  <tr>
   <td style="text-align:left;"> tidy </td>
   <td style="text-align:left;"> component </td>
  </tr>
  <tr>
   <td style="text-align:left;"> tidy </td>
   <td style="text-align:left;"> conf.int </td>
  </tr>
  <tr>
   <td style="text-align:left;"> tidy </td>
   <td style="text-align:left;"> conf.level </td>
  </tr>
  <tr>
   <td style="text-align:left;"> tidy </td>
   <td style="text-align:left;"> conf.method </td>
  </tr>
  <tr>
   <td style="text-align:left;"> tidy </td>
   <td style="text-align:left;"> conf.type </td>
  </tr>
  <tr>
   <td style="text-align:left;"> tidy </td>
   <td style="text-align:left;"> diagonal </td>
  </tr>
  <tr>
   <td style="text-align:left;"> tidy </td>
   <td style="text-align:left;"> droppars </td>
  </tr>
  <tr>
   <td style="text-align:left;"> tidy </td>
   <td style="text-align:left;"> effects </td>
  </tr>
  <tr>
   <td style="text-align:left;"> tidy </td>
   <td style="text-align:left;"> ess </td>
  </tr>
  <tr>
   <td style="text-align:left;"> tidy </td>
   <td style="text-align:left;"> estimate.method </td>
  </tr>
  <tr>
   <td style="text-align:left;"> tidy </td>
   <td style="text-align:left;"> exponentiate </td>
  </tr>
  <tr>
   <td style="text-align:left;"> tidy </td>
   <td style="text-align:left;"> fe </td>
  </tr>
  <tr>
   <td style="text-align:left;"> tidy </td>
   <td style="text-align:left;"> include_studies </td>
  </tr>
  <tr>
   <td style="text-align:left;"> tidy </td>
   <td style="text-align:left;"> instruments </td>
  </tr>
  <tr>
   <td style="text-align:left;"> tidy </td>
   <td style="text-align:left;"> intervals </td>
  </tr>
  <tr>
   <td style="text-align:left;"> tidy </td>
   <td style="text-align:left;"> matrix </td>
  </tr>
  <tr>
   <td style="text-align:left;"> tidy </td>
   <td style="text-align:left;"> measure </td>
  </tr>
  <tr>
   <td style="text-align:left;"> tidy </td>
   <td style="text-align:left;"> na.rm </td>
  </tr>
  <tr>
   <td style="text-align:left;"> tidy </td>
   <td style="text-align:left;"> object </td>
  </tr>
  <tr>
   <td style="text-align:left;"> tidy </td>
   <td style="text-align:left;"> p.values </td>
  </tr>
  <tr>
   <td style="text-align:left;"> tidy </td>
   <td style="text-align:left;"> par_type </td>
  </tr>
  <tr>
   <td style="text-align:left;"> tidy </td>
   <td style="text-align:left;"> parameters </td>
  </tr>
  <tr>
   <td style="text-align:left;"> tidy </td>
   <td style="text-align:left;"> parametric </td>
  </tr>
  <tr>
   <td style="text-align:left;"> tidy </td>
   <td style="text-align:left;"> pars </td>
  </tr>
  <tr>
   <td style="text-align:left;"> tidy </td>
   <td style="text-align:left;"> prob </td>
  </tr>
  <tr>
   <td style="text-align:left;"> tidy </td>
   <td style="text-align:left;"> region </td>
  </tr>
  <tr>
   <td style="text-align:left;"> tidy </td>
   <td style="text-align:left;"> return_zeros </td>
  </tr>
  <tr>
   <td style="text-align:left;"> tidy </td>
   <td style="text-align:left;"> rhat </td>
  </tr>
  <tr>
   <td style="text-align:left;"> tidy </td>
   <td style="text-align:left;"> robust </td>
  </tr>
  <tr>
   <td style="text-align:left;"> tidy </td>
   <td style="text-align:left;"> scales </td>
  </tr>
  <tr>
   <td style="text-align:left;"> tidy </td>
   <td style="text-align:left;"> se.type </td>
  </tr>
  <tr>
   <td style="text-align:left;"> tidy </td>
   <td style="text-align:left;"> strata </td>
  </tr>
  <tr>
   <td style="text-align:left;"> tidy </td>
   <td style="text-align:left;"> test </td>
  </tr>
  <tr>
   <td style="text-align:left;"> tidy </td>
   <td style="text-align:left;"> trim </td>
  </tr>
  <tr>
   <td style="text-align:left;"> tidy </td>
   <td style="text-align:left;"> upper </td>
  </tr>
  <tr>
   <td style="text-align:left;"> glance </td>
   <td style="text-align:left;"> deviance </td>
  </tr>
  <tr>
   <td style="text-align:left;"> glance </td>
   <td style="text-align:left;"> diagnostics </td>
  </tr>
  <tr>
   <td style="text-align:left;"> glance </td>
   <td style="text-align:left;"> looic </td>
  </tr>
  <tr>
   <td style="text-align:left;"> glance </td>
   <td style="text-align:left;"> mcmc </td>
  </tr>
  <tr>
   <td style="text-align:left;"> glance </td>
   <td style="text-align:left;"> test </td>
  </tr>
  <tr>
   <td style="text-align:left;"> glance </td>
   <td style="text-align:left;"> x </td>
  </tr>
  <tr>
   <td style="text-align:left;"> augment </td>
   <td style="text-align:left;"> data </td>
  </tr>
  <tr>
   <td style="text-align:left;"> augment </td>
   <td style="text-align:left;"> interval </td>
  </tr>
  <tr>
   <td style="text-align:left;"> augment </td>
   <td style="text-align:left;"> newdata </td>
  </tr>
  <tr>
   <td style="text-align:left;"> augment </td>
   <td style="text-align:left;"> se_fit </td>
  </tr>
  <tr>
   <td style="text-align:left;"> augment </td>
   <td style="text-align:left;"> type.predict </td>
  </tr>
  <tr>
   <td style="text-align:left;"> augment </td>
   <td style="text-align:left;"> type.residuals </td>
  </tr>
  <tr>
   <td style="text-align:left;"> augment </td>
   <td style="text-align:left;"> weights </td>
  </tr>
</tbody>
</table>

The currently acceptable column names by tidier method are:

<table class="table" style="width: auto !important; margin-left: auto; margin-right: auto;">
 <thead>
  <tr>
   <th style="text-align:left;"> Method </th>
   <th style="text-align:left;"> Column </th>
  </tr>
 </thead>
<tbody>
  <tr>
   <td style="text-align:left;"> tidy </td>
   <td style="text-align:left;"> acf </td>
  </tr>
  <tr>
   <td style="text-align:left;"> tidy </td>
   <td style="text-align:left;"> adj.p.value </td>
  </tr>
  <tr>
   <td style="text-align:left;"> tidy </td>
   <td style="text-align:left;"> alternative </td>
  </tr>
  <tr>
   <td style="text-align:left;"> tidy </td>
   <td style="text-align:left;"> at.value </td>
  </tr>
  <tr>
   <td style="text-align:left;"> tidy </td>
   <td style="text-align:left;"> at.variable </td>
  </tr>
  <tr>
   <td style="text-align:left;"> tidy </td>
   <td style="text-align:left;"> atmean </td>
  </tr>
  <tr>
   <td style="text-align:left;"> tidy </td>
   <td style="text-align:left;"> autocorrelation </td>
  </tr>
  <tr>
   <td style="text-align:left;"> tidy </td>
   <td style="text-align:left;"> bias </td>
  </tr>
  <tr>
   <td style="text-align:left;"> tidy </td>
   <td style="text-align:left;"> ci.width </td>
  </tr>
  <tr>
   <td style="text-align:left;"> tidy </td>
   <td style="text-align:left;"> class </td>
  </tr>
  <tr>
   <td style="text-align:left;"> tidy </td>
   <td style="text-align:left;"> cluster </td>
  </tr>
  <tr>
   <td style="text-align:left;"> tidy </td>
   <td style="text-align:left;"> coef.type </td>
  </tr>
  <tr>
   <td style="text-align:left;"> tidy </td>
   <td style="text-align:left;"> column1 </td>
  </tr>
  <tr>
   <td style="text-align:left;"> tidy </td>
   <td style="text-align:left;"> column2 </td>
  </tr>
  <tr>
   <td style="text-align:left;"> tidy </td>
   <td style="text-align:left;"> comp </td>
  </tr>
  <tr>
   <td style="text-align:left;"> tidy </td>
   <td style="text-align:left;"> comparison </td>
  </tr>
  <tr>
   <td style="text-align:left;"> tidy </td>
   <td style="text-align:left;"> component </td>
  </tr>
  <tr>
   <td style="text-align:left;"> tidy </td>
   <td style="text-align:left;"> conf.high </td>
  </tr>
  <tr>
   <td style="text-align:left;"> tidy </td>
   <td style="text-align:left;"> conf.low </td>
  </tr>
  <tr>
   <td style="text-align:left;"> tidy </td>
   <td style="text-align:left;"> contrast </td>
  </tr>
  <tr>
   <td style="text-align:left;"> tidy </td>
   <td style="text-align:left;"> cumulative </td>
  </tr>
  <tr>
   <td style="text-align:left;"> tidy </td>
   <td style="text-align:left;"> cutoff </td>
  </tr>
  <tr>
   <td style="text-align:left;"> tidy </td>
   <td style="text-align:left;"> delta </td>
  </tr>
  <tr>
   <td style="text-align:left;"> tidy </td>
   <td style="text-align:left;"> den.df </td>
  </tr>
  <tr>
   <td style="text-align:left;"> tidy </td>
   <td style="text-align:left;"> denominator </td>
  </tr>
  <tr>
   <td style="text-align:left;"> tidy </td>
   <td style="text-align:left;"> dev.ratio </td>
  </tr>
  <tr>
   <td style="text-align:left;"> tidy </td>
   <td style="text-align:left;"> df </td>
  </tr>
  <tr>
   <td style="text-align:left;"> tidy </td>
   <td style="text-align:left;"> distance </td>
  </tr>
  <tr>
   <td style="text-align:left;"> tidy </td>
   <td style="text-align:left;"> estimate </td>
  </tr>
  <tr>
   <td style="text-align:left;"> tidy </td>
   <td style="text-align:left;"> estimate1 </td>
  </tr>
  <tr>
   <td style="text-align:left;"> tidy </td>
   <td style="text-align:left;"> estimate2 </td>
  </tr>
  <tr>
   <td style="text-align:left;"> tidy </td>
   <td style="text-align:left;"> event </td>
  </tr>
  <tr>
   <td style="text-align:left;"> tidy </td>
   <td style="text-align:left;"> exp </td>
  </tr>
  <tr>
   <td style="text-align:left;"> tidy </td>
   <td style="text-align:left;"> expected </td>
  </tr>
  <tr>
   <td style="text-align:left;"> tidy </td>
   <td style="text-align:left;"> fpr </td>
  </tr>
  <tr>
   <td style="text-align:left;"> tidy </td>
   <td style="text-align:left;"> freq </td>
  </tr>
  <tr>
   <td style="text-align:left;"> tidy </td>
   <td style="text-align:left;"> GCV </td>
  </tr>
  <tr>
   <td style="text-align:left;"> tidy </td>
   <td style="text-align:left;"> group </td>
  </tr>
  <tr>
   <td style="text-align:left;"> tidy </td>
   <td style="text-align:left;"> group1 </td>
  </tr>
  <tr>
   <td style="text-align:left;"> tidy </td>
   <td style="text-align:left;"> group2 </td>
  </tr>
  <tr>
   <td style="text-align:left;"> tidy </td>
   <td style="text-align:left;"> index </td>
  </tr>
  <tr>
   <td style="text-align:left;"> tidy </td>
   <td style="text-align:left;"> item1 </td>
  </tr>
  <tr>
   <td style="text-align:left;"> tidy </td>
   <td style="text-align:left;"> item2 </td>
  </tr>
  <tr>
   <td style="text-align:left;"> tidy </td>
   <td style="text-align:left;"> kendall_score </td>
  </tr>
  <tr>
   <td style="text-align:left;"> tidy </td>
   <td style="text-align:left;"> lag </td>
  </tr>
  <tr>
   <td style="text-align:left;"> tidy </td>
   <td style="text-align:left;"> lambda </td>
  </tr>
  <tr>
   <td style="text-align:left;"> tidy </td>
   <td style="text-align:left;"> letters </td>
  </tr>
  <tr>
   <td style="text-align:left;"> tidy </td>
   <td style="text-align:left;"> lhs </td>
  </tr>
  <tr>
   <td style="text-align:left;"> tidy </td>
   <td style="text-align:left;"> logLik </td>
  </tr>
  <tr>
   <td style="text-align:left;"> tidy </td>
   <td style="text-align:left;"> mcmc.error </td>
  </tr>
  <tr>
   <td style="text-align:left;"> tidy </td>
   <td style="text-align:left;"> mean </td>
  </tr>
  <tr>
   <td style="text-align:left;"> tidy </td>
   <td style="text-align:left;"> meansq </td>
  </tr>
  <tr>
   <td style="text-align:left;"> tidy </td>
   <td style="text-align:left;"> method </td>
  </tr>
  <tr>
   <td style="text-align:left;"> tidy </td>
   <td style="text-align:left;"> n </td>
  </tr>
  <tr>
   <td style="text-align:left;"> tidy </td>
   <td style="text-align:left;"> N </td>
  </tr>
  <tr>
   <td style="text-align:left;"> tidy </td>
   <td style="text-align:left;"> n.censor </td>
  </tr>
  <tr>
   <td style="text-align:left;"> tidy </td>
   <td style="text-align:left;"> n.event </td>
  </tr>
  <tr>
   <td style="text-align:left;"> tidy </td>
   <td style="text-align:left;"> n.risk </td>
  </tr>
  <tr>
   <td style="text-align:left;"> tidy </td>
   <td style="text-align:left;"> null.value </td>
  </tr>
  <tr>
   <td style="text-align:left;"> tidy </td>
   <td style="text-align:left;"> num.df </td>
  </tr>
  <tr>
   <td style="text-align:left;"> tidy </td>
   <td style="text-align:left;"> nzero </td>
  </tr>
  <tr>
   <td style="text-align:left;"> tidy </td>
   <td style="text-align:left;"> obs </td>
  </tr>
  <tr>
   <td style="text-align:left;"> tidy </td>
   <td style="text-align:left;"> op </td>
  </tr>
  <tr>
   <td style="text-align:left;"> tidy </td>
   <td style="text-align:left;"> outcome </td>
  </tr>
  <tr>
   <td style="text-align:left;"> tidy </td>
   <td style="text-align:left;"> p </td>
  </tr>
  <tr>
   <td style="text-align:left;"> tidy </td>
   <td style="text-align:left;"> p.value </td>
  </tr>
  <tr>
   <td style="text-align:left;"> tidy </td>
   <td style="text-align:left;"> p.value.Sargan </td>
  </tr>
  <tr>
   <td style="text-align:left;"> tidy </td>
   <td style="text-align:left;"> p.value.weakinst </td>
  </tr>
  <tr>
   <td style="text-align:left;"> tidy </td>
   <td style="text-align:left;"> p.value.Wu.Hausman </td>
  </tr>
  <tr>
   <td style="text-align:left;"> tidy </td>
   <td style="text-align:left;"> parameter </td>
  </tr>
  <tr>
   <td style="text-align:left;"> tidy </td>
   <td style="text-align:left;"> PC </td>
  </tr>
  <tr>
   <td style="text-align:left;"> tidy </td>
   <td style="text-align:left;"> percent </td>
  </tr>
  <tr>
   <td style="text-align:left;"> tidy </td>
   <td style="text-align:left;"> power </td>
  </tr>
  <tr>
   <td style="text-align:left;"> tidy </td>
   <td style="text-align:left;"> proportion </td>
  </tr>
  <tr>
   <td style="text-align:left;"> tidy </td>
   <td style="text-align:left;"> pyears </td>
  </tr>
  <tr>
   <td style="text-align:left;"> tidy </td>
   <td style="text-align:left;"> response </td>
  </tr>
  <tr>
   <td style="text-align:left;"> tidy </td>
   <td style="text-align:left;"> rhs </td>
  </tr>
  <tr>
   <td style="text-align:left;"> tidy </td>
   <td style="text-align:left;"> robust.se </td>
  </tr>
  <tr>
   <td style="text-align:left;"> tidy </td>
   <td style="text-align:left;"> row </td>
  </tr>
  <tr>
   <td style="text-align:left;"> tidy </td>
   <td style="text-align:left;"> scale </td>
  </tr>
  <tr>
   <td style="text-align:left;"> tidy </td>
   <td style="text-align:left;"> sd </td>
  </tr>
  <tr>
   <td style="text-align:left;"> tidy </td>
   <td style="text-align:left;"> series </td>
  </tr>
  <tr>
   <td style="text-align:left;"> tidy </td>
   <td style="text-align:left;"> sig.level </td>
  </tr>
  <tr>
   <td style="text-align:left;"> tidy </td>
   <td style="text-align:left;"> size </td>
  </tr>
  <tr>
   <td style="text-align:left;"> tidy </td>
   <td style="text-align:left;"> spec </td>
  </tr>
  <tr>
   <td style="text-align:left;"> tidy </td>
   <td style="text-align:left;"> state </td>
  </tr>
  <tr>
   <td style="text-align:left;"> tidy </td>
   <td style="text-align:left;"> statistic </td>
  </tr>
  <tr>
   <td style="text-align:left;"> tidy </td>
   <td style="text-align:left;"> statistic.Sargan </td>
  </tr>
  <tr>
   <td style="text-align:left;"> tidy </td>
   <td style="text-align:left;"> statistic.weakinst </td>
  </tr>
  <tr>
   <td style="text-align:left;"> tidy </td>
   <td style="text-align:left;"> statistic.Wu.Hausman </td>
  </tr>
  <tr>
   <td style="text-align:left;"> tidy </td>
   <td style="text-align:left;"> std_estimate </td>
  </tr>
  <tr>
   <td style="text-align:left;"> tidy </td>
   <td style="text-align:left;"> std.all </td>
  </tr>
  <tr>
   <td style="text-align:left;"> tidy </td>
   <td style="text-align:left;"> std.dev </td>
  </tr>
  <tr>
   <td style="text-align:left;"> tidy </td>
   <td style="text-align:left;"> std.error </td>
  </tr>
  <tr>
   <td style="text-align:left;"> tidy </td>
   <td style="text-align:left;"> std.lv </td>
  </tr>
  <tr>
   <td style="text-align:left;"> tidy </td>
   <td style="text-align:left;"> std.nox </td>
  </tr>
  <tr>
   <td style="text-align:left;"> tidy </td>
   <td style="text-align:left;"> step </td>
  </tr>
  <tr>
   <td style="text-align:left;"> tidy </td>
   <td style="text-align:left;"> strata </td>
  </tr>
  <tr>
   <td style="text-align:left;"> tidy </td>
   <td style="text-align:left;"> stratum </td>
  </tr>
  <tr>
   <td style="text-align:left;"> tidy </td>
   <td style="text-align:left;"> study </td>
  </tr>
  <tr>
   <td style="text-align:left;"> tidy </td>
   <td style="text-align:left;"> sumsq </td>
  </tr>
  <tr>
   <td style="text-align:left;"> tidy </td>
   <td style="text-align:left;"> tau </td>
  </tr>
  <tr>
   <td style="text-align:left;"> tidy </td>
   <td style="text-align:left;"> term </td>
  </tr>
  <tr>
   <td style="text-align:left;"> tidy </td>
   <td style="text-align:left;"> time </td>
  </tr>
  <tr>
   <td style="text-align:left;"> tidy </td>
   <td style="text-align:left;"> tpr </td>
  </tr>
  <tr>
   <td style="text-align:left;"> tidy </td>
   <td style="text-align:left;"> type </td>
  </tr>
  <tr>
   <td style="text-align:left;"> tidy </td>
   <td style="text-align:left;"> uniqueness </td>
  </tr>
  <tr>
   <td style="text-align:left;"> tidy </td>
   <td style="text-align:left;"> value </td>
  </tr>
  <tr>
   <td style="text-align:left;"> tidy </td>
   <td style="text-align:left;"> var_kendall_score </td>
  </tr>
  <tr>
   <td style="text-align:left;"> tidy </td>
   <td style="text-align:left;"> variable </td>
  </tr>
  <tr>
   <td style="text-align:left;"> tidy </td>
   <td style="text-align:left;"> variance </td>
  </tr>
  <tr>
   <td style="text-align:left;"> tidy </td>
   <td style="text-align:left;"> withinss </td>
  </tr>
  <tr>
   <td style="text-align:left;"> tidy </td>
   <td style="text-align:left;"> y.level </td>
  </tr>
  <tr>
   <td style="text-align:left;"> tidy </td>
   <td style="text-align:left;"> y.value </td>
  </tr>
  <tr>
   <td style="text-align:left;"> tidy </td>
   <td style="text-align:left;"> z </td>
  </tr>
  <tr>
   <td style="text-align:left;"> glance </td>
   <td style="text-align:left;"> adj.r.squared </td>
  </tr>
  <tr>
   <td style="text-align:left;"> glance </td>
   <td style="text-align:left;"> agfi </td>
  </tr>
  <tr>
   <td style="text-align:left;"> glance </td>
   <td style="text-align:left;"> AIC </td>
  </tr>
  <tr>
   <td style="text-align:left;"> glance </td>
   <td style="text-align:left;"> AICc </td>
  </tr>
  <tr>
   <td style="text-align:left;"> glance </td>
   <td style="text-align:left;"> alpha </td>
  </tr>
  <tr>
   <td style="text-align:left;"> glance </td>
   <td style="text-align:left;"> alternative </td>
  </tr>
  <tr>
   <td style="text-align:left;"> glance </td>
   <td style="text-align:left;"> autocorrelation </td>
  </tr>
  <tr>
   <td style="text-align:left;"> glance </td>
   <td style="text-align:left;"> avg.silhouette.width </td>
  </tr>
  <tr>
   <td style="text-align:left;"> glance </td>
   <td style="text-align:left;"> betweenss </td>
  </tr>
  <tr>
   <td style="text-align:left;"> glance </td>
   <td style="text-align:left;"> BIC </td>
  </tr>
  <tr>
   <td style="text-align:left;"> glance </td>
   <td style="text-align:left;"> cfi </td>
  </tr>
  <tr>
   <td style="text-align:left;"> glance </td>
   <td style="text-align:left;"> chi.squared </td>
  </tr>
  <tr>
   <td style="text-align:left;"> glance </td>
   <td style="text-align:left;"> chisq </td>
  </tr>
  <tr>
   <td style="text-align:left;"> glance </td>
   <td style="text-align:left;"> cochran.qe </td>
  </tr>
  <tr>
   <td style="text-align:left;"> glance </td>
   <td style="text-align:left;"> cochran.qm </td>
  </tr>
  <tr>
   <td style="text-align:left;"> glance </td>
   <td style="text-align:left;"> conf.high </td>
  </tr>
  <tr>
   <td style="text-align:left;"> glance </td>
   <td style="text-align:left;"> conf.low </td>
  </tr>
  <tr>
   <td style="text-align:left;"> glance </td>
   <td style="text-align:left;"> converged </td>
  </tr>
  <tr>
   <td style="text-align:left;"> glance </td>
   <td style="text-align:left;"> convergence </td>
  </tr>
  <tr>
   <td style="text-align:left;"> glance </td>
   <td style="text-align:left;"> crit </td>
  </tr>
  <tr>
   <td style="text-align:left;"> glance </td>
   <td style="text-align:left;"> cv.crit </td>
  </tr>
  <tr>
   <td style="text-align:left;"> glance </td>
   <td style="text-align:left;"> den.df </td>
  </tr>
  <tr>
   <td style="text-align:left;"> glance </td>
   <td style="text-align:left;"> deviance </td>
  </tr>
  <tr>
   <td style="text-align:left;"> glance </td>
   <td style="text-align:left;"> df </td>
  </tr>
  <tr>
   <td style="text-align:left;"> glance </td>
   <td style="text-align:left;"> df.null </td>
  </tr>
  <tr>
   <td style="text-align:left;"> glance </td>
   <td style="text-align:left;"> df.residual </td>
  </tr>
  <tr>
   <td style="text-align:left;"> glance </td>
   <td style="text-align:left;"> dw.original </td>
  </tr>
  <tr>
   <td style="text-align:left;"> glance </td>
   <td style="text-align:left;"> dw.transformed </td>
  </tr>
  <tr>
   <td style="text-align:left;"> glance </td>
   <td style="text-align:left;"> edf </td>
  </tr>
  <tr>
   <td style="text-align:left;"> glance </td>
   <td style="text-align:left;"> estimator </td>
  </tr>
  <tr>
   <td style="text-align:left;"> glance </td>
   <td style="text-align:left;"> events </td>
  </tr>
  <tr>
   <td style="text-align:left;"> glance </td>
   <td style="text-align:left;"> finTol </td>
  </tr>
  <tr>
   <td style="text-align:left;"> glance </td>
   <td style="text-align:left;"> function.count </td>
  </tr>
  <tr>
   <td style="text-align:left;"> glance </td>
   <td style="text-align:left;"> G </td>
  </tr>
  <tr>
   <td style="text-align:left;"> glance </td>
   <td style="text-align:left;"> g.squared </td>
  </tr>
  <tr>
   <td style="text-align:left;"> glance </td>
   <td style="text-align:left;"> gamma </td>
  </tr>
  <tr>
   <td style="text-align:left;"> glance </td>
   <td style="text-align:left;"> gradient.count </td>
  </tr>
  <tr>
   <td style="text-align:left;"> glance </td>
   <td style="text-align:left;"> H </td>
  </tr>
  <tr>
   <td style="text-align:left;"> glance </td>
   <td style="text-align:left;"> h.squared </td>
  </tr>
  <tr>
   <td style="text-align:left;"> glance </td>
   <td style="text-align:left;"> hypvol </td>
  </tr>
  <tr>
   <td style="text-align:left;"> glance </td>
   <td style="text-align:left;"> i.squared </td>
  </tr>
  <tr>
   <td style="text-align:left;"> glance </td>
   <td style="text-align:left;"> independence </td>
  </tr>
  <tr>
   <td style="text-align:left;"> glance </td>
   <td style="text-align:left;"> isConv </td>
  </tr>
  <tr>
   <td style="text-align:left;"> glance </td>
   <td style="text-align:left;"> iter </td>
  </tr>
  <tr>
   <td style="text-align:left;"> glance </td>
   <td style="text-align:left;"> iterations </td>
  </tr>
  <tr>
   <td style="text-align:left;"> glance </td>
   <td style="text-align:left;"> kHKB </td>
  </tr>
  <tr>
   <td style="text-align:left;"> glance </td>
   <td style="text-align:left;"> kLW </td>
  </tr>
  <tr>
   <td style="text-align:left;"> glance </td>
   <td style="text-align:left;"> lambda </td>
  </tr>
  <tr>
   <td style="text-align:left;"> glance </td>
   <td style="text-align:left;"> lambda.1se </td>
  </tr>
  <tr>
   <td style="text-align:left;"> glance </td>
   <td style="text-align:left;"> lambda.min </td>
  </tr>
  <tr>
   <td style="text-align:left;"> glance </td>
   <td style="text-align:left;"> lambdaGCV </td>
  </tr>
  <tr>
   <td style="text-align:left;"> glance </td>
   <td style="text-align:left;"> logLik </td>
  </tr>
  <tr>
   <td style="text-align:left;"> glance </td>
   <td style="text-align:left;"> max.cluster.size </td>
  </tr>
  <tr>
   <td style="text-align:left;"> glance </td>
   <td style="text-align:left;"> max.hazard </td>
  </tr>
  <tr>
   <td style="text-align:left;"> glance </td>
   <td style="text-align:left;"> max.time </td>
  </tr>
  <tr>
   <td style="text-align:left;"> glance </td>
   <td style="text-align:left;"> maxit </td>
  </tr>
  <tr>
   <td style="text-align:left;"> glance </td>
   <td style="text-align:left;"> MCMC.burnin </td>
  </tr>
  <tr>
   <td style="text-align:left;"> glance </td>
   <td style="text-align:left;"> MCMC.interval </td>
  </tr>
  <tr>
   <td style="text-align:left;"> glance </td>
   <td style="text-align:left;"> MCMC.samplesize </td>
  </tr>
  <tr>
   <td style="text-align:left;"> glance </td>
   <td style="text-align:left;"> measure </td>
  </tr>
  <tr>
   <td style="text-align:left;"> glance </td>
   <td style="text-align:left;"> median </td>
  </tr>
  <tr>
   <td style="text-align:left;"> glance </td>
   <td style="text-align:left;"> method </td>
  </tr>
  <tr>
   <td style="text-align:left;"> glance </td>
   <td style="text-align:left;"> min.hazard </td>
  </tr>
  <tr>
   <td style="text-align:left;"> glance </td>
   <td style="text-align:left;"> min.time </td>
  </tr>
  <tr>
   <td style="text-align:left;"> glance </td>
   <td style="text-align:left;"> missing_method </td>
  </tr>
  <tr>
   <td style="text-align:left;"> glance </td>
   <td style="text-align:left;"> model </td>
  </tr>
  <tr>
   <td style="text-align:left;"> glance </td>
   <td style="text-align:left;"> n </td>
  </tr>
  <tr>
   <td style="text-align:left;"> glance </td>
   <td style="text-align:left;"> n.clusters </td>
  </tr>
  <tr>
   <td style="text-align:left;"> glance </td>
   <td style="text-align:left;"> n.factors </td>
  </tr>
  <tr>
   <td style="text-align:left;"> glance </td>
   <td style="text-align:left;"> n.max </td>
  </tr>
  <tr>
   <td style="text-align:left;"> glance </td>
   <td style="text-align:left;"> n.start </td>
  </tr>
  <tr>
   <td style="text-align:left;"> glance </td>
   <td style="text-align:left;"> nevent </td>
  </tr>
  <tr>
   <td style="text-align:left;"> glance </td>
   <td style="text-align:left;"> nexcluded </td>
  </tr>
  <tr>
   <td style="text-align:left;"> glance </td>
   <td style="text-align:left;"> ngroups </td>
  </tr>
  <tr>
   <td style="text-align:left;"> glance </td>
   <td style="text-align:left;"> nobs </td>
  </tr>
  <tr>
   <td style="text-align:left;"> glance </td>
   <td style="text-align:left;"> norig </td>
  </tr>
  <tr>
   <td style="text-align:left;"> glance </td>
   <td style="text-align:left;"> npar </td>
  </tr>
  <tr>
   <td style="text-align:left;"> glance </td>
   <td style="text-align:left;"> npasses </td>
  </tr>
  <tr>
   <td style="text-align:left;"> glance </td>
   <td style="text-align:left;"> null.deviance </td>
  </tr>
  <tr>
   <td style="text-align:left;"> glance </td>
   <td style="text-align:left;"> nulldev </td>
  </tr>
  <tr>
   <td style="text-align:left;"> glance </td>
   <td style="text-align:left;"> num.df </td>
  </tr>
  <tr>
   <td style="text-align:left;"> glance </td>
   <td style="text-align:left;"> number.interaction </td>
  </tr>
  <tr>
   <td style="text-align:left;"> glance </td>
   <td style="text-align:left;"> offtable </td>
  </tr>
  <tr>
   <td style="text-align:left;"> glance </td>
   <td style="text-align:left;"> p.value </td>
  </tr>
  <tr>
   <td style="text-align:left;"> glance </td>
   <td style="text-align:left;"> p.value.cochran.qe </td>
  </tr>
  <tr>
   <td style="text-align:left;"> glance </td>
   <td style="text-align:left;"> p.value.cochran.qm </td>
  </tr>
  <tr>
   <td style="text-align:left;"> glance </td>
   <td style="text-align:left;"> p.value.original </td>
  </tr>
  <tr>
   <td style="text-align:left;"> glance </td>
   <td style="text-align:left;"> p.value.Sargan </td>
  </tr>
  <tr>
   <td style="text-align:left;"> glance </td>
   <td style="text-align:left;"> p.value.transformed </td>
  </tr>
  <tr>
   <td style="text-align:left;"> glance </td>
   <td style="text-align:left;"> p.value.weak.instr </td>
  </tr>
  <tr>
   <td style="text-align:left;"> glance </td>
   <td style="text-align:left;"> p.value.Wu.Hausman </td>
  </tr>
  <tr>
   <td style="text-align:left;"> glance </td>
   <td style="text-align:left;"> parameter </td>
  </tr>
  <tr>
   <td style="text-align:left;"> glance </td>
   <td style="text-align:left;"> pen.crit </td>
  </tr>
  <tr>
   <td style="text-align:left;"> glance </td>
   <td style="text-align:left;"> power </td>
  </tr>
  <tr>
   <td style="text-align:left;"> glance </td>
   <td style="text-align:left;"> power.reached </td>
  </tr>
  <tr>
   <td style="text-align:left;"> glance </td>
   <td style="text-align:left;"> pseudo.r.squared </td>
  </tr>
  <tr>
   <td style="text-align:left;"> glance </td>
   <td style="text-align:left;"> r.squared </td>
  </tr>
  <tr>
   <td style="text-align:left;"> glance </td>
   <td style="text-align:left;"> records </td>
  </tr>
  <tr>
   <td style="text-align:left;"> glance </td>
   <td style="text-align:left;"> residual.deviance </td>
  </tr>
  <tr>
   <td style="text-align:left;"> glance </td>
   <td style="text-align:left;"> rho </td>
  </tr>
  <tr>
   <td style="text-align:left;"> glance </td>
   <td style="text-align:left;"> rho2 </td>
  </tr>
  <tr>
   <td style="text-align:left;"> glance </td>
   <td style="text-align:left;"> rho20 </td>
  </tr>
  <tr>
   <td style="text-align:left;"> glance </td>
   <td style="text-align:left;"> rmean </td>
  </tr>
  <tr>
   <td style="text-align:left;"> glance </td>
   <td style="text-align:left;"> rmean.std.error </td>
  </tr>
  <tr>
   <td style="text-align:left;"> glance </td>
   <td style="text-align:left;"> rmsea </td>
  </tr>
  <tr>
   <td style="text-align:left;"> glance </td>
   <td style="text-align:left;"> rmsea.conf.high </td>
  </tr>
  <tr>
   <td style="text-align:left;"> glance </td>
   <td style="text-align:left;"> rscore </td>
  </tr>
  <tr>
   <td style="text-align:left;"> glance </td>
   <td style="text-align:left;"> score </td>
  </tr>
  <tr>
   <td style="text-align:left;"> glance </td>
   <td style="text-align:left;"> sigma </td>
  </tr>
  <tr>
   <td style="text-align:left;"> glance </td>
   <td style="text-align:left;"> sigma2_j </td>
  </tr>
  <tr>
   <td style="text-align:left;"> glance </td>
   <td style="text-align:left;"> spar </td>
  </tr>
  <tr>
   <td style="text-align:left;"> glance </td>
   <td style="text-align:left;"> srmr </td>
  </tr>
  <tr>
   <td style="text-align:left;"> glance </td>
   <td style="text-align:left;"> statistic </td>
  </tr>
  <tr>
   <td style="text-align:left;"> glance </td>
   <td style="text-align:left;"> statistic.Sargan </td>
  </tr>
  <tr>
   <td style="text-align:left;"> glance </td>
   <td style="text-align:left;"> statistic.weak.instr </td>
  </tr>
  <tr>
   <td style="text-align:left;"> glance </td>
   <td style="text-align:left;"> statistic.Wu.Hausman </td>
  </tr>
  <tr>
   <td style="text-align:left;"> glance </td>
   <td style="text-align:left;"> tau </td>
  </tr>
  <tr>
   <td style="text-align:left;"> glance </td>
   <td style="text-align:left;"> tau.squared </td>
  </tr>
  <tr>
   <td style="text-align:left;"> glance </td>
   <td style="text-align:left;"> tau.squared.se </td>
  </tr>
  <tr>
   <td style="text-align:left;"> glance </td>
   <td style="text-align:left;"> theta </td>
  </tr>
  <tr>
   <td style="text-align:left;"> glance </td>
   <td style="text-align:left;"> timepoints </td>
  </tr>
  <tr>
   <td style="text-align:left;"> glance </td>
   <td style="text-align:left;"> tli </td>
  </tr>
  <tr>
   <td style="text-align:left;"> glance </td>
   <td style="text-align:left;"> tot.withinss </td>
  </tr>
  <tr>
   <td style="text-align:left;"> glance </td>
   <td style="text-align:left;"> total </td>
  </tr>
  <tr>
   <td style="text-align:left;"> glance </td>
   <td style="text-align:left;"> total.variance </td>
  </tr>
  <tr>
   <td style="text-align:left;"> glance </td>
   <td style="text-align:left;"> totss </td>
  </tr>
  <tr>
   <td style="text-align:left;"> glance </td>
   <td style="text-align:left;"> value </td>
  </tr>
  <tr>
   <td style="text-align:left;"> glance </td>
   <td style="text-align:left;"> within.r.squared </td>
  </tr>
  <tr>
   <td style="text-align:left;"> augment </td>
   <td style="text-align:left;"> .class </td>
  </tr>
  <tr>
   <td style="text-align:left;"> augment </td>
   <td style="text-align:left;"> .cluster </td>
  </tr>
  <tr>
   <td style="text-align:left;"> augment </td>
   <td style="text-align:left;"> .cochran.qe.loo </td>
  </tr>
  <tr>
   <td style="text-align:left;"> augment </td>
   <td style="text-align:left;"> .col.prop </td>
  </tr>
  <tr>
   <td style="text-align:left;"> augment </td>
   <td style="text-align:left;"> .conf.high </td>
  </tr>
  <tr>
   <td style="text-align:left;"> augment </td>
   <td style="text-align:left;"> .conf.low </td>
  </tr>
  <tr>
   <td style="text-align:left;"> augment </td>
   <td style="text-align:left;"> .cooksd </td>
  </tr>
  <tr>
   <td style="text-align:left;"> augment </td>
   <td style="text-align:left;"> .cov.ratio </td>
  </tr>
  <tr>
   <td style="text-align:left;"> augment </td>
   <td style="text-align:left;"> .cred.high </td>
  </tr>
  <tr>
   <td style="text-align:left;"> augment </td>
   <td style="text-align:left;"> .cred.low </td>
  </tr>
  <tr>
   <td style="text-align:left;"> augment </td>
   <td style="text-align:left;"> .dffits </td>
  </tr>
  <tr>
   <td style="text-align:left;"> augment </td>
   <td style="text-align:left;"> .expected </td>
  </tr>
  <tr>
   <td style="text-align:left;"> augment </td>
   <td style="text-align:left;"> .fitted </td>
  </tr>
  <tr>
   <td style="text-align:left;"> augment </td>
   <td style="text-align:left;"> .fitted_j_0 </td>
  </tr>
  <tr>
   <td style="text-align:left;"> augment </td>
   <td style="text-align:left;"> .fitted_j_1 </td>
  </tr>
  <tr>
   <td style="text-align:left;"> augment </td>
   <td style="text-align:left;"> .hat </td>
  </tr>
  <tr>
   <td style="text-align:left;"> augment </td>
   <td style="text-align:left;"> .lower </td>
  </tr>
  <tr>
   <td style="text-align:left;"> augment </td>
   <td style="text-align:left;"> .moderator </td>
  </tr>
  <tr>
   <td style="text-align:left;"> augment </td>
   <td style="text-align:left;"> .moderator.level </td>
  </tr>
  <tr>
   <td style="text-align:left;"> augment </td>
   <td style="text-align:left;"> .observed </td>
  </tr>
  <tr>
   <td style="text-align:left;"> augment </td>
   <td style="text-align:left;"> .probability </td>
  </tr>
  <tr>
   <td style="text-align:left;"> augment </td>
   <td style="text-align:left;"> .prop </td>
  </tr>
  <tr>
   <td style="text-align:left;"> augment </td>
   <td style="text-align:left;"> .remainder </td>
  </tr>
  <tr>
   <td style="text-align:left;"> augment </td>
   <td style="text-align:left;"> .resid </td>
  </tr>
  <tr>
   <td style="text-align:left;"> augment </td>
   <td style="text-align:left;"> .resid_j_0 </td>
  </tr>
  <tr>
   <td style="text-align:left;"> augment </td>
   <td style="text-align:left;"> .resid_j_1 </td>
  </tr>
  <tr>
   <td style="text-align:left;"> augment </td>
   <td style="text-align:left;"> .row.prop </td>
  </tr>
  <tr>
   <td style="text-align:left;"> augment </td>
   <td style="text-align:left;"> .rownames </td>
  </tr>
  <tr>
   <td style="text-align:left;"> augment </td>
   <td style="text-align:left;"> .se.fit </td>
  </tr>
  <tr>
   <td style="text-align:left;"> augment </td>
   <td style="text-align:left;"> .seasadj </td>
  </tr>
  <tr>
   <td style="text-align:left;"> augment </td>
   <td style="text-align:left;"> .seasonal </td>
  </tr>
  <tr>
   <td style="text-align:left;"> augment </td>
   <td style="text-align:left;"> .sigma </td>
  </tr>
  <tr>
   <td style="text-align:left;"> augment </td>
   <td style="text-align:left;"> .std.resid </td>
  </tr>
  <tr>
   <td style="text-align:left;"> augment </td>
   <td style="text-align:left;"> .tau </td>
  </tr>
  <tr>
   <td style="text-align:left;"> augment </td>
   <td style="text-align:left;"> .tau.squared.loo </td>
  </tr>
  <tr>
   <td style="text-align:left;"> augment </td>
   <td style="text-align:left;"> .trend </td>
  </tr>
  <tr>
   <td style="text-align:left;"> augment </td>
   <td style="text-align:left;"> .uncertainty </td>
  </tr>
  <tr>
   <td style="text-align:left;"> augment </td>
   <td style="text-align:left;"> .upper </td>
  </tr>
  <tr>
   <td style="text-align:left;"> augment </td>
   <td style="text-align:left;"> .weight </td>
  </tr>
</tbody>
</table>

The [alexpghayes/modeltests](https://github.com/alexpghayes/modeltests) package provides unit testing infrastructure to check your new tidier methods. Please file an issue there to request new arguments/columns to be added to the glossaries!

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
#>  generics   * 0.1.0   2020-10-31 [1] CRAN (R 4.0.3)
#>  ggplot2    * 3.3.2   2020-06-19 [1] CRAN (R 4.0.0)
#>  infer      * 0.5.3   2020-07-14 [1] CRAN (R 4.0.0)
#>  parsnip    * 0.1.4   2020-10-27 [1] CRAN (R 4.0.2)
#>  purrr      * 0.3.4   2020-04-17 [1] CRAN (R 4.0.0)
#>  recipes    * 0.1.15  2020-11-11 [1] CRAN (R 4.0.2)
#>  rlang        0.4.9   2020-11-26 [1] CRAN (R 4.0.2)
#>  rsample    * 0.0.8   2020-09-23 [1] CRAN (R 4.0.2)
#>  tibble     * 3.0.4   2020-10-12 [1] CRAN (R 4.0.2)
#>  tidymodels * 0.1.2   2020-11-22 [1] CRAN (R 4.0.2)
#>  tidyverse  * 1.3.0   2019-11-21 [1] CRAN (R 4.0.0)
#>  tune       * 0.1.2   2020-11-17 [1] CRAN (R 4.0.3)
#>  workflows  * 0.2.1   2020-10-08 [1] CRAN (R 4.0.2)
#>  yardstick  * 0.0.7   2020-07-13 [1] CRAN (R 4.0.2)
#> 
#> [1] /Library/Frameworks/R.framework/Versions/4.0/Resources/library
```
