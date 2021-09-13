---
title: "Working with model coefficients"
tags: [parsnip,tune,broom,workflows]
categories: [model fitting]
type: learn-subsection
weight: 5
description: | 
  Create models that use coefficients, extract them from fitted models, and visualize them.
---



## Introduction 

There are many types of statistical models with diverse kinds of structure. Some models have coefficients (a.k.a. weights) for each term in the model. Familiar examples of such models are linear or logistic regression, but more complex models (e.g. neural networks, MARS) can also have model coefficients. When we work with models that use weights or coefficients, we often want to examine the estimated coefficients. 

This article describes how to retrieve the estimated coefficients from models fit using tidymodels. To use the code in this article, you will need to install the following packages: glmnet and tidymodels.

## Linear regression

Let's start with a linear regression model: 

`$$\hat{y} = \hat{\beta}_0 + \hat{\beta}_1x_1 + \ldots + \hat{\beta}_px_p$$` 

The `\(\beta\)` values are the coefficients and the `\(x_j\)` are model predictors, or features. 

Let's use the [Chicago train data](https://bookdown.org/max/FES/chicago-intro.html) where we predict the ridership at the Clark and Lake station (column name: `ridership`) with the previous ridership data 14 days prior at three of the stations. 

The data are in the modeldata package:  


```r
library(tidymodels)
tidymodels_prefer()
theme_set(theme_bw())

data(Chicago)

Chicago <- Chicago %>% select(ridership, Clark_Lake, Austin, Harlem)
```

### A single model

Let's start by fitting only a single parsnip model object. We'll create a model specification using `linear_reg()`. 

{{% note %}} The default engine is `"lm"` so no call to `set_engine()` is required. {{%/ note %}}

The `fit()` function estimates the model coefficients, given a formula and data set. 



```r
lm_spec <- linear_reg()
lm_fit <- fit(lm_spec, ridership ~ ., data = Chicago)
lm_fit
#> parsnip model object
#> 
#> Fit time:  3ms 
#> 
#> Call:
#> stats::lm(formula = ridership ~ ., data = data)
#> 
#> Coefficients:
#> (Intercept)   Clark_Lake       Austin       Harlem  
#>       1.678        0.904        0.612       -0.555
```

The best way to retrieve the fitted parameters is to use the `tidy()` method. This function, in the broom package, returns the coefficients and their associated statistics in a data frame with standardized column names: 


```r
tidy(lm_fit)
#> # A tibble: 4 × 5
#>   term        estimate std.error statistic   p.value
#>   <chr>          <dbl>     <dbl>     <dbl>     <dbl>
#> 1 (Intercept)    1.68     0.156      10.7  1.11e- 26
#> 2 Clark_Lake     0.904    0.0280     32.3  5.14e-210
#> 3 Austin         0.612    0.320       1.91 5.59e-  2
#> 4 Harlem        -0.555    0.165      -3.36 7.85e-  4
```

We'll use this function in subsequent sections. 

### Resampled or tuned models

The tidymodels framework emphasizes the use of resampling methods to evaluate and characterize how well a model works. While time series resampling methods are appropriate for these data, we can also use the [bootstrap](https://www.tmwr.org/resampling.html#bootstrap) to resample the data. This is a standard resampling approach when evaluating the uncertainty in statistical estimates.  

We'll use five bootstrap resamples of the data to simplify the plots and output (normally, we would use a larger number of resamples for more reliable estimates).


```r
set.seed(123)
bt <- bootstraps(Chicago, times = 5)
```

With resampling, we fit the same model to the different simulated versions of the data set produced by resampling. The tidymodels function [`fit_resamples()`](https://www.tmwr.org/resampling.html#resampling-performance) is the recommended approach for doing so. 

{{% warning %}} The `fit_resamples()` function does not automatically save the model objects for each resample since these can be quite large and its main purpose is estimating performance. However, we can pass a function to `fit_resamples()` that _can_ save the model object or any other aspect of the fit. {{%/ warning %}}

This function takes a single argument that represents the fitted [workflow object](https://www.tmwr.org/workflows.html) (even if you don't give `fit_resamples()` a workflow).

From this, we can extract the model fit. There are two "levels" of model objects that are available: 

* The parsnip model object, which wraps the underlying model object. We retrieve this using the `extract_fit_parsnip()` function. 

* The underlying model object (a.k.a. the engine fit) via the `extract_fit_engine()`. 

We'll use the latter option and then tidy this model object as we did in the previous section. Let's add this to the control function so that we can re-use it. 


```r
get_lm_coefs <- function(x) {
  x %>% 
    # get the lm model object
    extract_fit_engine() %>% 
    # transform its format
    tidy()
}
tidy_ctrl <- control_grid(extract = get_lm_coefs)
```

This argument is then passed to `fit_resamples()`:


```r
lm_res <- 
  lm_spec %>% 
  fit_resamples(ridership ~ ., resamples = bt, control = tidy_ctrl)
lm_res
#> # Resampling results
#> # Bootstrap sampling 
#> # A tibble: 5 × 5
#>   splits              id         .metrics         .notes           .extracts    
#>   <list>              <chr>      <list>           <list>           <list>       
#> 1 <split [5698/2076]> Bootstrap1 <tibble [2 × 4]> <tibble [0 × 1]> <tibble [1 ×…
#> 2 <split [5698/2098]> Bootstrap2 <tibble [2 × 4]> <tibble [0 × 1]> <tibble [1 ×…
#> 3 <split [5698/2064]> Bootstrap3 <tibble [2 × 4]> <tibble [0 × 1]> <tibble [1 ×…
#> 4 <split [5698/2082]> Bootstrap4 <tibble [2 × 4]> <tibble [0 × 1]> <tibble [1 ×…
#> 5 <split [5698/2088]> Bootstrap5 <tibble [2 × 4]> <tibble [0 × 1]> <tibble [1 ×…
```

Note that there is a `.extracts` column in our resampling results. This object contains the output of our `get_lm_coefs()` function for each resample. The structure of the elements of this column is a little complex. Let's start by looking at the first element (which corresponds to the first resample): 



```r
lm_res$.extracts[[1]]
#> # A tibble: 1 × 2
#>   .extracts        .config             
#>   <list>           <chr>               
#> 1 <tibble [4 × 5]> Preprocessor1_Model1
```

There is _another_ column in this element called `.extracts` that has the results of the `tidy()` function call: 


```r
lm_res$.extracts[[1]]$.extracts[[1]]
#> # A tibble: 4 × 5
#>   term        estimate std.error statistic   p.value
#>   <chr>          <dbl>     <dbl>     <dbl>     <dbl>
#> 1 (Intercept)    1.40     0.157       8.90 7.23e- 19
#> 2 Clark_Lake     0.842    0.0280     30.1  2.39e-184
#> 3 Austin         1.46     0.320       4.54 5.70e-  6
#> 4 Harlem        -0.637    0.163      -3.92 9.01e-  5
```

These nested columns can be flattened via the purrr `unnest()` function: 


```r
lm_res %>% 
  select(id, .extracts) %>% 
  unnest(.extracts) 
#> # A tibble: 5 × 3
#>   id         .extracts        .config             
#>   <chr>      <list>           <chr>               
#> 1 Bootstrap1 <tibble [4 × 5]> Preprocessor1_Model1
#> 2 Bootstrap2 <tibble [4 × 5]> Preprocessor1_Model1
#> 3 Bootstrap3 <tibble [4 × 5]> Preprocessor1_Model1
#> 4 Bootstrap4 <tibble [4 × 5]> Preprocessor1_Model1
#> 5 Bootstrap5 <tibble [4 × 5]> Preprocessor1_Model1
```

We still have a column of nested tibbles, so we can run the same command again to get the data into a more useful format: 


```r
lm_coefs <- 
  lm_res %>% 
  select(id, .extracts) %>% 
  unnest(.extracts) %>% 
  unnest(.extracts)

lm_coefs %>% select(id, term, estimate, p.value)
#> # A tibble: 20 × 4
#>    id         term        estimate   p.value
#>    <chr>      <chr>          <dbl>     <dbl>
#>  1 Bootstrap1 (Intercept)    1.40  7.23e- 19
#>  2 Bootstrap1 Clark_Lake     0.842 2.39e-184
#>  3 Bootstrap1 Austin         1.46  5.70e-  6
#>  4 Bootstrap1 Harlem        -0.637 9.01e-  5
#>  5 Bootstrap2 (Intercept)    1.69  2.87e- 28
#>  6 Bootstrap2 Clark_Lake     0.911 1.06e-219
#>  7 Bootstrap2 Austin         0.595 5.93e-  2
#>  8 Bootstrap2 Harlem        -0.580 3.88e-  4
#>  9 Bootstrap3 (Intercept)    1.27  3.43e- 16
#> 10 Bootstrap3 Clark_Lake     0.859 5.03e-194
#> 11 Bootstrap3 Austin         1.09  6.77e-  4
#> 12 Bootstrap3 Harlem        -0.470 4.34e-  3
#> 13 Bootstrap4 (Intercept)    1.95  2.91e- 34
#> 14 Bootstrap4 Clark_Lake     0.974 1.47e-233
#> 15 Bootstrap4 Austin        -0.116 7.21e-  1
#> 16 Bootstrap4 Harlem        -0.620 2.11e-  4
#> 17 Bootstrap5 (Intercept)    1.87  1.98e- 33
#> 18 Bootstrap5 Clark_Lake     0.901 1.16e-210
#> 19 Bootstrap5 Austin         0.494 1.15e-  1
#> 20 Bootstrap5 Harlem        -0.512 1.73e-  3
```

That's better! Now, let's plot the model coefficients for each resample: 


```r
lm_coefs %>%
  filter(term != "(Intercept)") %>% 
  ggplot(aes(x = term, y = estimate, group = id, col = id)) +  
  geom_hline(yintercept = 0, lty = 3) + 
  geom_line(alpha = 0.3, lwd = 1.2) + 
  labs(y = "Coefficient", x = NULL) +
  theme(legend.position = "top")
```

<img src="figs/lm-plot-1.svg" width="672" />

There seems to be a lot of uncertainty in the coefficient for the Austin station data, but less for the other two. 

Looking at the code for unnesting the results, you may find the double-nesting structure excessive or cumbersome. However, the extraction functionality is flexible, and a simpler structure would prevent many use cases. 

## More complex: a glmnet model

The glmnet model can fit the same linear regression model structure shown above. It uses regularization (a.k.a penalization) to estimate the model parameters. This has the benefit of shrinking the coefficients towards zero, important in situations where there are strong correlations between predictors or if some feature selection is required. Both of these cases are true for our Chicago train data set. 

There are two types of penalization that this model uses: 

* Lasso (a.k.a. `\(L_1\)`) penalties can shrink the model terms so much that they are absolute zero (i.e. their effect is entirely removed from the model). 

* Weight decay (a.k.a ridge regression or `\(L_2\)`) uses a different type of penalty that is most useful for highly correlated predictors. 

The glmnet model has two primary tuning parameters, the total amount of penalization and the mixture of the two penalty types. For example, this specification:


```r
glmnet_spec <- 
  linear_reg(penalty = 0.1, mixture = 0.95) %>% 
  set_engine("glmnet")
```

has a penalty that is 95% lasso and 5% weight decay. The total amount of these two penalties is 0.1 (which is fairly high). 

{{% note %}} Models with regularization require that predictors are all on the same scale. The ridership at our three stations are very different, but glmnet [automatically centers and scales the data](https://parsnip.tidymodels.org/reference/details_linear_reg_glmnet.html). You can use recipes to [center and scale your data yourself](https://recipes.tidymodels.org/reference/step_normalize.html). {{%/ note %}}

Let's combine the model specification with a formula in a model `workflow()` and then fit the model to the data:


```r
glmnet_wflow <- 
  workflow() %>% 
  add_model(glmnet_spec) %>% 
  add_formula(ridership ~ .)

glmnet_fit <- fit(glmnet_wflow, Chicago)
glmnet_fit
#> ══ Workflow [trained] ════════════════════════════════════════════════
#> Preprocessor: Formula
#> Model: linear_reg()
#> 
#> ── Preprocessor ──────────────────────────────────────────────────────
#> ridership ~ .
#> 
#> ── Model ─────────────────────────────────────────────────────────────
#> 
#> Call:  glmnet::glmnet(x = maybe_matrix(x), y = y, family = "gaussian",      alpha = ~0.95) 
#> 
#>    Df %Dev Lambda
#> 1   0  0.0   6.10
#> 2   1 12.8   5.56
#> 3   1 23.4   5.07
#> 4   1 32.4   4.62
#> 5   1 40.0   4.21
#> 6   1 46.2   3.83
#> 7   1 51.5   3.49
#> 8   1 55.9   3.18
#> 9   1 59.6   2.90
#> 10  1 62.7   2.64
#> 11  2 65.3   2.41
#> 12  2 67.4   2.19
#> 13  2 69.2   2.00
#> 14  2 70.7   1.82
#> 15  2 72.0   1.66
#> 16  2 73.0   1.51
#> 17  2 73.9   1.38
#> 18  2 74.6   1.26
#> 19  2 75.2   1.14
#> 20  2 75.7   1.04
#> 21  2 76.1   0.95
#> 22  2 76.4   0.86
#> 23  2 76.7   0.79
#> 24  2 76.9   0.72
#> 25  2 77.1   0.66
#> 26  2 77.3   0.60
#> 27  2 77.4   0.54
#> 28  2 77.6   0.50
#> 29  2 77.6   0.45
#> 30  2 77.7   0.41
#> 31  2 77.8   0.38
#> 32  2 77.8   0.34
#> 33  2 77.9   0.31
#> 34  2 77.9   0.28
#> 35  2 78.0   0.26
#> 36  2 78.0   0.23
#> 37  2 78.0   0.21
#> 38  2 78.0   0.20
#> 39  2 78.0   0.18
#> 40  2 78.0   0.16
#> 41  2 78.0   0.15
#> 42  2 78.1   0.14
#> 43  2 78.1   0.12
#> 44  2 78.1   0.11
#> 45  2 78.1   0.10
#> 46  2 78.1   0.09
#> 
#> ...
#> and 9 more lines.
```

In this output, the term `lambda` is used to represent the penalty. 

Note that the output shows many values of the penalty despite our specification of `penalty = 0.1`. It turns out that this model fits a "path" of penalty values.  Even though we are interested in a value of 0.1, we can get the model coefficients for many associated values of the penalty from the same model object. 

Let's look at two different approaches to obtaining the coefficients. Both will use the `tidy()` method. One will tidy a glmnet object and the other will tidy a tidymodels object. 

### Using glmnet penalty values

This glmnet fit contains multiple penalty values which depend on the data set; changing the data (or the mixture amount) often produces a different set of values. For this data set, there are 55 penalties available. To get the set of penalties produced for this data set, we can extract the engine fit and tidy: 


```r
glmnet_fit %>% 
  extract_fit_engine() %>% 
  tidy() %>% 
  rename(penalty = lambda) %>%   # <- for consistent naming
  filter(term != "(Intercept)")
#> # A tibble: 99 × 5
#>    term        step estimate penalty dev.ratio
#>    <chr>      <dbl>    <dbl>   <dbl>     <dbl>
#>  1 Clark_Lake     2   0.0753    5.56     0.127
#>  2 Clark_Lake     3   0.145     5.07     0.234
#>  3 Clark_Lake     4   0.208     4.62     0.324
#>  4 Clark_Lake     5   0.266     4.21     0.400
#>  5 Clark_Lake     6   0.319     3.83     0.463
#>  6 Clark_Lake     7   0.368     3.49     0.515
#>  7 Clark_Lake     8   0.413     3.18     0.559
#>  8 Clark_Lake     9   0.454     2.90     0.596
#>  9 Clark_Lake    10   0.491     2.64     0.627
#> 10 Clark_Lake    11   0.526     2.41     0.653
#> # … with 89 more rows
```

This works well but, it turns out that our penalty value (0.1) is not in the list produced by the model! The underlying package has functions that use interpolation to produce coefficients for this specific value, but the `tidy()` method for glmnet objects does not use it. 

### Using specific penalty values

If we run the `tidy()` method on the workflow or parsnip object, a different function is used that returns the coefficients for the penalty value that we specified: 


```r
tidy(glmnet_fit)
#> # A tibble: 4 × 3
#>   term        estimate penalty
#>   <chr>          <dbl>   <dbl>
#> 1 (Intercept)    1.69      0.1
#> 2 Clark_Lake     0.846     0.1
#> 3 Austin         0.271     0.1
#> 4 Harlem         0         0.1
```

For any another (single) penalty, we can use an additional argument:


```r
tidy(glmnet_fit, penalty = 5.5620)  # A value from above
#> # A tibble: 4 × 3
#>   term        estimate penalty
#>   <chr>          <dbl>   <dbl>
#> 1 (Intercept)  12.6       5.56
#> 2 Clark_Lake    0.0753    5.56
#> 3 Austin        0         5.56
#> 4 Harlem        0         5.56
```

The reason for having two `tidy()` methods is that, with tidymodels, the focus is on using a specific penalty value. 


### Tuning a glmnet model

If we know a priori acceptable values for penalty and mixture, we can use the `fit_resamples()` function as we did before with linear regression. Otherwise, we can tune those parameters with the tidymodels `tune_*()` functions. 

Let's tune our glmnet model over both parameters with this grid: 


```r
pen_vals <- 10^seq(-3, 0, length.out = 10)
grid <- crossing(penalty = pen_vals, mixture = c(0.1, 1.0))
```

Here is where more glmnet-related complexity comes in: we know that each resample and each value of `mixture` will probably produce a different set of penalty values contained in the model object. _How can we look at the coefficients at the specific penalty values that we are using to tune?_

The approach that we suggest is to use the special `path_values` option for glmnet. Details are described in the [technical documentation about glmnet and tidymodels](https://parsnip.tidymodels.org/reference/glmnet-details.html#arguments) but in short, this parameter will assign the collection of penalty values used by each glmnet fit (regardless of the data or value of mixture). 

We can pass these as an engine argument and then update our previous workflow object:


```r
glmnet_tune_spec <- 
  linear_reg(penalty = tune(), mixture = tune()) %>% 
  set_engine("glmnet", path_values = pen_vals)

glmnet_wflow <- 
  glmnet_wflow %>% 
  update_model(glmnet_tune_spec)
```

Now we will use an extraction function similar to when we used ordinary least squares. We add an additional argument to retain coefficients that are shrunk to zero by the lasso penalty: 


```r
get_glmnet_coefs <- function(x) {
  x %>% 
    extract_fit_engine() %>% 
    tidy(return_zeros = TRUE) %>% 
    rename(penalty = lambda)
}
parsnip_ctrl <- control_grid(extract = get_glmnet_coefs)

glmnet_res <- 
  glmnet_wflow %>% 
  tune_grid(
    resamples = bt,
    grid = grid,
    control = parsnip_ctrl
  )
glmnet_res
#> # Tuning results
#> # Bootstrap sampling 
#> # A tibble: 5 × 5
#>   splits              id         .metrics          .notes           .extracts   
#>   <list>              <chr>      <list>            <list>           <list>      
#> 1 <split [5698/2076]> Bootstrap1 <tibble [40 × 6]> <tibble [0 × 1]> <tibble [20…
#> 2 <split [5698/2098]> Bootstrap2 <tibble [40 × 6]> <tibble [0 × 1]> <tibble [20…
#> 3 <split [5698/2064]> Bootstrap3 <tibble [40 × 6]> <tibble [0 × 1]> <tibble [20…
#> 4 <split [5698/2082]> Bootstrap4 <tibble [40 × 6]> <tibble [0 × 1]> <tibble [20…
#> 5 <split [5698/2088]> Bootstrap5 <tibble [40 × 6]> <tibble [0 × 1]> <tibble [20…
```

As noted before, the elements of the main `.extracts` column have an embedded list column with the results of `get_glmnet_coefs()`:  


```r
glmnet_res$.extracts[[1]] %>% head()
#> # A tibble: 6 × 4
#>   penalty mixture .extracts         .config              
#>     <dbl>   <dbl> <list>            <chr>                
#> 1       1     0.1 <tibble [40 × 5]> Preprocessor1_Model01
#> 2       1     0.1 <tibble [40 × 5]> Preprocessor1_Model02
#> 3       1     0.1 <tibble [40 × 5]> Preprocessor1_Model03
#> 4       1     0.1 <tibble [40 × 5]> Preprocessor1_Model04
#> 5       1     0.1 <tibble [40 × 5]> Preprocessor1_Model05
#> 6       1     0.1 <tibble [40 × 5]> Preprocessor1_Model06
glmnet_res$.extracts[[1]]$.extracts[[1]] %>% head()
#> # A tibble: 6 × 5
#>   term         step estimate penalty dev.ratio
#>   <chr>       <dbl>    <dbl>   <dbl>     <dbl>
#> 1 (Intercept)     1    0.568  1          0.769
#> 2 (Intercept)     2    0.432  0.464      0.775
#> 3 (Intercept)     3    0.607  0.215      0.779
#> 4 (Intercept)     4    0.846  0.1        0.781
#> 5 (Intercept)     5    1.06   0.0464     0.782
#> 6 (Intercept)     6    1.22   0.0215     0.783
```

As before, we'll have to use a double `unnest()`. Since the penalty value is in both the top-level and lower-level `.extracts`, we'll use `select()` to get rid of the first version (but keep `mixture`):


```r
glmnet_res %>% 
  select(id, .extracts) %>% 
  unnest(.extracts) %>% 
  select(id, mixture, .extracts) %>%  # <- removes the first penalty column
  unnest(.extracts)
```

But wait! We know that each glmnet fit contains all of the coefficients. This means, for a specific resample and value of `mixture`, the results are the same:  


```r
all.equal(
  # First bootstrap, first `mixture`, first `penalty`
  glmnet_res$.extracts[[1]]$.extracts[[1]],
  # First bootstrap, first `mixture`, second `penalty`
  glmnet_res$.extracts[[1]]$.extracts[[2]]
)
#> [1] TRUE
```

For this reason, we'll add a `slice(1)` when grouping by `id` and `mixture`. This will get rid of the replicated results. 


```r
glmnet_coefs <- 
  glmnet_res %>% 
  select(id, .extracts) %>% 
  unnest(.extracts) %>% 
  select(id, mixture, .extracts) %>% 
  group_by(id, mixture) %>%          # ┐
  slice(1) %>%                       # │ Remove the redundant results
  ungroup() %>%                      # ┘
  unnest(.extracts)

glmnet_coefs %>% 
  select(id, penalty, mixture, term, estimate) %>% 
  filter(term != "(Intercept)")
#> # A tibble: 300 × 5
#>    id         penalty mixture term       estimate
#>    <chr>        <dbl>   <dbl> <chr>         <dbl>
#>  1 Bootstrap1 1           0.1 Clark_Lake    0.391
#>  2 Bootstrap1 0.464       0.1 Clark_Lake    0.485
#>  3 Bootstrap1 0.215       0.1 Clark_Lake    0.590
#>  4 Bootstrap1 0.1         0.1 Clark_Lake    0.680
#>  5 Bootstrap1 0.0464      0.1 Clark_Lake    0.746
#>  6 Bootstrap1 0.0215      0.1 Clark_Lake    0.793
#>  7 Bootstrap1 0.01        0.1 Clark_Lake    0.817
#>  8 Bootstrap1 0.00464     0.1 Clark_Lake    0.828
#>  9 Bootstrap1 0.00215     0.1 Clark_Lake    0.834
#> 10 Bootstrap1 0.001       0.1 Clark_Lake    0.837
#> # … with 290 more rows
```

Now we have the coefficients. Let's look at how they behave as more regularization is used: 


```r
glmnet_coefs %>% 
  filter(term != "(Intercept)") %>% 
  mutate(mixture = format(mixture)) %>% 
  ggplot(aes(x = penalty, y = estimate, col = mixture, groups = id)) + 
  geom_hline(yintercept = 0, lty = 3) +
  geom_line(alpha = 0.5, lwd = 1.2) + 
  facet_wrap(~ term) + 
  scale_x_log10() +
  scale_color_brewer(palette = "Accent") +
  labs(y = "coefficient") +
  theme(legend.position = "top")
```

<img src="figs/glmnet-plot-1.svg" width="816" />

Notice a couple of things: 

* With a pure lasso model (i.e., `mixture = 1`), the Austin station predictor is selected out in each resample. With a mixture of both penalties, its influence increases. Also, as the penalty increases, the uncertainty in this coefficient decreases. 

* The Harlem predictor is either quickly selected out of the model or goes from negative to positive. 

## Session information


```
#> ─ Session info ───────────────────────────────────────────────────────────────
#>  setting  value                       
#>  version  R version 4.1.1 (2021-08-10)
#>  os       macOS Big Sur 11.5.2        
#>  system   aarch64, darwin20           
#>  ui       X11                         
#>  language (EN)                        
#>  collate  en_US.UTF-8                 
#>  ctype    en_US.UTF-8                 
#>  tz       America/Denver              
#>  date     2021-09-13                  
#> 
#> ─ Packages ───────────────────────────────────────────────────────────────────
#>  package    * version date       lib source        
#>  broom      * 0.7.9   2021-07-27 [1] CRAN (R 4.1.0)
#>  dials      * 0.0.10  2021-09-10 [1] CRAN (R 4.1.1)
#>  dplyr      * 1.0.7   2021-06-18 [1] CRAN (R 4.1.0)
#>  ggplot2    * 3.3.5   2021-06-25 [1] CRAN (R 4.1.0)
#>  glmnet     * 4.1-2   2021-06-24 [1] CRAN (R 4.1.0)
#>  infer      * 1.0.0   2021-08-13 [1] CRAN (R 4.1.1)
#>  parsnip    * 0.1.7   2021-07-21 [1] CRAN (R 4.1.0)
#>  purrr      * 0.3.4   2020-04-17 [1] CRAN (R 4.1.0)
#>  recipes    * 0.1.16  2021-04-16 [1] CRAN (R 4.1.0)
#>  rlang      * 0.4.11  2021-04-30 [1] CRAN (R 4.1.0)
#>  rsample    * 0.1.0   2021-05-08 [1] CRAN (R 4.1.1)
#>  tibble     * 3.1.4   2021-08-25 [1] CRAN (R 4.1.1)
#>  tidymodels * 0.1.3   2021-04-19 [1] CRAN (R 4.1.0)
#>  tune       * 0.1.6   2021-07-21 [1] CRAN (R 4.1.0)
#>  workflows  * 0.2.3   2021-07-16 [1] CRAN (R 4.1.0)
#>  yardstick  * 0.0.8   2021-03-28 [1] CRAN (R 4.1.0)
#> 
#> [1] /Library/Frameworks/R.framework/Versions/4.1-arm64/Resources/library
```
