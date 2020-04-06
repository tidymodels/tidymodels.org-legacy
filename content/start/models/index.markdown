---
title: "Building a model"
weight: 1
tags: [parsnip, broom]
categories: [model fitting]
---





This article requires that you have the following packages installed: readr, rstanarm, and tidymodels.

# Introduction

How do you create a statistical model using the tidymodels packages? Start with your data, learn how to specify and train models with different engines, and understand why these functions are designed this way.

# How to create and use a model

Let's use the data from [Constable (1993)](https://link.springer.com/article/10.1007/BF00349318) to explore how three different feeding regimes affect the size of sea urchins over time. The initial size of the sea urchins at the beginning of the experiment probably affects how big they grow as they are fed. 

To start, let's load the tidymodels packages, along with readr to read our urchins data into R. For each urchin, we know their initial size (volume), the feeding regime of the experiment, and the suture width at the end of the experiment:


```r
library(tidymodels)
library(readr)

urchins <-
  # Data were assembled for a tutorial 
  # at https://www.flutterbys.com.au/stats/tut/tut7.5a.html
  read_csv("https://bit.ly/urchin_data") %>% 
  # Change the names to be a little more verbose
  setNames(c("food_regime", "initial_volume", "width")) %>% 
  # Factors are very helpful for modeling, so we convert one column
  mutate(food_regime = factor(food_regime, levels = c("Initial", "Low", "High")))
#> Parsed with column specification:
#> cols(
#>   TREAT = col_character(),
#>   IV = col_double(),
#>   SUTW = col_double()
#> )
```

As a first step in modeling, it's always a good idea to plot the data: 


```r
theme_set(theme_bw())
ggplot(urchins, aes(x = initial_volume, y = width, group = food_regime, col = food_regime)) + 
  geom_point() + 
  geom_smooth(method = lm, se = FALSE)
#> `geom_smooth()` using formula 'y ~ x'
```

<img src="figs/urchin-plot-1.svg" width="672" />

A standard analysis of covariance ([ANCOVA](https://en.wikipedia.org/wiki/Analysis_of_covariance)) model makes sense for this dataset because we have both a continuous predictor and a categorical predictor. Since the slopes appear to be different for at least two of the feeding regimes, let's build a model that allows for two-way interactions. Specifying an R formula with our variables in this way: 


```r
width ~ (initial_volume + food_regime)^2
```

allows our regression model depending on initial volume to have separate slopes and intercepts for each food regime. 

For this kind of model, ordinary least squares is a good initial approach. With tidymodels, we start by specifying the _functional form_ of the model that we will be using. Since there is a numeric outcome and the model should be linear with slopes and intercepts, the model type is "linear regression". To declare this: 



```r
linear_reg()
#> Linear Regression Model Specification (regression)
```

That is pretty underwhelming since, on its own, it doesn't really do much. However, now that the type of model has been specified, a method for _fitting_ the model can be stated using the **engine**. The engine value is often a mash-up of the software that can be used to fit the model as well as the estimation method. For example, to use ordinary least squares, we can set the engine to be `lm`:


```r
linear_reg() %>% 
  set_engine("lm")
#> Linear Regression Model Specification (regression)
#> 
#> Computational engine: lm
```

The [documentation page for `linear_reg()`](https://tidymodels.github.io/parsnip/reference/linear_reg.html) lists the possible engines. 

From here, the model can be estimated using the [`fit()`](https://tidymodels.github.io/parsnip/reference/fit.html) function:


```r
lm_fit <- 
  linear_reg() %>% 
  set_engine("lm") %>% 
  fit(width ~ (initial_volume + food_regime)^2, data = urchins)
lm_fit
#> parsnip model object
#> 
#> Fit time:  4ms 
#> 
#> Call:
#> stats::lm(formula = formula, data = data)
#> 
#> Coefficients:
#>                    (Intercept)                  initial_volume  
#>                      0.0331216                       0.0015546  
#>                 food_regimeLow                 food_regimeHigh  
#>                      0.0197824                       0.0214111  
#>  initial_volume:food_regimeLow  initial_volume:food_regimeHigh  
#>                     -0.0012594                       0.0005254
```

This object has the `lm` model built-in. 

Perhaps our analysis requires a description of the model parameter estimates and their statistical properties. Although the `summary()` function for `lm` objects can provide that, it gives the results back in an unwieldy format. Many models have a `tidy()` method that provides the summary results in a more predictable and useful format (e.g. a data frame with standard column names): 


```r
tidy(lm_fit)
#> # A tibble: 6 x 5
#>   term                            estimate std.error statistic  p.value
#>   <chr>                              <dbl>     <dbl>     <dbl>    <dbl>
#> 1 (Intercept)                     0.0331    0.00962      3.44  0.00100 
#> 2 initial_volume                  0.00155   0.000398     3.91  0.000222
#> 3 food_regimeLow                  0.0198    0.0130       1.52  0.133   
#> 4 food_regimeHigh                 0.0214    0.0145       1.47  0.145   
#> 5 initial_volume:food_regimeLow  -0.00126   0.000510    -2.47  0.0162  
#> 6 initial_volume:food_regimeHigh  0.000525  0.000702     0.748 0.457
```

Suppose that, for a publication, it would be particularly interesting to make a plot of the mean body size for urchins that started the experiment with an initial volume of 20ml. To create such a graph, we start with some new example data that we will make predictions for, to show in our graph:


```r
new_points <- expand.grid(initial_volume = 20, food_regime = c("Initial", "Low", "High"))
new_points
#>   initial_volume food_regime
#> 1             20     Initial
#> 2             20         Low
#> 3             20        High
```

To get our predicted results, let's use the `predict()` function to find the mean values at 20ml. 

It is also important to communicate the variability, so we also need to find the predicted confidence intervals. If we had used `lm()` to fit the model directly, a few minutes of reading the documentation page for `predict.lm()` would explain how to do this. However, if we decide to use a different model to estimate urchin size (_spoilers:_ we will), it is likely that a completely different syntax would be required. 

Instead, with tidymodels, the types of predicted values are standardized so that we can use the same syntax to get these values. 

First, let's generate the mean body width values: 


```r
mean_pred <- predict(lm_fit, new_data = new_points)
mean_pred
#> # A tibble: 3 x 1
#>    .pred
#>    <dbl>
#> 1 0.0642
#> 2 0.0588
#> 3 0.0961
```

When making predictions, the tidymodels convention is to always produce a tibble of results with standardized column names. This makes it easy to combine the original data and the predictions in a usable format: 


```r
conf_int_pred <- predict(lm_fit, new_data = new_points, type = "conf_int")
conf_int_pred
#> # A tibble: 3 x 2
#>   .pred_lower .pred_upper
#>         <dbl>       <dbl>
#> 1      0.0555      0.0729
#> 2      0.0499      0.0678
#> 3      0.0870      0.105

# Now combine: 
plot_data <- 
  new_points %>% 
  bind_cols(mean_pred) %>% 
  bind_cols(conf_int_pred)

# and plot:
ggplot(plot_data, aes(x = food_regime)) + 
  geom_point(aes(y = .pred)) + 
  geom_errorbar(aes(ymin = .pred_lower, ymax = .pred_upper), width = .2) + 
  labs(y = "urchin size")
```

<img src="figs/lm-all-pred-1.svg" width="672" />

Every one on your team is happy with that plot _except_ that one person who just read their first book on [Bayesian analysis](https://bayesian.org/what-is-bayesian-analysis/). They are interested in knowing if the results would be different if the model were estimated using a Bayesian approach. In such an analysis, a [_prior distribution_](https://towardsdatascience.com/introduction-to-bayesian-linear-regression-e66e60791ea7) needs to be declared for each model parameter that represents the possible values of the parameters (before being exposed to the observed data). After some discussion, the group agrees that the priors should be bell-shaped but, since no one has any idea what the range of values should be, to take a conservative approach and make the priors _wide_ using a Cauchy distribution (which is the same as a t-distribution with a single degree of freedom).

The [documentation](http://mc-stan.org/rstanarm/articles/priors.html) on the rstanarm package shows us that the `stan_glm()` function can be used to estimate this model, and that the function arguments that need to be specified are called `prior` and `prior_intercept`. It turns out that `linear_reg()` has a `stan` engine. Since these prior distribution arguments are specific to the Stan software, they are passed when the engine is set. After that, the same exact `fit()` call is used:


```r
library(rstanarm)

prior_dist <- student_t(df = 1)

bayes_fit <- 
  linear_reg() %>% 
  set_engine("stan", prior_intercept = prior_dist, prior = prior_dist) %>% 
  fit(width ~ (initial_volume + food_regime)^2, data = urchins)
#> 
#> SAMPLING FOR MODEL 'continuous' NOW (CHAIN 1).
#> Chain 1: 
#> Chain 1: Gradient evaluation took 7.8e-05 seconds
#> Chain 1: 1000 transitions using 10 leapfrog steps per transition would take 0.78 seconds.
#> Chain 1: Adjust your expectations accordingly!
#> Chain 1: 
#> Chain 1: 
#> Chain 1: Iteration:    1 / 2000 [  0%]  (Warmup)
#> Chain 1: Iteration:  200 / 2000 [ 10%]  (Warmup)
#> Chain 1: Iteration:  400 / 2000 [ 20%]  (Warmup)
#> Chain 1: Iteration:  600 / 2000 [ 30%]  (Warmup)
#> Chain 1: Iteration:  800 / 2000 [ 40%]  (Warmup)
#> Chain 1: Iteration: 1000 / 2000 [ 50%]  (Warmup)
#> Chain 1: Iteration: 1001 / 2000 [ 50%]  (Sampling)
#> Chain 1: Iteration: 1200 / 2000 [ 60%]  (Sampling)
#> Chain 1: Iteration: 1400 / 2000 [ 70%]  (Sampling)
#> Chain 1: Iteration: 1600 / 2000 [ 80%]  (Sampling)
#> Chain 1: Iteration: 1800 / 2000 [ 90%]  (Sampling)
#> Chain 1: Iteration: 2000 / 2000 [100%]  (Sampling)
#> Chain 1: 
#> Chain 1:  Elapsed Time: 0.241357 seconds (Warm-up)
#> Chain 1:                0.167284 seconds (Sampling)
#> Chain 1:                0.408641 seconds (Total)
#> Chain 1: 
#> 
#> SAMPLING FOR MODEL 'continuous' NOW (CHAIN 2).
#> Chain 2: 
#> Chain 2: Gradient evaluation took 1.2e-05 seconds
#> Chain 2: 1000 transitions using 10 leapfrog steps per transition would take 0.12 seconds.
#> Chain 2: Adjust your expectations accordingly!
#> Chain 2: 
#> Chain 2: 
#> Chain 2: Iteration:    1 / 2000 [  0%]  (Warmup)
#> Chain 2: Iteration:  200 / 2000 [ 10%]  (Warmup)
#> Chain 2: Iteration:  400 / 2000 [ 20%]  (Warmup)
#> Chain 2: Iteration:  600 / 2000 [ 30%]  (Warmup)
#> Chain 2: Iteration:  800 / 2000 [ 40%]  (Warmup)
#> Chain 2: Iteration: 1000 / 2000 [ 50%]  (Warmup)
#> Chain 2: Iteration: 1001 / 2000 [ 50%]  (Sampling)
#> Chain 2: Iteration: 1200 / 2000 [ 60%]  (Sampling)
#> Chain 2: Iteration: 1400 / 2000 [ 70%]  (Sampling)
#> Chain 2: Iteration: 1600 / 2000 [ 80%]  (Sampling)
#> Chain 2: Iteration: 1800 / 2000 [ 90%]  (Sampling)
#> Chain 2: Iteration: 2000 / 2000 [100%]  (Sampling)
#> Chain 2: 
#> Chain 2:  Elapsed Time: 0.219654 seconds (Warm-up)
#> Chain 2:                0.181312 seconds (Sampling)
#> Chain 2:                0.400966 seconds (Total)
#> Chain 2: 
#> 
#> SAMPLING FOR MODEL 'continuous' NOW (CHAIN 3).
#> Chain 3: 
#> Chain 3: Gradient evaluation took 1e-05 seconds
#> Chain 3: 1000 transitions using 10 leapfrog steps per transition would take 0.1 seconds.
#> Chain 3: Adjust your expectations accordingly!
#> Chain 3: 
#> Chain 3: 
#> Chain 3: Iteration:    1 / 2000 [  0%]  (Warmup)
#> Chain 3: Iteration:  200 / 2000 [ 10%]  (Warmup)
#> Chain 3: Iteration:  400 / 2000 [ 20%]  (Warmup)
#> Chain 3: Iteration:  600 / 2000 [ 30%]  (Warmup)
#> Chain 3: Iteration:  800 / 2000 [ 40%]  (Warmup)
#> Chain 3: Iteration: 1000 / 2000 [ 50%]  (Warmup)
#> Chain 3: Iteration: 1001 / 2000 [ 50%]  (Sampling)
#> Chain 3: Iteration: 1200 / 2000 [ 60%]  (Sampling)
#> Chain 3: Iteration: 1400 / 2000 [ 70%]  (Sampling)
#> Chain 3: Iteration: 1600 / 2000 [ 80%]  (Sampling)
#> Chain 3: Iteration: 1800 / 2000 [ 90%]  (Sampling)
#> Chain 3: Iteration: 2000 / 2000 [100%]  (Sampling)
#> Chain 3: 
#> Chain 3:  Elapsed Time: 0.209971 seconds (Warm-up)
#> Chain 3:                0.161009 seconds (Sampling)
#> Chain 3:                0.37098 seconds (Total)
#> Chain 3: 
#> 
#> SAMPLING FOR MODEL 'continuous' NOW (CHAIN 4).
#> Chain 4: 
#> Chain 4: Gradient evaluation took 1.1e-05 seconds
#> Chain 4: 1000 transitions using 10 leapfrog steps per transition would take 0.11 seconds.
#> Chain 4: Adjust your expectations accordingly!
#> Chain 4: 
#> Chain 4: 
#> Chain 4: Iteration:    1 / 2000 [  0%]  (Warmup)
#> Chain 4: Iteration:  200 / 2000 [ 10%]  (Warmup)
#> Chain 4: Iteration:  400 / 2000 [ 20%]  (Warmup)
#> Chain 4: Iteration:  600 / 2000 [ 30%]  (Warmup)
#> Chain 4: Iteration:  800 / 2000 [ 40%]  (Warmup)
#> Chain 4: Iteration: 1000 / 2000 [ 50%]  (Warmup)
#> Chain 4: Iteration: 1001 / 2000 [ 50%]  (Sampling)
#> Chain 4: Iteration: 1200 / 2000 [ 60%]  (Sampling)
#> Chain 4: Iteration: 1400 / 2000 [ 70%]  (Sampling)
#> Chain 4: Iteration: 1600 / 2000 [ 80%]  (Sampling)
#> Chain 4: Iteration: 1800 / 2000 [ 90%]  (Sampling)
#> Chain 4: Iteration: 2000 / 2000 [100%]  (Sampling)
#> Chain 4: 
#> Chain 4:  Elapsed Time: 0.219139 seconds (Warm-up)
#> Chain 4:                0.156797 seconds (Sampling)
#> Chain 4:                0.375936 seconds (Total)
#> Chain 4:

print(bayes_fit, digits = 5)
#> parsnip model object
#> 
#> Fit time:  1.7s 
#> stan_glm
#>  family:       gaussian [identity]
#>  formula:      width ~ (initial_volume + food_regime)^2
#>  observations: 72
#>  predictors:   6
#> ------
#>                                Median   MAD_SD  
#> (Intercept)                     0.03506  0.00926
#> initial_volume                  0.00149  0.00039
#> food_regimeLow                  0.01743  0.01196
#> food_regimeHigh                 0.01886  0.01370
#> initial_volume:food_regimeLow  -0.00117  0.00049
#> initial_volume:food_regimeHigh  0.00062  0.00067
#> 
#> Auxiliary parameter(s):
#>       Median  MAD_SD 
#> sigma 0.02132 0.00188
#> 
#> ------
#> * For help interpreting the printed output see ?print.stanreg
#> * For info on the priors used see ?prior_summary.stanreg
```

To update the parameter table, the `tidy()` method is once again used: 


```r
tidy(bayes_fit, intervals = TRUE)
#> # A tibble: 6 x 5
#>   term                            estimate std.error     lower     upper
#>   <chr>                              <dbl>     <dbl>     <dbl>     <dbl>
#> 1 (Intercept)                     0.0351    0.00926   0.0195    0.0504  
#> 2 initial_volume                  0.00149   0.000391  0.000853  0.00211 
#> 3 food_regimeLow                  0.0174    0.0120   -0.00298   0.0380  
#> 4 food_regimeHigh                 0.0189    0.0137   -0.00351   0.0425  
#> 5 initial_volume:food_regimeLow  -0.00117   0.000486 -0.00197  -0.000362
#> 6 initial_volume:food_regimeHigh  0.000622  0.000669 -0.000503  0.00174
```

A goal of the tidymodels packages is that the **interfaces to common tasks are standardized** (as seen in the `tidy()` results above). The same is true for getting predictions; we can use the same code even though the underlying packages use very different syntax:


```r
bayes_plot_data <- 
  new_points %>% 
  bind_cols(predict(bayes_fit, new_data = new_points)) %>% 
  bind_cols(predict(bayes_fit, new_data = new_points, type = "conf_int"))

ggplot(bayes_plot_data, aes(x = food_regime)) + 
  geom_point(aes(y = .pred)) + 
  geom_errorbar(aes(ymin = .pred_lower, ymax = .pred_upper), width = .2) + 
  labs(y = "urchin size") + 
  ggtitle("Bayesian model with t(1) prior distribution")
```

<img src="figs/stan-pred-1.svg" width="672" />

This isn't very different from the non-Bayesian results (except in interpretation). 


# Why does it work that way? 

The extra step of defining the model using a function like `linear_reg()` might seem superfluous since a call to `lm()` is much more succinct. However, the problem with standard modeling functions is that they don't separate what you want to do from the execution. For example, the process of executing a formula has to happen repeatedly across model calls even when the formula does not change; we can't recycle those computations. 

Also, from a tidy point of view, we can do some interesting things by incrementally creating a model (instead of using single function call). Model tuning, the tidy way, uses the specification of the model (e.g. `linear_reg()` plus `set_engine()`) to declare what parts of the model should be tuned. That would be very difficult to do if `linear_reg()` immediately fit the model. 

If you are familiar with the tidyverse, you may have noticed that our modeling code uses the magrittr pipe (`%>%`). With dplyr and other tidyverse packages, the pipe works well because all of the functions take the _data_ as the first argument. For example: 


```r
iris %>% 
  select(starts_with("Sepal"), Species) %>% 
  pivot_longer(
    cols = c(starts_with("Sepal")),
    names_to = "vars",
    values_to = "values"
  ) %>% 
  group_by(vars) ## etc etc
```

whereas the modeling code uses the pipe to pass around the _model object_:


```r
linear_reg() %>% 
  set_engine("stan", prior_intercept = prior_dist, prior = prior_dist) %>% 
  fit(width ~ (initial_volume + food_regime)^2, data = urchins)
```

This may seem jarring if you have used dplyr a lot, but it is extremely similar to how ggplot2 operates:


```r
ggplot(iris, aes(Sepal.Width, Sepal.Length)) + # returns a ggplot object 
  geom_point() +                               # same
  geom_smooth() +                              # same
  labs(y = "Width", x = "Length")              # etc etc
```


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
#>  date     2020-04-03                  
#> 
#> ─ Packages ───────────────────────────────────────────────────────────────────
#>  package    * version    date       lib source                               
#>  broom      * 0.5.5      2020-02-29 [1] CRAN (R 3.6.0)                       
#>  dials      * 0.0.4      2019-12-02 [1] CRAN (R 3.6.0)                       
#>  dplyr      * 0.8.5      2020-03-07 [1] CRAN (R 3.6.0)                       
#>  ggplot2    * 3.3.0.9000 2020-02-21 [1] Github (tidyverse/ggplot2@b434351)   
#>  infer      * 0.5.1      2019-11-19 [1] CRAN (R 3.6.0)                       
#>  parsnip    * 0.0.5      2020-01-07 [1] CRAN (R 3.6.0)                       
#>  purrr      * 0.3.3      2019-10-18 [1] CRAN (R 3.6.0)                       
#>  readr      * 1.3.1      2018-12-21 [1] CRAN (R 3.6.0)                       
#>  recipes    * 0.1.9      2020-01-14 [1] Github (tidymodels/recipes@5e7c702)  
#>  rlang        0.4.5      2020-03-01 [1] CRAN (R 3.6.0)                       
#>  rsample    * 0.0.5.9000 2020-03-20 [1] Github (tidymodels/rsample@4fdbd6c)  
#>  rstanarm   * 2.19.3     2020-02-11 [1] CRAN (R 3.6.1)                       
#>  tibble     * 2.1.3      2019-06-06 [1] CRAN (R 3.6.0)                       
#>  tidymodels * 0.1.0      2020-02-16 [1] CRAN (R 3.6.0)                       
#>  tune       * 0.0.1.9000 2020-03-17 [1] Github (tidymodels/tune@93f7b2e)     
#>  workflows  * 0.1.0.9000 2020-01-14 [1] Github (tidymodels/workflows@c89bc0c)
#>  yardstick  * 0.0.5      2020-01-23 [1] CRAN (R 3.6.0)                       
#> 
#> [1] /Library/Frameworks/R.framework/Versions/3.6/Resources/library
```
