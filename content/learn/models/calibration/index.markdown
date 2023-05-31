---
title: "An introduction to calibration with tidymodels"
tags: [probably,yardstick]
categories: [classification,calibration]
type: learn-subsection
weight: 5
description: | 
  Learn how the probably package can improve classification and regression models.
---




To use the code in this article, you will need to install the following packages: discrim, klaR, probably, and tidymodels. The probably package should be version 1.0.0 or greater. 

This article demonstrates how to improve an existing model to make its predictions better. A model is well-calibrated if its probability estimate is consistent with the rate that the event occurs "in the wild." In practice though, it can be difficult to validate this definition directly. 

For more details: 

 - Kull, Meelis, Telmo M. Silva Filho, and Peter Flach. "[Beyond sigmoids: How to obtain well-calibrated probabilities from binary classifiers with beta calibration.](https://scholar.google.com/scholar?hl=en&as_sdt=0%2C7&q=%22Beyond+sigmoids%22+calibration&btnG=)" (2017): 5052-5080

- Niculescu-Mizil, Alexandru, and Rich Caruana. "[Predicting good probabilities with supervised learning](https://scholar.google.com/scholar?hl=en&as_sdt=0%2C7&q=%E2%80%9CPredicting+Good+Probabilities+with+Supervised+Learning%E2%80%9D&btnG=)." In _Proceedings of the 22nd international conference on Machine learning_, pp. 625-632. 2005.

To get started, load some packages: 


```r
library(tidymodels)
library(probably)
library(discrim)

tidymodels_prefer()
theme_set(theme_bw())
options(pillar.advice = FALSE, pillar.min_title_chars = Inf)
```

We'll use an old example for illustration.

## An example: predicting cell segmentation quality

The modeldata package contains a data set called `cells`. Initially distributed by [Hill and Haney (2007)](https://bmcbioinformatics.biomedcentral.com/articles/10.1186/1471-2105-8-340), they showed how to create models that predict the _quality_ of the image analysis of cells. The outcome has two levels `"PS"` (for poorly segmented images) or `"WS"` (well-segmented). There are 56 image features that can be used to build a classifier. 

Let's load the data, remove an unwanted column, and look at the outcome frequencies: 


```r
data(cells)
cells$case <- NULL

dim(cells)
#> [1] 2019   57
cells %>% count(class)
#> # A tibble: 2 × 2
#>   class     n
#>   <fct> <int>
#> 1 PS     1300
#> 2 WS      719
```

There is a class imbalance but that will not affect our work here. 

Let's make a 75%/25% split of the data into training and testing using `initial_split()`. We'll also create a set of 10-fold cross-validation indices for model resampling. 


```r
set.seed(8928)
split <- initial_split(cells, strata = class)
cells_tr <- training(split)
cells_te <- testing(split)

cells_rs <- vfold_cv(cells_tr, strata = class)
```

Now that there are data to be modeled, let's get to it!

## A naive Bayes model

We'll show the utility of calibration tools by using a model that, in this instance, is likely to produce a poorly calibrated model. The naive Bayes classifier is a well established model that assumes that the predictors are statistically _independent_ of one another (to simplify the calculations).  While that is certainly not the case for these data, the model can be effective at discriminating between the classes. Unfortunately, when there are many predictors in the model, it has a tendency to produce class probability distributions that are pathological. The predictions tend to gravitate to values neat zero or one, producing distributions that are "U"-shaped ([Kuhn and Johnson, 2013](https://scholar.google.com/scholar?hl=en&as_sdt=0%2C7&q=%22Applied+Predictive+Modeling%22&btnG=)). 

To demonstrate, let's set up the model:


```r
bayes_wflow <-
  workflow() %>%
  add_formula(class ~ .) %>%
  add_model(naive_Bayes())
```

We'll resample the model first so that we can get a good assessment of the results. During the resampling process, two metrics are used to judge how well the model worked. First, the area under the ROC curve is used to measure the ability of the model to separate the classes (using probability predictions). Second, the Brier score can measure how close the probability estimates are to the actual outcome values (zero or one). The `collect_metrics()` function shows the resampling estimates: 


```r
cls_met <- metric_set(roc_auc, brier_class)
# We'll save the out-of-sample predictions to visualize them. 
ctrl <- control_resamples(save_pred = TRUE)

bayes_res <-
  bayes_wflow %>%
  fit_resamples(cells_rs, metrics = cls_met, control = ctrl)

collect_metrics(bayes_res)
#> # A tibble: 2 × 6
#>   .metric     .estimator  mean     n std_err .config             
#>   <chr>       <chr>      <dbl> <int>   <dbl> <chr>               
#> 1 brier_class binary     0.202    10 0.0100  Preprocessor1_Model1
#> 2 roc_auc     binary     0.856    10 0.00936 Preprocessor1_Model1
```

The ROC score is impressive! However, the Brier value indicates that the probability values, while discriminating well, are not very realistic. A value of 0.25 is the "bad model" threshold when there are two classes (a value of zero being the best possible result). 

### But is it calibrated? 

Spoilers: no. It is not. 

The first clue is the extreme U-shaped distribution of the probability scores (facetted by the true class value): 


```r
collect_predictions(bayes_res) %>%
  ggplot(aes(.pred_PS)) +
  geom_histogram(col = "white", bins = 40) +
  facet_wrap(~ class, ncol = 1) +
  geom_rug(col = "blue", alpha = 1 / 2) + 
  labs(x = "Probability Estimate of PS")
```

<div class="figure" style="text-align: center">
<img src="figs/prob-hist-1.svg" alt="plot of chunk prob-hist" width="60%" />
<p class="caption">plot of chunk prob-hist</p>
</div>

There are almost no cells with moderate probability estimates. Furthermore, when the model is incorrect, it is "confidently incorrect". 

The probably package has tools for visualizing and correcting model with poor calibration properties. 

The most common plot is to break the predictions into about ten equally sized buckets and compute the actual event rate within each. For example, if a bin captures the samples predicted to be poorly segmented with probabilities between 20% and 30%, we should expect about a 25% event rate within that partition. Here's a plot with ten bins: 


```r
cal_plot_breaks(bayes_res)
```

<div class="figure" style="text-align: center">
<img src="figs/break-plot-1.svg" alt="plot of chunk break-plot" width="60%" />
<p class="caption">plot of chunk break-plot</p>
</div>

The probabilities are not showing very good accuracy. 

There is also a similar function that can use moving windows with overlapping partitions. This provides a little more detail: 


```r
cal_plot_windowed(bayes_res, step_size = 0.025)
```

<div class="figure" style="text-align: center">
<img src="figs/break-windowed-1.svg" alt="plot of chunk break-windowed" width="60%" />
<p class="caption">plot of chunk break-windowed</p>
</div>

Bad. Still bad. 

Finally, for two class outcomes, we can fit a logistic regression model use a generalized additive model and examine the trend. 


```r
cal_plot_logistic(bayes_res)
```

<div class="figure" style="text-align: center">
<img src="figs/break-logistic-1.svg" alt="plot of chunk break-logistic" width="60%" />
<p class="caption">plot of chunk break-logistic</p>
</div>

Ooof. 

## Remediation

The good news is that we can do something about this. There are tools to fix the probability estimates so that they have better properties. 

The most common approach is the fit a logistic regression model to the data (with the probability estimates as the predictor). The probability predictions from this model is then used as the calibrated estimate. By default, a generalized additive model is used for this fit, but the `smooth = FALSE` argument can use simple linear effects. 

How do we know if this works? There are a set of `validate` functions that can use holdout data to resample the model with and without the calibration tool of choice. Since we already resampled the model, we'll use those results to estimate 10 more logistic regressions and use the out-of-sample data to estimate performance. 

`collect_metrics()` can again be used to see the performance statistics. We'll also use `cal_plot_windowed()` on the calibrated holdout data to get a visual assessment:  


```r
logit_val <- cal_validate_logistic(bayes_res, metrics = cls_met, save_pred = TRUE)
collect_metrics(logit_val)
#> # A tibble: 4 × 7
#>   .metric     .type        .estimator  mean     n std_err .config
#>   <chr>       <chr>        <chr>      <dbl> <int>   <dbl> <chr>  
#> 1 brier_class uncalibrated binary     0.202    10 0.0100  config 
#> 2 roc_auc     uncalibrated binary     0.856    10 0.00936 config 
#> 3 brier_class calibrated   binary     0.154    10 0.00608 config 
#> 4 roc_auc     calibrated   binary     0.855    10 0.00968 config

collect_predictions(logit_val) %>%
  filter(.type == "calibrated") %>%
  cal_plot_windowed(truth = class, estimate = .pred_PS, step_size = 0.025) +
  ggtitle("Logistic calibration via GAM")
```

<div class="figure" style="text-align: center">
<img src="figs/logistic-cal-1.svg" alt="plot of chunk logistic-cal" width="60%" />
<p class="caption">plot of chunk logistic-cal</p>
</div>

That's a lot better but it is problematic that the calibrated predictions to not reach zero or one. 

A different approach is to use isotonic regression. This method can result in very few unique probability estimates. probably has a version of isotonic regression that resamples the process to produce more unique probabilities: 


```r
set.seed(1212)
iso_val <- cal_validate_isotonic_boot(bayes_res, metrics = cls_met, 
                                      save_pred = TRUE, times = 25)
collect_metrics(iso_val)
#> # A tibble: 4 × 7
#>   .metric     .type        .estimator  mean     n std_err .config
#>   <chr>       <chr>        <chr>      <dbl> <int>   <dbl> <chr>  
#> 1 brier_class uncalibrated binary     0.202    10 0.0100  config 
#> 2 roc_auc     uncalibrated binary     0.856    10 0.00936 config 
#> 3 brier_class calibrated   binary     0.150    10 0.00504 config 
#> 4 roc_auc     calibrated   binary     0.856    10 0.00928 config

collect_predictions(iso_val) %>%
  filter(.type == "calibrated") %>%
  cal_plot_windowed(truth = class, estimate = .pred_PS, step_size = 0.025) +
  ggtitle("Isotonic regression calibration")
```

<div class="figure" style="text-align: center">
<img src="figs/isoreg-cal-1.svg" alt="plot of chunk isoreg-cal" width="60%" />
<p class="caption">plot of chunk isoreg-cal</p>
</div>

Much better. There is a slight bias since the estimated points are consistently above the identity line on the 45 degree angle. 

Finally, we can also test out [Beta calibration](https://scholar.google.com/scholar?hl=en&as_sdt=0%2C7&q=%22Beyond+sigmoids%22+calibration&btnG=): 


```r
beta_val <- cal_validate_beta(bayes_res, metrics = cls_met, save_pred = TRUE)
collect_metrics(beta_val)
#> # A tibble: 4 × 7
#>   .metric     .type        .estimator  mean     n std_err .config
#>   <chr>       <chr>        <chr>      <dbl> <int>   <dbl> <chr>  
#> 1 brier_class uncalibrated binary     0.202    10 0.0100  config 
#> 2 roc_auc     uncalibrated binary     0.856    10 0.00936 config 
#> 3 brier_class calibrated   binary     0.145    10 0.00439 config 
#> 4 roc_auc     calibrated   binary     0.856    10 0.00933 config

collect_predictions(beta_val) %>%
  filter(.type == "calibrated") %>%
  cal_plot_windowed(truth = class, estimate = .pred_PS, step_size = 0.025) +
  ggtitle("Beta calibration")
```

<div class="figure" style="text-align: center">
<img src="figs/beta-cal-1.svg" alt="plot of chunk beta-cal" width="60%" />
<p class="caption">plot of chunk beta-cal</p>
</div>

Also a big improvement but it does poorly at the lower end of the scale. 

Isotonic regression appears to have the best results. We'll save a model that is trained using all of the out-of-sample predictions from the original naive Bayes resampling results. The `cell_cal` object can be used to enact the calibration for new predictions. 

We can also fit the final naive Bayes model to predict the test set: 


```r
cell_cal <- cal_estimate_isotonic(bayes_res, times = 25)
bayes_fit <- bayes_wflow %>% fit(data = cells_tr)
```

## Test set results

First, we make our ordinary predictions: 


```r
cell_test_pred <- augment(bayes_fit, new_data = cells_te)
cell_test_pred %>% cls_met(class, .pred_PS)
#> # A tibble: 2 × 3
#>   .metric     .estimator .estimate
#>   <chr>       <chr>          <dbl>
#> 1 roc_auc     binary         0.839
#> 2 brier_class binary         0.226
```

These metric estimates are very consistent with the resasmpled performance estimates. 

We can then use our `cell_cal` object with the `cal_apply()` function:


```r
cell_test_cal_pred <-
  cell_test_pred %>%
  cal_apply(cell_cal)
cell_test_cal_pred %>% dplyr::select(class, starts_with(".pred_"))
#> # A tibble: 505 × 4
#>    class .pred_class .pred_PS .pred_WS
#>    <fct> <fct>          <dbl>    <dbl>
#>  1 PS    PS            0.848    0.152 
#>  2 WS    WS            0.137    0.863 
#>  3 WS    WS            0.0290   0.971 
#>  4 PS    PS            0.791    0.209 
#>  5 PS    PS            0.921    0.0790
#>  6 WS    WS            0.0883   0.912 
#>  7 PS    PS            0.800    0.200 
#>  8 PS    PS            0.673    0.327 
#>  9 WS    WS            0.235    0.765 
#> 10 WS    PS            0.516    0.484 
#> # ℹ 495 more rows
```

Note that `cal_apply()` recomputed the hard class predictions in the `.pred_class` column. It is possible that the changes in the probability estimates could invalidate the original hard class estimates. 

What do the calibrated test set results show? 


```r
cell_test_cal_pred %>% cls_met(class, .pred_PS)
#> # A tibble: 2 × 3
#>   .metric     .estimator .estimate
#>   <chr>       <chr>          <dbl>
#> 1 roc_auc     binary         0.839
#> 2 brier_class binary         0.160
cell_test_cal_pred %>%
  cal_plot_windowed(truth = class, estimate = .pred_PS, step_size = 0.025)
```

<div class="figure" style="text-align: center">
<img src="figs/calibrated-res-1.svg" alt="plot of chunk calibrated-res" width="60%" />
<p class="caption">plot of chunk calibrated-res</p>
</div>

Much better. The test set results also agree with the results from `cal_validate_isotonic_boot().` 

## Other model types

probably can also calibrate classification models with more than two outcome levels. The functions `cal_*_multinomial()` uses a multinomial model in the same spirit as the logistic regression model. Also, isotonic and Beta calibration can also be used via a "one versus all" approach that builds a set of binary calibrators and normalizes their results at the end (to ensure that they add to one). 

For regression models, there is `cal_plot_regression()` and `cal_*_linear()`. The latter uses `lm()` or `mgcv::gam()` to create a calibrator object. 

## Future plans

The tidymodels group is currently working on adding post-processors to workflow objects. This will allow a model workflow to modify the predictions of a model. Calibration is an important feature for post-processing. 
