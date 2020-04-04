---
title: "Subsampling for class imbalances"
tags: [recipes, themis, discrim, parsnip]
categories: [model fitting, pre-processing]
description: | 
  Improve model performance in imbalanced data sets through undersampling or oversampling.
---






# Introduction

This article requires that you have the following packages installed: discrim, klaR, readr, ROSE, themis, and tidymodels.

Subsampling a training set (either undersampling or oversampling the appropriate class or classes) can be a helpful approach to dealing with classification data where one or more classes occur very infrequently. With such an approach, most models will overfit to the majority class and produce very good statistics for the class containing the frequently occurring classes while the minority classes have poor performance. 

This article describes this method for dealing with class imbalances. For better understanding, some knowledge of classification metrics like sensitivity, specificity, and receiver operating characteristic curves is required. See Section 3.2.2 for [Kuhn and Johnson (2019)](https://bookdown.org/max/FES/measuring-performance.html) for more information on these metrics.  

# Simulated data

Consider a two-class problem where the first class has a very low rate of occurrence. The data were simulated and can be imported into R using the code below:


```r
imbal_data <- 
  readr::read_csv("https://bit.ly/imbal_data") %>% 
  mutate(Class = factor(Class))
dim(imbal_data)
#> [1] 1200   16
table(imbal_data$Class)
#> 
#> Class1 Class2 
#>     60   1140
```

If "Class1" is the event of interest, it is very likely that a classification model would be able to achieve very good _specificity_ since almost all of the data are of the second class. _Sensitivity_, however, would likely be poor since the models will optimize accuracy (or other loss functions) by predicting everything to be the majority class. 

When there are two classes, one result of class imbalance is that the default probability cutoff of 50% is inappropriate; a different cutoff that is more extreme might be able to achieve good performance. 

# Subsampling the data

One way to alleviate this issue is to _subsample_ the data. There are a number of ways to do this but the most simple one is to _sample down_ (undersample) the majority class data until it occurs with the same frequency as the minority class. While it may seem counterintuitive, throwing out a large percentage of your data can be effective at producing a useful model that can recognize both the majority and minority class. In some cases, this even means that the overall performance of the model is better (e.g. improved area under the ROC curve). However, subsampling almost always produces models that are _better calibrated_, meaning that the distributions of the class probabilities are more well behaved. As a result, the default 50% cutoff is much more likely to produce better sensitivity and specificity values than they would otherwise. 

To demonstrate this, `themis::step_rose()` will be used in a recipe for the simulated data. It uses the ROSE (random over sampling examples) method from [Menardi, G. and Torelli, N. (2014)](https://scholar.google.com/scholar?hl=en&q=%22training+and+assessing+classification+rules+with+imbalanced+data%22). This is an example of an oversampling strategy, rather than undersampling.

In terms of workflow:

 * It is extremely important that subsampling occurs _inside of resampling_. Otherwise, the resampling process can produce [poor estimates of model performance](https://topepo.github.io/caret/subsampling-for-class-imbalances.html#resampling). 
 * The subsampling process should only be applied to the analysis set. The assessment set should reflect the event rates seen "in the wild" and, for this reason, the `skip` argument to `step_downsample()` has a default of `TRUE`. 

Here is a simple recipe implementing oversampling: 


```r
library(tidymodels)
library(themis)
imbal_rec <- 
  recipe(Class ~ ., data = imbal_data) %>%
  step_rose(Class)
```

For a model, let's use a [quadratic discriminant analysis](https://en.wikipedia.org/wiki/Quadratic_classifier#Quadratic_discriminant_analysis) (QDA) model. From the discrim package, this model can be specified using:


```r
library(discrim)
qda_mod <- 
  discrim_regularized(frac_common_cov = 0, frac_identity = 0) %>% 
  set_engine("klaR")
```

To keep these objects bound together, they are combined in a workflow:


```r
qda_rose_wflw <- 
  workflow() %>% 
  add_model(qda_mod) %>% 
  add_recipe(imbal_rec)
qda_rose_wflw
#> ══ Workflow ═══════════════════════════════════════════════════════════════
#> Preprocessor: Recipe
#> Model: discrim_regularized()
#> 
#> ── Preprocessor ───────────────────────────────────────────────────────────
#> 1 Recipe Step
#> 
#> ● step_rose()
#> 
#> ── Model ──────────────────────────────────────────────────────────────────
#> Regularized Discriminant Model Specification (classification)
#> 
#> Main Arguments:
#>   frac_common_cov = 0
#>   frac_identity = 0
#> 
#> Computational engine: klaR
```

## Model performance

Stratified, repeated 10-fold cross-validation is used to resample the model:


```r
set.seed(5732)
cv_folds <- vfold_cv(imbal_data, strata = "Class", repeats = 5)
```

To measure model performance, let's use two metrics:

 * The area under the [ROC curve](https://en.wikipedia.org/wiki/Receiver_operating_characteristic) is an overall assessment of performance across _all_ cutoffs. Values near one indicate very good results while values near 0.05 would imply that the model is very poor. 
 * The _J_ index (a.k.a. [Youden's _J_](https://en.wikipedia.org/wiki/Youden%27s_J_statistic) statistic) is `sensitivity + specificity - 1`. Values near one are once again best. 

If a model is poorly calibrated, the ROC curve value might not show diminished performance. However, the _J_ index would be lower for models with pathological distributions for the class probabilities. The yardstick package will be used to compute these metrics. 


```r
cls_metrics <- metric_set(roc_auc, j_index)
```

Now, we train the models and generate the results using `tune::fit_resamples()`:


```r
set.seed(2180)
qda_rose_res <- fit_resamples(qda_rose_wflw, resamples = cv_folds, metrics = cls_metrics)
collect_metrics(qda_rose_res)
#> # A tibble: 2 x 5
#>   .metric .estimator  mean     n std_err
#>   <chr>   <chr>      <dbl> <int>   <dbl>
#> 1 j_index binary     0.769    50 0.0236 
#> 2 roc_auc binary     0.950    50 0.00569
```

What do the results look like without using ROSE? We can create another workflow and fit the QDA model along the same resamples:


```r
qda_wflw <- 
  workflow() %>% 
  add_model(qda_mod) %>% 
  add_formula(Class ~ .)

set.seed(2180)
qda_only_res <- fit_resamples(qda_wflw, resamples = cv_folds, metrics = cls_metrics)
collect_metrics(qda_only_res)
#> # A tibble: 2 x 5
#>   .metric .estimator  mean     n std_err
#>   <chr>   <chr>      <dbl> <int>   <dbl>
#> 1 j_index binary     0.250    50 0.0288 
#> 2 roc_auc binary     0.953    50 0.00479
```

It looks like ROSE helped a lot, especially with the J-index. Class imbalance sampling methods tend to greatly improve metrics based on the hard class predictions (i.e., the categorical predictions) because the default cutoff tends to be a better balance of sensitivity and specificity. 

Let's plot the metrics for each resample to see how the individual results changed. 


```r
no_sampling <- 
  qda_only_res %>% 
  collect_metrics(summarize = FALSE) %>% 
  dplyr::select(-.estimator) %>% 
  mutate(sampling = "no_sampling")

with_sampling <- 
  qda_rose_res %>% 
  collect_metrics(summarize = FALSE) %>% 
  dplyr::select(-.estimator) %>% 
  mutate(sampling = "rose")

bind_rows(no_sampling, with_sampling) %>% 
  mutate(label = paste(id2, id)) %>%  
  ggplot(aes(x = sampling, y = .estimate, group = label)) + 
  geom_line(alpha = .4) + 
  facet_wrap(~ .metric, scales = "free_y")
```

<img src="figs/merge-metrics-1.svg" width="672" />

This visually demonstrates that the subsampling mostly affects metrics that use the hard class predictions. 

# Session information


```
#> ─ Session info ───────────────────────────────────────────────────────────────
#>  setting  value                       
#>  version  R version 3.6.0 (2019-04-26)
#>  os       macOS  10.15.4              
#>  system   x86_64, darwin15.6.0        
#>  ui       X11                         
#>  language (EN)                        
#>  collate  en_US.UTF-8                 
#>  ctype    en_US.UTF-8                 
#>  tz       America/New_York            
#>  date     2020-04-04                  
#> 
#> ─ Packages ───────────────────────────────────────────────────────────────────
#>  package    * version date       lib source                          
#>  broom      * 0.5.5   2020-02-29 [1] CRAN (R 3.6.0)                  
#>  dials      * 0.0.4   2019-12-02 [1] CRAN (R 3.6.0)                  
#>  discrim    * 0.0.1   2019-10-11 [1] CRAN (R 3.6.0)                  
#>  dplyr      * 0.8.5   2020-03-07 [1] CRAN (R 3.6.0)                  
#>  ggplot2    * 3.3.0   2020-03-05 [1] CRAN (R 3.6.0)                  
#>  infer      * 0.5.1   2019-11-19 [1] CRAN (R 3.6.0)                  
#>  klaR       * 0.6-15  2020-02-19 [1] CRAN (R 3.6.0)                  
#>  parsnip    * 0.0.5   2020-01-07 [1] CRAN (R 3.6.0)                  
#>  purrr      * 0.3.3   2019-10-18 [1] CRAN (R 3.6.0)                  
#>  readr      * 1.3.1   2018-12-21 [1] CRAN (R 3.6.0)                  
#>  recipes    * 0.1.10  2020-03-18 [1] CRAN (R 3.6.0)                  
#>  rlang        0.4.5   2020-03-01 [1] CRAN (R 3.6.0)                  
#>  ROSE       * 0.0-3   2014-07-15 [1] CRAN (R 3.6.0)                  
#>  rsample    * 0.0.5   2019-07-12 [1] CRAN (R 3.6.0)                  
#>  themis     * 0.1.0   2020-01-13 [1] CRAN (R 3.6.0)                  
#>  tibble     * 2.1.3   2019-06-06 [2] CRAN (R 3.6.0)                  
#>  tidymodels * 0.1.0   2020-02-16 [1] CRAN (R 3.6.0)                  
#>  tune       * 0.0.1   2020-01-24 [1] Github (tidymodels/tune@ba56ec5)
#>  workflows  * 0.1.1   2020-03-17 [1] CRAN (R 3.6.0)                  
#>  yardstick  * 0.0.6   2020-03-17 [1] CRAN (R 3.6.0)                  
#> 
#> [1] /Users/desireedeleon/Library/R/3.6/library
#> [2] /Library/Frameworks/R.framework/Versions/3.6/Resources/library
```

