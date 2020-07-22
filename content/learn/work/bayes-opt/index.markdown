---
title: "Iterative Bayesian optimization of a classification model"
tags: [tune, dials, parsnip, recipes, workflows]
categories: [model tuning]
type: learn-subsection
weight: 3
description: | 
  Identify the best hyperparameters for a model using Bayesian optimization of iterative search.
---


  


## Introduction

To use the code in this article, you will need to install the following packages: kernlab, modeldata, themis, and tidymodels.

Many of the examples for model tuning focus on [grid search](/learn/work/tune-svm/). For that method, all the candidate tuning parameter combinations are defined prior to evaluation. Alternatively, _iterative search_ can be used to analyze the existing tuning parameter results and then _predict_ which tuning parameters to try next. 

There are a variety of methods for iterative search and the focus in this article is on _Bayesian optimization_. For more information on this method, these resources might be helpful:

* [_Practical bayesian optimization of machine learning algorithms_](https://scholar.google.com/scholar?hl=en&as_sdt=0%2C7&q=Practical+Bayesian+Optimization+of+Machine+Learning+Algorithms&btnG=) (2012). J Snoek, H Larochelle, and RP Adams. Advances in neural information.  

* [_A Tutorial on Bayesian Optimization for Machine Learning_](https://www.cs.toronto.edu/~rgrosse/courses/csc411_f18/tutorials/tut8_adams_slides.pdf) (2018). R Adams.

 * [_Gaussian Processes for Machine Learning_](http://www.gaussianprocess.org/gpml/) (2006). C E Rasmussen and C Williams.

* [Other articles!](https://scholar.google.com/scholar?hl=en&as_sdt=0%2C7&q="Bayesian+Optimization"&btnG=)


## Cell segmenting revisited

To demonstrate this approach to tuning models, let's return to the cell segmentation data from the [Getting Started](/start/resampling/) article on resampling: 


```r
library(tidymodels)
library(modeldata)

# Load data
data(cells)

set.seed(2369)
tr_te_split <- initial_split(cells %>% select(-case), prop = 3/4)
cell_train <- training(tr_te_split)
cell_test  <- testing(tr_te_split)

set.seed(1697)
folds <- vfold_cv(cell_train, v = 10)
```

## The tuning scheme

Since the predictors are highly correlated, we can used a recipe to convert the original predictors to principal component scores. There is also slight class imbalance in these data; about 64% of the data are poorly segmented. To mitigate this, the data will be down-sampled at the end of the pre-processing so that the number of poorly and well segmented cells occur with equal frequency. We can use a recipe for all this pre-processing, but the number of principal components will need to be _tuned_ so that we have enough (but not too many) representations of the data. 


```r
library(themis)

cell_pre_proc <-
  recipe(class ~ ., data = cell_train) %>%
  step_YeoJohnson(all_predictors()) %>%
  step_normalize(all_predictors()) %>%
  step_pca(all_predictors(), num_comp = tune()) %>%
  step_downsample(class)
```

In this analysis, we will use a support vector machine to model the data. Let's use a radial basis function (RBF) kernel and tune its main parameter ($\sigma$). Additionally, the main SVM parameter, the cost value, also needs optimization. 


```r
svm_mod <-
  svm_rbf(mode = "classification", cost = tune(), rbf_sigma = tune()) %>%
  set_engine("kernlab")
```

These two objects (the recipe and model) will be combined into a single object via the `workflow()` function from the [workflows](https://tidymodels.github.io/workflows/) package; this object will be used in the optimization process. 


```r
svm_wflow <-
  workflow() %>%
  add_model(svm_mod) %>%
  add_recipe(cell_pre_proc)
```

From this object, we can derive information about what parameters are slated to be tuned. A parameter set is derived by: 


```r
svm_set <- parameters(svm_wflow)
svm_set
#> Collection of 3 parameters for tuning
#> 
#>         id parameter type object class
#>       cost           cost    nparam[+]
#>  rbf_sigma      rbf_sigma    nparam[+]
#>   num_comp       num_comp    nparam[+]
```

The default range for the number of PCA components is rather small for this data set. A member of the parameter set can be modified using the `update()` function. Let's constrain the search to one to twenty components by updating the `num_comp` parameter. Additionally, the lower bound of this parameter is set to zero which specifies that the original predictor set should also be evaluated (i.e., with no PCA step at all): 


```r
svm_set <- 
  svm_set %>% 
  update(num_comp = num_comp(c(0L, 20L)))
```

## Sequential tuning 

Bayesian optimization is a sequential method that uses a model to predict new candidate parameters for assessment. When scoring potential parameter value, the mean and variance of performance are predicted. The strategy used to define how these two statistical quantities are used is defined by an _acquisition function_. 

For example, one approach for scoring new candidates is to use a confidence bound. Suppose accuracy is being optimized. For a metric that we want to maximize, a lower confidence bound can be used. The multiplier on the standard error (denoted as `\(\kappa\)`) is a value that can be used to make trade-offs between **exploration** and **exploitation**. 

 * **Exploration** means that the search will consider candidates in untested space.

 * **Exploitation** focuses in areas where the previous best results occurred. 

The variance predicted by the Bayesian model is mostly spatial variation; the value will be large for candidate values that are not close to values that have already been evaluated. If the standard error multiplier is high, the search process will be more likely to avoid areas without candidate values in the vicinity. 

We'll use another acquisition function, _expected improvement_, that determines which candidates are likely to be helpful relative to the current best results. This is the default acquisition function. More information on these functions can be found in the [package vignette for acquisition functions](https://tidymodels.github.io/tune/articles/acquisition_functions.html). 


```r
set.seed(12)
search_res <-
  svm_wflow %>% 
  tune_bayes(
    resamples = folds,
    # To use non-default parameter ranges
    param_info = svm_set,
    # Generate five at semi-random to start
    initial = 5,
    iter = 50,
    # How to measure performance?
    metrics = metric_set(roc_auc),
    control = control_bayes(no_improve = 30, verbose = TRUE)
  )
#> 
#> >  Generating a set of 5 initial parameter results
#> ✓ Initialization complete
#> 
#> Optimizing roc_auc using the expected improvement
#> 
#> ── Iteration 1 ───────────────────────────────────────────────────────
#> 
#> i Current best:		roc_auc=0.8645 (@iter 0)
#> i Gaussian process model
#> ✓ Gaussian process model
#> i Generating 5000 candidates
#> i Predicted candidates
#> i cost=0.00798, rbf_sigma=1e-10, num_comp=1
#> i Estimating performance
#> ✓ Estimating performance
#> ⓧ Newest results:	roc_auc=0.2985 (+/-0.0374)
#> 
#> ── Iteration 2 ───────────────────────────────────────────────────────
#> 
#> i Current best:		roc_auc=0.8645 (@iter 0)
#> i Gaussian process model
#> ✓ Gaussian process model
#> i Generating 5000 candidates
#> i Predicted candidates
#> i cost=0.00108, rbf_sigma=0.949, num_comp=18
#> i Estimating performance
#> ✓ Estimating performance
#> ⓧ Newest results:	roc_auc=0.3641 (+/-0.0754)
#> 
#> ── Iteration 3 ───────────────────────────────────────────────────────
#> 
#> i Current best:		roc_auc=0.8645 (@iter 0)
#> i Gaussian process model
#> ✓ Gaussian process model
#> i Generating 5000 candidates
#> i Predicted candidates
#> i cost=31.5, rbf_sigma=0.617, num_comp=7
#> i Estimating performance
#> ✓ Estimating performance
#> ⓧ Newest results:	roc_auc=0.7847 (+/-0.00945)
#> 
#> ── Iteration 4 ───────────────────────────────────────────────────────
#> 
#> i Current best:		roc_auc=0.8645 (@iter 0)
#> i Gaussian process model
#> ✓ Gaussian process model
#> i Generating 5000 candidates
#> i Predicted candidates
#> i cost=28.8, rbf_sigma=1.01e-10, num_comp=8
#> i Estimating performance
#> ✓ Estimating performance
#> ⓧ Newest results:	roc_auc=0.2839 (+/-0.0983)
#> 
#> ── Iteration 5 ───────────────────────────────────────────────────────
#> 
#> i Current best:		roc_auc=0.8645 (@iter 0)
#> i Gaussian process model
#> ✓ Gaussian process model
#> i Generating 5000 candidates
#> i Predicted candidates
#> i cost=0.001, rbf_sigma=1.63e-08, num_comp=8
#> i Estimating performance
#> ✓ Estimating performance
#> ⓧ Newest results:	roc_auc=0.2097 (+/-0.0491)
#> 
#> ── Iteration 6 ───────────────────────────────────────────────────────
#> 
#> i Current best:		roc_auc=0.8645 (@iter 0)
#> i Gaussian process model
#> ✓ Gaussian process model
#> i Generating 5000 candidates
#> i Predicted candidates
#> i cost=5.7, rbf_sigma=0.0108, num_comp=19
#> i Estimating performance
#> ✓ Estimating performance
#> ♥ Newest results:	roc_auc=0.8796 (+/-0.00586)
#> 
#> ── Iteration 7 ───────────────────────────────────────────────────────
#> 
#> i Current best:		roc_auc=0.8796 (@iter 6)
#> i Gaussian process model
#> ✓ Gaussian process model
#> i Generating 5000 candidates
#> i Predicted candidates
#> i cost=0.805, rbf_sigma=1.13e-10, num_comp=0
#> i Estimating performance
#> ✓ Estimating performance
#> ⓧ Newest results:	roc_auc=0.2952 (+/-0.0951)
#> 
#> ── Iteration 8 ───────────────────────────────────────────────────────
#> 
#> i Current best:		roc_auc=0.8796 (@iter 6)
#> i Gaussian process model
#> ✓ Gaussian process model
#> i Generating 5000 candidates
#> i Predicted candidates
#> i cost=30.9, rbf_sigma=0.00365, num_comp=11
#> i Estimating performance
#> ✓ Estimating performance
#> ⓧ Newest results:	roc_auc=0.8761 (+/-0.00843)
#> 
#> ── Iteration 9 ───────────────────────────────────────────────────────
#> 
#> i Current best:		roc_auc=0.8796 (@iter 6)
#> i Gaussian process model
#> ✓ Gaussian process model
#> i Generating 5000 candidates
#> i Predicted candidates
#> i cost=0.00101, rbf_sigma=0.00363, num_comp=17
#> i Estimating performance
#> ✓ Estimating performance
#> ⓧ Newest results:	roc_auc=0.2853 (+/-0.0978)
#> 
#> ── Iteration 10 ──────────────────────────────────────────────────────
#> 
#> i Current best:		roc_auc=0.8796 (@iter 6)
#> i Gaussian process model
#> ✓ Gaussian process model
#> i Generating 5000 candidates
#> i Predicted candidates
#> i cost=0.317, rbf_sigma=2.02e-06, num_comp=6
#> i Estimating performance
#> ✓ Estimating performance
#> ⓧ Newest results:	roc_auc=0.284 (+/-0.0978)
#> 
#> ── Iteration 11 ──────────────────────────────────────────────────────
#> 
#> i Current best:		roc_auc=0.8796 (@iter 6)
#> i Gaussian process model
#> ✓ Gaussian process model
#> i Generating 5000 candidates
#> i Predicted candidates
#> i cost=31.6, rbf_sigma=2e-05, num_comp=8
#> i Estimating performance
#> ✓ Estimating performance
#> ⓧ Newest results:	roc_auc=0.8638 (+/-0.00927)
#> 
#> ── Iteration 12 ──────────────────────────────────────────────────────
#> 
#> i Current best:		roc_auc=0.8796 (@iter 6)
#> i Gaussian process model
#> ✓ Gaussian process model
#> i Generating 5000 candidates
#> i Predicted candidates
#> i cost=29.1, rbf_sigma=0.000135, num_comp=7
#> i Estimating performance
#> ✓ Estimating performance
#> ⓧ Newest results:	roc_auc=0.8638 (+/-0.00971)
#> 
#> ── Iteration 13 ──────────────────────────────────────────────────────
#> 
#> i Current best:		roc_auc=0.8796 (@iter 6)
#> i Gaussian process model
#> ✓ Gaussian process model
#> i Generating 5000 candidates
#> i Predicted candidates
#> i cost=29.9, rbf_sigma=2.24e-09, num_comp=9
#> i Estimating performance
#> ✓ Estimating performance
#> ⓧ Newest results:	roc_auc=0.2841 (+/-0.098)
#> 
#> ── Iteration 14 ──────────────────────────────────────────────────────
#> 
#> i Current best:		roc_auc=0.8796 (@iter 6)
#> i Gaussian process model
#> ✓ Gaussian process model
#> i Generating 5000 candidates
#> i Predicted candidates
#> i cost=0.000998, rbf_sigma=5.91e-07, num_comp=7
#> i Estimating performance
#> ✓ Estimating performance
#> ⓧ Newest results:	roc_auc=0.2836 (+/-0.0977)
#> 
#> ── Iteration 15 ──────────────────────────────────────────────────────
#> 
#> i Current best:		roc_auc=0.8796 (@iter 6)
#> i Gaussian process model
#> ✓ Gaussian process model
#> i Generating 5000 candidates
#> i Predicted candidates
#> i cost=0.00217, rbf_sigma=0.821, num_comp=0
#> i Estimating performance
#> ✓ Estimating performance
#> ⓧ Newest results:	roc_auc=0.3214 (+/-0.0623)
#> 
#> ── Iteration 16 ──────────────────────────────────────────────────────
#> 
#> i Current best:		roc_auc=0.8796 (@iter 6)
#> i Gaussian process model
#> ✓ Gaussian process model
#> i Generating 5000 candidates
#> i Predicted candidates
#> i cost=0.438, rbf_sigma=0.99, num_comp=19
#> i Estimating performance
#> ✓ Estimating performance
#> ⓧ Newest results:	roc_auc=0.3638 (+/-0.074)
#> 
#> ── Iteration 17 ──────────────────────────────────────────────────────
#> 
#> i Current best:		roc_auc=0.8796 (@iter 6)
#> i Gaussian process model
#> ✓ Gaussian process model
#> i Generating 5000 candidates
#> i Predicted candidates
#> i cost=10.1, rbf_sigma=0.0106, num_comp=1
#> i Estimating performance
#> ✓ Estimating performance
#> ⓧ Newest results:	roc_auc=0.7575 (+/-0.0177)
#> 
#> ── Iteration 18 ──────────────────────────────────────────────────────
#> 
#> i Current best:		roc_auc=0.8796 (@iter 6)
#> i Gaussian process model
#> ✓ Gaussian process model
#> i Generating 5000 candidates
#> i Predicted candidates
#> i cost=0.0354, rbf_sigma=1.39e-10, num_comp=20
#> i Estimating performance
#> ✓ Estimating performance
#> ⓧ Newest results:	roc_auc=0.2098 (+/-0.049)
#> 
#> ── Iteration 19 ──────────────────────────────────────────────────────
#> 
#> i Current best:		roc_auc=0.8796 (@iter 6)
#> i Gaussian process model
#> ✓ Gaussian process model
#> i Generating 5000 candidates
#> i Predicted candidates
#> i cost=0.027, rbf_sigma=5.24e-07, num_comp=20
#> i Estimating performance
#> ✓ Estimating performance
#> ⓧ Newest results:	roc_auc=0.285 (+/-0.0989)
#> 
#> ── Iteration 20 ──────────────────────────────────────────────────────
#> 
#> i Current best:		roc_auc=0.8796 (@iter 6)
#> i Gaussian process model
#> ✓ Gaussian process model
#> i Generating 5000 candidates
#> i Predicted candidates
#> i cost=2.65, rbf_sigma=2.23e-10, num_comp=20
#> i Estimating performance
#> ✓ Estimating performance
#> ⓧ Newest results:	roc_auc=0.285 (+/-0.0989)
#> 
#> ── Iteration 21 ──────────────────────────────────────────────────────
#> 
#> i Current best:		roc_auc=0.8796 (@iter 6)
#> i Gaussian process model
#> ✓ Gaussian process model
#> i Generating 5000 candidates
#> i Predicted candidates
#> i cost=0.0117, rbf_sigma=0.00185, num_comp=0
#> i Estimating performance
#> ✓ Estimating performance
#> ⓧ Newest results:	roc_auc=0.8517 (+/-0.0112)
#> 
#> ── Iteration 22 ──────────────────────────────────────────────────────
#> 
#> i Current best:		roc_auc=0.8796 (@iter 6)
#> i Gaussian process model
#> ✓ Gaussian process model
#> i Generating 5000 candidates
#> i Predicted candidates
#> i cost=5.71, rbf_sigma=1.82e-07, num_comp=0
#> i Estimating performance
#> ✓ Estimating performance
#> ⓧ Newest results:	roc_auc=0.8494 (+/-0.0116)
#> 
#> ── Iteration 23 ──────────────────────────────────────────────────────
#> 
#> i Current best:		roc_auc=0.8796 (@iter 6)
#> i Gaussian process model
#> ✓ Gaussian process model
#> i Generating 5000 candidates
#> i Predicted candidates
#> i cost=0.0248, rbf_sigma=0.0283, num_comp=20
#> i Estimating performance
#> ✓ Estimating performance
#> ⓧ Newest results:	roc_auc=0.8713 (+/-0.00838)
#> 
#> ── Iteration 24 ──────────────────────────────────────────────────────
#> 
#> i Current best:		roc_auc=0.8796 (@iter 6)
#> i Gaussian process model
#> ✓ Gaussian process model
#> i Generating 5000 candidates
#> i Predicted candidates
#> i cost=0.0275, rbf_sigma=0.00958, num_comp=5
#> i Estimating performance
#> ✓ Estimating performance
#> ⓧ Newest results:	roc_auc=0.86 (+/-0.00898)
#> 
#> ── Iteration 25 ──────────────────────────────────────────────────────
#> 
#> i Current best:		roc_auc=0.8796 (@iter 6)
#> i Gaussian process model
#> ✓ Gaussian process model
#> i Generating 5000 candidates
#> i Predicted candidates
#> i cost=0.0016, rbf_sigma=1.14e-10, num_comp=19
#> i Estimating performance
#> ✓ Estimating performance
#> ⓧ Newest results:	roc_auc=0.2105 (+/-0.0489)
#> 
#> ── Iteration 26 ──────────────────────────────────────────────────────
#> 
#> i Current best:		roc_auc=0.8796 (@iter 6)
#> i Gaussian process model
#> ✓ Gaussian process model
#> i Generating 5000 candidates
#> i Predicted candidates
#> i cost=0.253, rbf_sigma=7.6e-09, num_comp=20
#> i Estimating performance
#> ✓ Estimating performance
#> ⓧ Newest results:	roc_auc=0.285 (+/-0.0989)
#> 
#> ── Iteration 27 ──────────────────────────────────────────────────────
#> 
#> i Current best:		roc_auc=0.8796 (@iter 6)
#> i Gaussian process model
#> ✓ Gaussian process model
#> i Generating 5000 candidates
#> i Predicted candidates
#> i cost=6.44, rbf_sigma=4.11e-06, num_comp=0
#> i Estimating performance
#> ✓ Estimating performance
#> ⓧ Newest results:	roc_auc=0.8494 (+/-0.0116)
#> 
#> ── Iteration 28 ──────────────────────────────────────────────────────
#> 
#> i Current best:		roc_auc=0.8796 (@iter 6)
#> i Gaussian process model
#> ✓ Gaussian process model
#> i Generating 5000 candidates
#> i Predicted candidates
#> i cost=0.0654, rbf_sigma=0.000133, num_comp=20
#> i Estimating performance
#> ✓ Estimating performance
#> ⓧ Newest results:	roc_auc=0.8655 (+/-0.00849)
#> 
#> ── Iteration 29 ──────────────────────────────────────────────────────
#> 
#> i Current best:		roc_auc=0.8796 (@iter 6)
#> i Gaussian process model
#> ✓ Gaussian process model
#> i Generating 5000 candidates
#> i Predicted candidates
#> i cost=0.0523, rbf_sigma=0.00231, num_comp=19
#> i Estimating performance
#> ✓ Estimating performance
#> ⓧ Newest results:	roc_auc=0.8647 (+/-0.00859)
#> 
#> ── Iteration 30 ──────────────────────────────────────────────────────
#> 
#> i Current best:		roc_auc=0.8796 (@iter 6)
#> i Gaussian process model
#> ✓ Gaussian process model
#> i Generating 5000 candidates
#> i Predicted candidates
#> i cost=1.33, rbf_sigma=2.88e-06, num_comp=20
#> i Estimating performance
#> ✓ Estimating performance
#> ⓧ Newest results:	roc_auc=0.5585 (+/-0.121)
#> 
#> ── Iteration 31 ──────────────────────────────────────────────────────
#> 
#> i Current best:		roc_auc=0.8796 (@iter 6)
#> i Gaussian process model
#> ✓ Gaussian process model
#> i Generating 5000 candidates
#> i Predicted candidates
#> i cost=0.0266, rbf_sigma=5.08e-06, num_comp=0
#> i Estimating performance
#> ✓ Estimating performance
#> ⓧ Newest results:	roc_auc=0.2953 (+/-0.0951)
#> 
#> ── Iteration 32 ──────────────────────────────────────────────────────
#> 
#> i Current best:		roc_auc=0.8796 (@iter 6)
#> i Gaussian process model
#> ✓ Gaussian process model
#> i Generating 5000 candidates
#> i Predicted candidates
#> i cost=2.01, rbf_sigma=0.971, num_comp=1
#> i Estimating performance
#> ✓ Estimating performance
#> ⓧ Newest results:	roc_auc=0.73 (+/-0.0172)
#> 
#> ── Iteration 33 ──────────────────────────────────────────────────────
#> 
#> i Current best:		roc_auc=0.8796 (@iter 6)
#> i Gaussian process model
#> ✓ Gaussian process model
#> i Generating 5000 candidates
#> i Predicted candidates
#> i cost=0.000993, rbf_sigma=1.89e-07, num_comp=20
#> i Estimating performance
#> ✓ Estimating performance
#> ⓧ Newest results:	roc_auc=0.285 (+/-0.0989)
#> 
#> ── Iteration 34 ──────────────────────────────────────────────────────
#> 
#> i Current best:		roc_auc=0.8796 (@iter 6)
#> i Gaussian process model
#> ✓ Gaussian process model
#> i Generating 5000 candidates
#> i Predicted candidates
#> i cost=0.00105, rbf_sigma=0.000297, num_comp=0
#> i Estimating performance
#> ✓ Estimating performance
#> ⓧ Newest results:	roc_auc=0.2948 (+/-0.0952)
#> 
#> ── Iteration 35 ──────────────────────────────────────────────────────
#> 
#> i Current best:		roc_auc=0.8796 (@iter 6)
#> i Gaussian process model
#> ✓ Gaussian process model
#> i Generating 5000 candidates
#> i Predicted candidates
#> i cost=0.123, rbf_sigma=0.81, num_comp=16
#> i Estimating performance
#> ✓ Estimating performance
#> ⓧ Newest results:	roc_auc=0.3385 (+/-0.0807)
#> 
#> ── Iteration 36 ──────────────────────────────────────────────────────
#> 
#> i Current best:		roc_auc=0.8796 (@iter 6)
#> i Gaussian process model
#> ✓ Gaussian process model
#> i Generating 5000 candidates
#> i Predicted candidates
#> i cost=0.0191, rbf_sigma=0.00288, num_comp=17
#> i Estimating performance
#> ✓ Estimating performance
#> ⓧ Newest results:	roc_auc=0.8621 (+/-0.00874)
#> ! No improvement for 30 iterations; returning current results.
```

The resulting tibble is a stacked set of rows of the rsample object with an additional column for the iteration number:


```r
search_res
#> # Tuning results
#> # 10-fold cross-validation 
#> # A tibble: 370 x 5
#>    splits             id     .metrics         .notes           .iter
#>    <list>             <chr>  <list>           <list>           <dbl>
#>  1 <split [1.4K/152]> Fold01 <tibble [5 × 7]> <tibble [0 × 1]>     0
#>  2 <split [1.4K/152]> Fold02 <tibble [5 × 7]> <tibble [0 × 1]>     0
#>  3 <split [1.4K/152]> Fold03 <tibble [5 × 7]> <tibble [0 × 1]>     0
#>  4 <split [1.4K/152]> Fold04 <tibble [5 × 7]> <tibble [0 × 1]>     0
#>  5 <split [1.4K/152]> Fold05 <tibble [5 × 7]> <tibble [0 × 1]>     0
#>  6 <split [1.4K/151]> Fold06 <tibble [5 × 7]> <tibble [0 × 1]>     0
#>  7 <split [1.4K/151]> Fold07 <tibble [5 × 7]> <tibble [0 × 1]>     0
#>  8 <split [1.4K/151]> Fold08 <tibble [5 × 7]> <tibble [0 × 1]>     0
#>  9 <split [1.4K/151]> Fold09 <tibble [5 × 7]> <tibble [0 × 1]>     0
#> 10 <split [1.4K/151]> Fold10 <tibble [5 × 7]> <tibble [0 × 1]>     0
#> # … with 360 more rows
```

As with grid search, we can summarize the results over resamples:


```r
estimates <- 
  collect_metrics(search_res) %>% 
  arrange(.iter)

estimates
#> # A tibble: 41 x 10
#>       cost rbf_sigma num_comp .iter .metric .estimator  mean     n std_err
#>      <dbl>     <dbl>    <int> <dbl> <chr>   <chr>      <dbl> <int>   <dbl>
#>  1 2.07e-3  1.56e- 5       10     0 roc_auc binary     0.283    10 0.0979 
#>  2 3.48e-1  4.43e- 2        1     0 roc_auc binary     0.757    10 0.0177 
#>  3 1.55e+1  1.28e- 7       20     0 roc_auc binary     0.285    10 0.0989 
#>  4 1.45e+0  2.04e- 3       15     0 roc_auc binary     0.865    10 0.00962
#>  5 3.04e-2  6.41e- 9        5     0 roc_auc binary     0.287    10 0.0966 
#>  6 7.98e-3  1.00e-10        1     1 roc_auc binary     0.298    10 0.0374 
#>  7 1.08e-3  9.49e- 1       18     2 roc_auc binary     0.364    10 0.0754 
#>  8 3.15e+1  6.17e- 1        7     3 roc_auc binary     0.785    10 0.00945
#>  9 2.88e+1  1.01e-10        8     4 roc_auc binary     0.284    10 0.0983 
#> 10 1.00e-3  1.63e- 8        8     5 roc_auc binary     0.210    10 0.0491 
#> # … with 31 more rows, and 1 more variable: .config <chr>
```


The best performance of the initial set of candidate values was `AUC = 0.865 `. The best results were achieved at iteration 6 with a corresponding AUC value of 0.88. The five best results are:


```r
show_best(search_res, metric = "roc_auc")
#> # A tibble: 5 x 10
#>      cost rbf_sigma num_comp .iter .metric .estimator  mean     n std_err
#>     <dbl>     <dbl>    <int> <dbl> <chr>   <chr>      <dbl> <int>   <dbl>
#> 1  5.70    0.0108         19     6 roc_auc binary     0.880    10 0.00586
#> 2 30.9     0.00365        11     8 roc_auc binary     0.876    10 0.00843
#> 3  0.0248  0.0283         20    23 roc_auc binary     0.871    10 0.00838
#> 4  0.0654  0.000133       20    28 roc_auc binary     0.866    10 0.00849
#> 5  0.0523  0.00231        19    29 roc_auc binary     0.865    10 0.00859
#> # … with 1 more variable: .config <chr>
```

A plot of the search iterations can be created via:


```r
autoplot(search_res, type = "performance")
```

<img src="figs/bo-plot-1.svg" width="672" />

There are many parameter combinations have roughly equivalent results. 

How did the parameters change over iterations? 



```r
autoplot(search_res, type = "parameters") + 
  labs(x = "Iterations", y = NULL)
```

<img src="figs/bo-param-plot-1.svg" width="672" />




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
#>  kernlab    * 0.9-29  2019-11-12 [1] CRAN (R 4.0.2)
#>  modeldata  * 0.0.2   2020-06-22 [1] CRAN (R 4.0.2)
#>  parsnip    * 0.1.2   2020-07-03 [1] CRAN (R 4.0.1)
#>  purrr      * 0.3.4   2020-04-17 [1] CRAN (R 4.0.0)
#>  recipes    * 0.1.13  2020-06-23 [1] CRAN (R 4.0.0)
#>  rlang      * 0.4.7   2020-07-09 [1] CRAN (R 4.0.2)
#>  rsample    * 0.0.7   2020-06-04 [1] CRAN (R 4.0.0)
#>  themis     * 0.1.1   2020-05-17 [1] CRAN (R 4.0.2)
#>  tibble     * 3.0.3   2020-07-10 [1] CRAN (R 4.0.2)
#>  tidymodels * 0.1.1   2020-07-14 [1] CRAN (R 4.0.2)
#>  tune       * 0.1.1   2020-07-08 [1] CRAN (R 4.0.0)
#>  workflows  * 0.1.2   2020-07-07 [1] CRAN (R 4.0.0)
#>  yardstick  * 0.0.7   2020-07-13 [1] CRAN (R 4.0.2)
#> 
#> [1] /Library/Frameworks/R.framework/Versions/4.0/Resources/library
```
 
