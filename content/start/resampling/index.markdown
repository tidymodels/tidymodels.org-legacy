---
title: "Evaluate your model with resampling"
weight: 3
tags: [rsample, parsnip, tune, yardstick]
categories: [resampling]
description: | 
  Measure model performance by generating different versions of the training data through resampling.
---







This article requires that you have the following packages installed: modeldata, ranger, and tidymodels.

# Introduction

Once we have a model trained, we need a way to measure how well that model is doing. This tutorial explains how to characterize model performance based on **resampling** statistics. 

# Predicting image segmentation

Some biologists conduct experiments on cells. In drug discovery, a particular type of cell can be treated with either a drug or control and then observed to see what the effect is (if any). A common approach for this kind of measurement is cell imaging. Different parts of the cells can be colored so that the locations of a cell can be determined. 

For example, in top panel of this image of five cells, the green color is meant to define the boundary of the cell (coloring something called the cytoskeleton) while the blue color defines the nucleus of the cell. 

<img src="img/cells.png" width="242" style="display: block; margin: auto;" />

Using these colors, the cells in an image can be _segmented_ so that we know which pixels belong to which cell. If this is done well, the cell can be measured in different ways that are important to the biology. Sometimes the shape of the cell matters and different mathematical tools are used to summarize characteristics like the size or "oblongness" of the cell. 

The bottom panel shows some segmentation results. Cells 1 and 5 are fairly well segmented. However, cells 2 to 4 are bunched up together because the segmentation was not very good. The consequence of bad segmentation is data contamination; when the biologist analyzes the shape or size of these cells, the data are inaccurate and could lead to the wrong conclusion. 

A cell-based experiment might involve millions of cells so it is unfeasible to visually assess them all. Instead, a subsample can be created and these cells can be manually labeled by experts as either poorly segmented (`PS`) or well-segmented (`WS`). If we can predict these labels accurately, the larger data set can be improved by filtering out the cells most likely to be poorly segmented.

An example data set called `cells` is contained in the modeldata package. It has labeled data for 2019 cells. Each also has a total of 56 predictors based on automated image analysis measurements. For example, `avg_inten_ch_1` is the mean intensity of the data contained in the nucleus, `area_ch_1` is the total size of the cell, and so on (some predictors are fairly arcane in nature). 


```r
data(cells, package = "modeldata")
cells
#> # A tibble: 2,019 x 58
#>    case  class angle_ch_1 area_ch_1 avg_inten_ch_1 avg_inten_ch_2 avg_inten_ch_3
#>    <fct> <fct>      <dbl>     <int>          <dbl>          <dbl>          <dbl>
#>  1 Test  PS        143.         185           15.7           4.95           9.55
#>  2 Train PS        134.         819           31.9         207.            69.9 
#>  3 Train WS        107.         431           28.0         116.            63.9 
#>  4 Train PS         69.2        298           19.5         102.            28.2 
#>  5 Test  PS          2.89       285           24.3         112.            20.5 
#>  6 Test  WS         40.7        172          326.          654.           129.  
#>  7 Test  WS        174.         177          260.          596.           124.  
#>  8 Test  PS        180.         251           18.3           5.73          17.2 
#>  9 Test  WS         18.9        495           16.1          89.5           13.7 
#> 10 Test  WS        153.         384           17.7          89.9           20.4 
#> # … with 2,009 more rows, and 51 more variables: avg_inten_ch_4 <dbl>,
#> #   convex_hull_area_ratio_ch_1 <dbl>, convex_hull_perim_ratio_ch_1 <dbl>,
#> #   diff_inten_density_ch_1 <dbl>, diff_inten_density_ch_3 <dbl>,
#> #   diff_inten_density_ch_4 <dbl>, entropy_inten_ch_1 <dbl>,
#> #   entropy_inten_ch_3 <dbl>, entropy_inten_ch_4 <dbl>,
#> #   eq_circ_diam_ch_1 <dbl>, eq_ellipse_lwr_ch_1 <dbl>,
#> #   eq_ellipse_oblate_vol_ch_1 <dbl>, eq_ellipse_prolate_vol_ch_1 <dbl>,
#> #   eq_sphere_area_ch_1 <dbl>, eq_sphere_vol_ch_1 <dbl>,
#> #   fiber_align_2_ch_3 <dbl>, fiber_align_2_ch_4 <dbl>,
#> #   fiber_length_ch_1 <dbl>, fiber_width_ch_1 <dbl>, inten_cooc_asm_ch_3 <dbl>,
#> #   inten_cooc_asm_ch_4 <dbl>, inten_cooc_contrast_ch_3 <dbl>,
#> #   inten_cooc_contrast_ch_4 <dbl>, inten_cooc_entropy_ch_3 <dbl>,
#> #   inten_cooc_entropy_ch_4 <dbl>, inten_cooc_max_ch_3 <dbl>,
#> #   inten_cooc_max_ch_4 <dbl>, kurt_inten_ch_1 <dbl>, kurt_inten_ch_3 <dbl>,
#> #   kurt_inten_ch_4 <dbl>, length_ch_1 <dbl>, neighbor_avg_dist_ch_1 <dbl>,
#> #   neighbor_min_dist_ch_1 <dbl>, neighbor_var_dist_ch_1 <dbl>,
#> #   perim_ch_1 <dbl>, shape_bfr_ch_1 <dbl>, shape_lwr_ch_1 <dbl>,
#> #   shape_p_2_a_ch_1 <dbl>, skew_inten_ch_1 <dbl>, skew_inten_ch_3 <dbl>,
#> #   skew_inten_ch_4 <dbl>, spot_fiber_count_ch_3 <int>,
#> #   spot_fiber_count_ch_4 <dbl>, total_inten_ch_1 <int>,
#> #   total_inten_ch_2 <dbl>, total_inten_ch_3 <int>, total_inten_ch_4 <int>,
#> #   var_inten_ch_1 <dbl>, var_inten_ch_3 <dbl>, var_inten_ch_4 <dbl>,
#> #   width_ch_1 <dbl>
```

The rates of the classes are somewhat imbalanced; there are more poorly segmented cells than well-segmented cells:


```r
class_rates <- table(cells$class)/nrow(cells)
class_rates
#> 
#>        PS        WS 
#> 0.6438831 0.3561169
```

# Data splitting

When beginning a modeling project, it is common to [separate the data set](https://bookdown.org/max/FES/data-splitting.html) into two partitions: 

 * The _training set_ is used to estimate parameters, compare models and feature engineering techniques, tune models, etc.

 * The _test set_ is held in reserve until the end of the project, at which point there should only be one or two models under serious consideration. It is used as an unbiased source for measuring final model performance. 

There are different ways to create these partitions of the data. The most common approach is to use a random sample. Suppose that one quarter of the data were reserved for the test set. Random sampling would randomly select 25% for the test set and use the remainder for the training set. We can use the [rsample](https://tidymodels.github.io/rsample/) package for this purpose. In the original analysis, the authors made their own training/test set and that information is contained in the column `case`. To demonstrate how to make a split, we'll remove this column before we make our own split. 

Since random sampling uses random numbers, it is important to set the random number seed. This ensures that the random numbers can be reproduced at a later time (if needed). 

The function `rsample::initial_split()` takes the original data and saves the information on how to make the partitions. After that, the `training()` and `testing()` functions return the actual data sets:


```r
library(tidymodels) # Load the rsample package, along with the rest of tidymodels

set.seed(123)
cell_split <- initial_split(cells %>% select(-case), strata = class)
cell_train <- training(cell_split)
cell_test  <- testing(cell_split)

nrow(cell_train)
#> [1] 1515
nrow(cell_train)/nrow(cells)
#> [1] 0.7503715
```

The majority of the modeling work is then conducted on the training set data. 

# Modeling

Random forest models are ensembles of decisions trees. A large number of tree models are created for the ensemble based on slightly different versions of the training set. When creating the individual decision trees, the fitting process encourages them to be as diverse as possible. The collection of trees are combined into the random forest model and, when a new sample is predicted, the votes from each tree are used to calculate the final predicted value for the new sample. 

This model is very low maintenance; it requires very little preprocessing of the data and, while it can be tuned, the default parameters tend to give reasonable results. At the same time, the number of trees in the ensemble should be large (in the thousands) and this makes the model moderately expensive to compute. 

To fit a random forest model on the training set, let's use the [parsnip](https://tidymodels.github.io/parsnip/) package in conjunction with the ranger package. We first define the model that we want to create:


```r
rf_mod <- 
  rand_forest(trees = 1000) %>% 
  set_engine("ranger") %>% 
  set_mode("classification")
```

From this, the `fit()` function can be used with a simple model formula. Since random forest models use random numbers, we again set the seed prior to computing: 


```r
set.seed(5273)
rf_fit <- rf_mod %>% fit(class ~ ., data = cell_train)
rf_fit
#> parsnip model object
#> 
#> Fit time:  2.5s 
#> Ranger result
#> 
#> Call:
#>  ranger::ranger(formula = formula, data = data, num.trees = ~1000,      num.threads = 1, verbose = FALSE, seed = sample.int(10^5,          1), probability = TRUE) 
#> 
#> Type:                             Probability estimation 
#> Number of trees:                  1000 
#> Sample size:                      1515 
#> Number of independent variables:  56 
#> Mtry:                             7 
#> Target node size:                 10 
#> Variable importance mode:         none 
#> Splitrule:                        gini 
#> OOB prediction error (Brier s.):  0.1214473
```

# Estimating performance

During a modeling project, we might create a variety of different models. To choose between them, we need to consider how well these models do, as measured by some performance statistics. In our example in this article, some options we could use are: 

 * the area under the Receiver Operating Characteristic (ROC) curve, and
 
 * overall classification accuracy.
 
The ROC curve uses the class probability estimates to give us a sense of performance across the entire set of potential probability cutoffs. Overall accuracy uses the hard class predictions to measure performance (assuming a simple 50% probability cutoff to categorize a cell as poorly segmented). 

The yardstick package has functions for computing both of these measures called `roc_auc()` and `accuracy()`. 

At first glance, it might seem like a good idea to use the training set data to compute these statistics^[Spoiler alert: it is a very bad idea to do this.]. To do this, we call the `predict()` method to get both types of predictions (i.e. probabilities and hard class predictions).


```r
rf_training_pred <- 
  predict(rf_fit, cell_train) %>% 
  bind_cols(predict(rf_fit, cell_train, type = "prob")) %>% 
  # Add the true outcome data back in
  bind_cols(cell_train %>% 
              select(class))
```

Using the yardstick functions, this model has spectacular results, so spectacular that you might be starting to get suspicious: 


```r
roc_auc(rf_training_pred,  truth = class, .pred_PS)
#> # A tibble: 1 x 3
#>   .metric .estimator .estimate
#>   <chr>   <chr>          <dbl>
#> 1 roc_auc binary          1.00
accuracy(rf_training_pred, truth = class, .pred_class)
#> # A tibble: 1 x 3
#>   .metric  .estimator .estimate
#>   <chr>    <chr>          <dbl>
#> 1 accuracy binary         0.993
```

Now that we have this model with exceptional performance, we proceed to the test set. Unfortunately, we discover that, although our results aren't bad, they are certainly worse than what we initially thought based on predicting the training set: 


```r
rf_testing_pred <- 
  predict(rf_fit, cell_test) %>% 
  bind_cols(predict(rf_fit, cell_test, type = "prob")) %>% 
  bind_cols(cell_test %>% select(class))

roc_auc(rf_testing_pred,  truth = class, .pred_PS)
#> # A tibble: 1 x 3
#>   .metric .estimator .estimate
#>   <chr>   <chr>          <dbl>
#> 1 roc_auc binary         0.910
accuracy(rf_testing_pred, truth = class, .pred_class)
#> # A tibble: 1 x 3
#>   .metric  .estimator .estimate
#>   <chr>    <chr>          <dbl>
#> 1 accuracy binary         0.839
```

**What happened here?**

There are several reasons why training set statistics like the ones shown in this section can be unrealistically optimistic: 

 * Models like random forests, neural networks, and other black-box methods can essentially memorize the training set. Re-predicting that same set should always result in nearly perfect results^[Notice that this means our degree of optimism in training set statistics is model dependent.]. 

* The training set does not have the capacity to be a good arbiter of performance. It is not an independent piece of information; predicting the training set can only reflect what the model already knows. 

To understand that second point better, think about an analogy from teaching. Suppose you give a class a test, then give them the answers, then provide the same test. The student scores on the _second_ test do not accurately reflect what they know about the subject; these scores would probably be higher than their results on the first test. 



# Resampling to the rescue

Resampling methods, such as cross-validation and the bootstrap, are empirical simulation systems. They create a series of data sets similar to the training/testing split discussed previously; a subset of the data are used for creating the model and a different subset is used to measure performance. Resampling is always used with the _training set_. This schematic from [Kuhn and Johnson (2019)](https://bookdown.org/max/FES/resampling.html) illustrates data usage for resampling methods:

<img src="img/resampling.svg" width="70%" style="display: block; margin: auto;" />

Let's use 10-fold cross-validation (CV) in this example. This method randomly allocates the 1515 cells in the training set to 10 groups of roughly equal size, called "folds". For the first iteration of resampling, the first fold of about 151 cells are held out for the purpose of measuring performance. This is similar to a test set but, to avoid confusion, we call these data the _assessment set_. The other 90% of the data (about 1363 cells) are used to fit the model and is called the _analysis set_. This model is applied to the assessment set and performance statistics are computed on that assessment set. 

In this example, 10-fold CV moves iteratively through the folds and leaves a different 10% out each time for model assessment. At the end of this process, there are 10 sets of performance statistics that were created on 10 data sets that were not used in the modeling process. For the cell example, this means 10 accuracies and 10 areas under the ROC curve. While 10 models were created, these are not used further; we do not keep the models themselves trained on these folds because their only purpose is calculating performance metrics. 



The final resampling estimates for the model are the **averages** of the performance statistics replicates. For example, suppose for our data the results were: 

<table class="table" style="width: auto !important; margin-left: auto; margin-right: auto;">
 <thead>
  <tr>
   <th style="text-align:left;"> resample </th>
   <th style="text-align:right;"> accuracy </th>
   <th style="text-align:right;"> roc_auc </th>
   <th style="text-align:right;"> assessment size </th>
  </tr>
 </thead>
<tbody>
  <tr>
   <td style="text-align:left;"> Fold01 </td>
   <td style="text-align:right;"> 0.8355263 </td>
   <td style="text-align:right;"> 0.8944311 </td>
   <td style="text-align:right;"> 152 </td>
  </tr>
  <tr>
   <td style="text-align:left;"> Fold02 </td>
   <td style="text-align:right;"> 0.7631579 </td>
   <td style="text-align:right;"> 0.8259958 </td>
   <td style="text-align:right;"> 152 </td>
  </tr>
  <tr>
   <td style="text-align:left;"> Fold03 </td>
   <td style="text-align:right;"> 0.8092105 </td>
   <td style="text-align:right;"> 0.8962054 </td>
   <td style="text-align:right;"> 152 </td>
  </tr>
  <tr>
   <td style="text-align:left;"> Fold04 </td>
   <td style="text-align:right;"> 0.8157895 </td>
   <td style="text-align:right;"> 0.8925961 </td>
   <td style="text-align:right;"> 152 </td>
  </tr>
  <tr>
   <td style="text-align:left;"> Fold05 </td>
   <td style="text-align:right;"> 0.8421053 </td>
   <td style="text-align:right;"> 0.9243115 </td>
   <td style="text-align:right;"> 152 </td>
  </tr>
  <tr>
   <td style="text-align:left;"> Fold06 </td>
   <td style="text-align:right;"> 0.8410596 </td>
   <td style="text-align:right;"> 0.9039773 </td>
   <td style="text-align:right;"> 151 </td>
  </tr>
  <tr>
   <td style="text-align:left;"> Fold07 </td>
   <td style="text-align:right;"> 0.8410596 </td>
   <td style="text-align:right;"> 0.9331921 </td>
   <td style="text-align:right;"> 151 </td>
  </tr>
  <tr>
   <td style="text-align:left;"> Fold08 </td>
   <td style="text-align:right;"> 0.8013245 </td>
   <td style="text-align:right;"> 0.8892707 </td>
   <td style="text-align:right;"> 151 </td>
  </tr>
  <tr>
   <td style="text-align:left;"> Fold09 </td>
   <td style="text-align:right;"> 0.8609272 </td>
   <td style="text-align:right;"> 0.9190196 </td>
   <td style="text-align:right;"> 151 </td>
  </tr>
  <tr>
   <td style="text-align:left;"> Fold10 </td>
   <td style="text-align:right;"> 0.8476821 </td>
   <td style="text-align:right;"> 0.9440285 </td>
   <td style="text-align:right;"> 151 </td>
  </tr>
</tbody>
</table>

From these resampling statistics, the final estimate of performance for this random forest model would be 0.902 for the area under the ROC curve and 0.826 for accuracy. 

These resampling statistics are an effective method for measuring model performance _without_ predicting the training set directly as a whole. 

To generate these results, the first step is to create a resampling object using rsample. There are several resampling methods implemented in rsample; cross-validation folds can be created using `vfold_cv()`: 


```r
set.seed(1697)
folds <- vfold_cv(cell_train, v = 10)
folds
#> #  10-fold cross-validation 
#> # A tibble: 10 x 2
#>    splits             id    
#>    <named list>       <chr> 
#>  1 <split [1.4K/152]> Fold01
#>  2 <split [1.4K/152]> Fold02
#>  3 <split [1.4K/152]> Fold03
#>  4 <split [1.4K/152]> Fold04
#>  5 <split [1.4K/152]> Fold05
#>  6 <split [1.4K/151]> Fold06
#>  7 <split [1.4K/151]> Fold07
#>  8 <split [1.4K/151]> Fold08
#>  9 <split [1.4K/151]> Fold09
#> 10 <split [1.4K/151]> Fold10
```

The list column for `splits` contains the information on which rows belong in the analysis and assessment sets. There are functions that can be used to extract the individual resampled data called `analysis()` and `assessment()`. 

However, the tune package contains high-level functions that can do the required computations to resample a model for the purpose of measuring performance. You have several options for building an object for resampling; you can resample a model specification preprocessed with a formula or [recipe](/start/recipes/), or you can resample a [`workflow()`](https://tidymodels.github.io/workflows/) that bundles together a model specification and formula/recipe. For this example, let's use a `workflow()` that bundles together the random forest model and a formula. Whatever of these options you use, the syntax to resample is very similar to `fit()`: 


```r
set.seed(5273)
rf_wf <- 
  workflow() %>%
  add_model(rf_mod) %>%
  add_formula(class ~ .)

rf_fit_rs <- fit_resamples(rf_wf, folds)

```


```r
rf_fit_rs
#> #  10-fold cross-validation 
#> # A tibble: 10 x 4
#>    splits             id     .metrics         .notes          
#>    <list>             <chr>  <list>           <list>          
#>  1 <split [1.4K/152]> Fold01 <tibble [2 × 3]> <tibble [0 × 1]>
#>  2 <split [1.4K/152]> Fold02 <tibble [2 × 3]> <tibble [0 × 1]>
#>  3 <split [1.4K/152]> Fold03 <tibble [2 × 3]> <tibble [0 × 1]>
#>  4 <split [1.4K/152]> Fold04 <tibble [2 × 3]> <tibble [0 × 1]>
#>  5 <split [1.4K/152]> Fold05 <tibble [2 × 3]> <tibble [0 × 1]>
#>  6 <split [1.4K/151]> Fold06 <tibble [2 × 3]> <tibble [0 × 1]>
#>  7 <split [1.4K/151]> Fold07 <tibble [2 × 3]> <tibble [0 × 1]>
#>  8 <split [1.4K/151]> Fold08 <tibble [2 × 3]> <tibble [0 × 1]>
#>  9 <split [1.4K/151]> Fold09 <tibble [2 × 3]> <tibble [0 × 1]>
#> 10 <split [1.4K/151]> Fold10 <tibble [2 × 3]> <tibble [0 × 1]>
```

The results are similar to the `folds` results with some extra columns. The column `.metrics` contains the performance statistics created from the 10 assessment sets. These can be manually unnested but the tune package contains a number of simple functions that can extract these data: 
 

```r
collect_metrics(rf_fit_rs)
#> # A tibble: 2 x 5
#>   .metric  .estimator  mean     n std_err
#>   <chr>    <chr>      <dbl> <int>   <dbl>
#> 1 accuracy binary     0.826    10 0.00909
#> 2 roc_auc  binary     0.902    10 0.0104
```

Think about these values we now have for accuracy and AUC. These performance metrics are now more realistic (i.e. lower) than our ill-advised first attempt at computing performance metrics in the section above. If we wanted to try different model types for this data set, we could more confidently compare performance metrics computed using resampling to choose between models. Also, remember that at the end of our project, we return to our test set to estimate final model performance. We have looked at this once already before we started using resampling, but let's remind ourselves of the results:


```r
accuracy(rf_testing_pred, truth = class, .pred_class)
#> # A tibble: 1 x 3
#>   .metric  .estimator .estimate
#>   <chr>    <chr>          <dbl>
#> 1 accuracy binary         0.839
roc_auc(rf_testing_pred,  truth = class, .pred_PS)
#> # A tibble: 1 x 3
#>   .metric .estimator .estimate
#>   <chr>   <chr>          <dbl>
#> 1 roc_auc binary         0.910
```

The performance metrics from the test set are much closer to the performance metrics computed using resampling than our first ("bad idea") attempt. Resampling allows us to simulate how well our model will perform on new data, and the test set acts as the final, unbiased check for our model's performance.



# Session information


```
#> ─ Session info ───────────────────────────────────────────────────────────────
#>  setting  value                       
#>  version  R version 3.6.2 (2019-12-12)
#>  os       macOS Mojave 10.14.6        
#>  system   x86_64, darwin15.6.0        
#>  ui       X11                         
#>  language (EN)                        
#>  collate  en_US.UTF-8                 
#>  ctype    en_US.UTF-8                 
#>  tz       America/Denver              
#>  date     2020-04-09                  
#> 
#> ─ Packages ───────────────────────────────────────────────────────────────────
#>  package    * version     date       lib source                               
#>  broom      * 0.5.5       2020-02-29 [1] CRAN (R 3.6.0)                       
#>  dials      * 0.0.4.9000  2020-03-20 [1] local                                
#>  dplyr      * 0.8.5       2020-03-07 [1] CRAN (R 3.6.0)                       
#>  ggplot2    * 3.3.0       2020-03-05 [1] CRAN (R 3.6.0)                       
#>  infer      * 0.5.1       2019-11-19 [1] CRAN (R 3.6.0)                       
#>  modeldata  * 0.0.1       2019-12-06 [1] CRAN (R 3.6.0)                       
#>  parsnip    * 0.0.5.9001  2020-04-03 [1] Github (tidymodels/parsnip@0e83faf)  
#>  purrr      * 0.3.3       2019-10-18 [1] CRAN (R 3.6.0)                       
#>  ranger     * 0.12.1      2020-01-10 [1] CRAN (R 3.6.0)                       
#>  recipes    * 0.1.10.9000 2020-04-03 [1] local                                
#>  rlang        0.4.5.9000  2020-03-20 [1] Github (r-lib/rlang@a90b04b)         
#>  rsample    * 0.0.6       2020-03-31 [1] CRAN (R 3.6.2)                       
#>  tibble     * 3.0.0       2020-03-30 [1] CRAN (R 3.6.2)                       
#>  tidymodels * 0.1.0       2020-02-16 [1] CRAN (R 3.6.0)                       
#>  tune       * 0.1.0       2020-04-02 [1] CRAN (R 3.6.2)                       
#>  workflows  * 0.1.1.9000  2020-03-20 [1] Github (tidymodels/workflows@e995c18)
#>  yardstick  * 0.0.6       2020-03-17 [1] CRAN (R 3.6.0)                       
#> 
#> [1] /Library/Frameworks/R.framework/Versions/3.6/Resources/library
```
