---
title: "Correlation and regression fundamentals with tidy data principles"
tags: [broom]
categories: [statistical analysis]
type: learn-subsection
weight: 1
description: | 
  Analyze the results of correlation tests and simple regression models for many data sets at once.
---





## Introduction

This article only requires the tidymodels package.

While the tidymodels package [broom](https://broom.tidyverse.org/) is useful for summarizing the result of a single analysis in a consistent format, it is really designed for high-throughput applications, where you must combine results from multiple analyses. These could be subgroups of data, analyses using different models, bootstrap replicates, permutations, and so on. In particular, it plays well with the `nest()/unnest()` functions from [tidyr](https://tidyr.tidyverse.org/) and the `map()` function in [purrr](https://purrr.tidyverse.org/).

## Correlation analysis

Let's demonstrate this with a simple data set, the built-in `Orange`. We start by coercing `Orange` to a `tibble`. This gives a nicer print method that will be especially useful later on when we start working with list-columns.


```r
library(tidymodels)

data(Orange)

Orange <- as_tibble(Orange)
Orange
#> # A tibble: 35 x 3
#>    Tree    age circumference
#>    <ord> <dbl>         <dbl>
#>  1 1       118            30
#>  2 1       484            58
#>  3 1       664            87
#>  4 1      1004           115
#>  5 1      1231           120
#>  6 1      1372           142
#>  7 1      1582           145
#>  8 2       118            33
#>  9 2       484            69
#> 10 2       664           111
#> # … with 25 more rows
```

This contains 35 observations of three variables: `Tree`, `age`, and `circumference`. `Tree` is a factor with five levels describing five trees. As might be expected, age and circumference are correlated:


```r
cor(Orange$age, Orange$circumference)
#> [1] 0.914

library(ggplot2)

ggplot(Orange, aes(age, circumference, color = Tree)) +
  geom_line()
```

<img src="figs/unnamed-chunk-2-1.svg" width="672" />

Suppose you want to test for correlations individually *within* each tree. You can do this with dplyr's `group_by`:


```r
Orange %>% 
  group_by(Tree) %>%
  summarize(correlation = cor(age, circumference))
#> # A tibble: 5 x 2
#>   Tree  correlation
#>   <ord>       <dbl>
#> 1 3           0.988
#> 2 1           0.985
#> 3 5           0.988
#> 4 2           0.987
#> 5 4           0.984
```

(Note that the correlations are much higher than the aggregated one, and also we can now see the correlation is similar across trees).

Suppose that instead of simply estimating a correlation, we want to perform a hypothesis test with `cor.test()`:


```r
ct <- cor.test(Orange$age, Orange$circumference)
ct
#> 
#> 	Pearson's product-moment correlation
#> 
#> data:  Orange$age and Orange$circumference
#> t = 13, df = 33, p-value = 2e-14
#> alternative hypothesis: true correlation is not equal to 0
#> 95 percent confidence interval:
#>  0.834 0.956
#> sample estimates:
#>   cor 
#> 0.914
```

This test output contains multiple values we may be interested in. Some are vectors of length 1, such as the p-value and the estimate, and some are longer, such as the confidence interval. We can get this into a nicely organized tibble using the `tidy()` function:


```r
tidy(ct)
#> # A tibble: 1 x 8
#>   estimate statistic  p.value parameter conf.low conf.high method    alternative
#>      <dbl>     <dbl>    <dbl>     <int>    <dbl>     <dbl> <chr>     <chr>      
#> 1    0.914      12.9 1.93e-14        33    0.834     0.956 Pearson'… two.sided
```

Often, we want to perform multiple tests or fit multiple models, each on a different part of the data. In this case, we recommend a `nest-map-unnest` workflow. For example, suppose we want to perform correlation tests for each different tree. We start by `nest`ing our data based on the group of interest:


```r
nested <- 
  Orange %>% 
  nest(data = c(age, circumference))
```

Then we perform a correlation test for each nested tibble using `purrr::map()`:


```r
nested %>% 
  mutate(test = map(data, ~ cor.test(.x$age, .x$circumference)))
#> # A tibble: 5 x 3
#>   Tree  data             test   
#>   <ord> <list>           <list> 
#> 1 1     <tibble [7 × 2]> <htest>
#> 2 2     <tibble [7 × 2]> <htest>
#> 3 3     <tibble [7 × 2]> <htest>
#> 4 4     <tibble [7 × 2]> <htest>
#> 5 5     <tibble [7 × 2]> <htest>
```

This results in a list-column of S3 objects. We want to tidy each of the objects, which we can also do with `map()`.


```r
nested %>% 
  mutate(
    test = map(data, ~ cor.test(.x$age, .x$circumference)), # S3 list-col
    tidied = map(test, tidy)
  ) 
#> # A tibble: 5 x 4
#>   Tree  data             test    tidied          
#>   <ord> <list>           <list>  <list>          
#> 1 1     <tibble [7 × 2]> <htest> <tibble [1 × 8]>
#> 2 2     <tibble [7 × 2]> <htest> <tibble [1 × 8]>
#> 3 3     <tibble [7 × 2]> <htest> <tibble [1 × 8]>
#> 4 4     <tibble [7 × 2]> <htest> <tibble [1 × 8]>
#> 5 5     <tibble [7 × 2]> <htest> <tibble [1 × 8]>
```

Finally, we want to unnest the tidied data frames so we can see the results in a flat tibble. All together, this looks like:


```r
Orange %>% 
  nest(data = c(age, circumference)) %>% 
  mutate(
    test = map(data, ~ cor.test(.x$age, .x$circumference)), # S3 list-col
    tidied = map(test, tidy)
  ) %>% 
  unnest(cols = tidied) %>% 
  select(-data, -test)
#> # A tibble: 5 x 9
#>   Tree  estimate statistic p.value parameter conf.low conf.high method
#>   <ord>    <dbl>     <dbl>   <dbl>     <int>    <dbl>     <dbl> <chr> 
#> 1 1        0.985      13.0 4.85e-5         5    0.901     0.998 Pears…
#> 2 2        0.987      13.9 3.43e-5         5    0.914     0.998 Pears…
#> 3 3        0.988      14.4 2.90e-5         5    0.919     0.998 Pears…
#> 4 4        0.984      12.5 5.73e-5         5    0.895     0.998 Pears…
#> 5 5        0.988      14.1 3.18e-5         5    0.916     0.998 Pears…
#> # … with 1 more variable: alternative <chr>
```

## Regression models

This type of workflow becomes even more useful when applied to regressions. Untidy output for a regression looks like:


```r
lm_fit <- lm(age ~ circumference, data = Orange)
summary(lm_fit)
#> 
#> Call:
#> lm(formula = age ~ circumference, data = Orange)
#> 
#> Residuals:
#>    Min     1Q Median     3Q    Max 
#> -317.9 -140.9  -17.2   96.5  471.2 
#> 
#> Coefficients:
#>               Estimate Std. Error t value Pr(>|t|)    
#> (Intercept)     16.604     78.141    0.21     0.83    
#> circumference    7.816      0.606   12.90  1.9e-14 ***
#> ---
#> Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
#> 
#> Residual standard error: 203 on 33 degrees of freedom
#> Multiple R-squared:  0.835,	Adjusted R-squared:  0.83 
#> F-statistic:  166 on 1 and 33 DF,  p-value: 1.93e-14
```

When we tidy these results, we get multiple rows of output for each model:


```r
tidy(lm_fit)
#> # A tibble: 2 x 5
#>   term          estimate std.error statistic  p.value
#>   <chr>            <dbl>     <dbl>     <dbl>    <dbl>
#> 1 (Intercept)      16.6     78.1       0.212 8.33e- 1
#> 2 circumference     7.82     0.606    12.9   1.93e-14
```

Now we can handle multiple regressions at once using exactly the same workflow as before:


```r
Orange %>%
  nest(data = c(-Tree)) %>% 
  mutate(
    fit = map(data, ~ lm(age ~ circumference, data = .x)),
    tidied = map(fit, tidy)
  ) %>% 
  unnest(tidied) %>% 
  select(-data, -fit)
#> # A tibble: 10 x 6
#>    Tree  term          estimate std.error statistic   p.value
#>    <ord> <chr>            <dbl>     <dbl>     <dbl>     <dbl>
#>  1 1     (Intercept)    -265.      98.6      -2.68  0.0436   
#>  2 1     circumference    11.9      0.919    13.0   0.0000485
#>  3 2     (Intercept)    -132.      83.1      -1.59  0.172    
#>  4 2     circumference     7.80     0.560    13.9   0.0000343
#>  5 3     (Intercept)    -210.      85.3      -2.46  0.0574   
#>  6 3     circumference    12.0      0.835    14.4   0.0000290
#>  7 4     (Intercept)     -76.5     88.3      -0.867 0.426    
#>  8 4     circumference     7.17     0.572    12.5   0.0000573
#>  9 5     (Intercept)     -54.5     76.9      -0.709 0.510    
#> 10 5     circumference     8.79     0.621    14.1   0.0000318
```

You can just as easily use multiple predictors in the regressions, as shown here on the `mtcars` dataset. We nest the data into automatic vs. manual cars (the `am` column), then perform the regression within each nested tibble.


```r
data(mtcars)
mtcars <- as_tibble(mtcars)  # to play nicely with list-cols
mtcars
#> # A tibble: 32 x 11
#>      mpg   cyl  disp    hp  drat    wt  qsec    vs    am  gear  carb
#>    <dbl> <dbl> <dbl> <dbl> <dbl> <dbl> <dbl> <dbl> <dbl> <dbl> <dbl>
#>  1  21       6  160    110  3.9   2.62  16.5     0     1     4     4
#>  2  21       6  160    110  3.9   2.88  17.0     0     1     4     4
#>  3  22.8     4  108     93  3.85  2.32  18.6     1     1     4     1
#>  4  21.4     6  258    110  3.08  3.22  19.4     1     0     3     1
#>  5  18.7     8  360    175  3.15  3.44  17.0     0     0     3     2
#>  6  18.1     6  225    105  2.76  3.46  20.2     1     0     3     1
#>  7  14.3     8  360    245  3.21  3.57  15.8     0     0     3     4
#>  8  24.4     4  147.    62  3.69  3.19  20       1     0     4     2
#>  9  22.8     4  141.    95  3.92  3.15  22.9     1     0     4     2
#> 10  19.2     6  168.   123  3.92  3.44  18.3     1     0     4     4
#> # … with 22 more rows

mtcars %>%
  nest(data = c(-am)) %>% 
  mutate(
    fit = map(data, ~ lm(wt ~ mpg + qsec + gear, data = .x)),  # S3 list-col
    tidied = map(fit, tidy)
  ) %>% 
  unnest(tidied) %>% 
  select(-data, -fit)
#> # A tibble: 8 x 6
#>      am term        estimate std.error statistic  p.value
#>   <dbl> <chr>          <dbl>     <dbl>     <dbl>    <dbl>
#> 1     1 (Intercept)   4.28      3.46      1.24   0.247   
#> 2     1 mpg          -0.101     0.0294   -3.43   0.00750 
#> 3     1 qsec          0.0398    0.151     0.264  0.798   
#> 4     1 gear         -0.0229    0.349    -0.0656 0.949   
#> 5     0 (Intercept)   4.92      1.40      3.52   0.00309 
#> 6     0 mpg          -0.192     0.0443   -4.33   0.000591
#> 7     0 qsec          0.0919    0.0983    0.935  0.365   
#> 8     0 gear          0.147     0.368     0.398  0.696
```

What if you want not just the `tidy()` output, but the `augment()` and `glance()` outputs as well, while still performing each regression only once? Since we're using list-columns, we can just fit the model once and use multiple list-columns to store the tidied, glanced and augmented outputs.


```r
regressions <- 
  mtcars %>%
  nest(data = c(-am)) %>% 
  mutate(
    fit = map(data, ~ lm(wt ~ mpg + qsec + gear, data = .x)),
    tidied = map(fit, tidy),
    glanced = map(fit, glance),
    augmented = map(fit, augment)
  )

regressions %>% 
  select(tidied) %>% 
  unnest(tidied)
#> # A tibble: 8 x 5
#>   term        estimate std.error statistic  p.value
#>   <chr>          <dbl>     <dbl>     <dbl>    <dbl>
#> 1 (Intercept)   4.28      3.46      1.24   0.247   
#> 2 mpg          -0.101     0.0294   -3.43   0.00750 
#> 3 qsec          0.0398    0.151     0.264  0.798   
#> 4 gear         -0.0229    0.349    -0.0656 0.949   
#> 5 (Intercept)   4.92      1.40      3.52   0.00309 
#> 6 mpg          -0.192     0.0443   -4.33   0.000591
#> 7 qsec          0.0919    0.0983    0.935  0.365   
#> 8 gear          0.147     0.368     0.398  0.696

regressions %>% 
  select(glanced) %>% 
  unnest(glanced)
#> # A tibble: 2 x 12
#>   r.squared adj.r.squared sigma statistic p.value    df   logLik   AIC   BIC
#>       <dbl>         <dbl> <dbl>     <dbl>   <dbl> <dbl>    <dbl> <dbl> <dbl>
#> 1     0.833         0.778 0.291     15.0  7.59e-4     3 -5.80e-3  10.0  12.8
#> 2     0.625         0.550 0.522      8.32 1.70e-3     3 -1.24e+1  34.7  39.4
#> # … with 3 more variables: deviance <dbl>, df.residual <int>, nobs <int>

regressions %>% 
  select(augmented) %>% 
  unnest(augmented)
#> # A tibble: 32 x 10
#>       wt   mpg  qsec  gear .fitted  .resid .std.resid  .hat .sigma  .cooksd
#>    <dbl> <dbl> <dbl> <dbl>   <dbl>   <dbl>      <dbl> <dbl>  <dbl>    <dbl>
#>  1  2.62  21    16.5     4    2.73 -0.107     -0.527  0.517  0.304 0.0744  
#>  2  2.88  21    17.0     4    2.75  0.126      0.509  0.273  0.304 0.0243  
#>  3  2.32  22.8  18.6     4    2.63 -0.310     -1.29   0.312  0.279 0.188   
#>  4  2.2   32.4  19.5     4    1.70  0.505      1.97   0.223  0.233 0.278   
#>  5  1.62  30.4  18.5     4    1.86 -0.244     -0.982  0.269  0.292 0.0889  
#>  6  1.84  33.9  19.9     4    1.56  0.274      1.12   0.286  0.286 0.125   
#>  7  1.94  27.3  18.9     4    2.19 -0.253     -0.942  0.151  0.293 0.0394  
#>  8  2.14  26    16.7     5    2.21 -0.0683    -0.276  0.277  0.307 0.00732 
#>  9  1.51  30.4  16.9     5    1.77 -0.259     -1.18   0.430  0.284 0.263   
#> 10  3.17  15.8  14.5     5    3.15  0.0193     0.0789 0.292  0.308 0.000644
#> # … with 22 more rows
```

By combining the estimates and p-values across all groups into the same tidy data frame (instead of a list of output model objects), a new class of analyses and visualizations becomes straightforward. This includes:

- sorting by p-value or estimate to find the most significant terms across all tests,
- p-value histograms, and
- volcano plots comparing p-values to effect size estimates.

In each of these cases, we can easily filter, facet, or distinguish based on the `term` column. In short, this makes the tools of tidy data analysis available for the *results* of data analysis and models, not just the inputs.


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
#>  rlang        0.4.9   2020-11-26 [1] CRAN (R 4.0.2)
#>  rsample    * 0.0.8   2020-09-23 [1] CRAN (R 4.0.2)
#>  tibble     * 3.0.4   2020-10-12 [1] CRAN (R 4.0.2)
#>  tidymodels * 0.1.2   2020-11-22 [1] CRAN (R 4.0.2)
#>  tune       * 0.1.2   2020-11-17 [1] CRAN (R 4.0.3)
#>  workflows  * 0.2.1   2020-10-08 [1] CRAN (R 4.0.2)
#>  yardstick  * 0.0.7   2020-07-13 [1] CRAN (R 4.0.2)
#> 
#> [1] /Library/Frameworks/R.framework/Versions/4.0/Resources/library
```

