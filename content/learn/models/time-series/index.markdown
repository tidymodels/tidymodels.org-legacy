---
title: "Modeling time series with tidy resampling"
tags: [rsample]
categories: [model fitting, resampling]
type: learn-subsection
weight: 4
description: | 
  Calculate performance estimates for time series forecasts using resampling.
---






## Introduction

To use the code in this article, you will need to install the following packages: forecast, sweep, tidymodels, timetk, and zoo.

"[Demo Week: Tidy Forecasting with sweep](https://www.business-science.io/code-tools/2017/10/25/demo_week_sweep.html)" is an excellent article that uses tidy methods with time series. This article uses their analysis with rsample to find performance estimates for future observations using [rolling forecast origin resampling](https://robjhyndman.com/hyndsight/crossvalidation/). 

## Example data

The data for this article are sales of alcoholic beverages originally from [the Federal Reserve Bank of St. Louis website](https://fred.stlouisfed.org/series/S4248SM144NCEN).


```r
library(tidymodels)
library(modeldata)
data("drinks")
glimpse(drinks)
#> Rows: 309
#> Columns: 2
#> $ date           <date> 1992-01-01, 1992-02-01, 1992-03-01, 1992-04-01, 1992-…
#> $ S4248SM144NCEN <dbl> 3459, 3458, 4002, 4564, 4221, 4529, 4466, 4137, 4126, …
```

Each row represents one month of sales (in millions of US dollars). 

## Time series resampling

Suppose that we need predictions for one year ahead and our model should use the most recent data from the last 20 years. To set up this resampling scheme:


```r
roll_rs <- rolling_origin(
  drinks, 
  initial = 12 * 20, 
  assess = 12,
  cumulative = FALSE
  )

nrow(roll_rs)
#> [1] 58

roll_rs
#> # Rolling origin forecast resampling 
#> # A tibble: 58 x 2
#>    splits           id     
#>    <list>           <chr>  
#>  1 <split [240/12]> Slice01
#>  2 <split [240/12]> Slice02
#>  3 <split [240/12]> Slice03
#>  4 <split [240/12]> Slice04
#>  5 <split [240/12]> Slice05
#>  6 <split [240/12]> Slice06
#>  7 <split [240/12]> Slice07
#>  8 <split [240/12]> Slice08
#>  9 <split [240/12]> Slice09
#> 10 <split [240/12]> Slice10
#> # … with 48 more rows
```

Each `split` element contains the information about that resample:


```r
roll_rs$splits[[1]]
#> <Analysis/Assess/Total>
#> <240/12/309>
```

For plotting, let's index each split by the first day of the assessment set:


```r
get_date <- function(x) {
  min(assessment(x)$date)
}

start_date <- map(roll_rs$splits, get_date)
roll_rs$start_date <- do.call("c", start_date)
head(roll_rs$start_date)
#> [1] "2012-01-01" "2012-02-01" "2012-03-01" "2012-04-01" "2012-05-01"
#> [6] "2012-06-01"
```

This resampling scheme has 58 splits of the data so that there will be 58 ARIMA models that are fit. To create the models, we use the `auto.arima()` function from the forecast package. The rsample functions `analysis()` and `assessment()` return a data frame, so another step converts the data to a `ts` object called `mod_dat` using a function in the timetk package.


```r
library(forecast)  # for `auto.arima`
library(timetk)    # for `tk_ts`
library(zoo)       # for `as.yearmon`

fit_model <- function(x, ...) {
  # suggested by Matt Dancho:
  x %>%
    analysis() %>%
    # Since the first day changes over resamples, adjust it
    # based on the first date value in the data frame 
    tk_ts(start = .$date[[1]] %>% as.yearmon(), 
          frequency = 12, 
          silent = TRUE) %>%
    auto.arima(...)
}
```

Save each model in a new column:


```r
roll_rs$arima <- map(roll_rs$splits, fit_model)

# For example:
roll_rs$arima[[1]]
#> Series: . 
#> ARIMA(4,1,1)(0,1,2)[12] 
#> 
#> Coefficients:
#>          ar1     ar2    ar3     ar4     ma1    sma1    sma2
#>       -0.185  -0.024  0.358  -0.152  -0.831  -0.193  -0.324
#> s.e.   0.147   0.166  0.144   0.081   0.138   0.067   0.064
#> 
#> sigma^2 estimated as 72198:  log likelihood=-1591
#> AIC=3198   AICc=3199   BIC=3226
```

(There are some warnings produced by these regarding extra columns in the data that can be ignored.)

## Model performance

Using the model fits, let's measure performance in two ways:

 * _Interpolation_ error will measure how well the model fits to the data that were used to create the model. This is most likely optimistic since no holdout method is used. 
 * _Extrapolation_ or _forecast_ error evaluates the performance of the model on the data from the following year (that were not used in the model fit).
 
In each case, the mean absolute percent error (MAPE) is the statistic used to characterize the model fits. The interpolation error can be computed from the `Arima` object. To make things easy, let's use the sweep package's `sw_glance()` function:


```r
library(sweep)

roll_rs$interpolation <- map_dbl(
  roll_rs$arima,
  function(x) 
    sw_glance(x)[["MAPE"]]
  )

summary(roll_rs$interpolation)
#>    Min. 1st Qu.  Median    Mean 3rd Qu.    Max. 
#>    2.84    2.92    2.95    2.95    2.97    3.13
```

For the extrapolation error, the model and split objects are required. Using these:


```r
get_extrap <- function(split, mod) {
  n <- nrow(assessment(split))
  # Get assessment data
  pred_dat <- assessment(split) %>%
    mutate(
      pred = as.vector(forecast(mod, h = n)$mean),
      pct_error = ( S4248SM144NCEN - pred ) / S4248SM144NCEN * 100
    )
  mean(abs(pred_dat$pct_error))
}

roll_rs$extrapolation <- 
  map2_dbl(roll_rs$splits, roll_rs$arima, get_extrap)

summary(roll_rs$extrapolation)
#>    Min. 1st Qu.  Median    Mean 3rd Qu.    Max. 
#>    2.37    3.23    3.63    3.65    4.11    5.45
```

What do these error estimates look like over time?


```r
roll_rs %>%
  select(interpolation, extrapolation, start_date) %>%
  pivot_longer(cols = matches("ation"), names_to = "error", values_to = "MAPE") %>%
  ggplot(aes(x = start_date, y = MAPE, col = error)) + 
  geom_point() + 
  geom_line()
```

<img src="figs/plot-1.svg" width="672" />

It is likely that the interpolation error is an underestimate to some degree, as mentioned above. 

It is also worth noting that `rolling_origin()` can be used over calendar periods, rather than just over a fixed window size. This is especially useful for irregular series where a fixed window size might not make sense because of missing data points, or because of calendar features like different months having a different number of days.

The example below demonstrates this idea by splitting `drinks` into a nested set of 26 years, and rolling over years rather than months. Note that the end result accomplishes a different task than the original example; in this new case, each slice moves forward an entire year, rather than just one month.


```r
# The idea is to nest by the period to roll over,
# which in this case is the year.
roll_rs_annual <- drinks %>%
  mutate(year = as.POSIXlt(date)$year + 1900) %>%
  nest(data = c(date, S4248SM144NCEN)) %>%
  rolling_origin(
    initial = 20, 
    assess = 1, 
    cumulative = FALSE
  )

analysis(roll_rs_annual$splits[[1]])
#> # A tibble: 20 x 2
#>     year data             
#>    <dbl> <list>           
#>  1  1992 <tibble [12 × 2]>
#>  2  1993 <tibble [12 × 2]>
#>  3  1994 <tibble [12 × 2]>
#>  4  1995 <tibble [12 × 2]>
#>  5  1996 <tibble [12 × 2]>
#>  6  1997 <tibble [12 × 2]>
#>  7  1998 <tibble [12 × 2]>
#>  8  1999 <tibble [12 × 2]>
#>  9  2000 <tibble [12 × 2]>
#> 10  2001 <tibble [12 × 2]>
#> 11  2002 <tibble [12 × 2]>
#> 12  2003 <tibble [12 × 2]>
#> 13  2004 <tibble [12 × 2]>
#> 14  2005 <tibble [12 × 2]>
#> 15  2006 <tibble [12 × 2]>
#> 16  2007 <tibble [12 × 2]>
#> 17  2008 <tibble [12 × 2]>
#> 18  2009 <tibble [12 × 2]>
#> 19  2010 <tibble [12 × 2]>
#> 20  2011 <tibble [12 × 2]>
```

The workflow to access these calendar slices is to use `bind_rows()` to join
each analysis set together.


```r
mutate(
  roll_rs_annual,
  extracted_slice = map(splits, ~ bind_rows(analysis(.x)$data))
)
#> # Rolling origin forecast resampling 
#> # A tibble: 6 x 3
#>   splits         id     extracted_slice   
#>   <list>         <chr>  <list>            
#> 1 <split [20/1]> Slice1 <tibble [240 × 2]>
#> 2 <split [20/1]> Slice2 <tibble [240 × 2]>
#> 3 <split [20/1]> Slice3 <tibble [240 × 2]>
#> 4 <split [20/1]> Slice4 <tibble [240 × 2]>
#> 5 <split [20/1]> Slice5 <tibble [240 × 2]>
#> 6 <split [20/1]> Slice6 <tibble [240 × 2]>
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
#>  forecast   * 8.13    2020-09-12 [1] CRAN (R 4.0.2)
#>  ggplot2    * 3.3.2   2020-06-19 [1] CRAN (R 4.0.0)
#>  infer      * 0.5.3   2020-07-14 [1] CRAN (R 4.0.0)
#>  parsnip    * 0.1.4   2020-10-27 [1] CRAN (R 4.0.2)
#>  purrr      * 0.3.4   2020-04-17 [1] CRAN (R 4.0.0)
#>  recipes    * 0.1.15  2020-11-11 [1] CRAN (R 4.0.2)
#>  rlang        0.4.9   2020-11-26 [1] CRAN (R 4.0.2)
#>  rsample    * 0.0.8   2020-09-23 [1] CRAN (R 4.0.2)
#>  sweep      * 0.2.3   2020-07-10 [1] CRAN (R 4.0.2)
#>  tibble     * 3.0.4   2020-10-12 [1] CRAN (R 4.0.2)
#>  tidymodels * 0.1.2   2020-11-22 [1] CRAN (R 4.0.2)
#>  timetk     * 2.6.0   2020-11-21 [1] CRAN (R 4.0.2)
#>  tune       * 0.1.2   2020-11-17 [1] CRAN (R 4.0.3)
#>  workflows  * 0.2.1   2020-10-08 [1] CRAN (R 4.0.2)
#>  yardstick  * 0.0.7   2020-07-13 [1] CRAN (R 4.0.2)
#>  zoo        * 1.8-8   2020-05-02 [1] CRAN (R 4.0.0)
#> 
#> [1] /Library/Frameworks/R.framework/Versions/4.0/Resources/library
```
 
