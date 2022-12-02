---
title: "Hypothesis testing using resampling and tidy data"
tags: [infer]
categories: [statistical analysis]
type: learn-subsection
weight: 4
description: | 
  Perform common hypothesis tests for statistical inference using flexible functions.
---





## Introduction

This article only requires the tidymodels package. 

The tidymodels package [infer](https://infer.tidymodels.org/) implements an expressive grammar to perform statistical inference that coheres with the `tidyverse` design framework. Rather than providing methods for specific statistical tests, this package consolidates the principles that are shared among common hypothesis tests into a set of 4 main verbs (functions), supplemented with many utilities to visualize and extract information from their outputs.

Regardless of which hypothesis test we're using, we're still asking the same kind of question: 

>Is the effect or difference in our observed data real, or due to chance? 

To answer this question, we start by assuming that the observed data came from some world where "nothing is going on" (i.e. the observed effect was simply due to random chance), and call this assumption our **null hypothesis**. (In reality, we might not believe in the null hypothesis at all; the null hypothesis is in opposition to the **alternate hypothesis**, which supposes that the effect present in the observed data is actually due to the fact that "something is going on.") We then calculate a **test statistic** from our data that describes the observed effect. We can use this test statistic to calculate a **p-value**, giving the probability that our observed data could come about if the null hypothesis was true. If this probability is below some pre-defined **significance level** `alpha`, then we can reject our null hypothesis.

If you are new to hypothesis testing, take a look at 

* [Section 9.2 of _Statistical Inference via Data Science_](https://moderndive.com/9-hypothesis-testing.html#understanding-ht)
* The American Statistical Association's recent [statement on p-values](https://doi.org/10.1080/00031305.2016.1154108) 

The workflow of this package is designed around these ideas. Starting from some data set,

+ `specify()` allows you to specify the variable, or relationship between variables, that you're interested in,
+ `hypothesize()` allows you to declare the null hypothesis,
+ `generate()` allows you to generate data reflecting the null hypothesis, and
+ `calculate()` allows you to calculate a distribution of statistics from the generated data to form the null distribution.

Throughout this vignette, we make use of `gss`, a data set available in infer containing a sample of 500 observations of 11 variables from the *General Social Survey*. 


```r
library(tidymodels) # Includes the infer package

# load in the data set
data(gss)

# take a look at its structure
dplyr::glimpse(gss)
#> Rows: 500
#> Columns: 11
#> $ year    <dbl> 2014, 1994, 1998, 1996, 1994, 1996, 1990, 2016, 2000, 1998, 20…
#> $ age     <dbl> 36, 34, 24, 42, 31, 32, 48, 36, 30, 33, 21, 30, 38, 49, 25, 56…
#> $ sex     <fct> male, female, male, male, male, female, female, female, female…
#> $ college <fct> degree, no degree, degree, no degree, degree, no degree, no de…
#> $ partyid <fct> ind, rep, ind, ind, rep, rep, dem, ind, rep, dem, dem, ind, de…
#> $ hompop  <dbl> 3, 4, 1, 4, 2, 4, 2, 1, 5, 2, 4, 3, 4, 4, 2, 2, 3, 2, 1, 2, 5,…
#> $ hours   <dbl> 50, 31, 40, 40, 40, 53, 32, 20, 40, 40, 23, 52, 38, 72, 48, 40…
#> $ income  <ord> $25000 or more, $20000 - 24999, $25000 or more, $25000 or more…
#> $ class   <fct> middle class, working class, working class, working class, mid…
#> $ finrela <fct> below average, below average, below average, above average, ab…
#> $ weight  <dbl> 0.896, 1.083, 0.550, 1.086, 1.083, 1.086, 1.063, 0.478, 1.099,…
```

Each row is an individual survey response, containing some basic demographic information on the respondent as well as some additional variables. See `?gss` for more information on the variables included and their source. Note that this data (and our examples on it) are for demonstration purposes only, and will not necessarily provide accurate estimates unless weighted properly. For these examples, let's suppose that this data set is a representative sample of a population we want to learn about: American adults.

## Specify variables

The `specify()` function can be used to specify which of the variables in the data set you're interested in. If you're only interested in, say, the `age` of the respondents, you might write:


```r
gss %>%
  specify(response = age)
#> Response: age (numeric)
#> # A tibble: 500 × 1
#>      age
#>    <dbl>
#>  1    36
#>  2    34
#>  3    24
#>  4    42
#>  5    31
#>  6    32
#>  7    48
#>  8    36
#>  9    30
#> 10    33
#> # … with 490 more rows
```

On the front end, the output of `specify()` just looks like it selects off the columns in the dataframe that you've specified. What do we see if we check the class of this object, though?


```r
gss %>%
  specify(response = age) %>%
  class()
#> [1] "infer"      "tbl_df"     "tbl"        "data.frame"
```

We can see that the infer class has been appended on top of the dataframe classes; this new class stores some extra metadata.

If you're interested in two variables (`age` and `partyid`, for example) you can `specify()` their relationship in one of two (equivalent) ways:


```r
# as a formula
gss %>%
  specify(age ~ partyid)
#> Response: age (numeric)
#> Explanatory: partyid (factor)
#> # A tibble: 500 × 2
#>      age partyid
#>    <dbl> <fct>  
#>  1    36 ind    
#>  2    34 rep    
#>  3    24 ind    
#>  4    42 ind    
#>  5    31 rep    
#>  6    32 rep    
#>  7    48 dem    
#>  8    36 ind    
#>  9    30 rep    
#> 10    33 dem    
#> # … with 490 more rows

# with the named arguments
gss %>%
  specify(response = age, explanatory = partyid)
#> Response: age (numeric)
#> Explanatory: partyid (factor)
#> # A tibble: 500 × 2
#>      age partyid
#>    <dbl> <fct>  
#>  1    36 ind    
#>  2    34 rep    
#>  3    24 ind    
#>  4    42 ind    
#>  5    31 rep    
#>  6    32 rep    
#>  7    48 dem    
#>  8    36 ind    
#>  9    30 rep    
#> 10    33 dem    
#> # … with 490 more rows
```

If you're doing inference on one proportion or a difference in proportions, you will need to use the `success` argument to specify which level of your `response` variable is a success. For instance, if you're interested in the proportion of the population with a college degree, you might use the following code:


```r
# specifying for inference on proportions
gss %>%
  specify(response = college, success = "degree")
#> Response: college (factor)
#> # A tibble: 500 × 1
#>    college  
#>    <fct>    
#>  1 degree   
#>  2 no degree
#>  3 degree   
#>  4 no degree
#>  5 degree   
#>  6 no degree
#>  7 no degree
#>  8 degree   
#>  9 degree   
#> 10 no degree
#> # … with 490 more rows
```

## Declare the hypothesis

The next step in the infer pipeline is often to declare a null hypothesis using `hypothesize()`. The first step is to supply one of "independence" or "point" to the `null` argument. If your null hypothesis assumes independence between two variables, then this is all you need to supply to `hypothesize()`:


```r
gss %>%
  specify(college ~ partyid, success = "degree") %>%
  hypothesize(null = "independence")
#> Response: college (factor)
#> Explanatory: partyid (factor)
#> Null Hypothesis: independence
#> # A tibble: 500 × 2
#>    college   partyid
#>    <fct>     <fct>  
#>  1 degree    ind    
#>  2 no degree rep    
#>  3 degree    ind    
#>  4 no degree ind    
#>  5 degree    rep    
#>  6 no degree rep    
#>  7 no degree dem    
#>  8 degree    ind    
#>  9 degree    rep    
#> 10 no degree dem    
#> # … with 490 more rows
```

If you're doing inference on a point estimate, you will also need to provide one of `p` (the true proportion of successes, between 0 and 1), `mu` (the true mean), `med` (the true median), or `sigma` (the true standard deviation). For instance, if the null hypothesis is that the mean number of hours worked per week in our population is 40, we would write:


```r
gss %>%
  specify(response = hours) %>%
  hypothesize(null = "point", mu = 40)
#> Response: hours (numeric)
#> Null Hypothesis: point
#> # A tibble: 500 × 1
#>    hours
#>    <dbl>
#>  1    50
#>  2    31
#>  3    40
#>  4    40
#>  5    40
#>  6    53
#>  7    32
#>  8    20
#>  9    40
#> 10    40
#> # … with 490 more rows
```

Again, from the front-end, the dataframe outputted from `hypothesize()` looks almost exactly the same as it did when it came out of `specify()`, but infer now "knows" your null hypothesis.

## Generate the distribution

Once we've asserted our null hypothesis using `hypothesize()`, we can construct a null distribution based on this hypothesis. We can do this using one of several methods, supplied in the `type` argument:

* `bootstrap`: A bootstrap sample will be drawn for each replicate, where a sample of size equal to the input sample size is drawn (with replacement) from the input sample data.  
* `permute`: For each replicate, each input value will be randomly reassigned (without replacement) to a new output value in the sample.  
* `simulate`: A value will be sampled from a theoretical distribution with parameters specified in `hypothesize()` for each replicate. (This option is currently only applicable for testing point estimates.)  

Continuing on with our example above, about the average number of hours worked a week, we might write:


```r
gss %>%
  specify(response = hours) %>%
  hypothesize(null = "point", mu = 40) %>%
  generate(reps = 5000, type = "bootstrap")
#> Response: hours (numeric)
#> Null Hypothesis: point
#> # A tibble: 2,500,000 × 2
#> # Groups:   replicate [5,000]
#>    replicate hours
#>        <int> <dbl>
#>  1         1  58.6
#>  2         1  35.6
#>  3         1  28.6
#>  4         1  38.6
#>  5         1  28.6
#>  6         1  38.6
#>  7         1  38.6
#>  8         1  57.6
#>  9         1  58.6
#> 10         1  38.6
#> # … with 2,499,990 more rows
```

In the above example, we take 5000 bootstrap samples to form our null distribution.

To generate a null distribution for the independence of two variables, we could also randomly reshuffle the pairings of explanatory and response variables to break any existing association. For instance, to generate 5000 replicates that can be used to create a null distribution under the assumption that political party affiliation is not affected by age:


```r
gss %>%
  specify(partyid ~ age) %>%
  hypothesize(null = "independence") %>%
  generate(reps = 5000, type = "permute")
#> Response: partyid (factor)
#> Explanatory: age (numeric)
#> Null Hypothesis: independence
#> # A tibble: 2,500,000 × 3
#> # Groups:   replicate [5,000]
#>    partyid   age replicate
#>    <fct>   <dbl>     <int>
#>  1 ind        36         1
#>  2 ind        34         1
#>  3 ind        24         1
#>  4 rep        42         1
#>  5 dem        31         1
#>  6 dem        32         1
#>  7 dem        48         1
#>  8 rep        36         1
#>  9 ind        30         1
#> 10 dem        33         1
#> # … with 2,499,990 more rows
```

## Calculate statistics

Depending on whether you're carrying out computation-based inference or theory-based inference, you will either supply `calculate()` with the output of `generate()` or `hypothesize()`, respectively. The function, for one, takes in a `stat` argument, which is currently one of `"mean"`, `"median"`, `"sum"`, `"sd"`, `"prop"`, `"count"`, `"diff in means"`, `"diff in medians"`, `"diff in props"`, `"Chisq"`, `"F"`, `"t"`, `"z"`, `"slope"`, or `"correlation"`. For example, continuing our example above to calculate the null distribution of mean hours worked per week:


```r
gss %>%
  specify(response = hours) %>%
  hypothesize(null = "point", mu = 40) %>%
  generate(reps = 5000, type = "bootstrap") %>%
  calculate(stat = "mean")
#> Response: hours (numeric)
#> Null Hypothesis: point
#> # A tibble: 5,000 × 2
#>    replicate  stat
#>        <int> <dbl>
#>  1         1  39.8
#>  2         2  39.6
#>  3         3  39.8
#>  4         4  39.2
#>  5         5  39.0
#>  6         6  39.8
#>  7         7  40.6
#>  8         8  40.6
#>  9         9  40.4
#> 10        10  39.0
#> # … with 4,990 more rows
```

The output of `calculate()` here shows us the sample statistic (in this case, the mean) for each of our 1000 replicates. If you're carrying out inference on differences in means, medians, or proportions, or `"t"` and `"z"` statistics, you will need to supply an `order` argument, giving the order in which the explanatory variables should be subtracted. For instance, to find the difference in mean age of those that have a college degree and those that don't, we might write:


```r
gss %>%
  specify(age ~ college) %>%
  hypothesize(null = "independence") %>%
  generate(reps = 5000, type = "permute") %>%
  calculate("diff in means", order = c("degree", "no degree"))
#> Response: age (numeric)
#> Explanatory: college (factor)
#> Null Hypothesis: independence
#> # A tibble: 5,000 × 2
#>    replicate    stat
#>        <int>   <dbl>
#>  1         1 -0.0378
#>  2         2  1.55  
#>  3         3  0.465 
#>  4         4  1.39  
#>  5         5 -0.161 
#>  6         6 -0.179 
#>  7         7  0.0151
#>  8         8  0.914 
#>  9         9 -1.32  
#> 10        10 -0.426 
#> # … with 4,990 more rows
```

## Other utilities

The infer package also offers several utilities to extract meaning out of summary statistics and null distributions; the package provides functions to visualize where a statistic is relative to a distribution (with `visualize()`), calculate p-values (with `get_p_value()`), and calculate confidence intervals (with `get_confidence_interval()`).

To illustrate, we'll go back to the example of determining whether the mean number of hours worked per week is 40 hours.


```r
# find the point estimate
point_estimate <- gss %>%
  specify(response = hours) %>%
  calculate(stat = "mean")

# generate a null distribution
null_dist <- gss %>%
  specify(response = hours) %>%
  hypothesize(null = "point", mu = 40) %>%
  generate(reps = 5000, type = "bootstrap") %>%
  calculate(stat = "mean")
```

(Notice the warning: `Removed 1244 rows containing missing values.` This would be worth noting if you were actually carrying out this hypothesis test.)

Our point estimate 41.382 seems *pretty* close to 40, but a little bit different. We might wonder if this difference is just due to random chance, or if the mean number of hours worked per week in the population really isn't 40.

We could initially just visualize the null distribution.


```r
null_dist %>%
  visualize()
```

<img src="figs/visualize-1.svg" width="672" />

Where does our sample's observed statistic lie on this distribution? We can use the `obs_stat` argument to specify this.


```r
null_dist %>%
  visualize() +
  shade_p_value(obs_stat = point_estimate, direction = "two_sided")
```

<img src="figs/visualize2-1.svg" width="672" />

Notice that infer has also shaded the regions of the null distribution that are as (or more) extreme than our observed statistic. (Also, note that we now use the `+` operator to apply the `shade_p_value()` function. This is because `visualize()` outputs a plot object from ggplot2 instead of a dataframe, and the `+` operator is needed to add the p-value layer to the plot object.) The red bar looks like it's slightly far out on the right tail of the null distribution, so observing a sample mean of 41.382 hours would be somewhat unlikely if the mean was actually 40 hours. How unlikely, though?


```r
# get a two-tailed p-value
p_value <- null_dist %>%
  get_p_value(obs_stat = point_estimate, direction = "two_sided")

p_value
#> # A tibble: 1 × 1
#>   p_value
#>     <dbl>
#> 1   0.046
```

It looks like the p-value is 0.046, which is pretty small---if the true mean number of hours worked per week was actually 40, the probability of our sample mean being this far (1.382 hours) from 40 would be 0.046. This may or may not be statistically significantly different, depending on the significance level `alpha` you decided on *before* you ran this analysis. If you had set `alpha = .05`, then this difference would be statistically significant, but if you had set `alpha = .01`, then it would not be.

To get a confidence interval around our estimate, we can write:


```r
# start with the null distribution
null_dist %>%
  # calculate the confidence interval around the point estimate
  get_confidence_interval(point_estimate = point_estimate,
                          # at the 95% confidence level
                          level = .95,
                          # using the standard error
                          type = "se")
#> # A tibble: 1 × 2
#>   lower_ci upper_ci
#>      <dbl>    <dbl>
#> 1     40.1     42.7
```

As you can see, 40 hours per week is not contained in this interval, which aligns with our previous conclusion that this finding is significant at the confidence level `alpha = .05`.

## Theoretical methods

The infer package also provides functionality to use theoretical methods for `"Chisq"`, `"F"`, `"t"` and `"z"` distributions. 

Generally, to find a null distribution using theory-based methods, use the same code that you would use to find the observed statistic elsewhere, replacing calls to `calculate()` with `assume()`. For example, to calculate the observed `"t"` statistic (a standardized mean):


```r
# calculate an observed t statistic
obs_t <- gss %>%
  specify(response = hours) %>%
  hypothesize(null = "point", mu = 40) %>%
  calculate(stat = "t")
```

Then, to define a theoretical `"t"` distribution, we could write:


```r
# switch out `calculate()` with `assume()` to define a distribution
t_dist <- gss %>%
  specify(response = hours) %>%
  assume(distribution = "t")
```

From here, the theoretical distribution interfaces in the same way that simulation-based null distributions do. For example, to interface with p-values:


```r
# visualize the theoretical null distribution
visualize(t_dist) +
  shade_p_value(obs_stat = obs_t, direction = "greater")
```

<img src="figs/viz_t_dist-1.svg" width="672" />

```r

# more exactly, calculate the p-value
get_p_value(t_dist, obs_t, "greater")
#> # A tibble: 1 × 1
#>   p_value
#>     <dbl>
#> 1  0.0188
```

Confidence intervals lie on the scale of the data rather than on the standardized scale of the theoretical distribution, so be sure to use the unstandardized observed statistic when working with confidence intervals.


```r
# calculate the point estimate
obs_mean <- gss %>%
  specify(response = hours) %>%
  calculate(stat = "mean")

# find the theory-based confidence interval
theor_ci <- 
  get_confidence_interval(
    x = t_dist,
    level = .95,
    point_estimate = obs_mean
  )

theor_ci
#> # A tibble: 1 × 2
#>   lower_ci upper_ci
#>      <dbl>    <dbl>
#> 1     40.1     42.7
```

When visualized, the `"t"` distribution will be recentered and rescaled to align with the scale of the observed data.


```r
# visualize the theoretical sampling distribution
visualize(t_dist) +
  shade_confidence_interval(theor_ci)
```

<img src="figs/viz_t_ci-1.svg" width="672" />

## Multiple regression

To accommodate randomization-based inference with multiple explanatory variables, the package implements an alternative workflow based on model fitting. Rather than `calculate()`ing statistics from resampled data, this side of the package allows you to `fit()` linear models on data resampled according to the null hypothesis, supplying model coefficients for each explanatory variable. For the most part, you can just switch out `calculate()` for `fit()` in your `calculate()`-based workflows.

As an example, suppose that we want to fit `hours` worked per week using the respondent `age` and `college` completion status. We could first begin by fitting a linear model to the observed data.


```r
observed_fit <- gss %>%
  specify(hours ~ age + college) %>%
  fit()
```

Now, to generate null distributions for each of these terms, we can fit 1000 models to resamples of the `gss` dataset, where the response `hours` is permuted in each. Note that this code is the same as the above except for the addition of the `hypothesize()` and `generate()` step.


```r
null_fits <- gss %>%
  specify(hours ~ age + college) %>%
  hypothesize(null = "independence") %>%
  generate(reps = 1000, type = "permute") %>%
  fit()

null_fits
#> # A tibble: 3,000 × 3
#> # Groups:   replicate [1,000]
#>    replicate term           estimate
#>        <int> <chr>             <dbl>
#>  1         1 intercept     43.5     
#>  2         1 age           -0.0453  
#>  3         1 collegedegree -0.940   
#>  4         2 intercept     41.7     
#>  5         2 age           -0.000105
#>  6         2 collegedegree -1.03    
#>  7         3 intercept     39.4     
#>  8         3 age            0.0534  
#>  9         3 collegedegree -0.354   
#> 10         4 intercept     40.4     
#> # … with 2,990 more rows
```

To permute variables other than the response variable, the `variables` argument to `generate()` allows you to choose columns from the data to permute. Note that any derived effects that depend on these columns (e.g., interaction effects) will also be affected.

Beyond this point, observed fits and distributions from null fits interface exactly like analogous outputs from `calculate()`. For instance, we can use the following code to calculate a 95% confidence interval from these objects.


```r
get_confidence_interval(
  null_fits, 
  point_estimate = observed_fit, 
  level = .95
)
#> # A tibble: 3 × 3
#>   term          lower_ci upper_ci
#>   <chr>            <dbl>    <dbl>
#> 1 age            -0.0971   0.0872
#> 2 collegedegree  -2.79     2.61  
#> 3 intercept      37.8     45.5
```

Or, we can shade p-values for each of these observed regression coefficients from the observed data.


```r
visualize(null_fits) + 
  shade_p_value(observed_fit, direction = "both")
```

<img src="figs/viz_null_fits-1.svg" width="672" />

That's it! This vignette covers most all of the key functionality of infer. See `help(package = "infer")` for a full list of functions and vignettes.


## Session information


```
#> ─ Session info ─────────────────────────────────────────────────────
#>  setting  value
#>  version  R version 4.2.1 (2022-06-23)
#>  os       macOS Big Sur ... 10.16
#>  system   x86_64, darwin17.0
#>  ui       X11
#>  language (EN)
#>  collate  en_US.UTF-8
#>  ctype    en_US.UTF-8
#>  tz       America/Los_Angeles
#>  date     2022-12-02
#>  pandoc   2.19.2 @ /Applications/RStudio.app/Contents/MacOS/quarto/bin/tools/ (via rmarkdown)
#> 
#> ─ Packages ─────────────────────────────────────────────────────────
#>  package    * version date (UTC) lib source
#>  broom      * 1.0.1   2022-08-29 [1] CRAN (R 4.2.0)
#>  dials      * 1.1.0   2022-11-04 [1] CRAN (R 4.2.0)
#>  dplyr      * 1.0.10  2022-09-01 [1] CRAN (R 4.2.0)
#>  ggplot2    * 3.4.0   2022-11-04 [1] CRAN (R 4.2.0)
#>  infer      * 1.0.4   2022-12-02 [1] CRAN (R 4.2.1)
#>  parsnip    * 1.0.3   2022-11-11 [1] CRAN (R 4.2.0)
#>  purrr      * 0.3.5   2022-10-06 [1] CRAN (R 4.2.0)
#>  recipes    * 1.0.3   2022-11-09 [1] CRAN (R 4.2.0)
#>  rlang        1.0.6   2022-09-24 [1] CRAN (R 4.2.0)
#>  rsample    * 1.1.0   2022-08-08 [1] CRAN (R 4.2.0)
#>  tibble     * 3.1.8   2022-07-22 [1] CRAN (R 4.2.0)
#>  tidymodels * 1.0.0   2022-07-13 [1] CRAN (R 4.2.0)
#>  tune       * 1.0.1   2022-10-09 [1] CRAN (R 4.2.0)
#>  workflows  * 1.1.2   2022-11-16 [1] CRAN (R 4.2.0)
#>  yardstick  * 1.1.0   2022-09-07 [1] CRAN (R 4.2.0)
#> 
#>  [1] /Library/Frameworks/R.framework/Versions/4.2/Resources/library
#> 
#> ────────────────────────────────────────────────────────────────────
```
 
