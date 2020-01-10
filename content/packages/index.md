---
title: Tidymodels packages
---

## Installation and use

* Install many of the packages in the tidymodels organization by running `install.packages("tidymodels")`.

* Run `library(tidymodels)` to load the core packages and make it available in your current R session.

Learn more about the tidymodels package at <http://tidymodels.tidymodels.org>.

## Design goals

The goal of these packages are:

 * Encourage empirical validation and good methodology.

 * Smooth out diverse interfaces.
 
 * Build highly reusable infrastructure.

 * Enable a wider variety of methodologies.

 * Help package developers to quickly make high quality model packages of their own.

These package are guided by our [principles for creating modeling packages](https://tidymodels.github.io/model-implementation-principles/). 

<div class="package-section">

<div class="package-section-info">
  <h2 id="core-tidymodels">Core tidymodels</h2>
  <p>The core tidymodels packages work together to enable a wide variety of modeling approaches:</p>
</div>

<div class="packages">

  <div class="package">
    <img class="package-image" src="/images/recipes.png" alt=""></img>
    <div class="package-info">
      <h3><a href="https://tidymodels.github.io/rsample/"> recipes </a></h3>
      <p>recipes is a tidy interface to a large number of data pre-processing tools that can be used for feature engineering. <a href="https://tidymodels.github.io/rsample/" aria-hidden="true">Learn more ...</a></p>
    </div>
  </div>
  <div class="package">
    <img class="package-image" src="/images/parsnip.png" alt=""></img>
    <div class="package-info">
      <h3><a href="https://tidymodels.github.io/parsnip/"> parsnip </a></h3>
      <p>parsnip is a tidy, unified interface to models that can be used to try a range of models without getting bogged down in the syntactical minutiae of the underlying package. <a href="https://tidymodels.github.io/parsnip/" aria-hidden="true">Learn more ...</a></p>
    </div>
  </div>  
  <div class="package">
    <img class="package-image" src="/images/tune.png" alt=""></img>
    <div class="package-info">
      <h3><a href="https://tidymodels.github.io/tune/"> tune </a></h3>
      <p> tune helps optimize your the hyper-parameters of your model and pre-processing steps. <a href="https://tidymodels.github.io/tune/" aria-hidden="true">Learn more ...</a></p>
    </div>
  </div>  
    <div class="package">
    <img class="package-image" src="/images/rsample.png" alt=""></img>
    <div class="package-info">
      <h3><a href="https://tidymodels.github.io/rsample/">rsample</a></h3>
      <p>rsample provides infrastructure for data splitting and resampling <a href="https://tidymodels.github.io/rsample/" aria-hidden="true">Learn more ...</a></p>
    </div>
  </div>
  <div class="package">
    <img class="package-image" src="/images/yardstick.png" alt=""></img>
    <div class="package-info">
      <h3><a href="https://tidymodels.github.io/yardstick/"> yardstick </a></h3>
      <p>The effectiveness of a model can be measured using various performance metrics in yardstick <a href="https://tidymodels.github.io/yardstick/" aria-hidden="true">Learn more ...</a></p>
    </div>
  </div>
  <div class="package">
    <img class="package-image" src="/images/broom.png" alt=""></img>
    <div class="package-info">
      <h3><a href="https://broom.tidyverse.org/"> broom </a></h3>
      <p>broom can take common R objects and convert their information into useable and predictable formats. 
      <a href="https://broom.tidyverse.org/" aria-hidden="true">Learn more ...</a></p>
    </div>
  </div>
  <div class="package">
    <img class="package-image" src="/images/dials.png" alt=""></img>
    <div class="package-info">
      <h3><a href="https://dials.tidyverse.org/"> dials </a></h3>
      <p>dials works with tuning parameters and can create parameter grids. 
      <a href="https://tidymodels.github.io/dials/" aria-hidden="true">Learn more ...</a></p>
    </div>
  </div>  
  <div class="package">
    <img class="package-image" src="/images/tidymodels.png" alt=""></img>
    <div class="package-info">
      <h3><a href="https://tidymodels.tidymodels.org/"> tidymodels </a></h3>
      <p>tidymodels is a meta-package that installs and load the packages that you will need.  
      <a href="https://tidymodels.github.io/tidymodels/" aria-hidden="true">Learn more ...</a></p>
    </div>
  </div>
</div>
</div>

These packages can be installed and loaded _en mass_ using the `tidymodels` package. 

## Non-Core Packages

The tidymodels organization has a variety of other packages to facilitate data analysis and modeling. They are not loaded automatically with `library(tidymodels)`, so you'll need to load each one with its own call to `library()`. 

These packages include: 

* [`infer`](http://infer.netlify.com/) is a high-level API for for tidyverse-friendly statistical inference.

* [`tidypredict`](https://tidymodels.github.io/tidypredict/) and [`modeldb`](https://tidymodels.github.io/modeldb/) can convert prediction equations to different languages (e.g. SQL) and fit some models in-database. 

* [`hardhat`](https://tidymodels.github.io/hardhat/) helps beginners setup high-quality R packages for modeling. 

* There are several add-on packages for creating recipes. [`embed`](https://tidymodels.github.io/embed/) contains steps to create embeddings or projections of predictors. [`textrecipes`](https://tidymodels.github.io/textrecipes/) has extra steps for text processing, and [`themis`](https://tidymodels.github.io/themis/) can help alleviate class imbalances using sampling methods. 

* `parsnip` also has additional packages that contain model definitions. [`discrim`](https://tidymodels.github.io/discrim/) contains definitions for discriminant analysis models and [`rules`](https://tidymodels.github.io/rules/) does the same for rule-based classification and regression models. 

* Some R objects become abnormally large when saved to disk. The [`butcher`](https://tidymodels.github.io/butcher/) package can reduce the size of those objects by removing the sub-components. 

* To know whether the data that you are predicting are _extrapolations_ from the training set, [`applicable`](https://tidymodels.github.io/applicable/) can produce metrics the measure extrapolation. 

* The [`corrr`](https://tidymodels.github.io/corrr/) package has tidy interfaces for working with correlation matrices. 

* [`probably`](https://tidymodels.github.io/probably/) has tools for post-processing class probability estimates.

* The [`tidyposterior`](https://tidymodels.github.io/tidyposterior/) package can let users make formal statistical comparisons between models using resampling and Bayesian methods. 



## Get help

If you’re asking for R help, reporting a bug, or requesting a new feature, you’re more likely to succeed if you include a good reproducible example, which is precisely what the [reprex](http://reprex.tidymodels.org/) package is meant for. You can learn more about reprex, along with other tips on how to help others help you in the [help section](https://www.tidyverse.org/help/).
