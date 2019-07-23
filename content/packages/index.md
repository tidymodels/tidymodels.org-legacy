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
    <img class="package-image" src="/images/rsample.png" alt=""></img>
    <div class="package-info">
      <h3><a href="https://tidymodels.github.io/rsample/">rsample</a></h3>
      <p>rsample provides infrastructure for data splitting and resampling <a href="https://tidymodels.github.io/rsample/" aria-hidden="true">Learn more ...</a></p>
    </div>
  </div>
  <div class="package">
    <img class="package-image" src="/images/recipes.png" alt=""></img>
    <div class="package-info">
      <h3><a href="https://tidymodels.github.io/rsample/"> recipes </a></h3>
      <p>recipes is a tidy interface to a large number of data pre-processing tools that can be used for feature engineering. <a href="https://tidymodels.github.io/rsample/" aria-hidden="true">Learn more ...</a></p>
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
    <img class="package-image" src="/images/parsnip.png" alt=""></img>
    <div class="package-info">
      <h3><a href="https://tidymodels.github.io/parsnip/"> parsnip </a></h3>
      <p>parsnip is a tidy, unified interface to models that can be used to try a range of models without getting bogged down in the syntactical minutiae of the underlying package. <a href="https://tidymodels.github.io/parsnip/" aria-hidden="true">Learn more ...</a></p>
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
    <img class="package-image" src="/images/tidymodels.png" alt=""></img>
    <div class="package-info">
      <h3><a href="https://tidymodels.tidymodels.org/"> tidymodels </a></h3>
      <p>tidymodels is a meta-package that installs and load the packages that you will need.  
      <a href="https://tidymodels.tidymodels.org/" aria-hidden="true">Learn more ...</a></p>
    </div>
  </div>
</div>
</div>

The tidymodels also includes many other packages with more specialized usage. They are not loaded automatically with `library(tidymodels)`, so you'll need to load each one with its own call to `library()`. 


## Get help

If you’re asking for R help, reporting a bug, or requesting a new feature, you’re more likely to succeed if you include a good reproducible example, which is precisely what the [reprex](http://reprex.tidymodels.org/) package is meant for. You can learn more about reprex, along with other tips on how to help others help you in the [help section](https://www.tidyverse.org/help/).
