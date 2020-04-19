<a alt = "Netlify Deployments" href="https://app.netlify.com/sites/tidymodels-org/deploys"><img src="https://api.netlify.com/api/v1/badges/1979930f-1fd5-42cd-a097-c582d16c24d9/deploy-status" height = 20 /></a>
<a rel="license" href="http://creativecommons.org/licenses/by-sa/4.0/"><img alt="Creative Commons License" style="border-width:0" src="https://i.creativecommons.org/l/by-sa/4.0/88x31.png" height = 20 /></a>

# tidymodels.org

This repo is the source of <https://www.tidymodels.org>, and this readme tells you how it all works. 

* If you spot any small problems with the website, please feel empowered to fix 
  them directly with a PR. 
  
* If you see any larger problems, an issue is probably better: that way we can 
  discuss the problem before you commit any time to it.

This repo (and resulting website) is licensed as [CC BY-SA](LICENSE.md).

# Requirements to preview the site locally 

This blogdown site uses renv to create a project-specific library of packages. The [renv package](https://rstudio.github.io/renv/index.html) uses a [snapshot and restore](https://environments.rstudio.com/snapshot.html) strategy to create **r**eproducible **env**vironments for R projects. A project that uses renv has its own project-specific library that is separate from your personal library of packages. This helps contributors ensure we're all using the same version of packages and that the public site is also being built from these versions.

1. Get a local copy of the website source.
   * Users of devtools/usethis can do:
     ```r
     usethis::create_from_github(“tidymodels/tidymodels.org”)
     ```
     Note that `usethis::create_from_github()` works best when it can find a
     GitHub personal access token and usethis (git2r, really) is configured
     correctly for your preferred transport protocol (SSH vs HTTPS).
     [Setup advice](https://usethis.r-lib.org/articles/articles/usethis-setup.html).
   * Otherwise, use your favorite method to fork and clone or download the
     repo as a ZIP file and unpack.
1. Start R in your new `tidymodels.org/` directory. Expect to see some renv startup
   along these lines:
   ```
   * Project '~/rsites/tidymodels.org' loaded. [renv 0.9.3]
   Error in loadNamespace(name) : there is no package called ‘rmarkdown’
   ```
1. Run `renv::restore()`. This will print out "The following package(s) will be
   installed" followed by a long list of packages. Respond *"yes"*. renv will
   build the project-specific library containing packages at the correct
   versions.
1. Restart R.
1. You should now be able to render the site in all the usual ways for blogdown,
   such as `blogdown::serve_site()` or *Addins > Serve Site*.
