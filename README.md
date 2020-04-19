<a alt = "Netlify Deployments" href="https://app.netlify.com/sites/tidymodels-org/deploys"><img src="https://api.netlify.com/api/v1/badges/1979930f-1fd5-42cd-a097-c582d16c24d9/deploy-status" height = 20 /></a>
<a rel="license" href="http://creativecommons.org/licenses/by-sa/4.0/"><img alt="Creative Commons License" style="border-width:0" src="https://i.creativecommons.org/l/by-sa/4.0/88x31.png" height = 20 /></a>

# tidymodels.org

This repo is the source of <https://www.tidymodels.org>, and this readme tells you how it all works. 

* If you spot any small problems with the website, please feel empowered to fix 
  them directly with a PR. 
  
* If you see any larger problems, an issue is probably better: that way we can 
  discuss the problem before you commit any time to it.

This repo (and resulting website) is licensed as [CC BY-SA](LICENSE.md).

## Requirements to preview the site locally 

### R packages

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
   installed" followed by a long list of packages. Respond **"yes"**. renv will
   build the project-specific library containing packages at the correct
   versions.
1. Restart R.
1. You should now be able to render the site in all the usual ways for blogdown,
   such as `blogdown::serve_site()` or *Addins > Serve Site*.

### Hugo

In addition to R packages, you'll need to make sure that you are using the same version of Hugo that we use to build the site. If you are not familiar with Hugo, it is the static site generator that we are using via the R blogdown package. To check your local version of Hugo, you can do:

```R
# install.packages("blogdown") # if not using renv
blogdown::hugo_version()
```

Then check that against the version of Hugo we use to [build our site](https://github.com/tidymodels/tidymodels.org/blob/master/netlify.toml#L6).

If you have an older version, you can use:

```R
blogdown::update_hugo()
```

Once you are up-to-date, you can build the site locally using: 

```R
blogdown::serve_site()
```

 or *Addins > Serve Site* in the RStudio IDE.

This will open a preview of the site in your web browser, and it will 
automatically update whenever you modify one of the input files. For `.Rmarkdown` and `.Rmd` files, this will generate either a `.markdown` or an `.html` file. These rendered files need to be commited and pushed to GitHub to be published on the site.

## Structure

The source of the website is a collection of `.md`, `.Rmarkdown`, and `.Rmd` files stored in
[`content/`](content/), which are rendered for the site with 
[blogdown](https://bookdown.org/yihui/blogdown). 

* `content/packages/index.md`: this is a top-level page on the site rendered from a single `.md` file. If you only edit this page, you do not have to use `blogdown::serve_site()` locally to render.
  
* `content/start/`: these files make up a 5-part tutorial series to help users get started with tidymodels. Each article is an `.Rmarkdown` file as a page bundle, meaning that each article is in its own folder along with accompanying images, data, and rendered figures. If you edit a tutorial, please run `blogdown::serve_site()` locally to render the `.markdown` file, and be sure to commit the rendered file to the repo. No `*.Rmd` or `*.html` files should be committed in this directory. If you generate an `*.html` file locally during development, delete it once it's no longer useful to you. Keep it out of this repo. Also please make sure if you edit a file in this section that nothing is added to the `static/` folder- all accompanying files should be in the article page bundle.
    
* `content/learn/`: these files make up the articles presented in the learn section. This section is nested, meaning that inside this section, there are actually 4 subsections: `models`, `statistics`, `work`, `develop`. Each article is an `.Rmarkdown` file. If you edit or add an article, please run `blogdown::serve_site()` locally to render the `.markdown` file, and be sure to commit the rendered file to the repo. 

    When you do that, any new articles added will show up on the main `learn/` listing page automatically. By default, a maximum of 5 articles per subsection will show up in this list; use weights in the individual article YAML files to decide which 5 and their order. All articles with weights > 5 will show up when you click *“See all”* for that subsection. No `*.Rmd` or `*.html` files should be committed to this directory. If you generate an `*.html` file locally during development, delete it once it's no longer useful to you. Keep it out of this repo. Also please make sure if you edit a file in this section that nothing is added to the `static/` folder- all accompanying files should be in the article page bundle.

* `content/help/index.md`: this is a top-level page on the site rendered from a single `.md` file. If you only edit this page, you do not have to use `blogdown::serve_site()` locally to render.

* `content/contribute/index.md`: this is a top-level page on the site rendered from a single `.md` file. If you only edit this page, you do not have to use `blogdown::serve_site()` locally to render.

* `content/books/`: these files make up the books page, linked from resource stickies. To add a new book, create a new folder with a new `.markdown` file inside named `index.md`. An image file of the cover should be added in the same folder, named `cover.*`.

* `content/find/`: these files make up the find page, linked from the top navbar and resource stickies. Each of these pages is an `.Rmd` file. If you edit a page, please run `blogdown::serve_site()` locally to render the `.html` file, and be sure to commit the rendered file to the repo. Also please make sure if you edit a file in this section that nothing is added to the `static/` folder- all accompanying files should be in the article page bundle.

## Troubleshooting

If blogdown attempts to re-render posts (potentially on a massive scale), you need to make all the derived files look more recently modified than their respective source files. This affects (`.Rmarkdown`, `.markdown`) and (`.Rmd`, `.html`) file pairs. Do something like this:

```R
library(fs)

md <- dir_ls("content", recurse = TRUE, glob = "*.markdown")
file_touch(md)

html <- dir_ls("content", recurse = TRUE, glob = "*.html")
file_touch(html)
```

For other problems, consider that you need to update blogdown or to run `blogdown::update_hugo()` (perhaps in an R session launched with `sudo`).

Also, if you accidentally or intentionally knit or preview the content using another method than `blogdown::serve_site()` (e.g. click the **Preview** button in RStudio for `.[R]md`), make sure you don't commit an `.html` file from an **`.md`** file.
