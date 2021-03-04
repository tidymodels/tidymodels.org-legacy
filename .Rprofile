# REMEMBER to restart R after you modify and save this file!

# First, execute the global .Rprofile if it exists. You may configure blogdown
# options there, too, so they apply to any blogdown projects. Feel free to
# ignore this part if it sounds too complicated to you.
if (file.exists("~/.Rprofile")) {
  base::sys.source("~/.Rprofile", envir = environment())
}

library(magrittr)
hugo_version <- readLines("netlify.toml") %>% 
  stringr::str_subset("HUGO_VERSION") %>% 
  stringr::str_extract("[0-9]+\\.*[0-9]*\\.*[0-9]*") %>% 
  unique()

# fix Hugo version
options(blogdown.hugo.version = hugo_version)
