# R >= 4.4 (the notebook was run with R 4.4.1 under IRkernel)
pkgs <- c("lme4", "lmerTest", "dplyr", "readr", "tidyr", "ggplot2", "brms", "posterior", "logspline", "knitr", "IRkernel")
install.packages(setdiff(pkgs, rownames(installed.packages())))
# brms backend: cmdstanr (faster) if available, otherwise rstan
if (!requireNamespace("cmdstanr", quietly = TRUE)) {
  install.packages("cmdstanr", repos = c("https://stan-dev.r-universe.dev", getOption("repos")))
  cmdstanr::install_cmdstan()
}
# IRkernel::installspec()   # once, to use R in Jupyter
