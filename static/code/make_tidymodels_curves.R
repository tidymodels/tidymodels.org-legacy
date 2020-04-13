library(AppliedPredictiveModeling)
library(caret)
library(RColorBrewer)
library(tidyverse)
library(tidymodels)
library(ggpubr)

# alison added
library(beyonce)
library(showtext)
font_add_google("Lato")


set.seed(2115)
two_class_dat <- quadBoundaryFunc(1000)
two_class_dat$X1 <- -two_class_dat$X1

extras1 <- tibble(
  X1 = rnorm(100, mean = -1, sd = .5),
  X2 = rnorm(100, mean = -1.25), sd = .5,
  class = "Class2"
)

extras2 <- tibble(
  X1 = rnorm(100, mean = -2, sd = .5),
  X2 = rnorm(100, mean = 2), sd = .5,
  class = "Class2"
)

two_class_dat <- bind_rows(two_class_dat, extras1, extras2)

ggplot(two_class_dat, aes(x =X2, y = X1, col = class)) + geom_point()



rng_1 <- extendrange(two_class_dat$X1)
rng_2 <- extendrange(two_class_dat$X2)

set.seed(1)
contour_grid <- expand.grid(X1 = seq(rng_1[1], rng_1[2], length = 200),
                            X2 = seq(rng_2[1], rng_2[2], length = 200))

two_class_ctrl_rand <- trainControl(method = "none", classProbs = TRUE,
                                    search = "random", 
                                    summaryFunction = twoClassSummary)


set.seed(235)
tt <- contour_grid
for(i in 1:10) {
  
  
  bt_version <- two_class_dat[sample(1:nrow(two_class_dat), replace = TRUE),]
  
  

  nn_fit <- train(class ~ X1 + X2,
                  data = bt_version,
                  method = "nnet",
                  tuneGrid = data.frame(size = 4, decay = 2),
                  trace = FALSE,
                  metric = "ROC",
                  trControl = two_class_ctrl_rand)
  
  
  tt[[paste0("nnet", i)]] <- predict(nn_fit, contour_grid, type ="prob")[,1]
}


p <- ggplot(tt, aes(x = X2, y = X1)) + 
  theme_bw()+
  theme(
    legend.position = "none",
    axis.line = element_blank(),
    axis.text.x = element_blank(),
    axis.text.y = element_blank(),
    axis.ticks = element_blank(),
    axis.title.x = element_blank(),
    axis.title.y = element_blank(),
    panel.background = element_blank(),
    panel.border = element_blank(),
    panel.grid.major = element_blank(),
    panel.grid.minor = element_blank(),
    plot.background = element_rect(fill = "black")
  )

brewcols <- brewer.pal(9, "Set1")
cols <- beyonce_palette(66)[2:11]

for(i in seq_along(cols))
  p <- 
  p + 
  geom_contour(
    aes_string(z = paste0("nnet", i)),
    breaks = .5,
    col = cols[i],
    lwd = .7,
    alpha = 1
  )

card <- p + 
  theme_void() + 
  theme_transparent() +
  theme(panel.background = element_rect(fill = "#1a162d")) +
  expand_limits(y = c(-3, 1)) +
  coord_cartesian(xlim = c(-4.5, 4.5)) +
  annotate("text", x = 0, y = -1.5, label = "tidymodels", 
           colour = "white", size = 12, family = "Lato")

ggsave(here::here("static/code/curves_card.jpg"))

