install.packages("DAAG")
install.packages("plotly")
install.packages("dplyr")
install.packages("FactoMineR")
install.packages("factoextra")
install.packages("keras")
install.packages("tensorflow")




library(DAAG)
library(ggplot2)
library(dplyr)
library(plotly)
library(FactoMineR)
library(factoextra)
library(keras)
library(tensorflow)

ais <- ais
head(ais)

minmax <- function(x) (x - min(x)/ max(x) - min(x))
x_train <- apply(ais[, 1:11], 2, minmax)

pca <- prcomp(x_train)

fviz_screeplot(pca, ncp = 10)

qplot(x = 1:11, y = cumsum(pca$sdev)/sum(pca$sdev), geom = "line")

ggplot(as.data.frame(pca$x), aes(x = PC1, y = PC2, col = ais$sex)) + geom_point()

pca_plotly <- plot_ly(as.data.frame(pca$x), x = ~PC1, y = ~PC2, z = ~PC3, color = ~ais$sex) 

pca_plotly <- pca_plotly %>% add_markers()

pca_plotly <- pca_plotly %>% layout(scene = list(xaxis = list(title = 'PC1'),
                                   yaxis = list(title = 'PC2'),
                                   zaxis = list(title = 'PC3')))

pca_plotly


x_train <- as.matrix(x_train)

library(keras)

model <- keras_model_sequential()
model %>%
  layer_dense(units = 6, activation = "tanh", input_shape = ncol(x_train)) %>%
  layer_dense(units = 2, activation = "tanh", name = "bottleneck") %>%
  layer_dense(units = 6, activation = "tanh") %>% 
  layer_dense(units = ncol(x_train))

summary(model)

model %>% compile(
  loss = "mean_squared_error",
  optimizer = "adam"
)

model %>% fit(
  x = x_train, 
  y = x_train, 
  verbose = 0, 
  epochs = 100, 
  batch_size = 2
)

mse.ae2 <- evaluate(model, x_train, x_train)
mse.ae2
