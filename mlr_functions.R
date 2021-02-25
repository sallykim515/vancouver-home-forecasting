# custom functions

# scatter plot & lowess line
plot_and_lowess <- function(data, x_name, x_unit, l_delta) {
  
  ordering <- order(data[[x_name]])
  x1 <- data[[x_name]][ordering]
  y1 <- data[["Price"]][ordering]
  
  plot(y1~ x1, pch=16, 
       xlab = paste(x_name, x_unit),
       ylab = "Price",
       main = paste("Price vs.", x_name)
  )
  lines(lowess(x1, y1, delta=l_delta), col='red')
}

# highlight points on a plot
highlight_pts <- function(data, x_name, logical){
  points(data[[x_name]][logical],
         data$Price[logical], pch=16, col='red')
}

# calculate & print MAE and RMSE
calc_mae_rmse <- function(z){
  predict.y <- predict(z, newdata=mydata.valid)
  mydata.valid2 <- as.data.frame(cbind(mydata.valid, predict.y))
  mydata.valid2$errors.y <- mydata.valid2$Price - mydata.valid2$predict.y
  
  MAE.x <- sum(abs(mydata.valid2$errors.y))/nrow(mydata.valid2); MAE.x
  RMSE.x <- sqrt(sum(mydata.valid2$errors.y^2)/nrow(mydata.valid2)); RMSE.x
  
  cat("RMSE: " ,RMSE.x, "\n")
  cat("MAE: " ,MAE.x, "\n")
}