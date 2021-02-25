#setwd("C:/Users/sally/Documents/GitHub/vancouver-home-forecasting")

# 1. exploratory analysis
# 2. building & testing models
# 3. check MLR assumptions
# 4. prediction

source("functions.R")
library(car);
library(MASS);
library(MPV);


# Import the data:
mydata <- read.csv(file="House sale data Vancouver.csv", header=TRUE)

mydata.train <- mydata[1:1042,]
mydata.valid <- mydata[1043:nrow(mydata),]

###########################################################
# 1. Exploratory Data Analysis
###########################################################

# DAYS ON MARKET
x_name = "Days.on.market"
plot_and_lowess(mydata.train, x_name, "(days)", 10)
highlight_pts(mydata, x_name, 
              mydata[[x_name]] > 400)

# AGE
x_name = "Age"
plot_and_lowess(mydata.train, x_name, "(years)", 10)

# TOTAL FLOOR AREA
x_name = "Total.floor.area"
plot_and_lowess(mydata.train, x_name, "(square feet)", 10)

# outliers
highlight_pts(mydata, x_name, 
              mydata[[x_name]] < 1000 & mydata$Price > 2500000)
highlight_pts(mydata, x_name, 
              mydata[[x_name]] < 500 & mydata$Price > 2000000)
highlight_pts(mydata, x_name, 
              mydata[[x_name]]> 1200 & mydata[[x_name]] < 1800 
              & mydata$Price > 2500000)
highlight_pts(mydata, x_name, 
              mydata[[x_name]] > 2500 & mydata[[x_name]] < 3000 
              & mydata$Price < 1000000)

# LOT SIZE
x_name = "Lot.Size"
plot_and_lowess(mydata.train, x_name, "(square feet)", 10)
highlight_pts(mydata, x_name, 
              mydata[[x_name]] > 12000)
highlight_pts(mydata, x_name, 
              mydata[[x_name]] > 10000 & mydata$Price < 2500000)


# transformations
mydata.train$Days.on.market.log <- log(mydata.train$Days.on.market)
mydata.train$Age.sqrt <- sqrt(mydata.train$Age+1)
mydata.train$Lot.Size.log <- log(mydata.train$Lot.Size)

###########################################################
# 2. Fit Model
###########################################################
z.final <- lm(Price ~ Total.floor.area*Lot.Size, data=mydata.train)
summary(z.final)

###########################################################
# 3. Check Assumptions: Linearity, Equal variance, Normality
###########################################################

# prep
resid.final <- resid(z.final)
predict.final <- predict(z.final)

# Linearity assumption & Equal variance assumption
plot(resid.final ~ predict.final, 
     main='Residual plot',
     ylab='Residuals',
     xlab='Predicted Price',
     pch=16)
abline(0,0)

br <- (max(predict.final) - min(predict.final))/3
abline(v=min(predict.final)+br/2, lty=1, col='red')
abline(v=min(predict.final)+br*3/2, lty=1, col='red')
abline(v=min(predict.final)+br*5/2, lty=1, col='red')

# Normality assumption
#hist
hist(resid.final,
     main='Histogram of Residuals',
     xlab='Residuals'
)
#qqnorm
qqnorm(resid.final, ylab= "standardized residuals", xlab = "Normal scores", pch=16)
qqline(resid.final)

# Goodness of fit
cat("R^2: " , summary(z.final)$r.squared, "\n")
cat("Residual Standard Errors: " , summary(z.final)$sigma)

# VIF
vif(z.final) # VIF >> 5 but okay since the purpose is to predict

# Hypothesis testing
# Significance of the regression (F-test)
summary(z.final)  # reject H0, the regression is significant

# Significance of the interaction
drop1(z.final, test="F")  # reject H0, the interaction term is significant

###########################################################
# 4. Prediction
###########################################################
mydata.train <- mydata.train[order(mydata.train$Total.floor.area),]
mydata.valid <- mydata.valid[order(mydata.valid$Total.floor.area),]
price.pred <- data.frame(predict(z.final, newdata=mydata.valid, 
                                 interval="prediction", level=0.95))

plot(Price ~ Total.floor.area, data=mydata.valid,
     main="Over- and Under- valued Properties",
     xlab="Total Floor Area (sq.ft.)",
     ylab="Price ($)"
     )
lines(price.pred$fit ~ mydata.valid$Total.floor.area)
lines(price.pred$lwr ~ mydata.valid$Total.floor.area, lty=2, col='blue')
lines(price.pred$upr ~ mydata.valid$Total.floor.area, lty=2, col='blue')

# color under-valued properties (ie, outside 95% prediction interval)
highlight_pts(mydata.valid, "Total.floor.area", mydata.valid$Price < price.pred$lwr)
highlight_pts(mydata.valid, "Total.floor.area", mydata.valid$Price > price.pred$upr)

# under-valued properties
print(mydata.valid$Address[mydata.valid$Price < price.pred$lwr])

# over-valued properties
print(mydata.valid$Address[mydata.valid$Price > price.pred$upr])




