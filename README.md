# vancouver-home-forecasting
A model built as a part of final project for Descriptive and Predictive Analytics course during Master of Business Analytics program. The work had been modified to identify under-valued properties for potential investment candidates.

# Overview
Using readily available information about a home, a model is built to predict its sale price. The following are considered as potential explanatory variables:
* Days on market (days)
* Total floor area (square feet)
* Age (years)
* Lot Size (square feet)

The data set ("House sale data Vancouver.csv") used in the model was collected from sales of detached homes in Vancouver from 2019 to early 2020. The sales prices are capped at most $3 million. 

# Steps
1. Explore data set. Investigate any outliers and decide whether or not to include them.
2. Build a number of models. 
3. Check assumptions for Multiple Linear Regression: linearity, equal variances and normality of residuals. Test hypothesis for significance of the regressionand interaction as appropriate.
4. Predict on out-of-sample data, and identify under-valued properties. 

# Result
Model: (Predicted Value of Home Price) = 9182 + 553.9(Total Floor Area) + 183.9(Lot Size) - 0.0361(Total Floor Area)(Lot Size)

At 99% significance level, both null hypotheses for significance of the regression and significance of the interaction are rejected. (F statistics =212.4, df regression = 3, df error =1038, p-value < 0.001; F statistic = 9.4832, df removed = 1, df error = 1038, p-value = 0.0021)

# Notes
It is interesting to note that there is an interference interaction between Total Floor Area and Lot Size.

Further, although there is a strong multicollinearity between these two variables, noted by VIF values being greater than 5, this is okay as the primary purpose of the model is to predict home sale price. 

As illustrated by the table below, the final model does not have the smallest MAE nor root MSE values. It was decided the the benefit of slightly improving the model fit is not justfied by the added complexity of the model from both cost of extra data and the interpretation of the model.

| Model |	MAE |	root MSE | model statement | 
| ----- |:---:|:--------:| ---------------:|
| Final model | 366356 | 449129	| Price ~ (Total.floor.area) + (Lot.Size) + (Total.floor.area)(Lot.Size) | 
| Model A	| 355979	| 436575	| Price ~ (Total.floor.area) + (Lot.Size) + (Age.sqrt) + (Lot.Size)(Age.sqrt)| 
| Model B	| 361677	| 439417	| Price ~ (Total.floor.area) + (Lot.Size.log) + (Age.sqrt) + (Lot.Size.log)(Age.sqrt)| 
| Model C	| 365479	| 446768	| Price ~ (Total.floor.area) + (Lot.Size) + (Age.sqrt)| 
| Model D	| 369295	| 447062	| Price ~ (Total.floor.area) + (Lot.Size.log) + (Age.sqrt)| 
| Model E	| 370458	| 458778	| Price ~ (Total.floor.area) + (Lot.Size)| 
| Model F	| 373888	| 458161	| Price ~ (Total.floor.area) + (Lot.Size.log)| 
| Model G	| 376361	| 464999	| Price ~ (Total.floor.area)| 

# Conclusion
The below is a plot of 95% prediction intervals. 

![prediction](https://user-images.githubusercontent.com/39283556/108954231-59f29300-7621-11eb-8229-5785744a6d50.png)

The red dots indicate those outside the prediction intervals, suggesting their over- and under- valuation of the home price. The exact addresses of those properties are listed below.

![properties](https://user-images.githubusercontent.com/39283556/108954806-2feda080-7622-11eb-90c8-c105a048fb66.PNG)

# Extension
As an extension to the multiple linear regression model, a decision tree model was created in Python. The below is an impact of number of leaf nodes (i.e. depth of the tree) on MAE (Mean Average Error) values, in relative percentage form to the average of home Price. 

![optimal_leaf_nodes](https://user-images.githubusercontent.com/39283556/109229449-2a09d380-7778-11eb-8c66-7da2511c3fc1.PNG)

![decision_tree_results](https://user-images.githubusercontent.com/39283556/109229456-2b3b0080-7778-11eb-8065-4194b9e99caa.PNG)

Further Random Forest model was built with an improved prediction performance:
![random_forest_mae](https://user-images.githubusercontent.com/39283556/109353129-c42d5280-7830-11eb-8d01-c51254215987.PNG)

# Notes for Further Exploration
- Identify over- and under- valued properties using the decision tree model, and compare the results from the original MLR model
- Compare the performances of Multiple Linear Regression vs. Decision Tree Regression
