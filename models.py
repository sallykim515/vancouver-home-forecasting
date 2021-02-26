###########################################################
# applying the learning from Kaggle's micro course on "Intro to Machine Learning"
# on the previous work of MLR model in R
# Decision Tree and Random Forest modelling here
###########################################################
import models_functions as fn
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# import data
vanc_data = pd.read_csv("House sale data Vancouver.csv")
y = vanc_data.Price
X = vanc_data[["Total floor area", "Lot Size", "Age"]]

# split training and validation sets
train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=1)

# build decision tree, w/ default max_leaf_nodes
dt_model = fn.build_decision_tree(train_X, train_y, max_leaf_nodes=None, random_state=1)
dt_mae = fn.get_mae(dt_model, val_X, val_y)
print("%% Average MAE of default Decision Tree: \t\t\t %.2f %%" % (dt_mae / np.mean(y) * 100))

# re-build the decision tree, w/ optimal # of leaf nodes
best_leaf_num = fn.viz_maes_and_opt_leaf(train_X, val_X, train_y, val_y, display=False)
dt_model2 = fn.build_decision_tree(train_X, train_y, max_leaf_nodes=best_leaf_num, random_state=1)
dt_mae2 = fn.get_mae(dt_model2, val_X, val_y)
print("%% Average MAE of Decision Tree w/ optimal depth: \t %.2f %%" % (dt_mae2 / np.mean(y) * 100))

# build random_forest
rf_model = fn.build_random_forest(train_X, train_y, random_state=1)
rf_mae = fn.get_mae(rf_model, val_X, val_y)
print("%% Average MAE of Random Forest: \t %.2f %%" % (rf_mae / np.mean(y) * 100))
