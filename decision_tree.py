###########################################################
# re-do of the class project in Python, following Kaggle's micro course on "Intro to Machine Learning"
###########################################################
import decision_tree_functions as fn
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

# import data
vanc_data = pd.read_csv("House sale data Vancouver.csv")
y = vanc_data.Price
X = vanc_data[["Total floor area", "Lot Size", "Age"]]  # vanc_data.drop(["Price", "Address", "List Date"], axis=1)

# out of sample
train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=1)

# default max_leaf_nodes
val_y, predictions = fn.build_and_predict(train_X, val_X, train_y, val_y, max_leaf_nodes=None, random_state=1)
print("%% Average MAE of default Decision Tree: \t\t\t %.2f %%" % (mean_absolute_error(val_y, predictions) / np.mean(y) * 100))

# re-build the model using optimal # of leaves
best_leaf_num = fn.viz_maes_and_opt_leaf(train_X, val_X, train_y, val_y)
val_y, predictions2 = fn.build_and_predict(train_X, val_X, train_y, val_y, max_leaf_nodes=best_leaf_num, random_state=1)
print("%% Average MAE of Decision Tree w/ optimal depth: \t %.2f %%" % (mean_absolute_error(val_y, predictions2) / np.mean(y) * 100))
