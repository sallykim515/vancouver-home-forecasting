###########################################################
# re-do of the class project in Python, following Kaggle's micro course on "Intro to Machine Learning"
###########################################################
import decision_tree_functions as fn
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# import data
vanc_data = pd.read_csv("House sale data Vancouver.csv")
y = vanc_data.Price
X = vanc_data[["Total floor area", "Lot Size", "Age"]]

# split training and validation sets
train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=1)

# default max_leaf_nodes
model = fn.build_decision_tree(train_X, train_y, max_leaf_nodes=None, random_state=1)
mae = fn.get_mae(model, val_X, val_y)
print("%% Average MAE of default Decision Tree: \t\t\t %.2f %%" % (mae / np.mean(y) * 100))

# re-build the model using optimal # of leaf nodes
best_leaf_num = fn.viz_maes_and_opt_leaf(train_X, val_X, train_y, val_y)
model2 = fn.build_decision_tree(train_X, train_y, max_leaf_nodes=best_leaf_num, random_state=1)
mae2 = fn.get_mae(model2, val_X, val_y)
print("%% Average MAE of Decision Tree w/ optimal depth: \t %.2f %%" % (mae2 / np.mean(y) * 100))
