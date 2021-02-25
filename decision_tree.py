###########################################################
# re-do of the class project in Python, following Kaggle's micro course on "Intro to Machine Learning"
###########################################################
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt

# import data
vanc_data = pd.read_csv("House sale data Vancouver.csv")
y = vanc_data.Price
X = vanc_data[["Total floor area", "Lot Size", "Age"]]  # vanc_data.drop(["Price", "Address", "List Date"], axis=1)

# out-of-sample
train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=1)

# build the model
vanc_model = DecisionTreeRegressor(random_state=1)
vanc_model.fit(train_X, train_y)

# predict
predictions = vanc_model.predict(val_X)

# validate
print("%% Average MAE of default Decision Tree: \t\t\t %.2f %%" % (mean_absolute_error(val_y, predictions) / np.mean(y) * 100))


###########################################################
# under- vs. over- fitting
# code adopted from https://www.kaggle.com/dansbecker/underfitting-and-overfitting
###########################################################


def get_mae(max_leaf_nodes, train_X, val_X, train_y, val_y):
    model = DecisionTreeRegressor(max_leaf_nodes=max_leaf_nodes, random_state=1)
    model.fit(train_X, train_y)
    preds = model.predict(val_X)
    mae = mean_absolute_error(val_y, preds)
    return (mae)


nodes = []
maes = []
for max_leaf_node in range(2, 100, 1):
    nodes.append(max_leaf_node)
    maes.append(get_mae(max_leaf_node, train_X, val_X, train_y, val_y))

# visualization of mae
fig, ax = plt.subplots()
ax.plot(nodes, maes)
ax.set(title='MAE vs. Max # of Leaf Nodes',
       xlabel='Max # of Leaf Nodes',
       ylabel='MAE')
plt.grid(True, linestyle='--')
plt.show()

# re-build the model using optimal # of leaves
best_leaf_num = nodes[maes.index(min(maes))]
vanc_model2 = DecisionTreeRegressor(random_state=1, max_leaf_nodes=best_leaf_num)
vanc_model2.fit(train_X, train_y)

# predict
predictions2 = vanc_model2.predict(val_X)

# validate
print("%% Average MAE of Decision Tree w/ optimal depth: \t %.2f %%" % (mean_absolute_error(val_y, predictions2) / np.mean(y) * 100))
