import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt


# code adopted from https://www.kaggle.com/dansbecker/underfitting-and-overfitting
def get_mae(max_leaf_nodes, train_X, val_X, train_y, val_y):
    model = DecisionTreeRegressor(max_leaf_nodes=max_leaf_nodes, random_state=1)
    model.fit(train_X, train_y)
    preds = model.predict(val_X)
    mae = mean_absolute_error(val_y, preds)
    return (mae)


def build_and_predict(train_X, val_X, train_y, val_y, max_leaf_nodes, random_state):

    # build the model
    vanc_model = DecisionTreeRegressor(random_state=random_state, max_leaf_nodes=max_leaf_nodes)
    vanc_model.fit(train_X, train_y)

    # predict
    predictions = vanc_model.predict(val_X)

    return val_y, predictions


# visualize maes & return optimal # of leaf nodes
def viz_maes_and_opt_leaf(train_X, val_X, train_y, val_y):
    nodes = []
    maes = []
    for max_leaf_node in range(2, 100, 1):
        nodes.append(max_leaf_node)
        maes.append(get_mae(max_leaf_node, train_X, val_X, train_y, val_y))

    fig, ax = plt.subplots()
    ax.plot(nodes, maes)
    ax.set(title='MAE vs. Max # of Leaf Nodes',
           xlabel='Max # of Leaf Nodes',
           ylabel='MAE')
    plt.grid(True, linestyle='--')
    plt.show()

    return nodes[maes.index(min(maes))]
