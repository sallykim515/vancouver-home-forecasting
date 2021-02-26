from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt


def build_decision_tree(train_X, train_y, max_leaf_nodes=None, random_state=1):
    model = DecisionTreeRegressor(random_state=random_state, max_leaf_nodes=max_leaf_nodes)
    model.fit(train_X, train_y)
    return model


def get_mae(model, val_X,  val_y):
    preds = model.predict(val_X)
    mae = mean_absolute_error(val_y, preds)
    return mae


# visualize maes & return optimal # of leaf nodes
def viz_maes_and_opt_leaf(train_X, val_X, train_y, val_y, display=True):
    nodes = []
    maes = []
    for max_leaf_node in range(2, 100, 1):
        nodes.append(max_leaf_node)
        model=build_decision_tree(train_X, train_y, max_leaf_node, random_state=1)
        maes.append(get_mae(model, val_X, val_y))

    if display:
        fig, ax = plt.subplots()
        ax.plot(nodes, maes)
        ax.set(title='MAE vs. Max # of Leaf Nodes',
               xlabel='Max # of Leaf Nodes',
               ylabel='MAE')
        plt.grid(True, linestyle='--')
        plt.show()

    return nodes[maes.index(min(maes))]


def build_random_forest(train_X, train_y, random_state=1):
    model = RandomForestRegressor(random_state=random_state)
    model.fit(train_X, train_y)
    return model
