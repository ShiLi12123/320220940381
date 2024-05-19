import matplotlib as plt
import numpy as np
import pandas as pd

from sklearn.compose import make_column_selector, make_column_transformer
from sklearn.metrics import classification_report
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OrdinalEncoder
from sklearn.svm import LinearSVC
from sklearn.tree import plot_tree


def run_dt():
    wait = pd.read_csv('restaurant.csv')
    cat_columns = wait.columns.tolist()
    wait[cat_columns] = wait[cat_columns].astype("category")
    variables = wait[cat_columns[:-1]]
    target = wait[cat_columns[-1]]

    ordinal_encoder = make_column_transformer(
        (
            OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=np.nan),
            make_column_selector(dtype_include="category"),
        ),
        remainder="passthrough",
        verbose_feature_names_out=False,
    )


    dt_ordinal = make_pipeline(
        ordinal_encoder,
        DecisionTreeClassifier(
            criterion="entropy",
            splitter="best"
            )
    )

    tree = dt_ordinal.fit(variables, target)

    #  fig = plt.figure(figuresize=(25, 20))
    #  _ = tree.plot_tree(dt_ordinal[-1],
    #                     feature_names=cat_columns[:-1],
    #                     class_names=[cat_columns[-1]],
    #                     filled=True)

    print(classification_report(target, tree.predict(variables)))


if __name__ == "__main__":
    run_dt()
