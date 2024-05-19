import numpy as np
import pandas as pd

from sklearn.compose import make_column_selector, make_column_transformer
from sklearn.metrics import classification_report
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OrdinalEncoder
from sklearn.svm import LinearSVC


def run_svc():
    wait = pd.read_csv('restaurant.csv')
    cat_columns = wait.columns.tolist()
    wait[cat_columns] = wait[cat_columns].astype("category")
    variables = wait[cat_columns[:-1]]
    print(variables)
    target = wait[cat_columns[-1]]

    ordinal_encoder = make_column_transformer(
        (
            OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=999),
            make_column_selector(dtype_include="category"),
        ),
        remainder="passthrough",
        verbose_feature_names_out=False,
    )


    svc_ordinal = make_pipeline(
        ordinal_encoder,
        #  use SVC for testing different kernels
        LinearSVC(
            loss="squared_hinge",
            tol=1e-3,
            C=1.5,
            )
    )

    svc = svc_ordinal.fit(variables, target)

    print(classification_report(target, svc.predict(variables)))


if __name__ == "__main__":
    run_svc()
