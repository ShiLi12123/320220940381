import numpy as np
import pandas as pd

from sklearn.compose import make_column_selector, make_column_transformer
from sklearn.metrics import classification_report
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OrdinalEncoder
from sklearn.naive_bayes import GaussianNB


def run_gnb():
    wait = pd.read_csv('restaurant.csv')
    cat_columns = wait.columns.tolist()
    wait[cat_columns] = wait[cat_columns].astype("category")
    variables = wait[cat_columns[:-1]]
    print(variables)
    target = wait[cat_columns[-1]]

    ordinal_encoder = make_column_transformer(
        (
            OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=np.nan),
            make_column_selector(dtype_include="category"),
        ),
        remainder="passthrough",
        verbose_feature_names_out=False,
    )


    gnb_ordinal = make_pipeline(
        ordinal_encoder,
        GaussianNB(
            )
    )

    gnb = gnb_ordinal.fit(variables, target)

    print(classification_report(target, gnb.predict(variables)))


if __name__ == "__main__":
    run_gnb()
