import pandas as pd
from pgmpy.estimators import HillClimbSearch, BicScore
from pgmpy.models import BayesianNetwork
from pgmpy.inference import VariableElimination

from .config import PREPROCESSED_CSV_PATH


def main():
    df = pd.read_csv(PREPROCESSED_CSV_PATH)

    # For simplicity, discretize income and sleep duration
    if "Income" in df.columns:
        df["IncomeBucket"] = pd.qcut(df["Income"], q=4, labels=["Q1", "Q2", "Q3", "Q4"])

    if "Sleep duration(Hrs)" in df.columns:
        df["SleepBucket"] = pd.cut(df["Sleep duration(Hrs)"],
                                   bins=[0, 5, 7, 9, 24],
                                   labels=["<5", "5-7", "7-9", ">9"])

    # Select a subset of columns to keep BN manageable
    cols = [
        "Year",
        "Indian States",
        "Psychosocial Factors",
        "SleepBucket",
        "IncomeBucket",
        "Physical Activity",
        "Psychological Health"
    ]
    df_bn = df[cols].dropna()

    hc = HillClimbSearch(df_bn)
    best_model = hc.estimate(scoring_method=BicScore(df_bn))

    model = BayesianNetwork(best_model.edges())
    model.fit(df_bn)

    infer = VariableElimination(model)

    # Example inference: P(Psychological Health | IncomeBucket=Q4, SleepBucket='7-9')
    q = infer.query(
        variables=["Psychological Health"],
        evidence={"IncomeBucket": "Q4", "SleepBucket": "7-9"}
    )
    print(q)


if __name__ == "__main__":
    main()
