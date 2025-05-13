import kagglehub
import pandas as pd


def load_data():
    path = kagglehub.dataset_download(
        "abhishek14398/salary-dataset-simple-linear-regression"
    )

    df = pd.read_csv(f"{path}/Salary_dataset.csv")

    df = df[["YearsExperience", "Salary"]]

    x = df[["YearsExperience"]]

    y = df["Salary"]

    return x, y
