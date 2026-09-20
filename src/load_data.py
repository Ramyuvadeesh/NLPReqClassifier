import arff
import pandas as pd

def load_dataset():
    dataset = arff.load(open("dataset/original/nfr.arff"))

    df = pd.DataFrame(
        dataset["data"],
        columns=[attribute[0] for attribute in dataset["attributes"]]
    )

    return df


if __name__ == "__main__":
    df = load_dataset()

    print(df.head())
    print(df.shape)