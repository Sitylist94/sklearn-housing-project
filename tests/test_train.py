from sklearn.model_selection import train_test_split
from src.data_loader import load_data


def test_split():
    df = load_data()
    X = df.drop("target", axis=1)
    y = df["target"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    assert len(X_train) == int(0.8 * len(X))
    assert len(X_test) == int(0.2 * len(X))
    print("Test de split OK !")


test_split()