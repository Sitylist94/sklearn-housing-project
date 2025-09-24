from src.utils import *

def load_data():
    data = fetch_california_housing()
    df = pd.DataFrame(data.data, columns=data.feature_names)
    df["target"] = data.target
    return df

df = load_data()



