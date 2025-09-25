from src.data_loader import load_data
from src.model import modelLinear
from sklearn.model_selection import train_test_split

df = load_data()
X = df.drop("target", axis=1)
y = df["target"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model, scaler = modelLinear()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
