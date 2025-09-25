from src.pre_processing import X_train_scaled, X_test_scaled, y_train, y_test, model
from src.train_model import train_model
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

def main():
    trained_model = train_model()

    y_pred = trained_model.predict(X_test_scaled)
    print("R² :", r2_score(y_test, y_pred))
    print("MAE :", mean_absolute_error(y_test, y_pred))
    print("MSE :", mean_squared_error(y_test, y_pred))

if __name__ == "__main__":
    main()
