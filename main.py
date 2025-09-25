from src.pre_processing import X_train_scaled, X_test_scaled, y_train, y_test
from src.train_model import train_model
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
import pandas as pd

def main():
    trained_model = train_model()

    y_pred = trained_model.predict(X_test_scaled)

    print("R² :", r2_score(y_test, y_pred))
    print("MAE :", mean_absolute_error(y_test, y_pred))
    print("MSE :", mean_squared_error(y_test, y_pred))

    plot_predictions_vs_true(y_test, y_pred)
    plot_features_vs_target(X_test_scaled, y_test, y_pred)

def plot_predictions_vs_true(y_true, y_pred):
    plt.figure(figsize=(10, 6))
    plt.scatter(y_true, y_pred, alpha=0.5)
    plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', linewidth=2)
    plt.xlabel("Valeurs réelles")
    plt.ylabel("Prédictions")
    plt.title("Comparaison Prédictions vs Réel")
    plt.show()

def plot_features_vs_target(X_scaled, y_true, y_pred):
    feature_names = ["MedInc","HouseAge","AveRooms","AveBedrms","Population","AveOccup","Latitude","Longitude"]
    X_df = pd.DataFrame(X_scaled, columns=feature_names)

    for col in X_df.columns:
        plt.figure(figsize=(8,5))
        plt.scatter(X_df[col], y_true, alpha=0.5, label="Réel")
        plt.scatter(X_df[col], y_pred, alpha=0.5, label="Prédit")
        plt.xlabel(col)
        plt.ylabel("Target")
        plt.title(f"Target vs {col}")
        plt.legend()
        plt.show()

if __name__ == "__main__":
    main()
