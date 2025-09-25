from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from src.pre_processing import *
from src.train_model import trained_model

def evaluate_model():
    y_pred = trained_model.predict(X_test_scaled)

    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)

    print('R2 score:', r2)
    print('MAE score:', mae)

evaluate_model()