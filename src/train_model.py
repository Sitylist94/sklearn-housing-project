from src.pre_processing import X_train_scaled, y_train, model

def train_model():
    model.fit(X_train_scaled, y_train)
    return model

trained_model = train_model()