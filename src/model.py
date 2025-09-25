from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

def modelLinear():
    model = LinearRegression()
    scaler = StandardScaler()
    return model, scaler