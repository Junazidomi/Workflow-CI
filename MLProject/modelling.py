import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

data=pd.read_csv('data_clean.csv')

X=data.drop(columns=['price'])
y=data['price']

X_train,X_test, y_train,y_test=train_test_split(X,y, test_size=0.2, random_state=42)

mlflow.sklearn.autolog()

with mlflow.start_run():
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    predictions = model.predict(X_test)
    mse = mean_squared_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)
    mae=mean_absolute_error(y_test, predictions)
    
    print(f"MSE: {mse:.4f}, R2: {r2:.4f}, MAE: {mae:.4f}")
