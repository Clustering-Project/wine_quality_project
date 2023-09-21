from sklearn.metrics import mean_squared_error
from math import sqrt



def train_model(model, X_train, y_train, X_val, y_val):
    
    model.fit(X_train, y_train)
    
    train_preds = model.predict(X_train)
    
    train_rmse = eval_model(y_train, train_preds)
    
    val_preds = model.predict(X_val)
    
    val_rmse = eval_model(y_val, val_preds)
    
    print(f'The train RMSE is ${round(train_rmse):,}')
    print(f'The validate RMSE is ${round(val_rmse):,}')
    
    return model



def test_model(model, X_test, y_test):
    
    model.fit(X_test, y_test)
    
    test_preds = model.predict(X_test)
    
    test_rmse = eval_model(y_test, test_preds)
    
    print(f'The test RMSE is ${round(test_rmse):,}')
    
    return model



def eval_model(y_actual, y_hat):
    
    return sqrt(mean_squared_error(y_actual, y_hat))