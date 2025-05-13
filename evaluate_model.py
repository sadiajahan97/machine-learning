from sklearn.metrics import mean_squared_error, r2_score


def evaluate_model(model, x_test, y_test):
    y_pred = model.predict(x_test)

    mean_squared_error_value = mean_squared_error(y_test, y_pred)

    r2_score_value = r2_score(y_test, y_pred)

    gradient = model.coef_[0]

    y_intercept = model.intercept_

    return mean_squared_error_value, r2_score_value, gradient, y_intercept
