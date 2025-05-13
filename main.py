from evaluate_model import evaluate_model
from load_data import load_data
from plot_model_predictions import plot_model_predictions
from train_model import train_model

x, y = load_data()

model, x_test, y_test = train_model(x, y)

mean_squared_error_value, r2_score_value, gradient, y_intercept = evaluate_model(
    model, x_test, y_test
)

plot_model_predictions(model, x, y)

print(f"Model: y = {gradient} * x + {y_intercept}")

print(f"Mean Squared Error: {mean_squared_error_value}")

print(f"R2 Score: {r2_score_value}")
