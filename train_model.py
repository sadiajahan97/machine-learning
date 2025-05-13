from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split


def train_model(x, y):
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=42
    )

    model = LinearRegression()

    model.fit(x_train, y_train)

    return model, x_test, y_test
