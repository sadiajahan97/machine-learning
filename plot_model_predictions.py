import matplotlib.pyplot as plt


def plot_model_predictions(model, x, y):
    plt.scatter(x, y, color="blue")

    plt.plot(x, model.predict(x), color="red")

    plt.xlabel("Years of Experience")

    plt.ylabel("Salary")

    plt.title("Salary vs Years of Experience")

    plt.show()
