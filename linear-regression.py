import numpy as np
import matplotlib.pyplot as plt

class LinearRegression:
    def __init__(self, learning_rate=0.01, epochs=1000):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.beta_0 = 0
        self.beta_1 = 0

    def fit(self, X, y):
        m = len(X)
        for _ in range(self.epochs):
            y_pred = self.predict(X)
            error = y_pred - y
            grad_beta_0 = (1 / m) * np.sum(error)
            grad_beta_1 = (1 / m) * np.sum(error * X)
            self.beta_0 -= self.learning_rate * grad_beta_0
            self.beta_1 -= self.learning_rate * grad_beta_1

    def predict(self, X):
        return self.beta_0 + self.beta_1 * X

    def cost(self, X, y):
        m = len(X)
        y_pred = self.predict(X)
        return (1 / (2 * m)) * np.sum((y_pred - y) ** 2)

np.random.seed(0)
X = 2 * np.random.rand(100, 1)
y = 4 + 3 * X + np.random.randn(100, 1)

model = LinearRegression(learning_rate=0.1, epochs=1000)
model.fit(X, y)

y_pred = model.predict(X)

plt.figure(figsize=(8, 6))
plt.scatter(X, y, color='blue', label='Data Points')
plt.plot(X, y_pred, color='red', label='Regression Line')
plt.title("Linear Regression (from scratch)")
plt.xlabel("X")
plt.ylabel("y")
plt.legend()
plt.show()

print(f"Learned Parameters: beta_0 = {model.beta_0}, beta_1 = {model.beta_1}")
