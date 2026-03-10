import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load dataset
data = pd.read_csv("Advertising.csv")

X = data['TV'].values
y = data['Sales'].values

# LWR Function
def lwr(x_query, X, y, tau=50):
    m = len(X)

    weights = np.exp(-(X - x_query)**2 / (2 * tau**2))

    X_design = np.vstack([np.ones(m), X]).T
    W = np.diag(weights)

    theta = np.linalg.pinv(X_design.T @ W @ X_design) @ (X_design.T @ W @ y)

    return theta[0] + theta[1] * x_query


# Predict for plotting
y_pred = [lwr(x, X, y) for x in X]

# Plot graph
plt.scatter(X, y, label="Actual Data")
plt.plot(X, y_pred, label="LWR Curve")
plt.xlabel("TV Advertising Budget")
plt.ylabel("Sales")
plt.title("Locally Weighted Regression - Advertising Dataset")
plt.legend()

plt.show()