                                   #Syeda Bareeha Ali
# Polynomial Regression
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
dataset = pd.read_csv('C:\\Users\\DELL\\Downloads\\Machine Learning Course\\assignment 2\\monthlyexp vs incom.csv')

X = dataset.iloc[:, 0:1].values
y = dataset.iloc[:, 1:2].values

# Fitting Linear Regression to the dataset
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X, y)

# Fitting Polynomial Regression to the dataset
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 6)
X_poly = poly_reg.fit_transform(X)
poly_reg.fit(X_poly, y)
lin_reg_2 = LinearRegression()
lin_reg_2.fit(X_poly, y)

# Visualising the Linear Regression results
plt.scatter(X, y, color = 'black')
plt.plot(X, lin_reg.predict(X), color = 'darkorange')
plt.title('Monthly Experience VS Income (Linear Regression)', fontname='Source Code Pro', fontsize=10)
plt.xlabel('Monthly Experience', fontname='Source Code Pro', fontsize=10)
plt.ylabel('Income', fontname='Source Code Pro', fontsize=10)
plt.grid(color='grey', linestyle='dashed', linewidth=0.5, alpha=1)
plt.show()

# Visualising the Polynomial Regression results
plt.scatter(X, y, color = 'black')
plt.plot(X, lin_reg_2.predict(poly_reg.fit_transform(X)), color = 'darkorange')
plt.title('Monthly Experience VS Income (Polynomial Regression)', fontname='Source Code Pro', fontsize=10)
plt.xlabel('Monthly Experience', fontname='Source Code Pro', fontsize=10)
plt.ylabel('Income', fontname='Source Code Pro', fontsize=10)
plt.grid(color='grey', linestyle='dashed', linewidth=0.5, alpha=1)
plt.show()

# Visualising the Polynomial Regression results (for higher resolution and smoother curve)
X_grid = np.arange(min(X), max(X), 0.01)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X, y, color = 'black')
plt.plot(X_grid, lin_reg_2.predict(poly_reg.fit_transform(X_grid)), color = 'darkorange')
plt.title(' Monthly Experience VS Income (Polynomial Regression)', fontname='Source Code Pro', fontsize=10)
plt.xlabel('Monthly Experience', fontname='Source Code Pro', fontsize=10)
plt.ylabel('Income', fontname='Source Code Pro', fontsize=10)
plt.grid(color='grey', linestyle='dashed', linewidth=0.5, alpha=1)
plt.show()

# Predicting a new result with Linear Regression
print('Prediction for 6-Months-Experience (using LR) is:',lin_reg.predict([[6]]))
# Predicting a new result with Polynomial Regression
print("Prediction for 6-Months-Experience (using PR) is:",lin_reg_2.predict(poly_reg.fit_transform([[6]])))
