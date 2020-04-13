                                         #Syeda Bareeha Ali
# Polynomial Regression
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
dataset = pd.read_csv('C:\\Users\\DELL\\Downloads\\Machine Learning Course\\assignment 2\\global_co2.csv')


X = dataset.iloc[:, 0:1].values
y = dataset.iloc[:, 1:2].values

# Fitting Linear Regression to the dataset
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X, y)

# Fitting Polynomial Regression to the dataset
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 4)
X_poly = poly_reg.fit_transform(X)
poly_reg.fit(X_poly, y)
lin_reg_2 = LinearRegression()
lin_reg_2.fit(X_poly, y)

# Visualising the Linear Regression results
plt.scatter(X, y, color = 'darkorange')
plt.plot(X, lin_reg.predict(X), color = 'black')
plt.title('Annual Global CO2 Production (Linear Regression)')
plt.xlabel('Years')
plt.ylabel('Production')
plt.show()

# Visualising the Polynomial Regression results
plt.scatter(X, y, color = 'darkorange')
plt.plot(X, lin_reg_2.predict(poly_reg.fit_transform(X)), color = 'black')
plt.title('Annual Global CO2 Production  (Polynomial Regression)')
plt.xlabel('Years')
plt.ylabel('Production')
plt.show()

# Visualising the Polynomial Regression results (for higher resolution and smoother curve)
X_grid = np.arange(min(X), max(X), 0.1)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X, y, color = 'darkorange')
plt.plot(X_grid, lin_reg_2.predict(poly_reg.fit_transform(X_grid)), color = 'black')
plt.title('Annual Global CO2 Production  (Polynomial Regression)')
plt.xlabel('Years')
plt.ylabel('Production')
plt.show()

# Predicting a new result with Polynomial Regression
print('Prediction for 2011 using PR is',end=' ')
print(lin_reg_2.predict(poly_reg.fit_transform([[2011]])))
print('Prediction for 2012 using PR is',end=' ')
print(lin_reg_2.predict(poly_reg.fit_transform([[2012]])))
print('Prediction for 2013 using PR is',end=' ')
print(lin_reg_2.predict(poly_reg.fit_transform([[2013]])))
