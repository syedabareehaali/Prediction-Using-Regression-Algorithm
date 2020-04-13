                                          #Syeda Bareeha Ali
# Polynomial Regression
import matplotlib.pyplot as plt
import pandas as pd
dataset = pd.read_csv('C:\\Users\\DELL\\Downloads\\Machine Learning Course\\assignment 2\\housing price.csv')

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
plt.scatter(X, y, color = 'black')
plt.plot(X, lin_reg.predict(X), color = 'darkorange')
plt.title('House ID vs Price (Linear Regression)')
plt.xlabel('House ID')
plt.ylabel('Price')
plt.show()

# Visualising the Polynomial Regression results
plt.scatter(X, y, color = 'black')
plt.plot(X, lin_reg_2.predict(poly_reg.fit_transform(X)), color = 'darkorange')
plt.title('House ID vs Price(Polynomial Regression)')
plt.xlabel('House ID')
plt.ylabel('Price')
plt.show()

# Predicting a new result
houseID=int(input('Enter House ID:' ))
if houseID>2919 or houseID<1461:
    print("Invalid House ID")
else:    
   y_pred_LR = lin_reg.predict([[houseID]])
   y_pred_PR=lin_reg_2.predict(poly_reg.fit_transform([[houseID]]))
   print('House Price using LR is:' ,y_pred_LR)
   print('House Price using PR is:' ,y_pred_PR)
   
