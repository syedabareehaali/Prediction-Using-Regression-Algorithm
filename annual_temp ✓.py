                                         #Syeda Bareeha Ali
# Polynomial Regression
import matplotlib.pyplot as plt
import pandas as pd
dataset = pd.read_csv('C:\\Users\\DELL\\Downloads\\Machine Learning Course\\assignment 2\\annual_temp.csv')

''' ''' ''' ''' ''' ''' ''' ''' ''' ''' ''' ''' ''' ''' ''' ''' ''' ''' ''' ''' ''' ''' ''' ''' ''' ''' 
#For CGAG:
Xc= dataset.loc[(dataset.Source=='GCAG'),['Year']]
Yc = dataset.loc[(dataset.Source=='GCAG'),['Mean']]

# Fitting Linear Regression to the dataset
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(Xc, Yc)

# Fitting Polynomial Regression to the dataset
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 4)
Xc_poly = poly_reg.fit_transform(Xc)
poly_reg.fit(Xc_poly, Yc)
lin_reg_2 = LinearRegression()
lin_reg_2.fit(Xc_poly, Yc)

# Visualising the Linear Regression results
plt.scatter(Xc, Yc, color = 'darkorange')
plt.plot(Xc, lin_reg.predict(Xc), color = 'black')
plt.title('Years vs Mean Temperature of GCAG (Linear Regression)')
plt.xlabel('Year')
plt.ylabel('Mean Temperature')
plt.show()

# Visualising the Polynomial Regression results
plt.scatter(Xc, Yc, color = 'darkorange')
plt.plot(Xc, lin_reg_2.predict(poly_reg.fit_transform(Xc)), color = 'black')
plt.title('Years vs Mean Temperature of GCAG (Polynomial Regression)')
plt.xlabel('Years')
plt.ylabel('Mean Temperature')
plt.show()


# Predicting a new result with Polynomial Regression
print('Using PR Mean Temp of GCAG in 2016: ',lin_reg_2.predict(poly_reg.fit_transform([[2016]])))
print('Using PR Mean Temp of GCAG in 2017: ',lin_reg_2.predict(poly_reg.fit_transform([[2017]])))

''' ''' ''' ''' ''' ''' ''' ''' ''' ''' ''' ''' ''' ''' ''' ''' ''' ''' ''' ''' ''' ''' ''' ''' ''' ''' 
#For GISTEMP:
Xi= dataset.loc[(dataset.Source=='GISTEMP'),['Year']]
Yi = dataset.loc[(dataset.Source=='GISTEMP'),['Mean']]

# Fitting Linear Regression to the dataset
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(Xi, Yi)

# Fitting Polynomial Regression to the dataset
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 4)
Xi_poly = poly_reg.fit_transform(Xi)
poly_reg.fit(Xi_poly, Yi)
lin_reg_2 = LinearRegression()
lin_reg_2.fit(Xi_poly, Yi)

# Visualising the Linear Regression results
plt.scatter(Xi, Yi, color = 'darkorange')
plt.plot(Xi, lin_reg.predict(Xc), color = 'black')
plt.title('Years vs Mean Temperature of GISTEMP (Linear Regression)')
plt.xlabel('Year')
plt.ylabel('Mean Temperature')
plt.show()

# Visualising the Polynomial Regression results
plt.scatter(Xi, Yi, color = 'darkorange')
plt.plot(Xi, lin_reg_2.predict(poly_reg.fit_transform(Xi)), color = 'black')
plt.title('Years vs Mean Temperature of GISTEMPT (Polynomial Regression)')
plt.xlabel('Years')
plt.ylabel('Mean Temperature')
plt.show()


# Predicting a new result with Polynomial Regression
print('Using PR Mean Temp of GISTEMP in 2016: ',lin_reg_2.predict(poly_reg.fit_transform([[2016]])))
print('Using PR Mean Temp of GISTEMP in 2017: ',lin_reg_2.predict(poly_reg.fit_transform([[2017]])))


