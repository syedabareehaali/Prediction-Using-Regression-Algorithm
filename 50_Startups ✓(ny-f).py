                                           #Syeda Bareeha Ali
# Decision Tree Regression
import matplotlib.pyplot as plt
import pandas as pd
dataset = pd.read_csv('C:\\Users\\DELL\\Downloads\\Machine Learning Course\\assignment 2\\50_Startups.csv')
dataset['Sum']=dataset[['R&D Spend', 'Administration', 'Marketing Spend']].sum(axis=1)

#For NEW YORK:
Xny = dataset.loc[(dataset.State=='New York'),['Sum']]
Yny = dataset.loc[(dataset.State=='New York'),['Profit']]

from sklearn.model_selection import train_test_split
Xny_train, Xny_test, Yny_train, Yny_test = train_test_split(Xny, Yny, test_size = 0.2, random_state = 0)
from sklearn.tree import DecisionTreeRegressor
regressor = DecisionTreeRegressor(random_state = 0)
regressor.fit(Xny, Yny)
# Visualising the Decision Tree Regression results (higher resolution)
plt.scatter(Xny, Yny, color = 'darkorange')
plt.plot(Xny, regressor.predict(Xny), color = 'black')
plt.title('New York Spending vs Profit (Decision Tree Regression)')
plt.xlabel('Spending')
plt.ylabel('Profit')
plt.grid(color='grey', linestyle='dashed', linewidth=0.5, alpha=1)
plt.show()
print('New York\'s Profit for 900000 spending is:',regressor.predict([[900000]]))

#For FLORIDA:
Xf = dataset.loc[(dataset.State=='Florida'),['Sum']]
Yf = dataset.loc[(dataset.State=='Florida'),['Profit']]

from sklearn.model_selection import train_test_split
Xf_train, Xf_test, Yf_train, Yf_test = train_test_split(Xf, Yf, test_size = 0.2, random_state = 0)
from sklearn.tree import DecisionTreeRegressor
regressor = DecisionTreeRegressor(random_state = 0)
regressor.fit(Xf, Yf)
# Visualising the Decision Tree Regression results (higher resolution)
plt.scatter(Xf, Yf, color = 'darkorange')
plt.plot(Xf, regressor.predict(Xf), color = 'black')
plt.title('Florida Spending vs Profit (Decision Tree Regression)')
plt.xlabel('Spending')
plt.ylabel('Profit')
plt.grid(color='grey', linestyle='dashed', linewidth=0.5, alpha=1)
plt.show()
print('Florida\'s Profit for 900000 spending is:',regressor.predict([[900000]]))