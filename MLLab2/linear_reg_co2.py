
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plot
fig = plot.figure()
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import Imputer

df = pd.read_csv('global_co2.csv', sep=',')
df.dropna(inplace=True)
mt = df[['Year', 'Per Capita']]
x = mt['Year'].values.reshape(-1,1)
y = mt['Per Capita']

    
X_train, X_test, y_train, y_test = train_test_split(x, y)

regressor = LinearRegression()
regressor.fit(X_train, y_train)

xx = np.linspace(min(x),max(x))
yy = regressor.predict(xx.reshape(xx.shape[0], 1))

ax = fig.gca()
ax.scatter(X_train, y_train, color="blue")
ax.scatter(X_test, y_test, color="yellow")
ax.plot(xx, yy, color='magenta')
fig.savefig('./outputs/lin_co2_def.png')

print ("R-squared: {}.".format(regressor.score(X_test, y_test)))
