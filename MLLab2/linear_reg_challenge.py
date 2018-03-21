
import numpy as np
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plot
fig = plot.figure()
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

X = []
Y = []
for line in open('challenge_dataset.txt'):
    x, y = line.split(',')
    X.append([float(x)])
    Y.append([float(y)])
    
X_train, X_test, y_train, y_test = train_test_split(X, Y)

regressor = LinearRegression()
regressor.fit(X_train, y_train)

xx = np.linspace(min(X),max(X))
yy = regressor.predict(xx.reshape(xx.shape[0], 1))

ax = fig.gca()
ax.scatter(X_train, y_train, color="blue")
ax.scatter(X_test, y_test, color="yellow")
ax.plot(xx, yy, color='magenta')
fig.savefig('./outputs/lin_chal.png')

print ("R-squared: {}.".format(regressor.score(X_test, y_test)))
