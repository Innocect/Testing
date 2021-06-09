import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression

x = np.array([10,9,2,15,10,16,11,16])
y = np.array([95,80,10,50,45,98,38,93])

linreg = LinearRegression()

x = x.reshape(-1, 1)
linreg.fit(x, y)
y_pred = linreg.predict(x)

plt.scatter(x, y)
plt.plot(x, y_pred, color="red")
plt.show()

print(linreg.coef_)
print(linreg.intercept_)
