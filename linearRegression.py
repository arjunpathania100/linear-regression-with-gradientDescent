import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#loading dataset
data = pd.read_csv('synthetic_dataset.csv')

#dependent and independent variable
X = data['feature_1']
y = data['target']

#plot the scatter plot to see spread of data
plt.scatter(X,y)
plt.xlabel('Feature1')
plt.ylabel('Target')
plt.show()

#calculate m & c
xysum = np.sum(X*y)

xsum = np.sum(X)

ysum = np.sum(y)

xsqsum = np.sum(X*X)

n = len(X)

#m & c should be scalar values
m = (n*(xysum)-xsum*ysum)/(n*xsqsum-xsum**2)
c = y.mean()-m*X.mean() 

#calculate predicted value
y_predicted = m*X+c

#scatter plot
X = data['feature_1']
y = data['target']

plt.scatter(X,y)
plt.xlabel('Feature1')
plt.ylabel('Target')
#linear regesssion
plt.plot(X,y_predicted,color='red')
plt.show()

ss_total = np.sum((y - y.mean())**2)
ss_res = np.sum((y - y_predicted)**2)
r2 = 1 - (ss_res/ss_total)

print("R² Score:", round(r2,2))
