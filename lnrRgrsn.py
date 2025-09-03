import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#loading dataset
data = pd.read_csv('synthetic_dataset.csv')

# Convert the pandas DataFrame to a NumPy array before slicing
data_np = data.to_numpy()

# Now, you can slice the NumPy array
X = data_np[:, 0]
y = data_np[:, 1].reshape(y.size,1)

X = np.vstack((np.ones((X.size,)),X)).T

plt.scatter(X[:,1],y)

def model(X,y,learning_rate,iterations):
    m = y.size
    theta = np.zeros((2,1))
    
    for _ in range(iterations):
        y_pred = np.dot(X,theta)
        cost = (1/(2*m))*(np.sum(np.square(y_pred-y)))
        
        d_theta = (1/m)*(np.dot(X.T,y_pred-y))
        theta= theta - (learning_rate*d_theta)
    return theta

iterations = 600
learning_rate = 0.000007
theta = model(X,y,learning_rate,iterations)

np.dot([1,40],theta)