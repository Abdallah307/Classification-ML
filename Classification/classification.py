import numpy as np
import pandas as pd
from matplotlib import pyplot as plt



data = pd.read_csv('./marks.csv', names=['Exam1', 'Exam2', 'Admitted'])

# print(data.head())


positive = data[  data['Admitted'].isin([1]) ] 
negative = data[  data['Admitted'].isin([0]) ]



# fig, ax = plt.subplots(figsize=(8, 5))
# ax.scatter(positive['Exam1'], positive['Exam2'], s=50, c='g', 
#            marker='o', label='Admitted')


# ax.scatter(negative['Exam1'], negative['Exam2'], s=50, c='r', 
#            marker='x', label='Not Admitted')

# ax.legend()

# ax.set_xlabel('Exam1 Score')
# ax.set_ylabel('Exam2 Score')


def sigmoid(z) :
    return 1 / (1 + np.exp(-z))

    
nums = np.arange(-10, 10, step=1)


fig, ax = plt.subplots(figsize = (8, 5))

ax.plot(nums, sigmoid(nums), 'r')


def cost(thetav, Xv, yv) :
    thetav = np.matrix(thetav)
    first_term = np.multiply(-yv, np.log(sigmoid(Xv * thetav.T)))
    second_term = np.multiply((1 - yv), np.log(1 - sigmoid(Xv * thetav.T)))
    
    return np.sum(first_term - second_term) / (len(Xv))



data.insert(0, 'Ones', 1)



# set X (training data) and y (target data)

cols = data.shape[1]

X = data.iloc[:, :cols - 1]

y = data.iloc[:, cols-1:cols]


# convert to matrices and initialize the parameters arry theta

X = np.matrix(X.values)

y = np.matrix(y.values)

theta = np.zeros([X.shape[1]])



this_cost = cost(theta, X, y)


def gradient(thetav, Xv, yv) :
    thetav = np.matrix(thetav)
    
    parameters = int(thetav.ravel().shape[1])
    grad = np.zeros(parameters)
    
    error = sigmoid(Xv * thetav.T) - yv
    
    for i in range(parameters) :
        term = np.multiply(error, Xv[:, i])
        grad[i] = np.sum(term) / len(Xv)
        
    return grad    


import scipy.optimize as opt

result = opt.fmin_tnc(func=cost, x0=theta,
                      fprime=gradient, args=(X, y))


cost_after_optimize = cost(result[0], X, y)


print(f"cost_after_optimize : {cost_after_optimize}")


def predict(theta, X) :
    probability = sigmoid(X * theta.T)
    
    return [1 if x >= 0.5 else 0 for x in probability]

theta_min = np.matrix(result[0])
predictions = predict(theta_min, X)

correct = [1 if ((a ==1 and b ==1) or
                  (a ==0 and b ==0)) else 0
                 for (a, b) in zip(predictions, y)]

accuracy = (sum(map(int, correct)) % len(correct))

print('Accuracy = {0}%'.format(accuracy))                 


