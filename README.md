# EX-05 Implementation-of-Logistic-Regression-Using-Gradient-Descent
# DATE:05.10.2023

## AIM:
To write a program to implement the the Logistic Regression Using Gradient Descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import Libraries
2. Load and Prepare Data
3. Feature Scaling (Optional)
4. Define the Logistic Regression Model
5. Make Predictions
6. Evaluate the Model
7. Plot the Learning Curve (Optional)

## Program:
```
/*
Program to implement the the Logistic Regression Using Gradient Descent.
Developed by: SYED ADIL BASHA
RegisterNumber:  212221043008

import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize

data = np.loadtxt("/content/ex2data1 (1).txt",delimiter=',')
x=data[:,[0,1]]
y=data[:,2]

x[:5]

y[:5]

plt.figure()
plt.scatter(x[y==1][:,0],x[y==1][:,1],label="Admitted")
plt.scatter(x[y==0][:,0],x[y==0][:,1],label="Not admitted")
plt.xlabel("Exam 1 Score")
plt.ylabel("Exam 2 Score")
plt.legend()
plt.show()

def signoid(z):
  return 1/(1+np.exp(-z))

plt.plot()
x_plot=np.linspace(-10,10,100)
plt.plot(x_plot,signoid(x_plot))
plt.show()

def costFunction(theta,x,y):
  h=signoid(np.dot(x,theta))
  J=-(np.dot(y,np.log(h))+np.dot(1-y,np.log(1-h)))/x.shape[0]
  grad=np.dot(x.T,h-y)/x.shape[0]
  return J,grad

x_train=np.hstack((np.ones((x.shape[0],1)),x))
theta=np.array([0,0,0])
J,grad=costFunction(theta,x_train,y)
print(J)
print(grad)

x_train=np.hstack((np.ones((x.shape[0],1)),x))
theta=np.array([-24,0.2,0.2])
J,grad=costFunction(theta,x_train,y)
print(J)
print(grad)

def cost(theta,x,y):
  h=signoid(np.dot(x,theta))
  J=-(np.dot(y,np.log(h))+np.dot(1-y,np.log(1-h)))/x.shape[0]
  return J

def gradient(theta,x,y):
  h=signoid(np.dot(x,theta))
  grad=np.dot(x.T,h-y)/x.shape[0]
  return grad

x_train=np.hstack((np.ones((x.shape[0],1)),x))
theta=np.array([0,0,0])
res=optimize.minimize(fun=cost,x0=theta,args=(x_train,y),method="Newton-CG",jac=gradient)
print(res.fun)
print(res.x)

def plotDecisionBoundary(theta,x,y):
  x_min,x_max=x[:,0].min()-1,x[:,0].max()+1
  y_min,y_max=x[:,1].min()-1,x[:,1].max()+1
  xx,yy=np.meshgrid(np.arange(x_min,x_max,0.1),np.arange(y_min,y_max,0.1))
  x_plot=np.c_[xx.ravel(),yy.ravel()]
  x_plot=np.hstack((np.ones((x_plot.shape[0],1)),x_plot))
  y_plot=np.dot(x_plot,theta).reshape(xx.shape)
  plt.figure()
  plt.scatter(x[y==1][:,0],x[y==1][:,1],label="Admitted")
  plt.scatter(x[y==0][:,0],x[y==0][:,1],label="Not Admitted")
  plt.contour(xx,yy,y_plot,levels=[0])
  plt.xlabel("Exam 1 Score")
  plt.ylabel("Exam 2 Score")
  plt.legend()
  plt.show()

plotDecisionBoundary(res.x,x,y)

prob=signoid(np.dot(np.array([1,45,85]),res.x))
print(prob)

def predict(theta,x):
  x_train=np.hstack((np.ones((x.shape[0],1)),x))
  prob=signoid(np.dot(x_train,theta))
  return (prob>=0.5).astype(int)

np.mean(predict(res.x,x)==y)
*/
```

## Output:

# 1.Array Value of x:
![1](https://github.com/SYEDADILBASHA1/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/134796157/623cb045-b0c6-45e9-8d7c-6070c5023415)

# 2.Array Value of y:
![2](https://github.com/SYEDADILBASHA1/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/134796157/7d5cab4d-4f7e-4a7b-86d4-734c04081eed)

# 3.Exam 1-score graph:
![3](https://github.com/SYEDADILBASHA1/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/134796157/1babb14c-7eb9-4eb7-bb25-323a9516c9ea)

# 4.sigmoid function graph:
![4](https://github.com/SYEDADILBASHA1/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/134796157/9a7e703f-b5bf-4e19-ad8c-7941d6a193bb)

# 5.X_train_grad value:
![5](https://github.com/SYEDADILBASHA1/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/134796157/d2e61583-4348-4d4f-8bb9-dd68b6628d70)

# 6.Y_train_grad value:
![6](https://github.com/SYEDADILBASHA1/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/134796157/e68ba1ac-00c1-4add-9717-91d882fafa8d)

# 7.print res.x:
![7](https://github.com/SYEDADILBASHA1/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/134796157/88189644-a6af-4e9a-bd18-9127db594689)

# 8.Decision Boundary - graph for exam score:
![8](https://github.com/SYEDADILBASHA1/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/134796157/285d8112-4899-478f-9a24-854d2662d29a)

# 9.Probability Value:
![9](https://github.com/SYEDADILBASHA1/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/134796157/bfb3a0da-3835-403d-8e68-e651167fbe9e)

# 10.Prediction Value of mean:
![10](https://github.com/SYEDADILBASHA1/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/134796157/6f1cc1cb-06ea-4414-9454-7b0fc4d089c3)

## Result:
Thus the program to implement the the Logistic Regression Using Gradient Descent is written and verified using python programming.

