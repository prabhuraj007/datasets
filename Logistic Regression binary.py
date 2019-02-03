import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

#importing dataset
data_preprcessing=pd.read_csv(filepath_or_buffer="D:\\MachinelearningATOZ\\Logistic Regression\\Logistic_Regression\\Social_Network_Ads.csv")
print(data_preprcessing)
#here we are taking age and expected column as independent variables
X=data_preprcessing[["Age","EstimatedSalary"]]
#taking last column purchased column  which has two values 0 and 1 only
Y=data_preprcessing[["Purchased"]].values
#FEATURE SCALING
#in making machine learning models there is possibility of calculating distance
#sqrt((x2−x1)2+(y2−y1)2) in this dataset x1 and a2 are samples from  age column
#and y1,y2 are the data from salary column we know that salary column data in housands
#and age column data just in numbers for example y1-y2 be 31oo and square of this 
#would be 961000000 and x1-x2 be 21 and square be 441 so here 96100000 is dominating 441
#so we have toput the two columns in same scale that is both columns should be in range
#from -1 to 1
from sklearn.preprocessing import StandardScaler
#here we are scaling the all the columns including dummy vriable of the country
#which we made previously
#we are not only making feature scaling not only salary and age column but also reming columns for accuracy reasons
#so all the columns data are in the same range
sc_X=StandardScaler()
#here we are fitting sc_X object to traing set and transforming
X[["Age","EstimatedSalary"]]=sc_X.fit_transform(X[["Age","EstimatedSalary"]])
data_preprcessing[["Age","EstimatedSalary"]]=sc_X.fit_transform(data_preprcessing[["Age","EstimatedSalary"]])
import seaborn as sns
#this will plot the two independent variables and one dependent variable
#here fit_reg is nothing but framing regression line  this is default as we dont 
#want regression line we are making that false
sns.lmplot('Age', 'EstimatedSalary', data_preprcessing, hue='Purchased', fit_reg=False)
#this will  Get a reference to the current figure.
fig = plt.gcf()
#this will alter the size of the graph
fig.set_size_inches(15, 10)
plt.show()


(p, n) = X.shape
theta = np.zeros((n+1,1)) # intializing theta with all zeros
ones = np.ones((p,1))
#here we are appending one column with values of ones
X = np.hstack((ones, X))
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.25,random_state=0)
'''
y=theta.X...THIS IS THE VECTORIZED FORM

h(x)=sigmoid(y)=1\1+exp(-x.theta)

'''
#here m is no of training examples used in the formula like cost function,gradient descent
m=len(Y_train)
#compute sigmoid function 
def sigmoid(x):
  return 1/(1+np.exp(-x))
#compute the cost function
def costFunction(theta, X_train, Y_train):
    J = (-1/m) * np.sum(np.multiply(Y_train, np.log(sigmoid(X_train @ theta))) 
        + np.multiply((1-Y_train), np.log(1 - sigmoid(X_train @ theta))))
    return J
'''
here @ means python code  matrix multiplication 

'''
'''
Note that we have used the sigmoid function in the costFunction above.

There are multiple ways to code cost function. Whats more important is the underlying mathematical ideas and our ability to translate them into code.

'''


iterations = 1500
alpha = 0.01

'''
Note that while this gradient looks identical to the linear regression gradient, 
the formula is actually different because linear and logistic regression have 
different definitions of hypothesis functions.
'''
#gradient descent 
def gradientdescent(X_train, Y_train, theta, alpha, iterations):
    for _ in range(iterations):
        theta = theta - ((alpha/m) * X_train.T @ (sigmoid(X_train @ theta) - Y_train))
    return theta

#calling gradient descent
theta = gradientdescent(X_train, Y_train, theta, alpha, iterations)

#calling cost function 

J = costFunction(theta, X_train, Y_train)
print(J)

hypothesis_train=sigmoid(X_train @ theta)




print(X_train[:,[1]].min())
'''
A quick glance at the training set tells us the two classes are generally 
found above and below some straight line. In order to make predictions, 
the classifier will need to identify this decision boundary.
However, before prognosticating anything, we first need the classifier 
to somehow compute buying probabilities given age and salary

z=θ0x0+θ1x1+θ2x2

x1 and x2 are the features funds and rating, respectively

x0=1
θ0, θ1 and θ2 are real numbers

hθ(x)=σ(θ0x0+θ1x1+θ2x2)=1\1+e−(θ0x0+θ1x1+θ2x2)

Specifically, this hypothesis computes the probability that a candidate will 
bought a car  given a set of features. 
But how do we know if this is a suitable hypothesis? 
Well, we can begin by settling on a probability threshold for deciding 
whether a candidate will bought or not buy the car. 
An intuitive threshold for this decision is: 
predict a candidate will bought if the bought probability equals or exceeds 50% 
(i.e., hθ(x)≥0.5), otherwise he/she is predicted to buy . 
If hθ(x)=0.5, then z=0; 
substituting this latter value into the linear combination above 
and then rearranging yields an intriguing equation:

x2=−(θ0\θ2)−(θ1\θ2)x1
Notice that this is an equation of a line; in fact, it's the equation of a line that exists on the scatterplot from earlier. However, because this line is linked to the threshold for making predictions, logically, this must be the decision boundary of the classifier! It looks like our choice for the hypothesis wasn't arbitrary after all. Now all that remains is to figure out what are the parameters (θ0, θ1 and θ2) of the hypothesis in order to identify the exact decision boundary.


'''

x1_vals = np.linspace(X_train[:,[1]].min(), X_train[:,[1]].max(), iterations)

'''
linspace Return evenly spaced numbers over a specified interval.

Returns num evenly spaced samples, calculated over the interval [start, stop].

The endpoint of the interval can optionally be excluded.
'''

x2_vals = (-theta[0, 0] - (theta[1, 0] * x1_vals)) / theta[2, 0]

sns.lmplot('Age', 'EstimatedSalary', data_preprcessing, hue='Purchased', fit_reg=False)
#this will  Get a reference to the current figure.
fig = plt.gcf()
#this will alter the size of the graph
fig.set_size_inches(15, 10)


plt.plot(
    x1_vals,
    x2_vals,
    color='black'
)

plt.show()


'''
lets test the model on the sample 
'''
#below 7th sample taken from X_train
test_example = np.array([1, -0.158074, 2.18056])
hypothesis_test=sigmoid(test_example @ theta)

'''
 
According to the probability threshold we decided on earlier, 
 
the classifier predicts the user will buy
0.5 is threshold

if hθ(x)≥0.5, then we predict a buy, otherwise a not buy.

'''

'''
Now lets quantify our model accuracy for which we will write a function rightly called accuracy

According to the probability threshold we decided on earlier, the classifier predicts the candidate will win. We can establish the classification accuracy by examining how well the classifier performs on the training set. Again, we'll make use of the threshold: if hθ(x)≥0.5, then we predict a buy, otherwise a not buy.

'''

def predict(X_train, theta):
    return sigmoid(X_train @ theta) >= 0.5

accuracy=(predict(X_train, theta) == Y_train).sum() / m
accuracy_percentage=accuracy*100

def predict_test(X_test,theta):
     return sigmoid(X_test @ theta)
 

                                                                                                                                                                                                                                                                                                                                                                          











