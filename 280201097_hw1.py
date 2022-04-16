import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats
import math
import pandas as pd
from scipy.stats import multivariate_normal
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from sklearn.model_selection import train_test_split

dataset = pd.read_csv("german.data",sep =" ", header=None)

accuracy = 0  # total accuracy
losses =[]
losses_without_risk = []
accuracies=[] # to store individual accuracies
first_attribute = 1
second_attribute = 12
def plot_gaussian():
    # Create graph for class 1
    # Create grid and multivariate normal
    x = np.linspace(0, 55, 1000)
    y = np.linspace(0, 55, 1000)
    X, Y = np.meshgrid(x, y)
    pos = np.empty(X.shape + (2,))
    pos[:, :, 0] = X
    pos[:, :, 1] = Y
    rv = multivariate_normal(mean1, cov)
    rv2 = multivariate_normal(mean2, cov2)

    # Make a 3D plot
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.plot_surface(X, Y, rv.pdf(pos), cmap='plasma', linewidth=0)
    ax.set_xlabel('X axis')
    ax.set_ylabel('Y axis')
    ax.set_zlabel('Z axis')
    plt.show()
    # Create graph for class 2
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.plot_surface(X, Y, rv2.pdf(pos), cmap='viridis', linewidth=0)
    ax.set_xlabel('X axis')
    ax.set_ylabel('Y axis')
    ax.set_zlabel('Z axis')
    plt.show()


# to find maximum of the discriminant function results
def find_max_likelihood(g1, g2):
    likelihood_list = [g1, g2]
    return max(likelihood_list)

# to calculate un-normalized posteriors and do classification with respect to g1 and g2results
def do_classification():
    decision_list = []  # to store classification results to calculate accuracy later.
    decision_list_with_rejection = []
    for k in range(len(testing[first_attribute])):      #calculate g(x)
        g1x = -0.5 * (np.log(np.linalg.det(cov)) + (np.array([testing[first_attribute][k], testing[second_attribute][k]]) - mean1).T.dot(np.linalg.inv(cov)).dot((np.array([testing[first_attribute][k], testing[second_attribute][k]]) - mean1)) + 2 * np.log(2 * np.pi))
        g2x = -0.5 * (np.log(np.linalg.det(cov2)) + (np.array([testing[first_attribute][k], testing[second_attribute][k]]) - mean2).T.dot(np.linalg.inv(cov2)).dot((np.array([testing[first_attribute][k], testing[second_attribute][k]]) - mean2)) + 2 * np.log(2 * np.pi))

        posterior1 = np.log(prior1) + g1x
        posterior2 = np.log(prior2) + g2x

        if posterior1 == find_max_likelihood(posterior1, posterior2):
            if posterior1-posterior2 <= 1:# if the classification resulted as class 1(good customer) and the results are too close do not make decision.
                decision_list_with_rejection.append(0)#because of cost
            else:
                decision_list_with_rejection.append(1)
            decision_list.append(1)
        else:
            decision_list.append(2)
            decision_list_with_rejection.append(2)
    risk(decision_list_with_rejection)
    return decision_list

def accuracy_score(testing, classification_result):
    accuracyList = []
    totalLoss = 0
    for n, m in zip(testing, classification_result):#check the original results and our results together for accurate guesses.
        if n == m:
            accuracyList.append(n)
        else:       #calculate total loss according to loss matrix for inaccurate guesses
            if m == 1:
                totalLoss +=5
            else:
                totalLoss +=1

    losses.append(totalLoss*100/len(testing))
    accuracy = len(accuracyList)*100/len(testing)
    return accuracy

def risk(decision_list_with_rejection):# this list includes 0 as no decision

    totalLoss = 0
    for n, m in zip(testing,
                    decision_list_with_rejection):  # check the original results and our results together for accurate guesses.
        if n == m:
            totalLoss += 0
        else:  # calculate total loss according to loss matrix for inaccurate guesses
            if m == 1:
                totalLoss += 5

            else:
                totalLoss += 1

    losses_without_risk.append(totalLoss * 100 / len(testing))



#500 times for avareage
for i in range(0, 500):
    # split data train & test
    training, testing = train_test_split(dataset, test_size=0.33)
    # calculate MUs and COVs(MLE parameters) of each class
    c12, c11, c22 = 0, 0, 0
    training = training.reset_index()   #indexes of the data creates problem so i have to reset the indexes after splitting.
    testing = testing.reset_index()
    m11 = training[first_attribute].loc[training[20] == 1].mean()   #calculate the mean value of attribute 1 class 1
    m21 = training[second_attribute].loc[training[20] == 1].mean()  #calculate the mean value of attribute 2 class 1
    m12 = training[first_attribute].loc[training[20] == 2].mean()   #calculate the mean value of attribute 1 class 2
    m22 = training[second_attribute].loc[training[20] == 2].mean()  #calculate the mean value of attribute 1 class 2

    mean1 = np.array([m11, m21])    #mean vector for the calculations

    for j in range(len(training)):
        if training[20][j] == 1:        #covariance calculation current index - mean for both attributes
            c12 += (training[first_attribute][j] - m11) * (training[second_attribute][j] - m21)
            c11 += (training[first_attribute][j] - m11) * (training[first_attribute][j] - m11)
            c22 += (training[second_attribute][j] - m21) * (training[second_attribute][j] - m21)

    c11, c22, c12 = c11 / len(training), c22 / len(training), c12 / len(training)   #normalize the values

    cov = np.array([[c11, c12],     #sigma or known as covariance matrix
                    [c12, c22]])


    k12, k11, k22 = 0, 0, 0
    for j in range(len(training)):
        if training[20][j] == 2:    #same calculation for the second class(2)
            k12 += (training[first_attribute][j] - m12) * (training[second_attribute][j] - m22)
            k11 += (training[first_attribute][j] - m12) * (training[first_attribute][j] - m12)
            k22 += (training[second_attribute][j] - m22) * (training[second_attribute][j] - m22)

    k11, k22, k12 = k11 / len(training), k22 / len(training), k12 / len(training)

    cov2 = np.array([[k11, k12],        #second covariance matrix
                     [k12, k22]])
    mean2 = np.array([m12, m22])        #second mean vector

    len1 = len(dataset[first_attribute].loc[dataset[20] == 1])
    len2 = len(dataset[first_attribute].loc[dataset[20] == 2])

    prior1 = len1 / len(dataset[first_attribute])       #probabilty of the class all the values on dataset included.
    prior2 = len2 / len(dataset[first_attribute])

    # classification_result is an array that stores the result of classification
    classification_result = do_classification()

    # compare classification results with y_test
    individual_accuracy = accuracy_score(testing[20], classification_result)

    accuracy = accuracy + individual_accuracy
    accuracies.append(individual_accuracy)

print("Average Accuracy = %", (accuracy/500.0))
print("Max Accuracy = %", max(accuracies))
print("Avarage Loss = ",np.array(losses).mean())
print("Max Loss= ",max(losses))
print("Avarage Loss without risk = ",np.array(losses_without_risk).mean())
print("Max Loss without risk = ",max(losses_without_risk))
print("avarage decrease in loss with rejection % = ",np.array(losses).mean() * 100 / (np.array(losses).mean() + np.array(losses_without_risk).mean()))

plot_gaussian()
input()