from sklearn.datasets import make_blobs
from scipy.stats import norm
from numpy import mean, std

#Function for gaussian distribution (normal distribution)
def normal_dist(data):
    mu = mean(data)
    sigma = std(data)

    dist = norm(mu, sigma)
    return dist

#Naive Bayes is based on Bayes theorem
#Works under the condition that every feature is independent of one another, w/o the assumption the 
#calculation becomes computationally expensive
#P(H|E) = P(E|H) * P(E) / P(H) - the probability of H occuring when E has already occured
#P(yi|x1,x2...xn) = P(x1,x2...xn|yi) * P(yi) / P(x1,x2...xn) x = observation, y = class

#Funtion that calculates the individual conditional probability
def probability(sample, prior, obv1, obv2):
    return prior*obv1.pdf(sample[0])*obv2.pdf(sample[1])


#Datapoints along with the classes
samp1, samp2 = make_blobs(n_samples=100, n_features=2, centers=2, random_state=1)
#print(samp1) prints the array that contains 2 values 
#print(samp2) prints the classes of the array

#Seperating the datapoints class-wise
class0 = samp1[samp2 == 0]
class1 = samp1[samp2 == 1]
#print(class0.shape) contains the datapoints whose class is 0

#calculating the prior P(yi) = number of samples of an individual class / total number of samples
prior0 = len(class0) / len(samp1)
prior1 = len(class1) / len(samp1)

#calculating the conditional probability
#calculating P(samp1|class0)*P(samp1|class1)*P(samp2|class0)*P(samp2|class1)
x1class0 = normal_dist(class0[:,0])
x2class0 = normal_dist(class0[:,1])

x1class1 = normal_dist(class1[:,0])
x2class1 = normal_dist(class1[:,1])

#Test sample - the first of the data
test = samp1[0]
test_label = samp2[0]

#calculating individual conditional probability
cond0 = probability(test, prior0, x1class0, x2class0)
cond1 = probability(test, prior1, x1class1, x2class1)

print('P(y=0 | %s) = %.3f' % (test, cond0*100))
print('P(y=1 | %s) = %.3f' % (test, cond1*100))
print('Truth: y=%d' % test_label)

#It is predicting accurately. Since, the probability of the class being 0 is 0.3, we can accept that the predicted class is 0 itself