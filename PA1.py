import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#convergence criteria decides when to end the code
def convergenceCriteria(k,cost,episolon = 0.1):
	if(k < 2):
		return False
	changeInLoss = (abs(cost[k-2] - cost[k-1]) * 100)/cost[k-2]
	if(changeInLoss < episolon):
		return True
	else: return False

#function for the cost function of quadratic regularization
def CostFuction(X,y,theta):
    tobesummed = np.power(((X * theta.T)-y),2)
    reglzr = np.power(theta,2)
    reglzr = np.sum(reglzr)
    reglzr = (1/(2*len(y))) * reglzr
    return np.sum(tobesummed)/(2 * len(X)) + reglzr

#function for loss function of gradient descent with regularization
def CostFuctionWithLassoRegularization(X,y,theta):
    tobesummed = np.power(((X * theta.T)-y),2)
    part1 = np.sum(tobesummed)/(2 * len(y))
    part2 = (1/(2*len(y)))*np.sum(np.absolute(theta))
    return part1 + part2
        
#find the sign for theta
def sign(theta,index):
    if(theta[0,index] >= 0):
        return 1
    else:
        return -1
    
        
#gradient descent with lasso regularization     
def GradientDescentLassoRegulazation(X, y, theta,alpha, iters, myLambda):
	k = 0
	cost = []
	temp = np.matrix(np.zeros(theta.shape)) #temp store for theta
	parameters = int(theta.ravel().shape[1]) #number of features #16
	#costlasso = np.zeros(iters) #initialize cost to zeros same num as iter 1000 
	#for i in range(iters):
	while(not convergenceCriteria(k,cost)):
		error = (X * theta.T) - y 	#y - h(x)

		for j in range(parameters):
			total = sign(theta,j) * (1/(2*len(y)))
			term = np.multiply(error, X[:,j])
			temp[0,j] = theta[0,j] - ((alpha / len(y)) * (np.sum(term) + total))
		theta = temp
		cost.append(CostFuction(X, y, theta))
		k = k +1

	return theta, cost,k

        
#fuction for the cost fuction without regularization
def CostFuctionWithoutRegularization(X,y,theta):
	tobesummed = np.power(((X * theta.T)-y),2)
	return np.sum(tobesummed)/(2 * len(X))



#function for Gradient descent with quadratic regularization
def GradientDescentWQR(X,y,theta,alpha,iters,myLambda):
	k = 0
	cost= []
	temp = np.matrix(np.zeros(theta.shape)) #temp store for theta
	parameters = int(theta.ravel().shape[1]) #number of features #16
	#cost = np.zeros(iters) #initialize cost to zeros same num as iter 1000 
	#for i in range(iters):
	while(not convergenceCriteria(k,cost)):
		error = (X * theta.T) - y 	#y - h(x)

		for j in range(parameters):
			total = theta[0,j] * (1/len(X))
			term = np.multiply(error, X[:,j])
			temp[0,j] = theta[0,j] - ((alpha / len(X)) * (np.sum(term) + total))
		theta = temp
		k = k + 1
		cost.append(CostFuction(X, y, theta))


	return theta, cost,k



#find the gradient descent
def gradientDescent(X, y, theta, alpha, iters, myLambda):

	temp = np.matrix(np.zeros(theta.shape)) #temp store for theta
	parameters = int(theta.ravel().shape[1]) #number of features #16
	cost = np.zeros(iters) #initialize cost to zeros same num as iter 1000 
	for i in range(iters):
		error = (X * theta.T) - y 	#y - h(x)

		for j in range(parameters):
			total = theta[0,j] * (1/len(X))
			term = np.multiply(error, X[:,j])
			temp[0,j] = theta[0,j] - ((alpha / len(X)) * (np.sum(term) + total))
		theta = temp
		cost[i] = CostFuction(X, y, theta)

	return theta, cost

#normalize all the data between -0.5 < x < 0.5
def featureNormalization(data):
	return (data - data.mean())/data.std()

#plot the graph for Cost fuction to iterations
def PlotGraph(iters,cost):
	fig, ax = plt.subplots(figsize=(8,8))
	ax.plot(np.arange(iters), cost, 'r')
	ax.set_xlabel('Iterations')
	ax.set_ylabel('Cost')
	ax.set_title('Error vs. Training Epoch')
	plt.show()
	return

#Round each parameter to 0 if its absolute value is smaller than 5 * 10 ^-3
def RoundToZero(theta):
	value = 0.005
	found = 0
	#print(np.absolute(theta))
	parameters = int(theta.ravel().shape[1])
	for i in range(parameters):
		temp = np.absolute(theta[0,i])
		if(temp < value):
			theta[0,i] = 0;
			found = found + 1
	
	return found

def main():
    data = pd.read_csv('data.csv',names=["A1","A2","A3","A4","A5","A6","A7","A8","A9","A10","A11","A12","A13","A14","A15","B"])
    myData = featureNormalization(data)
    #print(myData)
    print('\n')

    cols = 17 #training
    rows = 48 #training

    col = 17 #testing
    row = 12 #testing

    #add ones as column
    myData.insert(0, 'A0', 1)
    
    #training data
    #make the matrix for X and Y
    matrix_X = myData.iloc[ 0 : rows-1 , 0 : cols-1]
    X = np.matrix(matrix_X.values)	
    matrix_Y = myData.iloc[0 : rows-1,cols-1:cols]
    Y = np.matrix(matrix_Y.values)    

    #testing data
    #make the matrix for testing data
    max_X = myData.iloc[rows:,0:col-1]
    testX = np.matrix(max_X.values)	
    max_Y = myData.iloc[rows:,col-1:col]
    testY = np.matrix(max_Y.values)

    #initialize theta alpha iter
    theta = np.matrix(np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]))
    theta1 = np.matrix(np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]))
    #print(theta[0,2])
    alpha = 0.01
    iters = 1000
    myLambda = 1
    
    #train 1
    #print(CostFuction(X,Y,theta)) 
    #print(CostFuctionWithoutRegularization(X,Y,theta))#no gradient descent
    gradient, cost , iters = GradientDescentWQR(X, Y, theta, alpha, iters, myLambda)
    
    squaredLoss = CostFuction(X,Y,gradient)
    print("squared loss trained ",squaredLoss) #error with gradient descent

    #test 1
    loss = CostFuctionWithoutRegularization(testX,testY,gradient)
    print("squared loss test",loss)
    ZeroFound1 = RoundToZero(gradient)
    #PlotGraph(iters,cost)

    #train 2
    #print(CostFuctionWithLassoRegularization(X,Y,theta1))
    lassoGradient,lassoCost, iters = GradientDescentLassoRegulazation(X, Y, theta1,alpha, iters, myLambda)
    print("squared loss laso trained ",CostFuctionWithLassoRegularization(X,Y,lassoGradient))
    #print(lassoGradient)
    #test 2 
    lassoLoss = CostFuctionWithoutRegularization(testX,testY,lassoGradient)
    print("squared loss laso test ",lassoLoss)
    PlotGraph(iters,lassoCost)
    ZeroFound2 = RoundToZero(lassoGradient)

    print("Gradient descent with quadratic regulalization has found: ", ZeroFound1, " found Zero")
    print("Gradient descent with lasso regulalization has found: ", ZeroFound2, " found Zero")
    return


if __name__ == "__main__":
	main()
