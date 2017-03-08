import numpy as np
from sklearn import datasets, linear_model
#hyper parameter

class Config:
    nn_input_dim = 2  # input layer dimensionality
    nn_output_dim = 1  # output layer dimensionality
    # Gradient descent parameters (I picked these by hand)
    epsilon = 0.01  # learning rate for gradient descent
    reg_lambda = 0.01  # regularization strength




		


def generate_data():
	X = np.matrix('1,1; 0,0; 1,0; 0,1')

	y = np.matrix('0;0;1;1')


	return X, y




def run(X, model):
	W1,b1,W2,b2,W3,b3 = model['w1'],model['b1'],model['w2'],model['b2'],model['w3'],model['b3']	
	#foward prop
	z1 = np.dot(X,W1) + b1
	a1 = np.tanh(z1)
	z2 = np.dot(a1,W2) + b2
	a2 = np.tanh(z2)
	z3 = np.dot(a2, W3) + b3
	yHat = np.tanh(z3)

	return yHat



def cost(Yhat , y):
	result = 0.5 * np.sum(np.square(y - Yhat),axis =0)
	return result



def create_model(input, hidden,hidden2, output):
	#the parameters are jsut the number of each layer
	#this function just creates a model with random weights

	w1 = np.random.rand(input, hidden)
	b1 = np.random.rand(1,hidden)
	w2 = np.random.rand(hidden, hidden2)
	b2 = np.random.rand(1, hidden2)
	w3 = np.random.rand(hidden2,output)
	b3 = np.random.rand(1,output)
	model ={
		'w1': w1,
		'b1': b1,
		'w2': w2,
		'b2':b2	,
		'w3': w3,
		'b3': b3}
	return model


def forward_prop(model, X):
	W1,b1,W2,b2,W3,b3 = model['w1'],model['b1'],model['w2'],model['b2'],model['w3'],model['b3']

	#foward prop
	z1 = np.dot(X,W1) + b1
	a1 = np.tanh(z1)
	z2 = np.dot(a1,W2) + b2
	a2 = np.tanh(z2)
	z3 = np.dot(a2, W3) + b3
	yHat = np.tanh(z3)

	return yHat


def tanh_prime(x):
	return 1 - np.tanh(x)**2
#this allows the function to be apply elemntwise
tanh_prime = np.vectorize(tanh_prime)

def cost(y ,yHat):
	result = 0.5 * np.sum(np.square(y - yHat))
	return result

def train(model,X,y, epochs):
	W1,b1,W2,b2,W3,b3 = model['w1'],model['b1'],model['w2'],model['b2'],model['w3'],model['b3']

	for i in range(0,epochs):
		#foward prop
		z1 = np.dot(X,W1) + b1
		a1 = np.tanh(z1)
		z2 = np.dot(a1,W2) + b2
		a2 = np.tanh(z2)
		z3 = np.dot(a2, W3) + b3
		yHat = np.tanh(z3)

		#back propagation


		delta4 = np.multiply(-(y - yHat), tanh_prime(z3) )

		djdw3 = (a2.T).dot(delta4)

		delta3 = np.multiply(delta4.dot(W3.T), tanh_prime(z2) )

		djdw2 = (a1.T).dot(delta3)

		delta2 = np.multiply(delta3.dot(W2.T), tanh_prime(z1) )

		djdw1 = (X.T).dot(delta2)

		db1 = np.sum(delta2,axis = 0)
		db2 = np.sum(delta3,axis=0)
		db3 = np.sum(delta4,axis =0)


		#now we add weight decay
		djdw1 += W1 * Config.reg_lambda
		djdw2 += W2 * Config.reg_lambda
		djdw3 += W3 * Config.reg_lambda
		

		# fianlly we update weights
		W1 -= djdw1 * Config.epsilon
		b1 -= db1 * Config.epsilon
		W2 -= djdw2 * Config.epsilon
		b2 -= db2 * Config.epsilon
		W3 -= djdw3 * Config.epsilon
		b3 -= db3 * Config.epsilon

		if(i % 1000 == 0):
			print cost(y,yHat)

	model ={
		'w1': W1,
		'b1': b1,
		'w2': W2,
		'b2':b2	,
		'w3': W3,
		'b3': b3}
	return model

X, y = generate_data()

predictor = create_model(2,6,4,1)

predictor  = train(predictor, X,y,10000)

examp = np.array([0,0])

print run(examp, predictor )

new_inp = raw_input()
new_inp2 = raw_input()

next_in = np.array([int(new_inp),int(new_inp2)])
print next_in

print run(next_in, predictor)
