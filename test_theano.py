import theano 
import theano.tensor as T
import numpy

x_data = numpy.float32(numpy.random.rand(2,10000))
y_data = numpy.dot([0.1000, 0.2000],x_data) + 0.3

X = T.matrix()
Y = T.vector()
b = theano.shared(numpy.random.uniform(-1,1),name ="b")
W = theano.shared(numpy.random.uniform(-1.0,1.0,(1,2)),name ="W")
y = W.dot(X) + b

cost = T.mean(T.sqr(y-Y))
gradientW = T.grad(cost=cost,wrt=W)
gradientB = T.grad(cost=cost,wrt=b)
updates = [[W,W - gradientW * 0.01],[b,b-gradientB * 0.01]]

train = theano.function(inputs=[X,Y],outputs = cost,updates= updates,allow_input_downcast=True)

for i in range(2,10000):
  train(x_data,y_data)
  print(W.get_value(),b.get_value())