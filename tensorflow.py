import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import numpy 


x_data = numpy.float32(numpy.random.rand(2,10000))
y_data = numpy.dot([0.1000, 0.2000],x_data) + 0.3

b = tf.Variable(tf.zeros([1]))
W = tf.Variable(tf.random.uniform([1,2],-1.0,1.0))
y = tf.matmul(W,x_data) +b
lost = tf.reduce_mean(tf.square(y-y_data))
optimizer = tf.train.GradientDescentOptimizer(0.01)

train = optimizer.minimize(lost)

init = tf.initialize_all_variables()
sess = tf.Session()
sess.run(init)


for step in range(0,10000):
  sess.run(train)