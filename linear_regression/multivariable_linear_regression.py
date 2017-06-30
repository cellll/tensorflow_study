import tensorflow as tf

x1_data = [1,0,3,0,5]
x2_data = [0,2,0,4,0]
y_data = [1,2,3,4,5]

w1 = tf.Variable(tf.random_uniform([1], -1.0, 1.0))
w2 = tf.Variable(tf.random_uniform([1], -1.0, 1.0))

b = tf.Variable(tf.random_uniform([1], -1.0, 1.0))

hypothesis = w1*x1_data + w2*x2_data + b

cost = tf.reduce_mean(tf.square(hypothesis - y_data))
learning_rate = tf.Variable(0.08)
optimizer = tf.train.GradientDescentOptimizer(learning_rate)
train = optimizer.minimize(cost)

init = tf.global_variables_initializer()

sess = tf.Session()
sess.run(init)


for step in range(2001):
	sess.run(train)

	print (step, sess.run(cost), sess.run(w1), sess.run(w2), sess.run(b))


sess.close()

