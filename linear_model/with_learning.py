import tensorflow as tf

# Model parameters
W = tf.Variable([.3], dtype=tf.float32)
b = tf.Variable([-.3], dtype=tf.float32)

#Model input and output
x = tf.placeholder(tf.float32)
y = tf.placeholder(tf.float32)

#Model
linear_model = W*x +b

#loss
loss = tf.reduce_sum(tf.square(linear_model - y))

#Optimizer
optimizer = tf.train.GradientDescentOptimizer(0.01)
train = optimizer.minimize(loss)

#training Data
x_train = [1, 2, 3, 4]
y_train = [0, -1, -2, -3]

#training loop
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init) #reset values

for i in range(1000):
    sess.run(train, {x: x_train, y: y_train})

#Print final weight and bias, and loss
final_W, final_b, final_loss = sess.run([W, b, loss], {x: x_train, y: y_train})
print("W: ", final_W, "b:", final_b, "loss:", final_loss)