import tensorflow as tf

W = tf.Variable([.3], dtype=tf.float32)
b = tf.Variable([-.3], dtype=tf.float32)
x = tf.placeholder(tf.float32)
y = tf.placeholder(tf.float32)

#Model
linear_model = W*x +b

#loss
squared_deltas = tf.square(linear_model - y)
loss = tf.reduce_sum(squared_deltas)

fixW = tf.assign(W,[-1.])
fixb = tf.assign(b, [1.])


print("W:",W)

init = tf.global_variables_initializer()

sess = tf.Session()
sess.run(init)
# sess.run([fixW,fixb])

#intial results using intial weight and bias
print("result for a simple linear model:", sess.run(linear_model, {x: [1.,2.,3.,4.]}))
print("loss for linear model:", sess.run(loss, {x: [1.,2.,3.,4.], y: [0.,-1.,-2.,-3.]}))

print("\n After fixing the weight and bias variable")
# manually fixing the weight and bias to reduce loss
sess.run([fixW,fixb])

print("result for a simple linear model:", sess.run(linear_model, {x: [1.,2.,3.,4.]}))
print("loss for linear model:", sess.run(loss, {x: [1.,2.,3.,4.], y: [0.,-1.,-2.,-3.]}))
