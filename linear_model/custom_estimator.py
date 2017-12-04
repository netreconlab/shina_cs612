import tensorflow as tf
import numpy as np

features = [tf.feature_column.numeric_column("x", shape = [1])]

#to define our own custom linear model
def model(features, labels, mode):
    W = tf.get_variable("W", [1], tf.float64)
    b = tf.get_variable("b", [1], tf.float64)
    y = W * features['x'] + b

    # loss

    loss = tf.reduce_sum(tf.square(y - labels))

    global_step_incrementer = tf.assign_add(tf.train.get_global_step(), 1)
    optimizer = tf.train.GradientDescentOptimizer(0.01)
    train = tf.group (optimizer.minimize(loss), global_step_incrementer)

    return tf.estimator.EstimatorSpec(
        predictions=y,
        loss= loss,
        train_op=train,
        mode= mode
    )

estimator = tf.estimator.Estimator(model)

#data_Sets
x_train = np.array([1., 2., 3., 4.])
y_train = np.array([0., -1., -2., -3.])
x_eval = np.array([2., 5., 8., 1.])
y_eval = np.array([-1.01, -4.1, -7, 0.])


input_fn = tf.estimator.inputs.numpy_input_fn({"x": x_train}, y = y_train, batch_size=4, num_epochs=None, shuffle= True)
train_input_fn = tf.estimator.inputs.numpy_input_fn({"x": x_train}, y = y_train, num_epochs= 1000, batch_size=4, shuffle= False)
eval_input_fn = tf.estimator.inputs.numpy_input_fn({"x": x_eval}, y = y_eval, num_epochs= 1000, batch_size=4, shuffle= False)


estimator.train(input_fn=input_fn, steps=1000)

train_metrics = estimator.evaluate(input_fn=train_input_fn)
evaluation_metrics = estimator.evaluate(input_fn=eval_input_fn)

print("train metrics: %r"% train_metrics)
print("evaluation metrics: %r"% evaluation_metrics)


