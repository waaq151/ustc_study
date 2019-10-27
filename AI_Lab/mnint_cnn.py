import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
trX, trY, teX, teY = mnist.train.images, mnist.train.labels, mnist.test.images, mnist.test.labels

def init_weights(shape):
    return tf.Variable(tf.random_normal(shape, stddev=0.01))

def model(X, w_h, w_o):
    h = tf.nn.relu(tf.matmul(X, w_h))                            # this is a basic mlp, think 2 stacked logistic regressions这是一个基本的mlp，考虑2个堆叠的逻辑回归
    return tf.matmul(h, w_o)                                      # note that we dont take the softmax at the end because our cost fn does that for us

X = tf.placeholder("float", [None, 784])
Y = tf.placeholder("float", [None, 10])

w_h = init_weights([784, 625])                              # create symbolic variables
w_o = init_weights([625, 10])

py_x = model(X, w_h, w_o)

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=py_x, labels=Y)) # compute costs
train_op = tf.train.RMSPropOptimizer(0.001).minimize(cost)                              # construct an optimizer
predict_op = tf.argmax(py_x, 1)

with tf.name_scope("accuracy"):
    correct_pred = tf.equal(tf.argmax(Y, 1), tf.argmax(py_x, 1))             # Count correct predictions
    acc_op = tf.reduce_mean(tf.cast(correct_pred, "float"))                 # Cast boolean to float to average
    # Add scalar summary for accuracy
    tf.summary.scalar("accuracy", acc_op)


# Launch the graph in a session
with tf.Session() as sess:
    merged = tf.summary.merge_all()  
    writer = tf.summary.FileWriter('./logs/my_logs',sess.graph)
    
    # you need to initialize all variables
    tf.global_variables_initializer().run()
    
    for i in range(10):
        for start, end in zip(range(0, len(trX), 128), range(128, len(trX)+1, 128)):
            sess.run(train_op, feed_dict={X: trX[start:end], Y: trY[start:end]})
        
        summary, acc = sess.run([merged, acc_op], feed_dict={X: teX, Y: teY})
        writer.add_summary(summary, i)  # Write summary
        print(i, np.mean(np.argmax(teY, axis=1) ==
                         sess.run(predict_op, feed_dict={X: teX})))