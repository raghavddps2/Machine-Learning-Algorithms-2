import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

# import input_data
#We use the tensorflow helper functions to pull down the data from the tensorflow website
mnist = input_data.read_data_sets("MNIST_data/",one_hot = True)

#Here, the image is flattened.
x = tf.placeholder(tf.float32,shape=[None,784])
y_ = tf.placeholder(tf.float32,shape=[None,10])

#define the weights and the balances
W = tf.Variable(tf.zeros([784,10]))
b = tf.Variable(tf.zeros([10]))

#define the model.
y = tf.nn.softmax(tf.matmul(x,W)+b)


#loss is cross entropy
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_,logits=y))

#each training step in gradient descent we want to minimize the loss.
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

#Now, we should start training the model.
#initilaizeing the global varibales
init = tf.global_variables_initializer()

sess = tf.Session()
sess.run(init)

for i in range(1000):
    batch_xs,batch_ys = mnist.train.next_batch(100)
    if i == 0:
        print(batch_ys[0])
        plt.imshow(batch_xs[0].reshape(28,28))
        # plt.imshow(batch_xs[0])
    sess.run(train_step,feed_dict={x:batch_xs,y_:batch_ys})

correct_prediction = tf.equal(tf.argmax(y,1),tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
test_accuracy = sess.run(accuracy,feed_dict={x:mnist.test.images,y_:mnist.test.labels})
print("Test Accuracy: {0}%".format(test_accuracy*100.0))
sess.close()
