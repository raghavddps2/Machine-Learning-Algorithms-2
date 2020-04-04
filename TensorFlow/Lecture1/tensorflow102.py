import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior() 
import math
import matplotlib.pyplot as plt
import matplotlib.animation as animation

#Generate some houses sized between 1000 and 3500
num_house = 160
np.random.seed(42)
house_size = np.random.randint(low=1000,high=3500,size=num_house)

# Generate house prices from house size with random noise added.
np.random.seed(42)
house_price = house_size*100 + np.random.randint(low=20000,high=70000,size=num_house)

# Plot generated house size and price
plt.plot(house_size,house_price,"bx")
plt.ylabel("Price")
plt.xlabel("Size")
plt.show()


# We need to normalize values to prevent overflow/underflow
def normalize(array):
    return (array-array.mean())/array.std()

# defining the number of training samples we are going to use
num_train_samples = math.floor(num_house*0.7)

# 1. Preparing the data

# define training data
train_house_size = np.asarray(house_size[:num_train_samples])
train_price = np.asanyarray(house_price[:num_train_samples])

train_house_size_norm = normalize(train_house_size)
train_price_norm = normalize(train_price)

# define test data
test_house_size = np.asarray(house_size[num_train_samples:])
test_price = np.asanyarray(house_price[num_train_samples:])

test_house_size_norm = normalize(test_house_size)
test_price_norm = normalize(test_price)

# Tensor Types
'''
    1. tf.constant
    2. tf.variable
    3. tf.placeholder
'''
# We set up the tensorflow placeholders, that gets updated as we descend down the gradient
tf_house_size = tf.placeholder("float",name="house_size")
tf_price = tf.placeholder("float",name="price")

# Define the variables holding the size_factor and the price offset that we set during training the model.
tf_size_factor = tf.Variable(np.random.randn(),name="size_factor")
tf_price_offset = tf.Variable(np.random.randn(),name="price_offset")

# Define the operations for predicting values.
tf_price_pred = tf.add(tf.multiply(tf_size_factor,tf_house_size),tf_price_offset)

# 3. Defining the loss function
# Define the loss function
tf_cost = tf.reduce_sum(tf.pow(tf_price_pred-tf_price,2))/(2*num_train_samples)

# 4. defining a gradient descent optimizer, that will minimize the loss defined in the operation "cost"
# Optimizer learning rate
learning_rate = 0.1
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(tf_cost)

#Initialize the variables.
init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)

    display_every = 2
    num_iterating_itr = 100
    
    #Keep iterating over training data
    for iteration in range(num_iterating_itr):

        #fit all training data
        for (x,y) in zip(train_house_size_norm,train_price_norm):

            #Placeholder to be replaced by the given dicts.
            sess.run(optimizer,feed_dict={tf_house_size:x,tf_price:y})
        
        #Display the current status
        if (iteration+1)%display_every == 0:
            c = sess.run(tf_cost,feed_dict={tf_house_size:train_house_size_norm,tf_price:train_price_norm})
            print("Iteration ",'%04d'%(iteration+1),"cost =","{:.9f}".format(c),"size_factor=",sess.run(tf_size_factor),"price_offset=",sess.run(tf_price_offset))

    print("-----Optimization finished--------")

    training_cost = sess.run(tf_cost,feed_dict={tf_house_size:train_house_size_norm,tf_price:train_price_norm})
    print("Trained cost =",training_cost,"size_factor=",sess.run(tf_size_factor),"price_offset=",sess.run(tf_price_offset))

    #denormalizing
    train_house_size_mean = train_house_size.mean()
    train_house_size_std = train_house_size.std()

    train_price_mean = train_price.mean()
    train_price_std = train_price.std()

    test_house_size_mean = test_house_size.mean()
    test_house_size_std = test_house_size.std()

    test_price_mean = test_price.mean()
    test_price_std = test_price.std()

    #Plotting the graph
    plt.rcParams["figure.figsize"] = (10,8)
    plt.figure()
    plt.ylabel("Price")
    plt.xlabel("Size (sq. ft")

    plt.plot(train_house_size,train_price,'go',label='Training data')
    plt.plot(test_house_size,test_price,'mo',label='Testing data')

    #The regression line.
    plt.plot(train_house_size_norm*train_house_size_std+train_house_size_mean,
    (sess.run(tf_size_factor)*train_house_size_norm + sess.run(tf_price_offset))*train_price_std
    +train_price_mean,label="Learned Regression")

    # plt.plot(test_house_size_norm*test_house_size_std+test_house_size_mean,
    # (sess.run(tf_size_factor)*test_house_size_norm + sess.run(tf_price_offset))*test_price_std
    # +test_price_mean,label="Learned Regression")

    plt.legend(loc='upper left')
    plt.show()
    sess.close()