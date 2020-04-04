# import tensorflow
import tensorflow as tf
import numpy as np
import math
import matplotlib.pyplot as plt
sess = tf.compat.v1.Session()
hello = tf.constant("Hello World from raghav")
print(hello)

a = tf.constant(4)
b = tf.constant(5)

c = a+b
print("Sum is ",c)
