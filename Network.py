import numpy as np
import tensorflow as tf
import pandas as pd

import random

import matplotlib.pyplot as plt
import os

# Hyper-Parameters
LEARNING_RATE = 0.5
TRAINING_ITERATIONS = 500
DROP_OUT = 0.5
BATCH_SIZE = 100
VALIDATION_SIZE = 2000


# from tensorflow.examples.tutorials.mnist import input_data
# mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

def weight_initializer(shape):
    initial = tf.truncated_normal(shape, stddev=0.01)
    return tf.Variable(initial)


def bias_initializer(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def dense_to_one_hot_vector(labels, number_of_classes=10):
    number_of_labels = len(labels)
    temp_labels = pd.DataFrame(np.zeros((number_of_labels, number_of_classes)))
    for i in range(number_of_labels):
        temp_labels.iloc[i, labels[i]] = 1
    return temp_labels


def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')  # PADDING SAME YANI SHAPE E INPUT = SHAPE E OUTPUT


def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


def next_batch():
    size = training_x.shape[0]
    temp_x = list()
    temp_y = list()
    for _ in range(BATCH_SIZE):
        random_number = random.randint(0, size - 1)
        temp_x.append(training_x.iloc[random_number])
        temp_y.append(training_y.iloc[random_number])
    return temp_x, temp_y


def next_batch_validation():
    size = validation_x.shape[0]
    temp_x = list()
    temp_y = list()
    for _ in range(BATCH_SIZE):
        random_number = random.randint(0, size - 1)
        temp_x.append(validation_x.iloc[random_number])
        temp_y.append(validation_y.iloc[random_number])
    return temp_x, temp_y


def save(filename, sess):
    """Save the neural network to the file ``filename``."""
    saver = tf.train.Saver()
    saver.save(sess, filename, global_step=1000)


def load(filename, dir_name):
    global W_conv1, W_fc1, W_fc2
    global b_conv1, b_fc1, b_fc2
    global y_conv, prediction

    sess = tf.Session()
    saver = tf.train.import_meta_graph(filename + '.meta')
    saver.restore(sess, tf.train.latest_checkpoint('./' + dir_name))

    graph = tf.get_default_graph()

    W_conv1 = graph.get_tensor_by_name("W_conv1:0")
    W_fc1 = graph.get_tensor_by_name("W_fc1:0")
    W_fc2 = graph.get_tensor_by_name("W_fc2:0")

    b_conv1 = graph.get_tensor_by_name("b_conv1:0")
    b_fc1 = graph.get_tensor_by_name("b_fc1:0")
    b_fc2 = graph.get_tensor_by_name("b_fc2:0")

    y_conv = graph.get_tensor_by_name("y_conv:0")
    prediction = graph.get_tensor_by_name("prediction:0")


training_data = pd.read_csv("train.csv")
x, y = training_data.drop('label', 1), training_data['label']
x = np.multiply(x, 1.0 / 255.0)

y = dense_to_one_hot_vector(y, 10)
y = y.astype(np.uint8)

validation_x, validation_y = x[:VALIDATION_SIZE], y[:VALIDATION_SIZE]
training_x, training_y = x[VALIDATION_SIZE:], y[VALIDATION_SIZE:]

# sess = tf.InteractiveSession()
X = tf.placeholder(tf.float32, [None, 784])
Y = tf.placeholder(tf.float32, [None, 10])

# W = weight_initializer([784, 10])
# b = bias_initializer([10])
#
# sess.run(tf.global_variables_initializer())
#
# y = tf.matmul(X, W) + b
# cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=Y, logits=y))
#
# train = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)
# # start_idx = 0
# for i in range(100):
#     # batch_x = training_x[start_idx:start_idx + BATCH_SIZE]
#     # batch_y = training_y[start_idx:start_idx + BATCH_SIZE]
#     # start_idx += BATCH_SIZE
#     batch_x = training_x[i:i + BATCH_SIZE]
#     batch_y = training_y[i:i + BATCH_SIZE]
#     train.run(feed_dict={X: batch_x, Y: batch_y})
#
# correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(Y, 1))
# accuracy = tf.reduce_mean(tf.cast(correct_prediction, dtype=tf.float32))
#
# print(accuracy.eval(feed_dict={X: validation_x, Y: validation_y}))

W_conv1 = weight_initializer([5, 5, 1, 32])
b_conv1 = bias_initializer([32])

x_image = tf.reshape(X, [-1, 28, 28, 1])

h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

# W_conv2 = weight_initializer([5, 5, 32, 64])
# b_conv2 = bias_initializer([64])
#
# h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
# h_pool2 = max_pool_2x2(h_conv2)

W_fc1 = weight_initializer([14 * 14 * 32, 1024])
b_fc1 = bias_initializer([1024])

h_pool2_flat = tf.reshape(h_pool1, [-1, 14 * 14 * 32])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

keep_prob = tf.placeholder(tf.float32)
h_fc1_dropout = tf.nn.dropout(h_fc1, keep_prob)

W_fc2 = weight_initializer([1024, 10])
b_fc2 = bias_initializer([10])

y_conv = tf.matmul(h_fc1_dropout, W_fc2) + b_fc2

test_x = pd.read_csv("test.csv").values
test_x = test_x.astype(np.float)

test_x = np.multiply(test_x, 1.0 / 255.0)

prediction = tf.argmax(y_conv, 1)
# Training && Evaluating Convolution Neural Network

cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=Y, logits=y_conv))
train_step = tf.train.GradientDescentOptimizer(LEARNING_RATE).minimize(cross_entropy)
# tf.contrib.layers.apply_regularization(tf.contrib.layers.l2_regularizer(tf.constant(5)), [W_conv1, W_fc1, W_fc2])
correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
validation_accuracy = 0.0
for j in range(10):
    train_accuracies = list()
    validation_accuracies = list()
    x_range = list()
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        # start_idx = 0
        for i in range(TRAINING_ITERATIONS):
            # batch = mnist.train.next_batch(BATCH_SIZE)
            batch = next_batch()
            # batch = (training_x[:BATCH_SIZE], training_y[:BATCH_SIZE])
            # batch_x = training_x[start_idx:start_idx + BATCH_SIZE]
            # batch_y = training_y[start_idx:start_idx + BATCH_SIZE]
            if i % 5 == 0:
                x_range.append(i)
                train_accuracy = accuracy.eval(feed_dict={X: batch[0], Y: batch[1], keep_prob: 1.0})
                print('step %d, training accuracy %g' % (i, train_accuracy))
                train_accuracies.append(train_accuracy)
                batch = next_batch_validation()
                validation_accuracy = accuracy.eval(feed_dict={X: batch[0], Y: batch[1], keep_prob: 1.0})
                validation_accuracies.append(validation_accuracy)
            train_step.run(feed_dict={X: batch[0], Y: batch[1], keep_prob: 0.5})
        validation_accuracy = accuracy.eval(feed_dict={X: validation_x, Y: validation_y, keep_prob: 1.0})
        print('validation accuracy %g' % validation_accuracy)
        os.makedirs('./model_{0}_{1}'.format(j, validation_accuracy))
        save("./model_{0}_{1}/model_tf_{2}".format(j, validation_accuracy, j), sess)

        predicted_y = prediction.eval(feed_dict={X: test_x, keep_prob: 1.0})

        np.savetxt('./model_{0}_{1}/submission_softmax_{2}.csv'.format(j, validation_accuracy, j),
                   np.c_[range(1, len(test_x) + 1), predicted_y],
                   delimiter=',',
                   header='ImageId,Label',
                   comments='',
                   fmt='%d')
    plt.plot(x_range, train_accuracies, '-b', label='Training Accuracy')
    plt.plot(x_range, validation_accuracies, '-g', label='Validation Accuracy')
    plt.legend(loc='lower right', frameon=False)
    plt.ylim(ymax=1.0, ymin=0.0)
    plt.ylabel('accuracy')
    plt.xlabel('step')
    plt.savefig('./model_{0}_{1}/model_{2}.png'.format(j, validation_accuracy, j), )
    print("*" * 100)
    plt.close()
    # plt.show()
    '''
    def save(self, filename):
        """Save the neural network to the file ``filename``."""
        data = {"sizes": self.sizes,
                "weights": [w.tolist() for w in self.weights],
                "biases": [b.tolist() for b in self.biases],
                "cost": str(self.cost.__name__)}
        f = open(filename, "w")
        json.dump(data, f)
        f.close()


    #### Loading a Network
    def load(filename):
        """Load a neural network from the file ``filename``.  Returns an
        instance of Network.

        """
        f = open(filename, "r")
        data = json.load(f)
        f.close()
        cost = getattr(sys.modules[__name__], data["cost"])
        net = Network(data["sizes"], cost=cost)
        net.weights = [np.array(w) for w in data["weights"]]
        net.biases = [np.array(b) for b in data["biases"]]
        return net
    '''
