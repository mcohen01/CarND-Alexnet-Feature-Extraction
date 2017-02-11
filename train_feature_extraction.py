import pickle
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from alexnet import AlexNet

with open('train.p', mode='rb') as f:
    data = pickle.load(f)

X_train, y_train = data['features'], data['labels']


# Split data into training and validation sets.
X_train, y_train = shuffle(X_train, y_train)
X_train, X_validation, y_train, y_validation = train_test_split(X_train, y_train, train_size=.75)


# Define placeholders and resize operation.
x = tf.placeholder(tf.float32, (None, 32, 32, 3))
y = tf.placeholder(tf.int64, None)
resized = tf.image.resize_images(x, [227, 227])
fc7 = AlexNet(resized, feature_extract=True)
fc7 = tf.stop_gradient(fc7)

# Add the final layer for traffic sign classification.
shape = (fc7.get_shape().as_list()[-1], 43)
weights = tf.Variable(tf.truncated_normal(shape=shape, mean=0, stddev=.1))
bias = tf.Variable(tf.zeros(43))
logits = tf.nn.xw_plus_b(fc7, weights, bias)


cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits, y)
loss = tf.reduce_mean(cross_entropy)

optimizer = tf.train.AdamOptimizer(learning_rate=0.001)
train_op = optimizer.minimize(loss, var_list=[weights, bias])
init_op = tf.global_variables_initializer()

correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(y, 1))
accuracy_op = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


def evaluate(session, X_data, y_data, batch_size):
    num_examples = len(X_data)
    total_accuracy = 0
    for offset in range(0, num_examples, batch_size):
        batch_x, batch_y = X_data[offset:offset + batch_size], y_data[offset:offset + batch_size]
        accuracy = session.run([loss, accuracy_op], feed_dict={x: batch_x, y: batch_y})
        total_accuracy += (accuracy * len(batch_x))
    return total_accuracy / num_examples


EPOCHS = 10
BATCH_SIZE = 128

with tf.Session() as sess:
    sess.run(init_op)
    for i in range(EPOCHS):
        X_train, y_train = shuffle(X_train, y_train)
        for offset in range(0, X_train.shape[0], BATCH_SIZE):
            end = offset + BATCH_SIZE
            batch_x, batch_y = X_train[offset:end], y_train[offset:end]
            sess.run(train_op, feed_dict={x: batch_x, y: batch_y})

        validation_accuracy = evaluate(sess, X_validation, y_validation, BATCH_SIZE)

        print("EPOCH {} ...".format(i + 1))
        print("Validation Accuracy = {:.3f}".format(validation_accuracy))
        print()
