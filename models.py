import tensorflow as tf

def conv_model(feature, target, mode):
    target = tf.one_hot(tf.cast(target, tf.int32), 10, 1, 0)
    feature = tf.reshape(feature, (-1, 32, 32, 3))
    with tf.variable_scope('conv_layer1'):
        h_conv1 = tf.layers.conv2d(feature, 32, kernel_size=[5, 5], activation=tf.nn.relu, padding="SAME")
        h_pool1 = tf.nn.max_pool(h_conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    with tf.variable_scope('conv_layer2'):
        h_conv2 = tf.layers.conv2d(h_pool1, 64, kernel_size=[5, 5], activation=tf.nn.relu, padding="SAME")
        h_pool2 = tf.nn.max_pool(h_conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
        h_pool2_flat = tf.reshape(h_pool2, [-1, 8 * 8 * 64])

    h_fc1 = tf.layers.dropout(tf.layers.dense(h_pool2_flat, 1024, activation=tf.nn.relu), rate=0.5,
                           training=mode == tf.estimator.ModeKeys.TRAIN)

    logits = tf.layers.dense(h_fc1, 10, activation=None)
    loss = tf.losses.softmax_cross_entropy(target, logits)

    return tf.argmax(logits, 1), loss
