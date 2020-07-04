from argparse import ArgumentParser

import horovod.tensorflow as hvd
import numpy as np
import tensorflow as tf
from load_data import load_data
from models import conv_model
tf.logging.set_verbosity(tf.logging.INFO)

argparser = ArgumentParser()
argparser.add_argument("--save-dir", type=str, default='checkpoints')
argparser.add_argument("--data-dir", type=str, default='data/cifar-10-batches-py')
argparser.add_argument("--log-dir", type=str, default='logs')
argparser.add_argument("--batch-size", type=int, default=1)
argparser.add_argument("--lr", type=float, default=3e-3)
argparser.add_argument("--use-gpu", type=bool, default=False)
args = argparser.parse_args()


def train_input_generator(x_train, y_train, batch_size):
    assert len(x_train) == len(y_train)
    while True:
        p = np.random.permutation(len(x_train))
        x_train, y_train = x_train[p], y_train[p]
        index = 0
        while index <= len(x_train) - batch_size:
            yield x_train[index:index + batch_size], \
                  y_train[index:index + batch_size],
            index += batch_size


def main(_):
    hvd.init()
    master_node = hvd.rank() == 0
    num_workers = hvd.size()

    config = tf.ConfigProto()
    if args.use_gpu:
        config.gpu_options.allow_growth = True
        config.gpu_options.visible_device_list = str(hvd.local_rank())



    with tf.name_scope('input'):
        image = tf.placeholder(tf.float32, [None, 1024, 3], name='image')
        label = tf.placeholder(tf.float32, [None], name='label')
    predict, loss = conv_model(image, label, tf.estimator.ModeKeys.TRAIN)
    optimizer = hvd.DistributedOptimizer(tf.train.MomentumOptimizer(learning_rate=args.lr * num_workers, momentum=0.9))
    global_step = tf.train.get_or_create_global_step()
    train_op = optimizer.minimize(loss, global_step=global_step)

    (x_train, y_train), (x_test, y_test) = load_data(args.data_dir)
    x_train, y_train = x_train[hvd.rank()::hvd.size()], y_train[hvd.rank()::hvd.size()]
    order = np.random.permutation(len(x_train))
    x_train, y_train = x_train[order], y_train[order]
    x_train = np.random.permutation(x_train)
    x_train = np.reshape(x_train, (-1, 1024, 3)) / 255.0
    x_test = np.reshape(x_test, (-1, 1024, 3)) / 255.0
    training_batch_generator = train_input_generator(x_train, y_train, args.batch_size)

    hooks = [hvd.BroadcastGlobalVariablesHook(0), tf.train.StopAtStepHook(last_step=len(x_train) // hvd.size()),
             tf.train.LoggingTensorHook(tensors={'step': global_step, 'loss': loss}, every_n_iter=10)]

    ckpt_dir = args.save_dir if master_node else None

    with tf.train.MonitoredTrainingSession(checkpoint_dir=ckpt_dir, config=config, hooks=hooks) as mon_sess:
        while not mon_sess.should_stop():
            image_, label_ = next(training_batch_generator)
            mon_sess.run(train_op, feed_dict={image: image_, label: np.squeeze(label_)})

if __name__ == "__main__":
    tf.app.run()