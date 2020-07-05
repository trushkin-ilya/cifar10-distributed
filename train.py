import os
from argparse import ArgumentParser
from datetime import datetime

import horovod.tensorflow as hvd
import numpy as np
import tensorflow as tf

tf.get_logger().propagate = False

from load_data import load_data
from models import cnn_model_fn

argparser = ArgumentParser()
argparser.add_argument("--save-dir", type=str, default='checkpoints')
argparser.add_argument("--epochs", type=str, default=10)
argparser.add_argument("--data-dir", type=str, default='data/cifar-10-batches-py')
argparser.add_argument("--batch-size", type=int, default=1)
argparser.add_argument("--lr", type=float, default=3e-3)
args = argparser.parse_args()


def main(_):
    hvd.init()
    chief = hvd.rank() == 0
    num_workers = hvd.size()

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.gpu_options.visible_device_list = str(hvd.local_rank())

    (x_train, y_train), (x_test, y_test) = load_data(args.data_dir)
    x_train, y_train = x_train[hvd.rank()::num_workers], y_train[hvd.rank()::num_workers]
    x_train = x_train / 255.0
    x_test = x_test / 255.0

    hooks = [hvd.BroadcastGlobalVariablesHook(0)]

    model_dir = os.path.join(args.save_dir, datetime.now().strftime("%Y%m%d-%H%M%S")) if chief else None

    estimator = tf.estimator.Estimator(model_fn=cnn_model_fn, model_dir=model_dir,
                                       config=tf.estimator.RunConfig(session_config=config))

    train_fn = tf.estimator.inputs.numpy_input_fn(x={"x": x_train}, y=np.squeeze(y_train),
                                                  batch_size=args.batch_size,
                                                  num_epochs=1, shuffle=True)
    for _ in range(args.epochs):
        if chief:
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor(max_workers=2) as e:
                e.submit(estimator.train, input_fn=train_fn, hooks=hooks)
                eval_fn = tf.estimator.inputs.numpy_input_fn(x={"x": x_test},
                                                             y=np.squeeze(y_test),
                                                             batch_size=1,
                                                             num_epochs=1,
                                                             shuffle=False)
                e.submit(estimator.evaluate, input_fn=eval_fn)
        else:
            estimator.train(input_fn=train_fn, hooks=hooks)


if __name__ == "__main__":
    tf.app.run()
