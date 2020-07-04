from argparse import ArgumentParser
from datetime import datetime

import horovod.tensorflow as hvd
import horovod.tensorflow.keras as hvd_K
import tensorflow as tf
from load_data import load_data

from models import CNN

if __name__ == "__main__":
    argparser = ArgumentParser()
    argparser.add_argument("--save-dir", type=str, default='checkpoints')
    argparser.add_argument("--data-dir", type=str, default='data/cifar-10-batches-py')
    argparser.add_argument("--log-dir", type=str, default='logs')
    argparser.add_argument("--batch-size", type=int, default=1)
    argparser.add_argument("--lr", type=float, default=3e-3)
    args = argparser.parse_args()
    hvd.init()
    master_node = hvd.rank() == 0

    model = CNN()
    num_workers = hvd.size()
    callbacks = [hvd_K.callbacks.BroadcastGlobalVariablesCallback(0),
                 hvd_K.callbacks.MetricAverageCallback(),
                 hvd_K.callbacks.LearningRateWarmupCallback(warmup_epochs=5, verbose=1),
                 tf.keras.callbacks.ReduceLROnPlateau(patience=10, verbose=1)]
    if master_node:
        callbacks.append(
            tf.keras.callbacks.ModelCheckpoint(args.save_dir + '/' + model.__class__.__name__ + '[{epoch}].h5'))
        callbacks.append(tf.keras.callbacks.TensorBoard(args.log_dir + '/' + datetime.now().strftime("%Y%m%d-%H%M%S")))

    (x_train, y_train), (x_test, y_test) = load_data(args.data_dir)

    train_dataset = tf.data.Dataset\
        .from_tensor_slices((x_train, y_train)) \
        .shard(hvd.size(), hvd.rank()) \
        .shuffle(len(x_train))\
        .batch(args.batch_size, drop_remainder=True)

    test_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test))\
        .shard(hvd.size(), hvd.rank())\
        .batch(args.batch_size)

    optimizer = hvd.DistributedOptimizer(tf.keras.optimizers.SGD(lr=args.lr * num_workers))
    model.compile(optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model.fit(train_dataset, epochs=10, validation_data=test_dataset if master_node else None,
              callbacks=callbacks, verbose=1 if master_node else 0)
