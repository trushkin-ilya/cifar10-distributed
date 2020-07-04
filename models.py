import tensorflow as tf

layers = tf.keras.layers


class CNN(tf.keras.models.Sequential):
    def __init__(self, num_classes=10, input_shape=(32, 32, 3)):
        super(CNN, self).__init__([
            layers.Conv2D(32, (3, 3), padding='same', input_shape=input_shape),
            layers.Activation('relu'),
            layers.Conv2D(32, (3, 3)),
            layers.Activation('relu'),
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.Dropout(0.25),

            layers.Conv2D(64, (3, 3), padding='same'),
            layers.Activation('relu'),
            layers.Conv2D(64, (3, 3)),
            layers.Activation('relu'),
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.Dropout(0.25),

            layers.Flatten(),
            layers.Dense(512),
            layers.Activation('relu'),
            layers.Dropout(0.5),
            layers.Dense(num_classes),
            layers.Activation('softmax')])
