import tensorflow as tf


class Trainer:
    """
    Manage the train step
    """
    def __init__(self, model, optimizer, loss):
        self.model = model
        self.optimizer = optimizer
        self.loss = loss
        self.train_loss = tf.keras.metrics.Mean(name='train_loss')
        self.train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')

    def get_step(self):
        @tf.function
        def train_step(images, labels):
            with tf.GradientTape() as tape:
                predictions = self.model(images)
                loss = self.loss(labels, predictions)
            gradients = tape.gradient(loss, self.model.trainable_variables)
            self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))

            self.train_loss(loss)
            self.train_accuracy(labels, predictions)

        return train_step


class Validator:
    """
    Manage the validation step
    """
    def __init__(self, model, loss):
        self.model = model
        self.loss = loss
        self.test_loss = tf.keras.metrics.Mean(name='test_loss')
        self.test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')

    def get_step(self):
        @tf.function
        def test_step(images, labels):
            predictions = self.model(images)
            t_loss = self.loss(labels, predictions)

            self.test_loss(t_loss)
            self.test_accuracy(labels, predictions)

        return test_step
