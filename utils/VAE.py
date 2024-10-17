import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from keras import regularizers


class Sampling(layers.Layer):
    """Uses (z_mean, z_log_var) to sample z, the vector encoding a digit."""

    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon

    
    
    
    
    
class VAE(keras.Model):
    def __init__(self, encoder, decoder, **kwargs):
        super().__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
        self.total_loss_tracker = keras.metrics.Mean(name="total_loss")
        self.reconstruction_loss_tracker = keras.metrics.Mean(
            name="reconstruction_loss"
        )
        self.kl_loss_tracker = keras.metrics.Mean(name="kl_loss")

    @property
    def metrics(self):
        return [
            self.total_loss_tracker,
            self.reconstruction_loss_tracker,
            self.kl_loss_tracker,
        ]

    def train_step(self, data):
        I,x = data
        with tf.GradientTape() as tape:
            z_mean, z_log_var, z = self.encoder(I)
            reconstruction = self.decoder(z)
            reconstruction_loss = tf.reduce_mean(
                tf.reduce_sum(
                    keras.losses.MSE(x, reconstruction)
                    #keras.losses.binary_crossentropy(x, reconstruction)
                )
            )
            kl_loss = -0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
            kl_loss = 0.01 * tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1))
            total_loss = reconstruction_loss + kl_loss
        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        return {
            "loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
        }

    # def get_config(self):
    #     config = super(VAE, self).get_config()
    #     config.update({"encoder": self.encoder, "decoder": self.decoder, "total_loss_tracker":self.total_loss_tracker,
    #                    "reconstruction_loss_tracker": self.reconstruction_loss_tracker, "kl_loss_tracker": self.kl_loss_tracker})
    #     return config
    
    
    


def train(data, latent_dim = 2, epoch = 30, bs = 128,
          num_layers = [32, 64],lambda_act=0, lambda_weight=0):
    
    dim = data.shape[1]
    depth = len(num_layers)
    
    encoder_inputs = keras.Input(shape=(dim,))

    for i in range(depth):
        if i == 0:
            x = layers.Dense(num_layers[depth-i-1], activation="relu",activity_regularizer=regularizers.l2(lambda_act),
                            kernel_regularizer=regularizers.l2(lambda_weight))(encoder_inputs)
        else:
            x = layers.Dense(num_layers[depth-i-1], activation="relu",activity_regularizer=regularizers.l2(lambda_act),
                            kernel_regularizer=regularizers.l2(lambda_weight))(x)

    z_mean = layers.Dense(latent_dim, name="z_mean")(x)
    z_log_var = layers.Dense(latent_dim, name="z_log_var")(x)

    z = Sampling()([z_mean, z_log_var])
    encoder = keras.Model(encoder_inputs, [z_mean, z_log_var, z], name="encoder")

    latent_inputs = keras.Input(shape=(latent_dim,))

    for i in range(depth):
        if i == 0:
            x = layers.Dense(num_layers[i], activation="relu",activity_regularizer=regularizers.l2(lambda_act),
                            kernel_regularizer=regularizers.l2(lambda_weight))(latent_inputs)
        else:
            x = layers.Dense(num_layers[i], activation="relu",activity_regularizer=regularizers.l2(lambda_act),
                            kernel_regularizer=regularizers.l2(lambda_weight))(x)

    decoder_outputs = layers.Dense(dim, activation="sigmoid",activity_regularizer=regularizers.l2(lambda_act),
                                  kernel_regularizer=regularizers.l2(lambda_weight))(x)

    decoder = keras.Model(latent_inputs, decoder_outputs, name="decoder")

    model = VAE(encoder, decoder)
    model.compile(optimizer=keras.optimizers.Adam())

    callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=10)
    model.fit(data, data, epochs=epoch, batch_size=bs, callbacks=[callback])
    
    return model
    
    
    
# ====================================End_of_code=========================================    


