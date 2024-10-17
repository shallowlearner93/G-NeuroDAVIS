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

    
    
    
    
    
class NGDAVIS(keras.Model):
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

    
    
    


def train(data_in, data_out, latent_dim = 2, epoch = 30, bs = 128,
          num_layers = [32, 64],lambda_act=0, lambda_weight=0):
    
    in_dim = data_in.shape[1]
    out_dim = data_out.shape[1]
    
    encoder_inputs = keras.Input(shape=(in_dim,))

    z_mean = layers.Dense(latent_dim, name="z_mean")(encoder_inputs)
    z_log_var = layers.Dense(latent_dim, name="z_log_var")(encoder_inputs)

    z = Sampling()([z_mean, z_log_var])
    encoder = keras.Model(encoder_inputs, [z_mean, z_log_var, z], name="encoder")

    latent_inputs = keras.Input(shape=(latent_dim,))

    for i in range(len(num_layers)):
        if i == 0:
            x = layers.Dense(num_layers[i], activation="relu",activity_regularizer=regularizers.l2(lambda_act),
                            kernel_regularizer=regularizers.l2(lambda_weight))(latent_inputs)
        else:
            x = layers.Dense(num_layers[i], activation="relu",activity_regularizer=regularizers.l2(lambda_act),
                            kernel_regularizer=regularizers.l2(lambda_weight))(x)

    decoder_outputs = layers.Dense(out_dim, activation="sigmoid",activity_regularizer=regularizers.l2(lambda_act),
                                  kernel_regularizer=regularizers.l2(lambda_weight))(x)

    decoder = keras.Model(latent_inputs, decoder_outputs, name="decoder")

    model = NGDAVIS(encoder, decoder)
    model.compile(optimizer=keras.optimizers.Adam())

    callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=100)
    model.fit(data_in, data_out, epochs=epoch, batch_size=bs, callbacks=[callback])
    
    return model
    
    
def predict(data_in, data_out, model, latent_dim=2, epoch=30, bs=128,
            num_layers = [32, 64],lambda_act=0, lambda_weight=0):
    
    in_dim = data_in.shape[1]
    out_dim = data_out.shape[1]
    
    inputs = keras.Input(shape=(in_dim,))

    z = layers.Dense(latent_dim)(inputs)
    #z_log_var = layers.Dense(latent_dim, name="z_log_var")(encoder_inputs)

    #z = Sampling()([z_mean, z_log_var])
    #encoder = keras.Model(encoder_inputs, z, name="encoder")

    #latent_inputs = keras.Input(shape=(latent_dim,))
    
    weights = model.get_weights()

    for i in range(len(num_layers)):
        if i == 0:
            x = layers.Dense(num_layers[i], activation="relu",activity_regularizer=regularizers.l2(lambda_act),
                            kernel_regularizer=regularizers.l2(lambda_weight),
                            kernel_initializer=keras.initializers.Constant(weights[0]),
                            bias_initializer = keras.initializers.Constant(weights[1]),
                            trainable=False)(z)
        else:
            x = layers.Dense(num_layers[i], activation="relu",activity_regularizer=regularizers.l2(lambda_act),
                            kernel_regularizer=regularizers.l2(lambda_weight),
                            kernel_initializer=keras.initializers.Constant(weights[2*i]),
                            bias_initializer = keras.initializers.Constant(weights[2*i+1]),
                            trainable=False)(x)

    outputs = layers.Dense(out_dim, activation="sigmoid",activity_regularizer=regularizers.l2(lambda_act),
                           kernel_regularizer=regularizers.l2(lambda_weight),
                           kernel_initializer=keras.initializers.Constant(weights[-2]),
                           bias_initializer = keras.initializers.Constant(weights[-1]),
                           trainable=False)(x)

    model = keras.Model(inputs, outputs)
    encoder = keras.Model(inputs, z)

    #model = NGDAVIS(encoder, decoder)
    model.compile(loss=keras.losses.MeanSquaredError(), optimizer=keras.optimizers.Adam())

    callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=100)
    model.fit(data_in, data_out, epochs=epoch, batch_size=bs, callbacks=[callback])
    
    return encoder
    
    
    
# ====================================End_of_code=========================================    