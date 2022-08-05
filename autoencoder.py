import tensorflow as tf


class Sampling(tf.keras.layers.Layer):
    ''' Uses (z_mean, z_log_var) to sample z, the vector enconding a digit'''

    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon

    def get_config(self):
        config = super(Sampling, self).get_config()
        return config


def get_vae(input_shape, latent_dim):
    inputs = tf.keras.Input(shape=input_shape)

    # Encoder
    x = tf.keras.layers.Conv1D(64, 2,
                               name='conv1D_layer_1_1')(inputs)
    x = tf.keras.layers.PReLU()(x)
    x = tf.keras.layers.MaxPooling1D(2, 2, padding='same')(x)
    x = tf.keras.layers.BatchNormalization(name='batch_norm_1')(x)

    x = tf.keras.layers.Conv1D(32, 2, name='conv1D_layer_1_2')(x)
    x = tf.keras.layers.PReLU()(x)
    x = tf.keras.layers.MaxPooling1D(3, 2, padding='same')(x)
    x = tf.keras.layers.BatchNormalization(name='batch_norm_2')(x)
    x = tf.keras.layers.SpatialDropout1D(0.1)(x)

    x = tf.keras.layers.Conv1D(32, 2, name='conv1D_layer_1_3')(x)
    x = tf.keras.layers.PReLU()(x)
    x = tf.keras.layers.MaxPooling1D(5, 2, padding='same')(x)
    x = tf.keras.layers.BatchNormalization(name='batch_norm_3')(x)

    x = tf.keras.layers.Flatten(name='flatten_layer')(x)

    # Bottleneck
    z_mean = tf.keras.layers.Dense(latent_dim)(x)
    z_log_var = tf.keras.layers.Dense(latent_dim)(x)
    x = Sampling()((z_mean, z_log_var))

    x = tf.keras.layers.Dense(latent_dim, activation='sigmoid', name='bottleneck_layer')(x)
    x = tf.keras.layers.Dropout(0.1, name='dropout_layer')(x)
    x = tf.keras.layers.Reshape((-1, 1), name='reshape_layer')(x)

    # Decoder
    x = tf.keras.layers.Conv1D(32, 1,
                               name='conv1D_layer_2_1')(x)
    x = tf.keras.layers.PReLU()(x)
    x = tf.keras.layers.UpSampling1D(2)(x)

    x = tf.keras.layers.Conv1D(32, 1,
                               name='conv1D_layer_2_2')(x)
    x = tf.keras.layers.PReLU()(x)
    x = tf.keras.layers.UpSampling1D(2)(x)

    outputs = tf.keras.layers.Conv1D(1, 1, activation='tanh',
                                     name='conv1D_layer_2_3')(x)

    model = tf.keras.Model(inputs, outputs)

    kl_loss = -0.5 * tf.reduce_mean(z_log_var - tf.square(z_mean) - tf.exp(z_log_var) + 1) * 0.0001
    model.add_loss(kl_loss)

    return model
