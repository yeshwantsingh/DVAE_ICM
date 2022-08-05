import kapre
import tensorflow as tf


class Sampling(tf.keras.layers.Layer):
    ''' Uses (z_mean, z_log_var) to sample z, the vector enconding a digit'''
    def __init__(self, **kwargs):
        super(Sampling, self).__init__(**kwargs)

    def call(self, inputs):
        z_mean, z_log_var = inputs
        # batch = tf.shape(z_mean)[1]
        # dim = tf.shape(z_mean)[2]
        epsilon = tf.keras.backend.random_normal(shape=tf.shape(z_mean))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon

    def get_config(self):
        config = super(Sampling, self).get_config()
        return config


def get_vae(input_shape, latent_dim):
    inputs = tf.keras.Input(shape=input_shape)

    x = kapre.time_frequency.STFT(n_fft=2048 * 4, win_length=2048 * 4, hop_length=512, window_name=None,
                                  pad_begin=False, pad_end=False, input_data_format='default',
                                  output_data_format='default', name='STFT')(inputs)
    x = tf.math.abs(x)
    x = tf.where(tf.math.is_nan(x), tf.zeros_like(x), x)

    x = tf.keras.layers.Conv2D(32, (1, 3), activation='relu', name='conv_1_1')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    inp = tf.keras.layers.MaxPool2D((1, 2), name='pool_1')(x)

    x1 = tf.keras.layers.Conv2D(32, (1, 1), activation='relu', name='conv_1_2')(inp)
    x1 = tf.keras.layers.BatchNormalization()(x1)
    x2 = tf.keras.layers.Conv2D(32, (1, 1), activation='relu', name='conv_1_3')(inp)

    add_1 = tf.keras.layers.Add(name='add_1')([x1, x2])
    skip_1 = tf.keras.layers.Add(name='skip_1')([inp, add_1])

    x = tf.keras.layers.Conv2D(64, (1, 3), activation='relu', name='conv_2_1')(skip_1)
    x = tf.keras.layers.BatchNormalization()(x)
    inp = tf.keras.layers.MaxPool2D((1, 2), name='pool_2_1')(x)

    x1 = tf.keras.layers.Conv2D(64, (1, 1), activation='relu', name='conv_2_2')(inp)
    x1 = tf.keras.layers.BatchNormalization()(x1)
    x2 = tf.keras.layers.Conv2D(64, (1, 1), activation='relu', name='conv_2_3')(inp)

    add_2 = tf.keras.layers.Add(name='add_2')([x1, x2])
    skip_2 = tf.keras.layers.Add(name='skip_2')([inp, add_2])

    x = tf.keras.layers.Conv2D(64, (1, 3), activation='relu', name='conv_3_1')(skip_2)
    x = tf.keras.layers.BatchNormalization()(x)
    inp = tf.keras.layers.MaxPool2D((1, 2), name='pool_3_1')(x)

    x1 = tf.keras.layers.Conv2D(64, (1, 1), activation='relu', name='conv_3_2')(inp)
    x1 = tf.keras.layers.BatchNormalization()(x1)
    x2 = tf.keras.layers.Conv2D(64, (1, 1), activation='relu', name='conv_3_3')(inp)

    add_3 = tf.keras.layers.Add(name='add_3')([x1, x2])
    skip_3 = tf.keras.layers.Add(name='skip_3')([inp, add_3])

    x = tf.keras.layers.MaxPool2D((1, 2), name='pool_4_1')(skip_3)
    x = tf.keras.layers.Conv2D(128, (1, 3), activation='relu', name='conv_4_1')(x)
    x = tf.keras.layers.BatchNormalization()(x)

    x = tf.keras.layers.MaxPool2D((1, 2), name='pool_5_1')(x)
    x = tf.keras.layers.Conv2D(128, (1, 3), activation='relu', name='conv_5_1')(x)
    x = tf.keras.layers.BatchNormalization()(x)

    # Bottleneck
    z_mean = tf.keras.layers.Conv2D(latent_dim, (1, 1))(x)
    z_log_var = tf.keras.layers.Conv2D(latent_dim, (1, 1))(x)
    x = Sampling(name='Bottleneck')((z_mean, z_log_var))
    x = tf.keras.layers.Flatten()(x)
    # x = tf.keras.layers.Conv2D(64, (1, 10), activation='relu', name='conv_6_1')(x)
    # x = tf.keras.layers.UpSampling2D((1, 4), name='upsample_1')(x)
    #
    # x = tf.keras.layers.Conv2D(64, (1, 5), activation='relu', name='conv_7_1')(x)
    # x = tf.keras.layers.UpSampling2D((1, 3), name='upsample_2')(x)
    #
    # x = tf.keras.layers.Conv2D(1, (1, 3), activation='relu', name='conv_8_1')(x)
    # x = tf.keras.layers.UpSampling2D((1, 3), name='upsample_3')(x)
    #
    # x = tf.keras.layers.Conv2D(1, (1, 2), activation='relu', name='conv_9_1')(x)
    #
    # x = tf.cast(x, tf.complex64)
    # outputs = kapre.time_frequency.InverseSTFT(n_fft=2048 * 4, win_length=2048 * 4 + 68, hop_length=512,
    #                                            forward_window_name=None, name='ISTFT')(x)
    x = tf.expand_dims(x, -1)
    x = tf.keras.layers.Conv1D(4, 50, activation='tanh')(x)
    x = tf.keras.layers.UpSampling1D(2)(x)

    x = tf.keras.layers.Conv1D(4, 50, activation='tanh')(x)
    x = tf.keras.layers.UpSampling1D(2)(x)

    x = tf.keras.layers.Conv1D(1, 70, activation='tanh')(x)
    outputs = tf.keras.layers.UpSampling1D(2)(x)

    model = tf.keras.Model(inputs, outputs)
    kl_loss = -0.5 * tf.reduce_mean(z_log_var - tf.square(z_mean) - tf.exp(z_log_var) + 1) * 0.0001
    model.add_loss(kl_loss)

    return model