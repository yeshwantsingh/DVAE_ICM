import matplotlib.pyplot as plt
import tensorflow as tf
from mel_ae import get_vae

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)

AUTOTUNE = tf.data.AUTOTUNE


def plot_predictions(model, test_ds):
    predictions = model.predict(test_ds)
    for samples in test_ds.take(1):
        for i in range(0, 6):
            plt.subplot(6, 1, i + 1)
            plt.plot(samples[0][100 + i])
            frame1 = plt.gca()
            if i != 5:
                frame1.axes.get_xaxis().set_ticks([])
            plt.subplot(6, 1, i + 1)
            plt.plot(predictions[100 + i], 'r')
        plt.tight_layout()
        plt.show()


def get_waveform(file_path):
    # print(file_path.numpy())
    audio_binary = tf.io.read_file(file_path)
    waveform, _ = tf.audio.decode_wav(contents=audio_binary,
                                      desired_channels=1)
    waveform = tf.squeeze(waveform, axis=-1)
    frames = tf.signal.frame(waveform, 44100, 22050, pad_end=True)
    return frames


def make_dataset(filenames, train=False):
    files_ds = tf.data.Dataset.from_tensor_slices(filenames)
    waveform_ds = files_ds.map(map_func=get_waveform,
                               num_parallel_calls=AUTOTUNE)

    waveform_ds = waveform_ds.flat_map(lambda x: tf.data.Dataset.from_tensor_slices(x))

    if train:
        waveform_ds = waveform_ds.map(map_func=lambda x: (tf.random.normal(mean=0, stddev=.01, shape=x.shape) + x, x),
                                      num_parallel_calls=AUTOTUNE)
    else:
        waveform_ds = waveform_ds.map(map_func=lambda x: (x, x),
                                      num_parallel_calls=AUTOTUNE)

    ds = waveform_ds.batch(20)
    ds = ds.prefetch(AUTOTUNE)
    return ds


def get_callbacks():
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath='mel_ae_weights_{epoch:02d}.h5',
            monitor='loss',
            verbose=2,
            save_best_only=True,
            save_weights_only=True,
            mode='auto',
            save_freq='epoch',
        ),

        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='loss',
            factor=0.1,
            patience=10,
            verbose=0,
            mode='auto',
            min_delta=0.0001,
            cooldown=0,
            min_lr=0,
        ),
        tf.keras.callbacks.TensorBoard(
            log_dir='./logs',
            histogram_freq=1,
            write_images=True,
            update_freq='epoch',
        )

    ]
    return callbacks


if __name__ == '__main__':
    filenames = tf.io.gfile.glob(
        '/media/B/multitask_indian_music_classification/data/raw/Indian Hindustani Classical Music/*/*.wav')

    # filenames = tf.random.shuffle(filenames)
    filenames.sort()
    filenames = filenames[:1]

    train_files = filenames
    train_ds = make_dataset(train_files, train=True)

    model = get_vae((44100, 1), 1)
    # tf.keras.utils.plot_model(model, 'model.png', dpi=600, show_shapes=True)
    model.summary()
    # model.load_weights('weights.h5')

    model.compile(optimizer='adam', loss='mse')
    # model.summary(expand_nested=True)
    # train = np.random.randn(10000, 1024)
    # train = (train - train.mean()) / (train.max() - train.min())
    # test = np.random.randn(100, 1024)
    # test = (test - test.mean()) / (test.max() - test.min())
    # model.fit(train_ds, epochs=100, callbacks=get_callbacks())
    # model.save_weights("weights.h5")
    # optimizer_config = model.optimizer.get_config()
    # with open('opt.pkl', 'wb') as out:
    #     pickle.dump(optimizer_config, out)
    model.fit(train_ds, epochs=1, callbacks=get_callbacks())
    # model.evaluate(test_ds)
    # plot_predictions(model, test_ds)
