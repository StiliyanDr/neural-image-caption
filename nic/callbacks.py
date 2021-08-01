from tensorflow import keras


CHECKPOINT_PATTERN = ("perplexity={val_perplexity:.2f}_"
                      "loss={val_loss:.2f}_"
                      "epoch={epoch:02d}.ckpt")


def tensor_board(log_dir, update_frequency="epoch"):
    """
    Creates and returns a TensorBoard callback.

    :param log_dir: a str - the path of the directory where to log
    data for TensorBoard; it should be exclusively used by this
    callback.
    :param update_frequency: 'batch' or 'epoch' or integer. When using
    'batch', writes the losses and metrics to TensorBoard after each
    batch. The same applies for 'epoch'. If using an integer, let's say
    1000, the callback will write the metrics and losses to TensorBoard
    every 1000 batches. Note that writing too frequently to TensorBoard
    can slow down your training. Defaults to 'epoch'.
    :returns: a `TensorBoard` callback.
    """
    return keras.callbacks.TensorBoard(
        log_dir=log_dir,
        histogram_freq=1,
        write_graph=True,
        write_steps_per_second=True,
        update_freq=update_frequency,
    )


def checkpoint(path_pattern=CHECKPOINT_PATTERN,
               frequency="epoch"):
    """
    Creates and returns a ModelChekpoint callback.

    :param path_pattern: a str - a pattern for the paths of the
    checkpoints. Defaults to `CHECKPOINT_PATTERN`. Note that the
    directory where checkpoints are saved should be used exclisively
    by this callback.
    :param frequency: 'epoch' (the default) or integer. When using
    'epoch', the callback checks whether to save the model after each
    epoch. When using an integer, the callback checks after that many
    batches.
    :returns: a `ModelChekpoint`.
    """
    return keras.callbacks.ModelCheckpoint(
        filepath=path_pattern,
        monitor="val_perplexity",
        save_best_only=True,
        mode="min",
        save_weights_only=True,
        save_freq=frequency,
        verbose=1,
    )


def learning_rate_reduction(factor=0.9,
                            patience=5,
                            delta=0.001,
                            cooldown=0,
                            min_learning_rate=0.):
    """
    Creates and returns a `ReduceLROnPlateau` callback which reduces
    the learning rate when the model's perplexity stops improving.

    :param factor: a float - the factor by which the learning rate will
    be reduced (new = factor * old). Defaults to 0.9.
    :param patience: an int - number of epochs with no improvement after
    which learning rate will be reduced. Defaults to 5.
    :param delta: a float - the minimum perplexity improvement required.
    Defaults to 0.001.
    :param cooldown: an int - number of epochs to wait before resuming
    normal operation after the learning rate has been reduced. Defaults
    to 0.
    :param min_learning_rate: a float - the min learning rate.
    :returns: a `ReduceLROnPlateau`.
    """
    return keras.callbacks.ReduceLROnPlateau(
        monitor="val_perplexity",
        factor=factor,
        patience=patience,
        verbose=1,
        mode="min",
        min_delta=delta,
        cooldown=cooldown,
        min_lr=min_learning_rate
    )


def early_stopping(delta=0.001, patience=3):
    """
    Creates and returns an `EarlyStopping` callback.

    :param delta: a float - the minimum change in perplexity that
    qualifies as an improvement. Defaults to 0.001.
    :param patience: an int - the number of epochs with no improvement
    after which training will be stopped. Defaults to 3.
    :returns: an `EarlyStopping` callback.
    """
    return keras.callbacks.EarlyStopping(
        monitor="val_perplexity",
        min_delta=delta,
        patience=patience,
        verbose=1,
        mode="min"
    )


def terminate_on_nan():
    """
    :returns: a `TerminateOnNaN` callback.
    """
    return keras.callbacks.TerminateOnNaN()
