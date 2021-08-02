import os

from tensorflow import keras

from nic import (
    callbacks as cbs,
    metrics as mcs,
)
from nic.datapreparation.data import load_data


def compile_decoder(model,
                    learning_rate=0.0001):
    """
    Compiles and returns the decoder module of the NIC model. This is
    useful when training the decoder with the encoder still frozen.

    :param model: a model created with `define_decoder_model`.
    :param learning_rate: a float - the learning rate.
    :returns: the compiled model. Its optimizer is `Adam`, its loss is
    sparse categorical cross entropy, and its only additional metric is
    `Perplexity`.
    """
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss=mcs.create_loss_object(),
        metrics=[
            mcs.Perplexity(),
        ]
    )

    return model


def train_decoder(*,
                  model,
                  path_to_data,
                  batch_size=32,
                  buffer_size=1024,
                  tensor_board_dir,
                  tensor_board_update_freq="epoch",
                  checkpoint_dir,
                  checkpoint_freq="epoch",
                  learning_rate_decay=0.9,
                  decay_patience=3,
                  perplexity_delta=0.001,
                  min_learning_rate=0.,
                  early_stop_patience=3,
                  max_epochs=10,
                  shuffle_for_each_epoch=False):
    """
    :param model: a compiled model. If a fresh training process is
    started, this should be the result of `compile_decoder`. For extra
    training, this should be the result of `restore_model`.
    :param path_to_data: a str - the path of the directory where the
    preprocessed data is stored.
    :param batch_size: an int - the batch size. Defaults to 32.
    :param buffer_size: an int - the size for the buffer used to shuffle
    the train data.
    :param tensor_board_dir: a str - the path of the directory where to
    log data for TensorBoard; it should be exclusively for that.
    :param tensor_board_update_freq: 'batch' or 'epoch' or integer. When
    using 'batch', the losses and metrics are written to TensorBoard
    after each batch. The same applies for 'epoch'. If using an integer,
    let's say 1000, the the metrics and losses are written to TensorBoard
    every 1000 batches. Note that writing too frequently to TensorBoard
    can slow down your training. Defaults to 'epoch'.
    :param checkpoint_dir: a str - the path of the directory where to
    store model checkpoints. It should be be used exclusively for that.
    :param checkpoint_freq: 'epoch' (the default) or integer. When using
    'epoch', the function checks whether to save the model after each
    epoch. When using an integer, it is done after that many batches.
    Note that a new checkpoint is made only if the model's validation
    perplexity has improved.
    :param learning_rate_decay: a float - the factor by which the
    learning rate will be reduced (new = factor * old) when the model's
    validation perplexity stops improving. Defaults to 0.9.
    :param decay_patience: an int - number of epochs with no improvement
    after which the learning rate will be reduced. Defaults to 3.
    :param perplexity_delta: a float - the minimum perplexity improvement
    required. Defaults to 0.001.
    :param min_learning_rate: a float - the min learning rate. Defaults
    to 0.
    :param early_stop_patience: an int - the number of learning rate
    decays to wait until terminating the training process. Defaultst to
    3 which means that if the learning rate is decayed 3 times in a row
    (as validation perplexity does not improve), the process will be
    terminated.
    :param max_epochs: an int - the maximum number of epochs during
    training. Defaults to 10.
    :param shuffle_for_each_epoch: a boolean value indicating whether to
    shuffle the training data prior to each epoch. Defaults to False.
    :returns: a pair of the training history (as returned by Model.fit)
    and a dict mapping metric names to the corresponding test values.
    """
    train_data, val_data, test_data = _load_all_data(path_to_data,
                                                     batch_size,
                                                     buffer_size)
    callbacks = _create_callbacks(
        tensor_board_dir,
        tensor_board_update_freq,
        checkpoint_dir,
        checkpoint_freq,
        learning_rate_decay,
        decay_patience,
        perplexity_delta,
        decay_cooldown=0,
        min_learning_rate=min_learning_rate,
        early_stop_patience=early_stop_patience * decay_patience
    )

    history = model.fit(
        train_data,
        epochs=max_epochs,
        validation_data=val_data,
        callbacks=callbacks,
        shuffle=shuffle_for_each_epoch,
    )

    results = model.evaluate(test_data)

    return (history, dict(zip(model.metrics_names, results)))


def _load_all_data(path_to_data, batch_size, buffer_size):
    train = load_data(
        path_to_data,
        type="train",
        load_as_features=True
    ).shuffle(buffer_size).batch(batch_size)
    val = load_data(
        path_to_data,
        type="val",
        load_as_features=True
    ).batch(batch_size)
    test = load_data(
        path_to_data,
        type="test",
        load_as_features=True
    ).batch(batch_size)

    return (train, val, test)


def _create_callbacks(tensor_board_dir,
                      tensor_board_update_freq,
                      checkpoint_dir,
                      checkpoint_freq,
                      learning_rate_decay,
                      decay_patience,
                      perplexity_delta,
                      decay_cooldown,
                      min_learning_rate,
                      early_stop_patience):
    return [
        cbs.tensor_board(
            tensor_board_dir,
            tensor_board_update_freq
        ),
        cbs.checkpoint(
            os.path.join(checkpoint_dir, cbs.CHECKPOINT_PATTERN),
            checkpoint_freq,
        ),
        cbs.learning_rate_reduction(
            learning_rate_decay,
            decay_patience,
            perplexity_delta,
            decay_cooldown,
            min_learning_rate,
        ),
        cbs.early_stopping(
            perplexity_delta,
            early_stop_patience,
        ),
        cbs.terminate_on_nan(),
    ]


def restore_model(checkpoint_dir, restore_best=False):
    """
    :param checkpoint_dir: a str - the path of the directory where
    model checkpoints are stored.
    :param restore_best: a boolean value indicating whether the model
    with best recoded validation perplexity should be restored. If not,
    the latest checkpoint is loaded.
    :returns: the compiled keras.Model.
    :raises RuntimeError: if there are no checkpoints.
    """
    optimal, key = (
        (min, _perplexity_of)
        if (restore_best)
        else (max, os.path.getctime)
    )

    checkpoints = [os.path.join(checkpoint_dir, name)
                   for name in os.listdir(checkpoint_dir)]

    if (checkpoints):
        path = optimal(checkpoints, key=key)
        return keras.models.load_model(path)
    else:
        raise RuntimeError("No checkpoints available!")


def _perplexity_of(path):
    name = _short_name_of(path)
    start = name.find("=")
    end = name.find("_")
    return float(name[start + 1:end])


def _short_name_of(path):
    start = path.rfind(os.sep)
    return (path[start + 1:]
            if (start != -1)
            else path)
