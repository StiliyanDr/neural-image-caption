from tqdm import tqdm

from nic import (
    captioning as cptn,
    datapreparation as dp,
    metrics as mcs,
)


def bleu_score_of(model,
                  *,
                  is_decoder_only=True,
                  path_to_data,
                  data_type="test",
                  meta_tokens=dp.MetaTokens(),
                  caption_limit=None,
                  verbose=True):
    """
    :param model: an instance of the NIC model created with
    `define_decoder_model`, `define_model` or `connect`.
    :param is_decoder_only: a boolean value indicating whether
    `model` was defined with `define_decoder_model`. Defaults to
    True.
    :param path_to_data: a str - the path of the directory where
    preprocessed data is stored.
    :param data_type: a str - the type of data on which to evaluate the
    model. Should be 'test' (the default), 'val' or 'train'.
    :param meta_tokens: an instance of MetaTokens - the meta tokens
    with which the data was preprocessed.
    :param caption_limit: an unsigned int - a limit for the predicted
    captions' length in tokens. If omitted or `None`, defaults to
    `CAPTION_LIMIT`.
    :param verbose: a boolean value indicating whether to show a status
    bar for the progress. Defaults to `True`.

    :returns: a float in the range [0, 100] - the BLEU-4 score of the
    model.
    """
    images, images_count = dp.load_images(path_to_data,
                                          data_type,
                                          is_decoder_only)
    captions = dp.load_captions(path_to_data, data_type)
    tokenizer = dp.load_tokenizer(path_to_data)
    generator = cptn.CaptionGenerator(
        model,
        meta_tokens,
        tokenizer,
        is_decoder_only
    )
    reference, predicted = [], []

    if (verbose):
        print("Computing BLEU-4 score of", model.name, "on",
              images_count, data_type, "images.")
        images = tqdm(images)

    for image, id in images:
        reference.append([
            c.split()[1:-1]
            for c in captions[int(id)]
        ])
        predicted.append(
            generator(image, caption_limit)
        )

    return mcs.bleu_score_of(predicted, reference)
