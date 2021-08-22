import itertools

import tensorflow as tf

from nic import datapreparation as dp
from nic.datapreparation.preprocessing import prepare_image


_CAPTION_LIMIT = 100


class _Captions:
    """
    Non-public class.

    Represents the state of a batch of captions during their generation.
    """
    def __init__(self, meta_tokens, batch_size):
        """
        :param meta_tokens: an instance of MetaTokens.
        :param batch_size: an int.
        """
        self.__meta_tokens = meta_tokens
        self.__set_initial_captions(meta_tokens.start, batch_size)

    def __set_initial_captions(self, start_token=None, batch_size=None):
        if (start_token is None):
            start_token = self.__meta_tokens.start
            batch_size = len(self.__captions)

        self.__captions = [[start_token] for _ in range(batch_size)]
        self.__max_length = 0

    @property
    def current_words(self):
        """
        :returns: a list of strs.
        """
        return [c[-1] for c in self.__captions]

    def extend(self, new_words):
        """
        :param new_words: a list of strs.
        """
        end = self.__meta_tokens.end

        for cap, word in zip(self.__captions, new_words):
            if (cap[-1] != end):
                cap.append(word)

        self.__max_length += int(any(w != end for w in new_words))

    @property
    def all_finished(self):
        end = self.__meta_tokens.end

        return all(c[-1] == end for c in self.__captions)

    def extract(self):
        """
        :returns: a list of lists of strs - the captions without any
        meta tokens.
        """
        end = self.__meta_tokens.end

        captions = [
            (c[1:-1]
             if (c[-1] == end)
             else c[1:])
            for c in self.__captions
        ]
        self.__set_initial_captions()

        return captions

    @property
    def max_length(self):
        return self.__max_length


class CaptionGenerator:
    """
    Generates captions for batches of images.

    A generator can be created from a NIC model or the decoder module
    only. In the latter case, the generator can only be used with image
    features.
    """
    def __init__(self,
                 model,
                 meta_tokens,
                 tokenizer,
                 is_decoder_only=True,
                 image_options=None):
        """
        :param model: an instance of the NIC model created with
        `define_decoder_model`, `define_model` or `connect`.
        :param meta_tokens: an instance of MetaTokens - the meta tokens
        with which the data was preprocessed.
        :param tokenizer: the tokenizer built from the data, see
        `load_tokenizer`.
        :param is_decoder_only: a boolean value indicating whether
        `model` was defined with `define_decoder_model`. Defaults to
        True.
        :param image_options: an instance of ImageOptions or None. When
        an instance of ImageOptions, it should be the instance used when
        preprocessing the data; in this case the object supports loading
        and preparing images when given paths. When None, the object
        must only be given tf.Tensors which represent already prepared
        images. Note that this is ignored if `is_decoder_only` is True.
        """
        hidden_state_output = self.__set_up_encoder(model,
                                                    is_decoder_only)
        self.__set_up_decoder(model,
                              hidden_state_output,
                              is_decoder_only)
        self.__meta_tokens = meta_tokens
        self.__tokenizer = tokenizer
        self.__image_encoder, self.__target_size = (
            (getattr(tf.keras.applications, image_options.model_name),
             image_options.target_size)
            if (image_options is not None and not is_decoder_only)
            else (None, None)
        )

    def __set_up_encoder(self, model, is_decoder_only):
        image_input = model.inputs[0]

        if (is_decoder_only):
            features_transformation = model.get_layer(
                "features-transformation"
            )
            hidden_state = features_transformation(image_input)
        else:
            encoder = model.layers[1]
            decoder = model.layers[-1]
            features_transformation = decoder.get_layer(
                "features-transformation"
            )
            features = encoder(image_input)
            hidden_state = features_transformation(features)

        self.__encoder = tf.keras.Model(inputs=image_input,
                                        outputs=hidden_state)

        return hidden_state

    def __set_up_decoder(self, model, hidden_state, is_decoder_only):
        decoder = (model
                   if (is_decoder_only)
                   else model.layers[-1])
        captions_input = decoder.inputs[1]
        hidden_state_input = tf.keras.Input(
            shape=hidden_state.shape[-1],
            dtype=hidden_state.dtype
        )
        c_state_input = tf.keras.Input(
            shape=hidden_state.shape[-1],
            dtype=hidden_state.dtype
        )
        embedding = decoder.get_layer("word-embedding")
        rnn = decoder.get_layer("rnn-decoder")
        transformation = decoder.get_layer(
            "pre-projection-transformation"
        )
        projection = decoder.get_layer("word-projection")

        embedded_captions = embedding(captions_input)
        _, h_state, c_state = rnn(
            embedded_captions,
            initial_state=[hidden_state_input, c_state_input]
        )
        word_projections = projection(transformation(h_state))

        self.__decoder = tf.keras.Model(
            inputs=[captions_input, hidden_state_input, c_state_input],
            outputs=[word_projections, h_state, c_state]
        )

    @property
    def prepares_images(self):
        """
        :returns: a boolean value indicating whether the object is able
        to load and prepare images.
        """
        return self.__image_encoder is not None

    def __call__(self, images, limit=None):
        """
        :param images: a tf.Tensor. If `images.dtype` is `tf.string`,
        the tensor is assumed to contain image paths and the images are
        loaded and prepared (for the CNN encoder) as tensors. Otherwise
        the `images` tensor is assumed to already have a shape of
        (batch_size, *image_shape). image_shape is either the image
        shape expected by the CNN encoder or the shape of the features
        extracted by it. That is, image tensors must be preprocessed for
        the CNN encoder or be the features extracted by it, if the
        generator was created from the decoder module.
        :param limit: an unsigned int - a limit for the captions'
        length in tokens. If omitted or `None`, defaults to
        `CAPTION_LIMIT`.
        :returns: a list of lists of tokens - the generated captions,
        in the same order.
        :raises RuntimeError: if:
         - `images` contains paths but the object does not support image
           loading
         - `images` does not have the expected shape
        """
        images = self.__prepare_if_paths(images)
        self.__validate_shape_of(images)

        return self.__caption_images(
            images,
            (limit
             if (limit is not None)
             else _CAPTION_LIMIT)
        )

    def __prepare_if_paths(self, images):
        if (images.dtype == tf.string):
            if (self.prepares_images):
                images = tf.constant([
                    prepare_image(path.numpy().decode(),
                                  self.__image_encoder,
                                  self.__target_size).numpy()
                    for path in images
                ])
            else:
                raise RuntimeError("Can't handle paths!")

        return images

    def __validate_shape_of(self, images):
        target_shape = self.__encoder.inputs[0].shape[1:]
        actual_shape = images.shape[1:]

        if (actual_shape != target_shape):
            raise RuntimeError(
                f"Images shape {tuple(actual_shape)} does not "
                f"match expected shape {tuple(target_shape)}!"
            )

    def __caption_images(self, images, limit):
        h_state = self.__encoder(images)
        c_state = tf.zeros_like(h_state)
        captions = _Captions(self.__meta_tokens,
                             batch_size=images.shape[0])

        while (captions.max_length < limit and
               not captions.all_finished):
            distribution, h_state, c_state = self.__distribution_after(
                captions.current_words,
                h_state,
                c_state
            )
            new_words = self.__most_probable_words(distribution)
            captions.extend(new_words)

        return captions.extract()

    def __distribution_after(self,
                             current_words,
                             h_state,
                             c_state):
        return self.__decoder([
            tf.constant([
                [self.__tokenizer.word_index[w]]
                for w in current_words
            ],
                dtype=tf.int32
            ),
            h_state,
            c_state,
        ])

    def __most_probable_words(self, distribution):
        samples = tf.random.categorical(distribution, num_samples=1)
        return [self.__tokenizer.index_word[index.numpy()]
                for index in samples[:, 0]]


def generate_captions(images, generator, limit=None):
    """
    A convenience function which takes an iterable of batches of images
    and a CaptionGenerator and invokes the generator on each batch,
    producing a Python generator of batches of captions.

    :param images: an iterable of tf.Tensors - batches of strs (image
    paths) or of tf.Tensors (prepared images or image features).
    :param generator: an instance of CaptionGenerator. Note that it must
    support image loading if `images` contains paths.
    :param limit: an unsigned int - a limit for a caption's length in
    tokens. If omitted or `None`, defaults to `CAPTION_LIMIT`.
    :returns: a generator which invokes `generator` on each tensor in
    `images`.
    """
    return (generator(ims, limit) for ims in images)


def generate_captions_from_tensors(
    images,
    model,
    meta_tokens,
    tokenizer,
    is_decoder_only=True,
    caption_limit=None,
):
    """
    A convenience function which creates a CaptionGenerator and invokes
    it on each batch of images in an iterable of tf.Tensors. See
    CaptionGenerator and `generate_captions`.
    """
    generator = CaptionGenerator(
        model,
        meta_tokens,
        tokenizer,
        is_decoder_only
    )

    return generate_captions(images, generator, caption_limit)


def generate_captions_from_paths(image_paths,
                                 model,
                                 path_to_data,
                                 batch_size=32,
                                 meta_tokens=dp.MetaTokens(),
                                 image_options=dp.ImageOptions(),
                                 caption_limit=None):
    """
    :param image_paths: an iterable of strs - paths of image files.
    The supported image formats are JPEG, PNG, GIF, BMP.
    :param model: an instance of the NIC model created with
    `define_model` or `connect`. That is, an entire NIC model.
    :param path_to_data: a str - the path of the directory where
    preprocessed data is stored.
    :param batch_size: an int - the batch size in which to process
    images. Defaults to 32.
    :param meta_tokens: an instance of MetaTokens - the meta tokens
    with which the data was preprocessed.
    :param image_options: the instance of ImageOptions used when
    preprocessing the data.
    :param caption_limit: an unsigned int - a limit for a caption's
    length in tokens. If omitted or `None`, defaults to `CAPTION_LIMIT`.
    :returns: a generator which yields strs - the captions generated
    for the given images, in the same order.
    """
    generator = CaptionGenerator(
        model,
        meta_tokens,
        dp.load_tokenizer(path_to_data),
        is_decoder_only=False,
        image_options=image_options
    )
    batches_of_paths = _chopped_into_batches(image_paths, batch_size)

    batches_of_captions = generate_captions(batches_of_paths,
                                            generator,
                                            caption_limit)
    tokenized_captions = itertools.chain.from_iterable(
        batches_of_captions
    )

    return (" ".join(cap) for cap in tokenized_captions)


def _chopped_into_batches(paths, batch_size):
    return (
        tf.constant(paths[start : start + batch_size])
        for start in range(0, len(paths), batch_size)
    )
