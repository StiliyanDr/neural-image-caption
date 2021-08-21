import tensorflow as tf

from nic.datapreparation.preprocessing import prepare_image


_CAPTION_LIMIT = 100


class CaptionGenerator:
    """
    Generates captions given images.

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

    def __call__(self, image, limit=None):
        """
        :param image: a str or a tf.Tensor. If a str, the image is
        loaded and prepared (for the CNN encoder) as a tensor. Otherwise
        it is already a tensor. The tensor must be preprocessed for the
        CNN encoder or be the features extracted from an encoder, if the
        generator was created from the decoder module.
        :param limit: an unsigned int - a limit for the caption's
        length in tokens. If omitted or `None`, defaults to
        `CAPTION_LIMIT`.
        :returns: a list of tokens - the caption.
        :raises RuntimeError: if:
         - `image` is str and the object does not support image loading
         - `image` is a tensor (or loaded into a tensor) which does not
           have the expected shape
        """
        image = self.__prepare_if_path(image)
        self.__validate_shape_of(image)

        return self.__caption_image(
            image,
            (limit
             if (limit is not None)
             else _CAPTION_LIMIT)
        )

    def __prepare_if_path(self, image):
        if (isinstance(image, str)):
            if (self.prepares_images):
                image = prepare_image(image,
                                      self.__image_encoder,
                                      self.__target_size)
            else:
                raise RuntimeError("Can't handle paths!")

        return image

    def __validate_shape_of(self, image):
        target_shape = self.__encoder.inputs[0].shape[1:]

        if (image.shape != target_shape):
            raise RuntimeError(
                f"Image shape {tuple(image.shape)} does not "
                f"match expected shape {tuple(target_shape)}!"
            )

    def __caption_image(self, image, limit):
        h_state = self.__encoder(tf.expand_dims(image, axis=0))
        c_state = tf.zeros_like(h_state)
        translation = []
        word = self.__meta_tokens.start

        while (len(translation) < limit):
            distribution, h_state, c_state = self.__distribution_after(
                word,
                h_state,
                c_state
            )
            word = self.__most_probable_word(distribution)

            if (word != self.__meta_tokens.end):
                translation.append(word)
            else:
                break

        return translation

    def __distribution_after(self,
                             word,
                             h_state,
                             c_state):
        index = self.__tokenizer.word_index[word]
        return self.__decoder([
            tf.constant([[index]]),
            h_state,
            c_state,
        ])

    def __most_probable_word(self, probabilities):
        samples = tf.random.categorical(probabilities, num_samples=1)
        index = samples[0][0].numpy()

        return self.__tokenizer.index_word[index]


def generate_captions(images, generator, limit=None):
    """
    A convenience function which takes an iterable of images and a
    CaptionGenerator and invokes the generator on each image, producing
    a Python generator of captions.

    :param images: an iterable of strs (image paths) or of tf.Tensors
    (prepared images or image features).
    :param generator: an instance of CaptionGenerator. Note that it must
    support image loading if `images` is an iterable of paths.
    :param limit: an unsigned int - a limit for a caption's length in
    tokens. If omitted or `None`, defaults to `CAPTION_LIMIT`.
    :returns: a generator which invokes `generator` on each of the
    images.
    """
    return (generator(i, limit) for i in images)


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
    it on each image in an iterable of tf.Tensors. See CaptionGenerator
    and `generate_captions`.
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
                                 meta_tokens,
                                 tokenizer,
                                 image_options,
                                 caption_limit=None):
    """
    A convenience function which creates a CaptionGenerator and invokes
    it on each image in an iterable of image paths. See CaptionGenerator
    and `generate_captions`.

    Note that:
     - `model` must be the entire model, including the encoder module
     - the supported image formats are JPEG, PNG, GIF, BMP.
    """
    generator = CaptionGenerator(
        model,
        meta_tokens,
        tokenizer,
        is_decoder_only=False,
        image_options=image_options
    )

    return generate_captions(image_paths, generator, caption_limit)
