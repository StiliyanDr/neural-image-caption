import tensorflow as tf

from nic.datapreparation.preprocessing import prepare_image


_CAPTION_LIMIT = 100


class CaptionGenerator:
    """
    Generates captions given images.
    """
    def __init__(self,
                 model,
                 encoder_name,
                 decoder_name,
                 meta_tokens,
                 tokenizer,
                 image_options=None):
        """
        :param model: an instance of the NIC model created with
        `define_model` or `connect`.
        :param encoder_name: a str - the name of the CNN encoder module
        of the model.
        :param decoder_name: a str - the name of the decoder module
        of the model.
        :param meta_tokens: an instance of MetaTokens - the meta tokens
        with which the data was preprocessed.
        :param tokenizer: the tokenizer built from the data, see
        `load_tokenizer`.
        :param image_options: an instance of ImageOptions or None. When
        an instance of ImageOptions, it should be the instance used when
        preprocessing the data; in this case the object supports loading
        and preparing images when given paths. When None, the object
        must only be given tf.Tensors which represent already prepared
        images.
        """
        hidden_state_output = self.__set_up_encoder(model,
                                                    encoder_name,
                                                    decoder_name)
        self.__set_up_decoder(model, decoder_name, hidden_state_output)
        self.__meta_tokens = meta_tokens
        self.__tokenizer = tokenizer
        self.__image_encoder, self.__target_size = (
            (getattr(tf.keras.applications, image_options.model_name),
             image_options.target_size)
            if (image_options is not None)
            else (None, None)
        )

    def __set_up_encoder(self, model, encoder_name, decoder_name):
        image_input = model.inputs[0]
        encoder = model.get_layer(encoder_name)
        decoder = model.get_layer(decoder_name)
        features_transformation = decoder.get_layer(
            "features-transformation"
        )
        features = encoder(image_input)
        hidden_state = features_transformation(features)
        self.__encoder = tf.keras.Model(inputs=image_input,
                                        outputs=hidden_state)
        return hidden_state

    def __set_up_decoder(self, model, decoder_name, hidden_state):
        captions_input = model.inputs[1]
        hidden_state_input = tf.keras.Input(
            shape=hidden_state.shape[-1],
            dtype=hidden_state.dtype
        )
        c_state_input = tf.keras.Input(
            shape=hidden_state.shape[-1],
            dtype=hidden_state.dtype
        )
        decoder = model.get_layer(decoder_name)
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
        :param image: a str or a tf.Tensor. If a tensor, it is assumed
        to be prepared for the model. If a str, the image is loaded and
        prepared before a caption is generated for it.
        :param limit: an unsigned int - a limit for the caption's
        length in tokens. If omitted or `None`, defaults to
        `CAPTION_LIMIT`.
        :returns: a list of tokens - the caption.
        :raises RuntimeError: if `image` is str and the object does not
        support image loading.
        """
        return self.__caption_image(
            self.__prepare_if_path(image),
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

    def __caption_image(self, image, limit):
        h_state = self.__encoder(image)
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
        return self.__decoder(
            tf.constant([[index]]),
            h_state,
            c_state
        )

    def __most_probable_word(self, probabilities):
        samples = tf.random.categorical(probabilities, num_samples=1)
        index = samples[0][0].numpy()

        return self.__tokenizer.index_word[index]
