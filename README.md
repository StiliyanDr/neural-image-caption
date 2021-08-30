# neural-image-caption

A simple Python API built on top of [TensorFlow](https://www.tensorflow.org/) for neural image captioning with [MSCOCO](https://cocodataset.org/) data.  

## Table of contents

 * [Description](#description)
 * [Installation](#installation)
 * [MSCOCO API](#mscoco-api)
 * [NIC Model](#nic-model)
 * [Training on Google Colab](#training-on-colab)
 * [References](#references)

<a name="description"></a>

## Description

The **nic** API has two main purposes:  

* working with the MSCOCO dataset  
  The data can be downloaded, preprocessed and then loaded into Python objects as expected by TensorFlow.  
* training a neural network model for image captioning  
  A deep neural network model with sequence-to-sequence architecture can be easily defined, trained on the dataset and then used to caption images.  

These are discussed in more detail in the following sections.  

<a name="installation"></a>

## Installation

The API is available on PYPI and can be istalled with pip:  
`pip install nic`  

<a name="mscoco-api"></a>

## MSCOCO API

The MSCOCO dataset consists of more than 100 000 captioned images. Each image is "paired" with a few descriptions (in English) of what can be seen on it.  

The **nic** API makes it possible to download the dataset, preprocess the data and load it into Python objects used to train neural networks. We'll look into each of these next.  

Note that the dataset is very big so downloading and preprocessing it will take up a lot of space. At the time of writing this, **an archive file** of the dataset is between 10 and 20 GB. This is why getting rid of the original data might be a good idea once it is preprocessed.  

First we need to import the API (and TensorFlow).  

```python
import tensorflow as tf

import nic
```

### Downloading

Then we can download the dataset (from [here](http://images.cocodataset.org)).  

```python
mscoco_dir = r"mscoco"
version = "2017"
nic.dp.download_mscoco(mscoco_dir, version)
```

The dataset has train and validation splits so we will create a test split from the train data. Usually 20% of the samples are used for testing:  

```python
nic.dp.split_out_test_data(mscoco_dir,
                           split=0.2,
                           version=version,
                           verbose=True)
```

------

A note for those who may want to use the original MSCOCO data for something else too:   
The train images (randomly) selected for testing are moved from *mscoco/train2017* to a separate directory named *mscoco/test2017*. Their annotations are extracted from *annotations/captions_train2017.json* to *annotations/captions_test2017.json* but this extraction simply removes the annotations from the 'annotations' list in the first file and creates the second file which only contains the extracted annotations like so: `{"annotations": <annotations>}`.  
A copy of the original train captions file is created as back up so the original structure of the dataset can be restored by moving the images back to *train2017*, deleting the *captions_test2017.json* file and restoring the backup file with train captions.  

------

### Preprocessing

Next, we preprocess the dataset by calling the `preprocess_data` function. We provide this function with the path of the MSCOCO directory, the path where to store the preprocessed data, the meta tokens to be used when preprocessing captions, the maximum number of words (if needed) to include in the dictionary extracted from the captions, and some image options.  

The image options describe the way in which images are preprocessed. Image preprocessing involves 'preparing' images for a specific CNN encoder and optionally extracting features for images by running them through the encoder. The second part is useful when doing transfer learning with the CNN encoder module of the model being frozen. Extracting the features once and reusing them to train the other model layers is much more efficient than the alternative.  

The image options are as follows:  

- model_name  
  The name of the CNN encoder to preprocess the images for. This model is looked up in `tf.keras.applications` and its `preprocess_input` method is called on batches of images
- target_size  
  The spatial size of the image, as expected by the chosen CNN encoder
- feature_extractor  
  A callable taking and returning a `tf.Tensor`. If provided, it will extract features for batches of preprocessed images
- batch_size  
  The batch size to use when preprocessing (and extracting features for) the images

------

As we will see in a moment, the API provides a function that loads preprocessed data into a `tf.data.Dataset`. For those interested, here is how the preprocessed data looks on disk:  

 - the data is stored in a directory `D` which has three subdirectories - *train*, *test* and *val*
 - each of the subdirectories has a subdirectory named *images* which stores preprocessed images and optionally a subdirectory named *features* which stores features extracted for the images. Preprocessed images and image features are pickled `tf.Tensor`s, the file names are simply `<image_id>`*.pcl*
 - each of the subdirectories also contains a file named *captions.pcl*. It contains a pickled dictionary mapping image ids (int) to a list of str captions (the original captions enclosed with the start and end meta tokens)
 - the *train* subdirectory has another file - *tokenizer.json*. This is the JSON representation of a `tf.keras.preprocessing.text.Tokenizer` created from the train captions

------

In this example we will preprocess the data for [Inception ResNet v2](https://arxiv.org/abs/1602.07261).  

```python
data_dir = "data"

encoder = tf.keras.applications.inception_resnet_v2.InceptionResNetV2(
    include_top=False,
    weights="imagenet",
    pooling="max"
)
encoder = tf.keras.Model(encoder.input,
                         encoder.layers[-1].output,
                         name="inception-resnet-v2")

image_options = nic.dp.ImageOptions(
    model_name="inception_resnet_v2",
    target_size=(299, 299),
    feature_extractor=encoder,
    batch_size=16
)
meta_tokens = nic.dp.MetaTokens(
    start="<start>",
    end="<end>",
    unknown="<unk>",
    padding="<pad>",
)
max_words = None
nic.dp.preprocess_data(source_dir=mscoco_dir,
                       target_dir=data_dir,
                       version=version,
                       image_options=image_options,
                       meta_tokens=meta_tokens,
                       max_words=max_words,
                       verbose=True)
```

### Loading preprocessed data

Preprocessed data can be loaded with the `load_data` function. It takes the path of the directory where preprocessed data is stored, the type of data to load (`'train'`, `'val'` or `'test'`) and a boolean value indicating whether to load features or preprocessed images:  

```python
train_data = nic.dp.load_data(data_dir, type="train", load_as_features=True)
test_data = nic.dp.load_data(data_dir, type="test", load_as_features=False)
```

The data is loaded into a `tf.data.Dataset` which yields 3-tuples whose components are `tf.Tensor`s:  
* the 3D image tensor or features vector (if `load_as_features` is set to `True`)
* an integer vector which represents a caption for the image, without the end meta token at the end
* an integer vector which represents the same caption but this time without the start meta token in front

The shape of the caption vectors is `(max_caption_length,)` and shorter captions are post-padded with 0 (the index of the padding meta token). The shape of the image or features tensor depends on the chosen CNN encoder.  

------

Keras models are typically trained with `tf.data.Dataset` objects which yield elements with a different structure (not  3-tuples). In order to train a model with the datasets returned by `load_data`, we'd need to customise `fit`, as explained [here](https://keras.io/guides/customizing_what_happens_in_fit/).  

As described [later](#nic-model), the **nic** API can also be used to define and train a model with the MSCOCO dataset. The `CustomModel` class defined within **nic** can be used as an example for customizing `fit` to work with the datasets returned by `load_data`.  

------

There are a few more API functions that work with preprocessed data. The [tokenizer](https://www.tensorflow.org/api_docs/python/tf/keras/preprocessing/text/Tokenizer) can be loaded like this:  

```python
tokenizer = nic.dp.load_tokenizer(data_dir)
```

Captions can be loaded into a dictionary mapping integers (image ids) to lists of strings (the original captions enclosed with the start and end meta tokens):  

```python
val_captions = nic.dp.load_captions(data_dir, type="val")
```

Images (preprocessed for the chosen CNN encoder) or their corresponding features can be loaded into a `tf.data.Dataset` which yields pairs of the images/features and the image id:  

```python
test_images, count = nic.dp.load_images(data_dir, type="test", load_as_features=False)
```

Vocabulary and features sizes can also be obtained:  

```python
vocabulary_size = nic.dp.vocabulary_size(data_dir)
features_size = nic.dp.features_size(data_dir)
```

<a name="nic-model"></a>

## NIC Model

The other main part of the **nic** API is a neural network model that can be easily defined, trained on the MSCOCO dataset and then used to caption images.  

The model has a Seq2Seq architecture which is depicted below.  
![model_architecture](docs/architecture.png)

Images are represented as 3D tensors which are fed into a CNN. The resulting feature vectors are transformed and fed into an RNN, as the initial hidden state vectors.  

Captions are tokenized and each token is represented as a vector from a word embedding. The word embeddings are fed into the RNN as inputs.  

The hidden state vectors at each time point (caption length) are transformed and projected over the vocabulary words/terms.  

During training, the word projections are used to calculate the loss (categorical cross entropy). During inference, the projections are used to generate a word distribution that is used to select the next word in the caption.  

The CNN image encoder is typically a pretrained model, like Inception ResNet v2. The rest of the model, visualised with `tf.keras.utils.plot_model`, looks like this (the RNN's hidden size is 512 in this case):  
![decoder_functional](docs/decoder.png)

The model is largely similar to the one described [here](https://ieeexplore.ieee.org/document/7505636?denied=).  

### Defining the model

First we need to import the API:  

```python
import nic
```

The CNN encoder module can be any model (built with TensorFlow 2) that transforms an image (3D tensor) into a vector. Remember that the encoder is important when preprocessing data too, as mentioned in the [MSCOCO](#mscoco-api) section. The **nic** API makes it easy to use Inception ResNet v2 via the following function call:  

```python
pooling = "max"
encoder = nic.define_encoder_model(pooling)
```

This returns the [Inception ResNet v2](https://www.tensorflow.org/api_docs/python/tf/keras/applications/inception_resnet_v2/InceptionResNetV2) model trained on ImageNet with the top layer removed and global pooling applied to the last convolutional layer so that the output is a vector. `pooling` can be `"max"` or `"avg"`.  

The rest of the model (the RNN, word embeddings and so on) is referred to as 'decoder' below for simplicity (even though that is not what is typically called a decoder).  

The decoder can be defined with the `define_decoder_model` function. It needs to be passed the features size, vocabulary size, embedding size and some options for the RNN module. The first two can be obtained via the API from preprocessed data; docs are available for the `RNNOptions` (as well as for every public object from **nic**), use `help(nic.RNNOptions)` in an interpreter. The embedding size defaults to the RNN's hidden size.  

```python
data_dir = r"data"
rnn_options = nic.RNNOptions(size=256)
embedding_size = None
decoder = nic.define_decoder_model(
    nic.dp.features_size(data_dir),
    nic.dp.vocabulary_size(data_dir),
    rnn_options,
    embedding_size,
    name="nic-decoder"
)
```

The two modules can be connected into a single model:  

```python
model = nic.connect(
    decoder,
    image_shape=(299, 299, 3),
    encoder=encoder,
    name="nic"
)
```

The `image_shape` argument must be a three-tuple of integers - the shape of the input images, as expected by the encoder.  

If the encoder module is going to be the default one - Inception ResNet v2, the model can be defined like so:  

```python
model = nic.define_model(nic.dp.vocabulary_size(data_dir),
                         rnn_options,
                         embedding_size,
                         pooling)
```

### Training the model

The model, or the decoder module only, can be trained on preprocessed MSCOCO data. First, the model (or decoder) needs to be compiled:  

```python
compiled_model = nic.compile_model(
    model,
    learning_rate=0.0001
)
```

A compiled model can be trained with `train_model`:  

```python
checkpoint_dir = r"training/checkpoints"
tensor_board_dir = r"training/tensor_board"

history, test_metrics = nic.train_model(
    model=compiled_model,
    path_to_data=data_dir,
    is_decoder_only=False,
    batch_size=32,
    buffer_size=1024,
    tensor_board_dir=tensor_board_dir,
    tensor_board_update_freq="epoch",
    checkpoint_dir=checkpoint_dir,
    checkpoint_freq="epoch",
    learning_rate_decay=0.9,
    decay_patience=3,
    perplexity_delta=0.001,
    min_learning_rate=0.,
    early_stop_patience=3,
    max_epochs=10,
    shuffle_for_each_epoch=False,
    initial_epoch=0
)
```

This function trains the compiled model for at most `max_epochs` epochs, possibly shuffling the train data prior to each epoch (`shuffle_for_each_epoch`).  Resuming a training process is as easy as setting `initial_epoch` to the number of the last completed epoch and **increasing** `max_epochs`.   

The initial learning rate is the compiled model's learning rate, if the process is started from scratch; restored models (mention in a bit) come with their optimizers which include the latest learning rate. If the validation perplexity does not improve with at least `perplexity_delta` for `decay_patience` epochs in a row, the learning rate is reduced my multiplying it with `learning_rate_decay` ($lr = decay * lr$​​). If `early_stop_patience` learning rate changes still lead to no perplexity improvement (or the loss becomes NaN), the training process is terminated.  

TensorBoard logs go to `tensor_board_dir` with `tensor_board_update_freq` frequency. Checkpoints (`SavedModel`s) go to `checkpoint_dir` with `checkpoint_freq` frequency.  

`buffer_size` is the size of the buffer used to shuffle the train data before training is started.  

More details are available in the function's docs (`help(nic.train_model)`).  

A model checkpoint can be restored like so:  

```python
compiled_model = nic.restore_model(checkpoint_dir, restore_best=False)
```

Setting `restore_best` to `True` would restore the model with the best validation perplexity. Otherwise, the latest checkpoint is loaded.  

### Evaluating the model

A compiled model (which can also be the decoder module only) can be evaluated by computing its [BLEU-4](https://aclanthology.org/P02-1040.pdf) score:  

```python
meta_tokens = nic.dp.MetaTokens()
bleu_score = nic.bleu_score_of(
    compiled_model,
    is_decoder_only=False,
    path_to_data=data_dir,
    batch_size=32,
    data_type="test",
    meta_tokens=meta_tokens,
    caption_limit=100,
    verbose=True
)
```

`data_type` can also be `"val"` or even `"train"`. `meta_tokens` should be the meta tokens used when preprocessing data, which are typically the default ones so this can be omitted. `caption_limit` is a limit for the captions generated from `data_type` images. Omitting it means that there is no limit which is not a good idea for models that have not been trained for much time (as the captions are still pretty random and can be very long).  

### Generating captions

Captions can be generated by using `nic.CaptionGenerator`. It can be created from the entire model or the decoder module only.  

Creating it from the decoder is restrictive as this means it will need to be fed image features, as returned by the encoder. This is useful when training the decoder as we can evaluate it without needing the images (which take up a lot of space).      

Creating it from the entire model is useful when evaluating the model or at inference time, as we will need to process images (not image features). Here's an example:  

```python
image_options = nic.dp.ImageOptions()
generator = nic.CaptionGenerator(
    compiled_model,
    meta_tokens=meta_tokens,
    tokenizer=nic.dp.load_tokenizer(data_dir),
    is_decoder_only=False,
    image_options=image_options
)
```

Again, `image_options` should be the image options used when preprocessing the data (same for `meta_tokens`).  

A `nic.CaptionGenerator` instance generates captions for batches of images. A batch is represented as a `tf.Tensor` of:  

* image paths or 3D image tensors, when the instance is created from the entire model
* features vectors, when the instance is created from the decoder module only

For example, we can call the above generator on a batch of image paths like this:  

```python
import tensorflow as tf

image_paths = [
    "images/cat.jpg",
    "images/car.jpg",
]
captions = generator(tf.constant(image_paths), limit=None)
```

A list of captions (lists of `str` tokens) is returned. To limit the length of the captions, we can set `limit`.   

The following example shows a batch of 3D image tensors being passed to the generator:  

```python
images, count = nic.dp.load_images(data_dir, type="val")
images = images.batch(10)
images_batch, ids_batch = next(iter(images))
captions = generator(images_batch, limit=100)
```

Similarly, we could create the generator from the decoder module:  

```python
generator = nic.CaptionGenerator(
    decoder,
    meta_tokens=meta_tokens,
    tokenizer=nic.dp.load_tokenizer(data_dir),
    is_decoder_only=True
)
```

There's no need for image options as this generator will be working with image features only:  

```python
images, count = nic.dp.load_images(data_dir, type="val", load_as_features=True)
images = images.batch(10)
images_batch, ids_batch = next(iter(images))
captions = generator(images_batch, limit=100)
```

Once we have an iterable of batches, we can create a Python generator that calls the caption generator on each batch:  

```python
for captions_batch in nic.generate_captions(images, generator, limit=100):
    pass
```

There's also a convenience function that creates the caption generator and then returns a Python generator that calls the caption generator on each batch of tensors in an iterable:  

```python
batches_of_captions = list(nic.generate_captions_from_tensors(
    images,
    decoder,
    meta_tokens=meta_tokens,
    tokenizer=nic.dp.load_tokenizer(data_dir),
    is_decoder_only=True,
    caption_limit=100,
))
```

Finally, there's a high-level function that generates captions when given image paths. We also need to give it the entire model, the path to the preprocessed data, meta tokens and image options (if they are not the default ones), as well as the batch size to use and a caption limit:  

```python
images_paths = [
    "images/cat.jpg",
    "images/car.jpg",
    "images/nature.jpg",
]
captions = nic.generate_captions_from_paths(
    images_paths,
    compiled_model,
    path_to_data=data_dir,
    batch_size=32,
    meta_tokens=meta_tokens,
    image_options=image_options,
    caption_limit=100
)
image_captions = dict(zip(image_paths, captions))
```

The returned value is a Python generator which yields `str`s - the captions generated for the given images, in the same order. In the example above we pair each of the paths with the corresponding caption and create a mapping from the pairs.  

<a name="training-on-colab"></a>

## Training on Google Colab

[Google Colab](https://colab.research.google.com/) offers a Python environment with preinstalled packages like TensorFlow. It is also possible to request a GPU for a user allocated runtime. The runtimes have limited resources and even though Google Drive can be mounted, it most definitely wouldn't fit the entire MSCOCO dataset (the images in particular).  

To take advantage of Colab, we can:  

* preprocess the dataset on our machines once
* create an archive file containing image features
* upload it to Google Drive
* extract the features into the runtime
* train and evaluate a model using a GPU

In fact, the *data* directory contains preprocessed MSCOCO data with image features extracted with Inception ResNet v2. Each of the two archives - *max.zip* and *avg.zip* - is a separate instance of preprocessed data; the name indicates the global pooling applied to the output of the last convolutional block when extracting image features. The [neural_image_caption](neural_image_caption.ipynb) notebook can be used with any of the archive files to train and evaluate the decoder module of a model on Google Colab.  

<a name="References"></a>

## References

[O. Vinyals, A. Toshev, S. Bengio and D. Erhan](https://ieeexplore.ieee.org/document/7505636?denied=), "Show and Tell: Lessons Learned from the 2015 MSCOCO Image Captioning Challenge," in IEEE Transactions on Pattern Analysis and Machine Intelligence, vol. 39, no. 4, pp. 652-663, 1 April 2017, doi: 10.1109/TPAMI.2016.2587640.
