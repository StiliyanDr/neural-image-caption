# neural-image-caption

A simple Python API built on top of [TensorFlow](https://www.tensorflow.org/) for neural image captioning with [MSCOCO](https://cocodataset.org/) data.  

## Table of contents

 * [Description](#description)
 * [Installation](#installation)
 * [MSCOCO API](#mscoco-api)
 * [NIC Model](#nic-model)
 * [Training on Google Colab](#training-on-colab)

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

First we need to import the API.  

```python
import nic
```

### Downloading

Then we can download the dataset (from [here]("http://images.cocodataset.org")).  

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

<a name="training-on-colab"></a>

## Training on Google Colab

[Google Colab](https://colab.research.google.com/) offers a Python environment with preinstalled packages like TensorFlow. It is also possible to request a GPU for a user allocated runtime. The runtimes have limited resources and even though Google Drive can be mounted, it most definitely wouldn't fit the entire MSCOCO dataset (the images in particular).  

To take advantage of Colab, we can:  

* preprocess the dataset on our machines once
* create an archive file containing image features
* upload it to Google Drive
* extract the features into the runtime
* train and evaluate a model using a GPU

In fact, the *data.zip* file contains preprocessed MSCOCO data with image features extracted with Inception ResNet v2. The *neural_image_caption.ipynb* notebook can be used with this archive file to train and evaluate the decoder module of a model on Google Colab.  
