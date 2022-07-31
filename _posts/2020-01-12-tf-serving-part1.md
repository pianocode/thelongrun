---
toc: true
layout: post
description: A minimal example of using markdown with fastpages.
categories: [markdown]
title: Building a REST API with TensorFlow Serving (part 1)
---

One of the features that I personally think is undervalued from TensorFlow is the capability of serving TensorFlow models. At the moment of writing this post, the API that helps you do that is named TensorFlow Serving and is part of the TensorFlow Extended (TFX) ecosystem.

This post and the following one, will show you how to build a REST API with TensorFlow Serving. From serializing a TensorFlow object up to testing the API endpoint.

## What are *servables*?
Functions, embeddings and saved models are all objects that can be used as servables. How are servables defined in TensorFlow?

This is up to you but they must be able to be saved in what's called the **SavedModel format**. This format preserves the components of the TensorFlow object in the same state when the object is loaded in a different environment. The components of a TensorFlow object can be weights, the graph, additional assets, etc.

This post covers two types of objects:
* TensorFlow functions
* Keras models

## TensorFlow function as servable

TensorFlow functions are saved as valid servables if are defined in this way:
```{python}
class Adder(tf.Module):
    @tf.function(input_signature=[tf.TensorSpec(shape=[None, 3], dtype=tf.flaot32, name='x')])
    def sum_two(self, x):
        return x + 2
```

* Function definition inside Python class
* The parent class has to be `tf.Module`
* The `@tf.function` decorator translates the function definition into a TensorFlow graph
* The `input_signature` argument defines the type and shape of tensors that are accepted to be passed in the function

To return tensors with two dimensions, its shape has three elements being the second the length of the second dimension. Another example of a TensorFlow function:

```{python}
class Randomizer(tf.Module):
    @tf.function
    def fun_runif(self, N):
        return tf.random.uniform(shape=(N,))
```

Notice that `input_signature` in the decorator is not mandatory but always is good to include some safety tests when functions go into production. Now we create instances of these two object and save them in the local filesystem. For more information about `tf.saved_model` check this ![link](https://www.tensorflow.org/api_docs/python/tf/saved_model)

```{python}
# For the first function
myfun = Adder()
tf.saved_model.save(myfun, "tmp/sum_two/1")
 
# For the second function
myfun2 = Randomizer()
tf.saved_model.save(myfun2, "tmp/fun_runif/1")
```

## Keras model as servable

The same can be done with Keras models. This code snippet downloads a pretrained model for image classification from TensorFlow Hub. A custom class is created to preprocess external images.

```{python}
class CustomMobileNet_string(tf.keras.Model):
    model_handler = "https://tfhub.dev/google/imagenet/mobilenet_v2_035_224/classification/4"
     
    def __init__(self):
        super(CustomMobileNet_string, self).__init__()
        self.model = hub.load(self.__class__.model_handler)
        self.labels = None
         
    # Design you API with 'tf.function' decorator
    @tf.function(input_signature=[tf.TensorSpec(shape=None, dtype=tf.string)])
    def call(self, input_img):
        def _preprocess(img_file):
            img_bytes = tf.reshape(img_file, [])
            img = tf.io.decode_jpeg(img_bytes, channels=3)
            img = tf.image.convert_image_dtype(img, tf.float32)
            return tf.image.resize(img, (224, 224))
 
        labels = tf.io.read_file(self.labels)
        labels = tf.strings.split(labels, sep='\n')
        img = _preprocess(input_img)[tf.newaxis,:]
        logits = self.model(img)
        get_class = lambda x: labels[tf.argmax(x)]
        class_text = tf.map_fn(get_class, logits, tf.string)
        return class_text # index of the class
```

The class inherits from tf.keras.Model and there are few things to discuss about it:
1. The input to the model is a string of bytes, which come in a JSON file. More on that in the second part of the tutorial.
2. `tf.reshape` is at the top of the preprocessing stage due to shape restrictions set in the `@tf.function` decorator.
3. The attribute `labels` store *ImageNet* labels (available ![here](https://storage.googleapis.com/download.tensorflow.org/data/ImageNetLabels.txt)) because we want the model to return the label as text.

There's a slight change in the code when we save this servable:

```{python}
model_string = CustomMobileNet_string()
# Save the image labels as an asset, saved in 'Assets' folder
model_string.labels = tf.saved_model.Asset("data/labels/ImageNetLabels.txt")
tf.saved_model.save(model_string, "tmp/mobilenet_v2_test/1/")
```

To add more components besides the model into the SavedModel object, we need an `Asset`. The way it's done here is by adding the asset as an attribute of the class instance before saving the model.

## Further details

When the model is saved, you can navigate to the directory and should see the following directory structure:

![]({{site.baseurl}}/images/tf_savedmodel_dir.png "Directory saved model")

The files generated are:
* the graph of the function or model, saved in a Protobuf file with extension `.pb`
* the weights of the model or any TensorFlow Variable used in the servable, saved in the `variables` folder
* extra components are saved in the `assets` folder but it is empty in our examples

There are some questions that may arise when you build your own functions or models:

**What’s the reasoning behind the choice of parent classes?**

Attaching tf.Module class to a tf.function allows the latter to be saved with tf.saved_model. The same goes for the tf.keras.Model. You can find more info ![here](https://www.tensorflow.org/guide/saved_model#reusing_savedmodels_in_python).
**Why you add /1 in the model’s path?**

Servables must have an ID indicating the version of the model we are running inside the container. It’s helpful to keep track of multiple versions of your model when you are monitoring their metrics. You can a more in-depth explanation in the following ![link](https://stackoverflow.com/a/45552938).

Now, take a break and be ready to tackle TensorFlow Serving in part 2.
