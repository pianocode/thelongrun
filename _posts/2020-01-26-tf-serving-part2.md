---
toc: true
layout: post
description: A minimal example of using markdown with fastpages.
categories: [markdown]
title: Building a REST API with TensorFlow Serving (part 2)
---

*This post is the second part of the tutorial of Tensorflow Serving in order to productionize Tensorflow objects and build a REST API to make calls to them. Part 1 is located [here]({{site.baseUrl}}{% link _posts/2020-01-12-tf-serving-part1.md %}).*

Once these Tensorflow objects have been generated, it’s time to make them publicly available to everyone. By building a REST API around the object, people will be able to use your service in their project.

## Docker in a nutshell

Docker is a tool to build isolated environments (containers) in your computer in such a way that it doesn’t get into conflict with any file or program in your local filesystem (the host). Among all its features, I would highlight these:

* You can run containers with only what’s strictly necessary to run your piece of code. Containers are a lighter version of virtual machines.
* Docker networking capabilities allows you easily communicate multiple containers to each other.
* Even if your OS is not fully compatible with the tool you want to use, with containers you don’t run into compatibility issues anymore.
* Docker containers will run in the same way regardless of the hosting environment, be in your computer or a server running in a cloud service.

TensorFlow Serving has a quick start [tutorial](https://github.com/tensorflow/serving) that's good introduction to the package.

```{bash}
# Download the TensorFlow Serving Docker image and repo
docker pull tensorflow/serving
 
git clone https://github.com/tensorflow/serving
# Location of demo models
TESTDATA="$(pwd)/serving/tensorflow_serving/servables/tensorflow/testdata"
 
# Start TensorFlow Serving container and open the REST API port
docker run -t --rm -p 8501:8501 \
    -v "$TESTDATA/saved_model_half_plus_two_cpu:/models/half_plus_two" \
    -e MODEL_NAME=half_plus_two \
    tensorflow/serving &
 
# Query the model using the predict API
curl -d '{"instances": [1.0, 2.0, 5.0]}' \
    -X POST http://localhost:8501/v1/models/half_plus_two:predict
 
# Returns => { "predictions": [2.5, 3.0, 4.5] }
```

Pay attention to the arguments passed to the `docker run` command, specifically the ones accepting external values:
* `-p 8501:8501`, publishes the container’s port specified at the right of the colon, and is mapped to the same port in the host, specified at the left of the colon. For REST API, Tensorflow Serving makes use of this port, **so don’t change this parameter in your experiments**.
* `-v "$TESTDATA/saved_model_half_plus_two_cpu:/models/half_plus_two"`, attaches a volume to the container. This volume contains a copy of the folder where you saved your Tensorflow object. Just a level above the folder named `/1/`. This folder will appear in the container, under `/models/`.
* `-e MODEL_NAME=half_plus_two`, defines an environment variable. This variable is required to serve your model. For convenience, **use the same identifier as the container’s folder name where you attached your model**.

## Deploying servables in containers

You can design an API for your servable, but TensorFlow Serving abstracts away this step thanks to Docker. Once you deploy the container, you can make a request to the server to perform some kind of computation. Within the body of the request you may attach data (required to run the servable) and obtain some an output in return.

To make the computation you need to specify the endpoint URL of the servable in your request. In the example shown above this endpoint URL is `http://localhost:8501/v1/models/half_plus_two:predict`. Now everything is ready to run our TensorFlow objects. We will start with the Keras model:

```{bash}
docker run -t --rm -p 8501:8501 -v "$(pwd)/mobilenet_v2_test:/models/mobilenet_v2_test" -e MODEL_NAME=mobilenet_v2_test tensorflow/serving &
```

When this command was executed, the current directory was tmp/ (where I put all my models). and this is what the terminal returned:

![]({{site.baseurl}}/images/docker_run_tf_serving.png "Terminal run TF Serving")

The model is up and ready to send request to.

## Making requests to servables

Now that the container is up and running we can send requests with an image to be classified. I’ll show you two ways to achieve that.

### With `curl` library

First I made a little shell script (download it from [here](https://gist.github.com/mlgxmez/6cd3b5824567ba69edd4468e8de97f1f)) that receives the path of an image file as an argument and makes the call itself with the library `curl`. We're going to send the image of this chilling panda:

![]({{site.baseurl}}/images/imagenes-osos-panda.jpg "Panda bear image")

And this is how we make the call with the API we built:

![]({{site.baseurl}}/images/tf_serving_req1.png "Sending request shell")

The second example involves the servable that adds 2 to every element of the vector:

![]({{site.baseurl}}/images/tf_serving_req2.png "Sending request curl")

### With `requests` library

The library `requests` allows you doing the same thing but using Python code.

```{python}
import json
import requests
import base64
 
data = {}
with open('../../Downloads/imagenes-osos-panda.jpg', mode='rb') as file:
    img = file.read()
data = {"inputs":[{"b64":base64.encodebytes(img).decode("utf-8")}]}
 
# Making the request
r = requests.post("http://localhost:8501/v1/models/mobilenet_v2_test:predict", data=json.dumps(data))
r.content
# And returns:
# b'{\n    "outputs": [\n        "giant panda"\n    ]\n}'
```
In this piece of code, the input image is parsed as a JSON file using Base64 encoding before sending the request. More details on how to accomplish this is explained in the TensorFlow [documentation](https://www.tensorflow.org/tfx/serving/api_rest#predict_api).

Building a REST API with TensorFlow Serving is the stepping stone to use more advanced features. These will be covered in a future post. Stay tuned for more content on TensorFlow Serving!