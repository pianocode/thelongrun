---
toc: true
layout: post
description: A minimal example of using markdown with fastpages.
categories: [markdown]
title: Introduction to random variables for non-data scientists
---

# Introduction to random variables for non-data scientists

![Books](https://source.unsplash.com/v1QCJQoD03k/640x426)

**Me:** *Hey, just took a pic in the library! Look at how old and dusty are these books!*

**You:** *Are these the oldest ones you found in there*

**Me:** *No, I didn't inspect all of them. just picked these five.*

**You:** *Look! In the background you can see plenty of them. How many books do you think there were in the library?*

**Me:** *Uh, I don't know...Maybe between 3000 and 5000?*

:clapper: Cut!

That's not a scene from a movie. My intention was to develop some intiuition on what random variables are, because this concept will appear again and again in this blog.

To define a random variable, it's better to start defining the opposite of a random variable.
* A **deterministic variable** is a variable we are 100% sure about its value.
* A **random variable** is characterized by absence of complete data, but we can provide an estimate of its true value based on intuition or past experiences.

In short, a deterministic variable is defined by a single value (like a number, text or date). A random variable is defined with a probability distribution over a range of values. This distribution represents what are the possible values a variable can get, each value has probability linked to.

Going back to the short dialogue, there were five books. We call the number of books in the picture a *deterministic variable*. This is how we represent that variable with a probability distribution.

![]({{site.baseurl}}/images/deterministic_var.png "Deterministic variable")

The number of books in the whole library is actually higher. I didn't count all of them. I provided an estimate. The factors that led to that estimate (between 3000 and 5000) could be the size of the library, or average thickness of book's spine. In any case, that estimate could be represented with the followng probability distribution.

![]({{site.baseurl}}/images/uniform_random_var.png "Uniform random variable")

That's a **uniform distribution**. In simple terms, it means *"the probability of the library having 3000 books is as likely as having 5000 or any other quantity in between"*.

![]({{site.baseurl}}/images/normal_random_var.png "Gaussian random variable")

That's a **normal distribution**. It can be interpreted as *"the number of books is between 3000 and 5000 but we are more certain that this quantity is somewhere in the middle"*.

Probability distributions have the following properties:
* Probability values are bounded between 0 and 1.
* The area under the curve of normalized probability distribution is equal to 1

> Warning 
When the value range is lower than one the first property is violated in favor of the second one.

## Random variables with TensorFlow Probability

TensorFlow is one of the most popular frameworks for numerical computation and machine learning. What makes TensorFlow Probability unique are the capabilities to do probabilistic programming. A programming paradigm where you can define random variables.

```python
import tensorflow as tf
import tensorflow_probability as tf

deterministic_var = tf.Variable(5.) # Books in the picture
random_var = tfd.Normal(4000.0, 500.0) # Books in the library
```

The normal distribution models the number of books in the library. `random_var` requires two arguments: the mean and the standard deviation of the distribution.

TensorFlow Probability has a list of functions to interact with random variables. Use them carefully:

```python
# Returns the mean of the random variable
random_var.mean()
 
# Returns the mode of the random variable
random_var.mode()
 
# Likelihood (probability) that the random variable is equal to 3300
random_var.prob(3300)
 
# Probability that the random variable is lower than or equal to 3300
random_var.cdf(3300)

# The 20th percentile fo the random variable 
random_var.quantile(0.2)
 
# Generate 10 samples from the distribution of the random variable
random_var.sample(10)
```

Random variables and distributions linked to them are some of the essential components of probabilistic programming. We will dive deeper in future posts.

## Further reading 
We’ve barely scratched the surface of random variables. Depending on what we are trying to describe some distributions will fit better than others. [Here](https://en.wikipedia.org/wiki/List_of_probability_distributions) is a list with (mostly) all the different probability distributions.

In the TensorFlow website there’s a great variety of [tutorials](https://www.tensorflow.org/tutorials/), and also the instructions to install this library in your computer. Similarly for Tensorflow [Probability](https://www.tensorflow.org/probability/overview).