# Unbias

Keras based framework to build unbiased models from biased data.

-----

## Workflow

Suppose you are building a neural network that makes hiring decisions by looking at resumes. You train your model on data which contains past resumes and hiring decisions. You don't want any biases that were involved in making the past decisions be encoded in your model. This is how you do it:

1) Prepare your data:

Vectorize your resumes. Convert hiring decisions to one hots. Your data should look like this:

![data](pics/xy.png)


