# Unbias

Keras based framework to build unbiased models from biased data.

-----

## Workflow

Suppose you are building a neural network that makes hiring decisions by looking at resumes. You train your model on data which contains past resumes and hiring decisions. You don't want any biases that were involved in making the past decisions be encoded in your model. This is how you do it:

1) Prepare your data:

Vectorize your resumes. Convert hiring decisions to one hots. Your data should look like this:

![data](pics/xy.png)


2) Normally you would train a Keras model directly on this data like this:

```python
from keras.models import Sequential
from keras.layers import *

X = ...
Y = ...

model = Sequential()
model.add(....)
model.add(....)
model.compile(...)
model.fit(X, Y)
```

But your data is probably biased, and your model will learn these biases as well. For e.g, your model might hire female candidates for HR positions even if they have programming experience. To resolve this, you need explicit labels for each axis along which your model might be biased. An axis of bias could be gender, race, age, or any feature which should not affect the final output of your model. In the real world, you might want to spend some time manually labelling each resume :(.

![data with labels](pics/xly.png)

Note that your model can learn biases along such axes even if they are not among of your input features. For e.g, you may not have a 'gender' feature in your data, but gender can be inferred from other features, such as sports, words used in the resume (especially if you used word embeddings to encode free text), whether graduated from a boys' only school etc.

