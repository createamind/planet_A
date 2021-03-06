# Copyright 2019 The PlaNet Authors. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
from tensorflow_probability import distributions as tfd

from planet import tools

from planet import IMG_SIZE, NUM_CHANNELS
obs_size = IMG_SIZE
num_channels_x = NUM_CHANNELS

# from keras.applications import VGG16, MobileNetV2
# conv_base = MobileNetV2(include_top=False, input_shape=(96, 96, 3))
# conv_base.trainable = False
# import tensornets as nets

def encoder(obs):
  """Extract deterministic features from an observation."""
  kwargs2 = dict(strides=2, activation=tf.nn.relu)
  kwargs3 = dict(strides=3, activation=tf.nn.relu)
  kwargs1 = dict(strides=1, activation=tf.nn.relu)
  kwargs = dict(strides=2, activation=tf.nn.elu)
  hidden = tf.reshape(obs['image'], [-1] + obs['image'].shape[2:].as_list())   # (50,50,64,64,3) reshape to (2500,64,64,3)

  if obs_size == (32, 32):
    hidden = tf.layers.conv2d(hidden, 32, 4, **kwargs2)
    hidden = tf.layers.conv2d(hidden, 64, 4, **kwargs1)
    hidden = tf.layers.conv2d(hidden, 128, 4, **kwargs2)
    hidden = tf.layers.conv2d(hidden, 256, 4, **kwargs1)

  elif obs_size == (64, 64):
    hidden = tf.layers.conv2d(hidden, 32, 4, **kwargs2)
    hidden = tf.layers.conv2d(hidden, 64, 4, **kwargs2)
    hidden = tf.layers.conv2d(hidden, 128, 4, **kwargs2)
    hidden = tf.layers.conv2d(hidden, 256, 4, **kwargs2)

  elif obs_size == (96, 96):
    hidden = tf.layers.conv2d(hidden, 32, 8, **kwargs)
    hidden = tf.layers.conv2d(hidden, 64, 5, **kwargs)
    hidden = tf.layers.conv2d(hidden, 72, 5, **kwargs)
    hidden = tf.layers.conv2d(hidden, 128, 5, **kwargs)
    hidden = tf.layers.conv2d(hidden, 1024, 3, strides=1)
  # elif obs_size == (96, 96):
  #   # conv_base = MobileNetV2(include_top=False, input_shape=(96, 96, 3))
  #   # conv_base.trainable = False
  #   # hidden = conv_base(hidden)    # shape=(50, 3, 3, 1280)
  #   hidden = tf.layers.conv2d(hidden, 32, 8, **kwargs)
  #   hidden = tf.layers.conv2d(hidden, 64, 5, **kwargs)
  #   hidden = tf.layers.conv2d(hidden, 72, 5, **kwargs)
  #   hidden = tf.layers.conv2d(hidden, 128, 5, **kwargs)
  #
  #   hidden = tf.layers.conv2d(hidden, 1024, 3, strides=(1, 1), activation=tf.nn.elu)
  #   hidden = tf.layers.flatten(hidden)
  #   hidden = tf.layers.Dense(1024, tf.nn.elu)(hidden)
  #   # hidden = tf.layers.Dense(hidden, 1024, tf.nn.elu)

  elif obs_size == (128, 128):
    hidden = tf.layers.conv2d(hidden, 32, 4, **kwargs2)
    hidden = tf.layers.conv2d(hidden, 32, 3, **kwargs1)
    hidden = tf.layers.conv2d(hidden, 64, 3, **kwargs2)
    hidden = tf.layers.conv2d(hidden, 64, 3, **kwargs1)
    hidden = tf.layers.conv2d(hidden, 128, 3, **kwargs2)
    hidden = tf.layers.conv2d(hidden, 128, 3, **kwargs1)
    hidden = tf.layers.conv2d(hidden, 256, 3, **kwargs2)
    hidden = tf.layers.conv2d(hidden, 256, 3, **kwargs2)

  hidden = tf.layers.flatten(hidden)
  assert hidden.shape[1:].as_list() == [1024], hidden.shape.as_list()
  hidden = tf.reshape(hidden, tools.shape(obs['image'])[:2] + [
      np.prod(hidden.shape[1:].as_list())])
  return hidden                                                                # shape(50,50,1024)


def decoder(state, data_shape):
  """Compute the data distribution of an observation from its state."""
  kwargs2 = dict(strides=2, activation=tf.nn.relu)
  kwargs3 = dict(strides=3, activation=tf.nn.relu)
  kwargs1 = dict(strides=1, activation=tf.nn.relu)
  kwargs = dict(strides=2, activation=tf.nn.elu)
  hidden = tf.layers.dense(state, 1024, None)
  hidden = tf.reshape(hidden, [-1, 1, 1, hidden.shape[-1].value])

  if obs_size == (32, 32):
    hidden = tf.layers.conv2d_transpose(hidden, 128, 5, **kwargs1)
    hidden = tf.layers.conv2d_transpose(hidden, 64, 4, **kwargs2)
    hidden = tf.layers.conv2d_transpose(hidden, 32, 4, **kwargs1)
    hidden = tf.layers.conv2d_transpose(hidden, num_channels_x, 4, strides=2)

  elif obs_size == (64, 64):
    hidden = tf.layers.conv2d_transpose(hidden, 128, 5, **kwargs2)
    hidden = tf.layers.conv2d_transpose(hidden, 64, 5, **kwargs2)
    hidden = tf.layers.conv2d_transpose(hidden, 32, 6, **kwargs2)
    hidden = tf.layers.conv2d_transpose(hidden, num_channels_x, 6, strides=2)

  elif obs_size == (96, 96):

    hidden = tf.layers.dense(state, 1024, None)
    hidden = tf.reshape(hidden, [-1, 1, 1, hidden.shape[-1].value])
    hidden = tf.layers.conv2d_transpose(hidden, 128, 3, **kwargs)
    hidden = tf.layers.conv2d_transpose(hidden, 64, 5, **kwargs)
    hidden = tf.layers.conv2d_transpose(hidden, 48, 6, **kwargs)
    hidden = tf.layers.conv2d_transpose(hidden, 32, 5, **kwargs)
    hidden = tf.layers.conv2d_transpose(hidden, num_channels_x, 4, strides=2)

  elif obs_size == (128,128):
    hidden = tf.layers.conv2d_transpose(hidden, 256, 4, **kwargs1)
    hidden = tf.layers.conv2d_transpose(hidden, 256, 4, **kwargs1)
    hidden = tf.layers.conv2d_transpose(hidden, 128, 4, **kwargs1)
    hidden = tf.layers.conv2d_transpose(hidden, 128, 3, **kwargs1)
    hidden = tf.layers.conv2d_transpose(hidden, 64, 4, **kwargs2)
    hidden = tf.layers.conv2d_transpose(hidden, 64, 4, **kwargs1)   # ~=  pixels * stride + kernel_size
    hidden = tf.layers.conv2d_transpose(hidden, 32, 4, **kwargs2)
    hidden = tf.layers.conv2d_transpose(hidden, 32, 4, **kwargs1)
    hidden = tf.layers.conv2d_transpose(hidden, num_channels_x, 4, strides=2)


  mean = hidden

  if obs_size == (32,32):
    assert mean.shape[1:].as_list() == [32, 32, num_channels_x], mean.shape
  elif obs_size == (64,64):
    assert mean.shape[1:].as_list() == [64, 64, num_channels_x], mean.shape
  elif obs_size == (96, 96):
    assert mean.shape[1:].as_list() == [96, 96, num_channels_x], mean.shape
  elif obs_size == (128,128):
    assert mean.shape[1:].as_list() == [128, 128, num_channels_x], mean.shape

  mean = tf.reshape(mean, tools.shape(state)[:-1] + data_shape)
  dist = tools.MSEDistribution(mean)
  dist = tfd.Independent(dist, len(data_shape))
  return dist
