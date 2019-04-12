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

from tensorflow_probability import distributions as tfd
import tensorflow as tf

from planet.control import discounted_return
from planet import tools
COMMAND_ORDINAL = {
    "REACH_GOAL": 0,
    "GO_STRAIGHT": 1,
    "TURN_RIGHT": 2,
    "TURN_LEFT": 3,
    "LANE_FOLLOW": 4,
}

def cross_entropy_method(
    cell, objective_fn, state, obs_shape, action_shape, horizon,
    amount=1000, topk=100, iterations=10, discount=0.99,
    min_action=-1, max_action=1, command=1):
  obs_shape, action_shape = tuple(obs_shape), tuple(action_shape)
  original_batch = tools.shape(tools.nested.flatten(state)[0])[0]
  initial_state = tools.nested.map(lambda tensor: tf.tile(
      tensor, [amount] + [1] * (tensor.shape.ndims - 1)), state)
  extended_batch = tools.shape(tools.nested.flatten(initial_state)[0])[0]
  use_obs = tf.zeros([extended_batch, horizon, 1], tf.bool)
  obs = tf.zeros((extended_batch, horizon) + obs_shape)
  length = tf.ones([extended_batch], dtype=tf.int32) * horizon

  def iteration(mean_and_stddev, _):
    mean, stddev, command = mean_and_stddev
    # mean 1 12 2
    # stddev 1 12 2
    # Sample action proposals from belief.
    normal = tf.random_normal((original_batch, amount, horizon) + action_shape)
    action = normal * stddev[:, None] + mean[:, None]
    action = tf.clip_by_value(action, min_action, max_action)
    # Evaluate proposal actions.
    action = tf.reshape(
        action, (extended_batch, horizon) + action_shape)
    (_, state), _ = tf.nn.dynamic_rnn(
        cell, (0 * obs, action, use_obs), initial_state=initial_state)
    # action
    # Tensor(
    #     "graph/collection/should_collect_carla/simulate-1/train-carla-cem-12/scan/while/simulate/scan/while/Reshape:0",
    #     shape=(1000, 12, 2), dtype=float32)
    reward = objective_fn(state)
    bond_turn = tf.reshape(tf.reduce_sum(action[:, :, 1], axis=1), [1, 1000])
    bond_turn = tf.clip_by_value(bond_turn, -10, 10)
    bond_keep = tf.reshape(tf.reduce_sum(action[:, :, 0], axis=1), [1, 1000])
    bond_straight = tf.reshape(tf.reduce_sum(action[:, :, 0], axis=1), [1, 1000]) - \
                    tf.abs(tf.reshape(tf.reduce_sum(action[:, :, 1], axis=1), [1, 1000]))
    bond_straight = tf.clip_by_value(bond_straight, -8, 8)
    bond_keep = tf.clip_by_value(bond_keep, -8, 8)

    def f1(): return bond_straight   # go straight bond

    def f2(): return bond_turn + 0.2 * bond_keep      # right turn bond

    def f3(): return -bond_turn + 0.2 * bond_keep     # left turn bond

    def f4(): return bond_keep       # lane keep bond

    # bond = tf.case({tf.reduce_all(tf.equal(command, 2)): f1,
    #                 tf.reduce_all(tf.equal(command, 3)): f2}, default=f3, exclusive=True)
    bond = tf.case({tf.reduce_all(tf.equal(command, 2)): f2,
                    tf.reduce_all(tf.equal(command, 3)): f3,
                    tf.reduce_all(tf.equal(command, 4)): f4}, default=f1, exclusive=True)

    return_ = discounted_return.discounted_return(
        reward, length, discount)[:, 0]
    return_ = tf.reshape(return_, (original_batch, amount))
    return_ += bond
    # Re-fit belief to the best ones.
    _, indices = tf.nn.top_k(return_, topk, sorted=False)
    indices += tf.range(original_batch)[:, None] * amount
    best_actions = tf.gather(action, indices)
    mean, variance = tf.nn.moments(best_actions, 1)
    stddev = tf.sqrt(variance + 1e-6)
    return mean, stddev, command

  mean = tf.zeros((original_batch, horizon) + action_shape)
  # print('>>>>>>>>>>>>>>>>>>><<<<<<<<<<<<<<<<<<<\n'*10)

  def f1():
      x = tf.concat([mean[:, :, 0]+0.6, mean[:, :, 1]], 0)
      return tf.expand_dims(tf.transpose(x), 0)

  def f2():
      x = tf.concat([mean[:, :, 0]+0.2, mean[:, :, 1]+0.3], 0)
      return tf.expand_dims(tf.transpose(x), 0)

  def f3():
      x = tf.concat([mean[:, :, 0]+0.2, mean[:, :, 1]-0.3], 0)
      return tf.expand_dims(tf.transpose(x), 0)
  command = tf.reshape(command, (1, -1))
  mean = tf.case({tf.reduce_all(tf.equal(command, 2)): f2,
                  tf.reduce_all(tf.equal(command, 3)): f3}, default=f1, exclusive=True)

  stddev = tf.ones((original_batch, horizon) + action_shape)

  mean, stddev, command = tf.scan(
      iteration, tf.range(iterations), (mean, stddev, command * tf.ones([1, 12, 2], dtype=tf.int32)), back_prop=False)
  mean, stddev = mean[-1], stddev[-1]  # Select belief at last iterations.
  return mean
