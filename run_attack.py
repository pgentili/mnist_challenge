"""Evaluates a model against examples from a .npy file as specified
   in config.json"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
import json
import math
import os
import sys
import time

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

import numpy as np

from model import Model
from pgd_attack import class_attack_path

def run_attack(checkpoint, x_adv, epsilon):
  mnist = input_data.read_data_sets('MNIST_data', one_hot=False)

  model = Model()

  saver = tf.train.Saver()

  num_eval_examples = 10000
  eval_batch_size = 64

  num_batches = int(math.ceil(num_eval_examples / eval_batch_size))
  total_corr = 0
  conf_mat = np.zeros([10, 10], dtype=np.int32)

  x_nat = mnist.test.images
  l_inf = np.amax(np.abs(x_nat - x_adv))
  
  if l_inf > epsilon + 0.0001:
    print('maximum perturbation found: {}'.format(l_inf))
    print('maximum perturbation allowed: {}'.format(epsilon))
    return

  y_pred = [] # label accumulator

  with tf.Session() as sess:
    # Restore the checkpoint
    saver.restore(sess, checkpoint)

    # Iterate over the samples batch-by-batch
    for ibatch in range(num_batches):
      bstart = ibatch * eval_batch_size
      bend = min(bstart + eval_batch_size, num_eval_examples)

      x_batch = x_adv[bstart:bend, :]
      y_batch = mnist.test.labels[bstart:bend]

      dict_adv = {model.x_input: x_batch,
                  model.y_input: y_batch}
      cur_corr, y_pred_batch, conf_mat_batch = sess.run([model.num_correct,
                                                         model.y_pred,
                                                         model.conf_mat],
                                                        feed_dict=dict_adv)

      total_corr += cur_corr
      y_pred.append(y_pred_batch)
      conf_mat += conf_mat_batch

  accuracy = total_corr / num_eval_examples
  conf_mat = conf_mat.astype(float) / conf_mat.sum(axis=1)[:, np.newaxis]

  print('Accuracy: {:.2f}%'.format(100.0 * accuracy))
  y_pred = np.concatenate(y_pred, axis=0)
  np.save('pred.npy', y_pred)
  print('Output saved at pred.npy')

  return conf_mat

def main():
  import json

  with open('config.json') as config_file:
    config = json.load(config_file)

  model_dir = config['model_dir']

  checkpoint = tf.train.latest_checkpoint(model_dir)

  if not config['class_attack']:
    x_adv = np.load(config['store_adv_path'])

    if checkpoint is None:
      print('No checkpoint found')
    elif x_adv.shape != (10000, 784):
      print('Invalid shape: expected (10000,784), found {}'.format(x_adv.shape))
    elif np.amax(x_adv) > 1.0001 or \
         np.amin(x_adv) < -0.0001 or \
         np.isnan(np.amax(x_adv)):
      print('Invalid pixel range. Expected [0, 1], found [{}, {}]'.format(
                                                                np.amin(x_adv),
                                                                np.amax(x_adv)))
    else:
      run_attack(checkpoint, x_adv, config['epsilon'])
  else:
    combined_conf_mat = np.zeros([10, 10], dtype=np.float32)
    for i in range(10):
      path = class_attack_path(config, i)
      x_adv = np.load(path)

      if checkpoint is None:
        print('No checkpoint found')
      elif x_adv.shape != (10000, 784):
        print('Invalid shape: expected (10000,784), found {}'.format(x_adv.shape))
      elif np.amax(x_adv) > 1.0001 or \
           np.amin(x_adv) < -0.0001 or \
           np.isnan(np.amax(x_adv)):
        print('Invalid pixel range. Expected [0, 1], found [{}, {}]'.format(
                                                                  np.amin(x_adv),
                                                                  np.amax(x_adv)))
      else:
        tf.reset_default_graph()
        conf_mat = run_attack(checkpoint, x_adv, config['epsilon'])
        combined_conf_mat[:, i] = conf_mat[:, i]
    print('Final confusion matrix:\n{}'.format(np.around(combined_conf_mat, 4)))
    return combined_conf_mat

if __name__ == '__main__':
  main()
