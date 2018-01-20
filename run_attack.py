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

NUM_CLASSES = 10

def run_attack(checkpoint, x_adv, epsilon):
  mnist = input_data.read_data_sets('MNIST_data', one_hot=False)

  model = Model()

  saver = tf.train.Saver()

  num_eval_examples = 10000
  eval_batch_size = 64

  num_batches = int(math.ceil(num_eval_examples / eval_batch_size))
  total_corr = 0

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
      cur_corr, y_pred_batch = sess.run([model.num_correct, model.y_pred],
                                        feed_dict=dict_adv)

      total_corr += cur_corr
      y_pred.append(y_pred_batch)

  accuracy = total_corr / num_eval_examples

  print('Accuracy: {:.2f}%'.format(100.0 * accuracy))
  y_pred = np.concatenate(y_pred, axis=0)
  np.save('pred.npy', y_pred)
  print('Output saved at pred.npy')

def run_class_attack(checkpoint, x_adv_all, epsilon):
  mnist = input_data.read_data_sets('MNIST_data', one_hot=False)

  model = Model()

  saver = tf.train.Saver()

  num_eval_examples = 10000
  eval_batch_size = 64

  num_batches = int(math.ceil(num_eval_examples / eval_batch_size))
  
  comb_conf_mat = np.zeros([NUM_CLASSES, NUM_CLASSES], dtype=np.float64)
  # Number of predictable classes within epsilon of each eval example
  dist_preds = np.zeros(num_eval_examples, dtype=np.int32)

  x_nat = mnist.test.images

  for x_adv in x_adv_all:
    l_inf = np.amax(np.abs(x_nat - x_adv))
    
    if l_inf > epsilon + 0.0001:
      print('maximum perturbation found: {}'.format(l_inf))
      print('maximum perturbation allowed: {}'.format(epsilon))
      return

  with tf.Session() as sess:
    # Restore the checkpoint
    saver.restore(sess, checkpoint)

    # Iterate over the samples batch-by-batch
    for (i, x_adv) in enumerate(x_adv_all):
      # Start with one confusion matrix for each attack dataset
      conf_mat = np.zeros([NUM_CLASSES, NUM_CLASSES], dtype=np.int32)

      for ibatch in range(num_batches):
        bstart = ibatch * eval_batch_size
        bend = min(bstart + eval_batch_size, num_eval_examples)

        x_batch = x_adv[bstart:bend, :]
        y_batch = mnist.test.labels[bstart:bend]

        dict_adv = {model.x_input: x_batch,
                    model.y_input: y_batch}
        y_pred_batch, conf_mat_batch = sess.run([model.y_pred,
                                                 model.conf_mat],
                                                feed_dict=dict_adv)

        conf_mat += conf_mat_batch
        dist_preds[bstart:bend] += (y_pred_batch == i)

      # Divide by counts of each class
      conf_mat = conf_mat.astype(np.float64) / conf_mat.sum(axis=1)[:, np.newaxis]

      # Take i-th column of i-th confusion matrix, corresponding to proportion of
      # each class that gets classified as i during the class i attack)
      comb_conf_mat[:, i] = conf_mat[:, i]

  return comb_conf_mat, dist_preds

def run_class_attack_ext(model_dir, adv_path, epsilon):
  """e.g. for use in jupyter notebooks to avoid config file."""
  checkpoint = tf.train.latest_checkpoint(model_dir)
  x_adv_list = []

  for i in range(NUM_CLASSES):
    path = class_attack_path(adv_path, i)
    x_adv = np.load(path)
    x_adv_list.append(x_adv)

  x_adv_all = np.stack(x_adv_list, axis=0)

  return run_class_attack(checkpoint, x_adv_all, epsilon)

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
    if checkpoint is None:
      print('No checkpoint found')
      return

    x_adv_list = []

    for i in range(NUM_CLASSES):
      path = class_attack_path(config['store_adv_path'], i)
      x_adv = np.load(path)
      x_adv_list.append(x_adv)

      if x_adv.shape != (10000, 784):
        print('Invalid shape: expected (10000,784), found {}'.format(x_adv.shape))
        return
      elif np.amax(x_adv) > 1.0001 or \
           np.amin(x_adv) < -0.0001 or \
           np.isnan(np.amax(x_adv)):
        print('Invalid pixel range. Expected [0, 1], found [{}, {}]'.format(
                                                                  np.amin(x_adv),
                                                                  np.amax(x_adv)))
        return 

    x_adv_all = np.stack(x_adv_list, axis=0)
    conf_mat, dist_preds = run_class_attack(checkpoint,
                                            x_adv_all,
                                            config['epsilon'])
    print('Final confusion matrix:\n{}'.format(np.around(conf_mat, 4)))
    return conf_mat, dist_preds

if __name__ == '__main__':
  main()
