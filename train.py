# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Binary for training translation models and decoding from them.

Running this program without --decode will download the WMT corpus into
the directory specified as --data_dir and tokenize it in a very basic way,
and then start training a model saving checkpoints to --train_dir.

Running with --decode starts an interactive loop so you can see how
the current checkpoint translates English sentences into French.

See the following papers for more information on neural translation models.
 * http://arxiv.org/abs/1409.3215
 * http://arxiv.org/abs/1409.0473
 * http://arxiv.org/abs/1412.2007
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import os
import random
import sys
import time
import logging

import numpy as np
import tensorflow as tf

from model import Seq2SeqModel
import preprocess


tf.app.flags.DEFINE_float("learning_rate", 0.5, "Learning rate.")
tf.app.flags.DEFINE_float("learning_rate_decay_factor", 0.99, "Learning rate decays by this much.")
tf.app.flags.DEFINE_float("max_gradient_norm", 5.0, "Clip gradients to this norm.")

tf.app.flags.DEFINE_integer("batch_size", 64, "Batch size to use during training.")
tf.app.flags.DEFINE_integer("rnn_size", 512, "Size of each model layer.")
tf.app.flags.DEFINE_integer("num_layers", 2, "Number of layers in the model.")
tf.app.flags.DEFINE_integer("num_samples", 512, "Number of samples for sampled softmax.")

tf.app.flags.DEFINE_integer("src_vocab_size", 50000, "English vocabulary size.")
tf.app.flags.DEFINE_integer("tgt_vocab_size", 50000, "French vocabulary size.")
tf.app.flags.DEFINE_integer("max_training_size", -1, "Maximum number of training samples to take, -1 for unlimited.")

tf.app.flags.DEFINE_string("model_path", "/data/xinyu/cmv/counterarg/question_generation/model/vanilla-seq2seq/tf-seq2seq/", "Training directory.")
tf.app.flags.DEFINE_string("data_path", "/data/xinyu/cmv/counterarg/question_generation/trainable/tf-seq2seq/", "Data directory.")

tf.app.flags.DEFINE_boolean("decode", False,
                            "Set to True for interactive decoding.")
tf.app.flags.DEFINE_boolean("self_test", False,
                            "Run a self-test if this is set to True.")

FLAGS = tf.app.flags.FLAGS

# We use a number of buckets and pad to the closest one for efficiency.
# See seq2seq_model.Seq2SeqModel for details of how they work.
#_buckets = [(200, 20), (300, 30), (400, 40), (500,50), (1700, 60)]
_buckets = [(200, 20), (300, 30), (400, 40)]

def read_data(source_path, target_path, max_size=-1):
  """Read data from source and target files and put into buckets.

  Args:
    source_path: path to the files with token-ids for the source language.
    target_path: path to the file with token-ids for the target language;
      it must be aligned with the source file: n-th line contains the desired
      output for n-th line from the source_path.
    max_size: maximum number of lines to read, all other will be ignored;
      if 0 or None, data files will be read completely (no limit).

  Returns:
    data_set: a list of length len(_buckets); data_set[n] contains a list of
      (source, target) pairs read from the provided data files that fit
      into the n-th bucket, i.e., such that len(source) < _buckets[n][0] and
      len(target) < _buckets[n][1]; source and target are lists of token-ids.
  """
  data_set = [[] for _ in _buckets]
  fin_src = open(source_path)
  fin_tgt = open(target_path)
  src_content = [line.strip().split() for line in fin_src]
  tgt_content = [line.strip().split() for line in fin_tgt]
  counter = 0
  for idx, src_line in enumerate(src_content):
      tgt_line = tgt_content[idx]
      if src_line and tgt_line and (max_size < 0 or counter < max_size):
          counter += 1
          if counter % 100000 == 0:
              print(" reading data line %d" % counter)
          src_ids = [int(x) for x in src_line]
          tgt_ids = [int(x) for x in tgt_line]
          tgt_ids.append(preprocess.END_ID)
          for bucket_id, (src_size, tgt_size) in enumerate(_buckets):
              if len(src_ids) < src_size and len(tgt_ids) < tgt_size:
                  data_set[bucket_id].append([src_ids, tgt_ids])
                  break
  return data_set


def create_model(session, mode='train'):
  """Create translation model and initialize or load parameters in session."""
  print('building model...')
  model = Seq2SeqModel(
      FLAGS.src_vocab_size,
      FLAGS.tgt_vocab_size,
      _buckets,
      FLAGS.rnn_size,
      FLAGS.num_layers,
      FLAGS.max_gradient_norm,
      FLAGS.batch_size,
      FLAGS.learning_rate,
      FLAGS.learning_rate_decay_factor,
      FLAGS.num_samples,
      mode)
  print('seq2seq model built')
  ckpt = tf.train.get_checkpoint_state(FLAGS.model_path)
  if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
    print("Reading model parameters from %s" % ckpt.model_checkpoint_path)
    model.saver.restore(session, ckpt.model_checkpoint_path)
  else:
    print("Created model with fresh parameters.")
    session.run(tf.global_variables_initializer())
  return model


def train():
    train_set = read_data(source_path=FLAGS.data_path + "src-train.ids", target_path=FLAGS.data_path + "tgt-train.ids", max_size=FLAGS.max_training_size)
    valid_set = read_data(source_path=FLAGS.data_path + "src-train.ids", target_path=FLAGS.data_path + "tgt-train.ids", max_size=FLAGS.max_training_size)
    with tf.Session() as sess:
        model = create_model(sess, mode='train')
        train_summary_writer = tf.summary.FileWriter(FLAGS.model_path)
        train_bucket_sizes = [len(train_set[b]) for b in xrange(len(_buckets))]
        train_total_size = float(sum(train_bucket_sizes))
        step_num_for_one_epoch = train_total_size // FLAGS.batch_size
        step_num_per_checkpoint = step_num_for_one_epoch        # 5 checkpoints for 1 epoch
        print('train_bucket_sizes:', train_bucket_sizes)
        print('train_total_size:', train_total_size)
        print('number of steps for 1 epoch:', step_num_for_one_epoch)

        train_buckets_scale = [sum(train_bucket_sizes[:i + 1]) / train_total_size for i in xrange(len(train_bucket_sizes))]
        step_time, loss = 0.0, 0.0
        current_step, current_epoch, stride = 0, 0, 0
        previous_losses = []
        while True:
            random_number_01 = np.random.random_sample()
            bucket_id = min([i for i in xrange(len(train_buckets_scale)) if train_buckets_scale[i] > random_number_01])
            start_time = time.time()
            encoder_inputs, decoder_inputs, target_weights = model.get_batch(train_set, bucket_id)
            _, step_loss, _, summary = model.step(sess, encoder_inputs, decoder_inputs, target_weights, bucket_id, False)
            step_time += (time.time() - start_time) / step_num_per_checkpoint
            loss += step_loss / step_num_per_checkpoint
            tf.scalar_summary(summary, current_step)
            current_step += 1

            if current_step % 500 == 0:
                print('current step: %d' % current_step)

            if current_step % step_num_per_checkpoint == 0:
                if stride == 1:
                    stride = 0
                    current_epoch += 1
                else:
                    stride += 1

                train_summary_writer.add_summary(summary, model.global_step.eval())
                # Print statistics for the previous epoch.
                perplexity = math.exp(float(loss)) if loss < 300 else float("inf")
                print("current step %d\tcurrent epoch %d" % (current_step, current_epoch))
                print ("global step %d learning rate %.4f step-time %.2f perplexity %.2f loss %.4f"
                       % (model.global_step.eval(), model.learning_rate.eval(), step_time, perplexity, float(loss)))
                # Decrease learning rate if no improvement was seen over last 3 times.
                if len(previous_losses) > 2 and loss > max(previous_losses[-3:]):
                    sess.run(model.learning_rate_decay_op)
                previous_losses.append(loss)
                # Save checkpoint and zero timer and loss.
                checkpoint_path = os.path.join(FLAGS.model_path, "seq2seq.ckpt")
                model.saver.save(sess, checkpoint_path, global_step=model.global_step)
                step_time, loss = 0.0, 0.0
                # Run evals on development set and print their perplexity.
                for bucket_id in xrange(len(_buckets)):
                    if len(valid_set[bucket_id]) == 0:
                        print("  eval: empty bucket %d" % (bucket_id))
                        continue
                encoder_inputs, decoder_inputs, target_weights = model.get_batch(
                    valid_set, bucket_id)
                _, eval_loss, _ = model.step(sess, encoder_inputs, decoder_inputs,
                                             target_weights, bucket_id, True)
                eval_ppx = math.exp(float(eval_loss)) if eval_loss < 300 else float(
                    "inf")
                print("  eval: bucket %d perplexity %.2f loss %.4f" % (bucket_id, eval_ppx, float(eval_loss)))
                sys.stdout.flush()

def decode():
    return

def self_test():
  """Test the translation model."""
  with tf.Session() as sess:
    print("Self-test for neural translation model.")
    # Create model with vocabularies of 10, 2 small buckets, 2 layers of 32.
    model = Seq2SeqModel(10, 10, [(3, 3), (6, 6)], 32, 2,
                                       5.0, 4, 0.3, 0.99, num_samples=8)
    sess.run(tf.global_variables_initializer())

    # Fake data set for both the (3, 3) and (6, 6) bucket.
    data_set = ([([1, 1], [2, 2]),
                 ([3, 3], [4]),
                 ([5], [6])],
                [([1, 1, 1, 1, 1], [2, 2, 2, 2, 2]),
                 ([3, 3, 3], [5, 6])])
    for i in xrange(500):  # Train the fake model for 5 steps.
        bucket_id = random.choice([0])
        encoder_inputs, decoder_inputs, target_weights = model.get_batch(data_set, bucket_id)
        _, loss, _, summary = model.step(sess, encoder_inputs, decoder_inputs, target_weights, bucket_id, False)
        if i % 100 == 0:
            print('step %d, loss: %.4f' % (i, loss))
            print('bucket_id: %d' % bucket_id)
            print('encoder_inputs\n', encoder_inputs)
            print('decoder_inputs\n', decoder_inputs)
            print('target_weights\n', target_weights)



def main(_):
  if FLAGS.self_test:
    self_test()
  elif FLAGS.decode:
    decode()
  else:
    train()

if __name__ == "__main__":
  tf.app.run()