# Author: Xinyu Hua
# Last modified: 2017-10-13
# Modified from tensorflow official tutorial
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import random

import numpy as np
import tensorflow as tf
import preprocess


class Seq2SeqModel(object):
  """Sequence-to-sequence model with attention and for multiple buckets.

  This class implements a multi-layer recurrent neural network as encoder,
  and an attention-based decoder. This is the same as the model described in
  this paper: http://arxiv.org/abs/1412.7449 - please look there for details,
  or into the seq2seq library for complete model implementation.
  This class also allows to use GRU cells in addition to LSTM cells, and
  sampled softmax to handle large output vocabulary size. A single-layer
  version of this model, but with bi-directional encoder, was presented in
    http://arxiv.org/abs/1409.0473
  and sampled softmax is described in Section 3 of the following paper.
    http://arxiv.org/abs/1412.2007
  """

  def __init__(self,
               source_vocab_size,
               target_vocab_size,
               buckets,
               size,
               num_layers,
               max_gradient_norm,
               batch_size,
               learning_rate,
               learning_rate_decay_factor,
               num_samples=512,
               mode='train'):

    """Create the model.

    Args:
      source_vocab_size: size of the source vocabulary.
      target_vocab_size: size of the target vocabulary.
      buckets: a list of pairs (I, O), where I specifies maximum input length
        that will be processed in that bucket, and O specifies maximum output
        length. Training instances that have inputs longer than I or outputs
        longer than O will be pushed to the next bucket and padded accordingly.
        We assume that the list is sorted, e.g., [(2, 4), (8, 16)].
      size: number of units in each layer of the model.
      num_layers: number of layers in the model.
      max_gradient_norm: gradients will be clipped to maximally this norm.
      batch_size: the size of the batches used during training;
        the model construction is independent of batch_size, so it can be
        changed after initialization if this is convenient, e.g., for decoding.
      learning_rate: learning rate to start with.
      learning_rate_decay_factor: decay learning rate by this much when needed.
      use_lstm: if true, we use LSTM cells instead of GRU cells.
      num_samples: number of samples for sampled softmax.
      forward_only: if set, we do not construct the backward pass in the model.
      dtype: the data type to use to store internal variables.
    """
    self.embed_size = 300
    self.source_vocab_size = source_vocab_size
    self.target_vocab_size = target_vocab_size
    self.batch_size = batch_size
    self.buckets = buckets
    self.num_samples = num_samples
    self.rnn_size = size
    self.num_layers = num_layers
    self.max_gradient_norm = max_gradient_norm
    self.mode = mode
    self.learning_rate = tf.Variable(float(learning_rate), trainable=False, dtype=tf.float32)
    self.learning_rate_decay_op = self.learning_rate.assign(self.learning_rate * learning_rate_decay_factor)
    self.global_step = tf.Variable(0, trainable=False)
    self._add_placeholder()
    self._add_model()
    self._add_train_op()
    self.saver = tf.train.Saver(tf.global_variables())


  def _add_placeholder(self):
      # Feeds for inputs.
      self.encoder_inputs = []
      self.decoder_inputs = []
      self.target_weights = []
      for i in xrange(self.buckets[-1][0]):  # Last bucket is the biggest one.
          self.encoder_inputs.append(tf.placeholder(tf.int32, shape=[None],
                                                    name="encoder{0}".format(i)))
      for i in xrange(self.buckets[-1][1] + 1):
          self.decoder_inputs.append(tf.placeholder(tf.int32, shape=[None],
                                                    name="decoder{0}".format(i)))
          self.target_weights.append(tf.placeholder(tf.float32, shape=[None],
                                                    name="weight{0}".format(i)))

      # Our targets are decoder inputs shifted by one.
      self.targets = [self.decoder_inputs[i + 1]
                 for i in xrange(len(self.decoder_inputs) - 1)]



  def _add_model(self):

      # If we use sampled softmax, we need an output projection.
      self.output_projection = None
      softmax_loss_function = None
      # Sampled softmax only makes sense if we sample less than vocabulary size.
      if self.num_samples > 0 and self.num_samples < self.target_vocab_size:
          w_t = tf.get_variable("proj_w", [self.target_vocab_size, self.rnn_size], dtype=tf.float32)
          w = tf.transpose(w_t)
          b = tf.get_variable("proj_b", [self.target_vocab_size], dtype=tf.float32)
          self.output_projection = (w, b)

          def sampled_loss(labels, logits):
              labels = tf.reshape(labels, [-1, 1])
              # We need to compute the sampled_softmax_loss using 32bit floats to
              # avoid numerical instabilities.
              local_w_t = tf.cast(w_t, tf.float32)
              local_b = tf.cast(b, tf.float32)
              local_inputs = tf.cast(logits, tf.float32)
              return tf.cast(
                  tf.nn.sampled_softmax_loss(
                      weights=local_w_t,
                      biases=local_b,
                      labels=labels,
                      inputs=local_inputs,
                      num_sampled=self.num_samples,
                      num_classes=self.target_vocab_size),
                  tf.float32)

          softmax_loss_function = sampled_loss


      # The seq2seq function: we use embedding for the input and attention.
      def seq2seq_f(encoder_inputs, decoder_inputs, do_decode):
          def single_cell():
              return tf.contrib.rnn.BasicLSTMCell(self.rnn_size, reuse=tf.get_variable_scope().reuse)
          cell = single_cell()
          if self.num_layers > 1:
              cell = tf.contrib.rnn.MultiRNNCell([single_cell() for _ in range(self.num_layers)])
          return tf.contrib.legacy_seq2seq.embedding_attention_seq2seq(
              encoder_inputs,
              decoder_inputs,
              cell,
              num_encoder_symbols=self.source_vocab_size,
              num_decoder_symbols=self.target_vocab_size,
              embedding_size=self.embed_size,
              output_projection=self.output_projection,
              feed_previous=do_decode,
              dtype=tf.float32)

      if not self.mode == 'train':
          self.outputs, self.losses = tf.contrib.legacy_seq2seq.model_with_buckets(
              self.encoder_inputs, self.decoder_inputs, self.targets,
              self.target_weights, self.buckets, lambda x, y: seq2seq_f(x, y, True),
              softmax_loss_function=softmax_loss_function)
          # If we use output projection, we need to project outputs for decoding.
          if self.output_projection is not None:
              for b in xrange(len(self.buckets)):
                  self.outputs[b] = [
                      tf.matmul(output, self.output_projection[0]) + self.output_projection[1]
                      for output in self.outputs[b]
                      ]
      else:
          self.outputs, self.losses = tf.contrib.legacy_seq2seq.model_with_buckets(
              self.encoder_inputs, self.decoder_inputs, self.targets,
              self.target_weights, self.buckets,
              lambda x, y: seq2seq_f(x, y, False),
              softmax_loss_function=softmax_loss_function)
          tf.summary.scalar("loss", self.losses[0])
          self.summary_op = tf.summary.merge_all()
      print("seq2seq model variables created")

  def _add_train_op(self):
      # Gradients and SGD update operation for training the model.
      params = tf.trainable_variables()
      if self.mode == 'train':
          self.gradient_norms = []
          self.updates = []
          opt = tf.train.GradientDescentOptimizer(self.learning_rate)
          for b in xrange(len(self.buckets)):
              gradients = tf.gradients(self.losses[b], params)
              clipped_gradients, norm = tf.clip_by_global_norm(gradients,
                                                               self.max_gradient_norm)
              self.gradient_norms.append(norm)
              self.updates.append(opt.apply_gradients(
                  zip(clipped_gradients, params), global_step=self.global_step))

  def step(self, session, encoder_inputs, decoder_inputs, target_weights,
           bucket_id, forward_only):
    """Run a step of the model feeding the given inputs.

    Args:
      session: tensorflow session to use.
      encoder_inputs: list of numpy int vectors to feed as encoder inputs.
      decoder_inputs: list of numpy int vectors to feed as decoder inputs.
      target_weights: list of numpy float vectors to feed as target weights.
      bucket_id: which bucket of the model to use.
      forward_only: whether to do the backward step or only forward.

    Returns:
      A triple consisting of gradient norm (or None if we did not do backward),
      average perplexity, and the outputs.

    Raises:
      ValueError: if length of encoder_inputs, decoder_inputs, or
        target_weights disagrees with bucket size for the specified bucket_id.
    """
    # Check if the sizes match.
    encoder_size, decoder_size = self.buckets[bucket_id]

    if len(encoder_inputs) != encoder_size:
      raise ValueError("Encoder length must be equal to the one in bucket,"
                       " %d != %d." % (len(encoder_inputs), encoder_size))
    if len(decoder_inputs) != decoder_size:
      raise ValueError("Decoder length must be equal to the one in bucket,"
                       " %d != %d." % (len(decoder_inputs), decoder_size))
    if len(target_weights) != decoder_size:
      raise ValueError("Weights length must be equal to the one in bucket,"
                       " %d != %d." % (len(target_weights), decoder_size))
    # print('encoder_inputs:')
    # print(encoder_inputs)

    # Input feed: encoder inputs, decoder inputs, target_weights, as provided.
    input_feed = {}
    for l in xrange(encoder_size):
      input_feed[self.encoder_inputs[l].name] = encoder_inputs[l]

    for l in xrange(decoder_size):
      input_feed[self.decoder_inputs[l].name] = decoder_inputs[l]
      input_feed[self.target_weights[l].name] = target_weights[l]

    # Since our targets are decoder inputs shifted by one, we need one more.
    last_target = self.decoder_inputs[decoder_size].name
    input_feed[last_target] = np.zeros([self.batch_size], dtype=np.int32)

    # Output feed: depends on whether we do a backward step or not.
    if not forward_only:
      output_feed = [self.updates[bucket_id],  # Update Op that does SGD.
                     self.gradient_norms[bucket_id],  # Gradient norm.
                     self.losses[bucket_id],
                     self.summary_op]  # Loss for this batch.
    else:
      output_feed = [self.losses[bucket_id]]  # Loss for this batch.
      for l in xrange(decoder_size):  # Output logits.
        output_feed.append(self.outputs[bucket_id][l])

    outputs = session.run(output_feed, input_feed)
    if not forward_only:
      return outputs[1], outputs[2], None, outputs[3]  # Gradient norm, loss, no outputs.
    else:
      return None, outputs[0], outputs[1:]  # No gradient norm, loss, outputs.

  def get_batch(self, data, bucket_id):
    """Get a random batch of data from the specified bucket, prepare for step.

    To feed data in step(..) it must be a list of batch-major vectors, while
    data here contains single length-major cases. So the main logic of this
    function is to re-index data cases to be in the proper format for feeding.

    Args:
      data: a tuple of size len(self.buckets) in which each element contains
        lists of pairs of input and output data that we use to create a batch.
      bucket_id: integer, which bucket to get the batch for.

    Returns:
      The triple (encoder_inputs, decoder_inputs, target_weights) for
      the constructed batch that has the proper format to call step(...) later.
    """
    encoder_size, decoder_size = self.buckets[bucket_id]
    encoder_inputs, decoder_inputs = [], []

    # Get a random batch of encoder and decoder inputs from data,
    # pad them if needed, reverse encoder inputs and add GO to decoder.
    for _ in xrange(self.batch_size):
      encoder_input, decoder_input = random.choice(data[bucket_id])

      # Encoder inputs are padded and then reversed.
      encoder_pad = [preprocess.PAD_ID] * (encoder_size - len(encoder_input))
      encoder_inputs.append(list(reversed(encoder_input + encoder_pad)))

      # Decoder inputs get an extra "GO" symbol, and are padded then.
      decoder_pad_size = decoder_size - len(decoder_input) - 1
      decoder_inputs.append([preprocess.GO_ID] + decoder_input +
                            [preprocess.PAD_ID] * decoder_pad_size)

    # Now we create batch-major vectors from the data selected above.
    batch_encoder_inputs, batch_decoder_inputs, batch_weights = [], [], []

    # Batch encoder inputs are just re-indexed encoder_inputs.
    for length_idx in xrange(encoder_size):
      batch_encoder_inputs.append(
          np.array([encoder_inputs[batch_idx][length_idx]
                    for batch_idx in xrange(self.batch_size)], dtype=np.int32))

    # Batch decoder inputs are re-indexed decoder_inputs, we create weights.
    for length_idx in xrange(decoder_size):
      batch_decoder_inputs.append(
          np.array([decoder_inputs[batch_idx][length_idx]
                    for batch_idx in xrange(self.batch_size)], dtype=np.int32))

      # Create target_weights to be 0 for targets that are padding.
      batch_weight = np.ones(self.batch_size, dtype=np.float32)
      for batch_idx in xrange(self.batch_size):
        # We set weight to 0 if the corresponding target is a PAD symbol.
        # The corresponding target is decoder_input shifted by 1 forward.
        if length_idx < decoder_size - 1:
          target = decoder_inputs[batch_idx][length_idx + 1]
        if length_idx == decoder_size - 1 or target == preprocess.PAD_ID:
          batch_weight[batch_idx] = 0.0
      batch_weights.append(batch_weight)
    return batch_encoder_inputs, batch_decoder_inputs, batch_weights