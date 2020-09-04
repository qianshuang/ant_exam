# -*- coding: utf-8 -*-

import tensorflow as tf


class TCNNConfig(object):
  embedding_dim = 128

  num_classes = 2  # 类别数

  dropout_keep_prob = 0.5  # dropout保留比例
  learning_rate = 1e-3  # 学习率

  batch_size = 64  # 每批训练大小
  num_epochs = 100  # 总迭代轮次

  print_per_batch = 100  # 每多少轮输出一次结果


class TextCNN(object):
  """文本分类，CNN模型"""

  def __init__(self, config):
    self.config = config

    # 三个待输入的数据
    self.feat_index = tf.placeholder(tf.int32, shape=[None, None], name='feat_index')
    self.feat_value = tf.placeholder(tf.float32, shape=[None, None], name='feat_value')
    self.numeric_value = tf.placeholder(tf.float32, [None, None], name='num_value')
    self.label = tf.placeholder(tf.float32, shape=[None, self.config.num_classes], name='label')
    self.keep_prob = tf.placeholder_with_default(1.0, shape=())

    self.cnn()

  def cnn(self):
    embedding = tf.get_variable('embedding', [self.config.cate_feature_size, self.config.embedding_dim])
    embedding_inputs = tf.nn.embedding_lookup(embedding, self.feat_index)

    feat_value = tf.reshape(self.feat_value, shape=[-1, self.config.field_size, 1])
    embedding_inputs = tf.multiply(embedding_inputs, feat_value)

    # 卷积
    filter_w = tf.Variable(tf.truncated_normal([3, self.config.embedding_dim, 128], stddev=0.1))
    conv = tf.nn.conv1d(embedding_inputs, filter_w, 1, padding='SAME')

    filter_w_1 = tf.Variable(tf.truncated_normal([3, 128, 256], stddev=0.1))
    conv_1 = tf.nn.conv1d(conv, filter_w_1, 1, padding='SAME')

    gmp = tf.reduce_max(conv_1, reduction_indices=[1], name='gmp')
    gmp = tf.concat([self.numeric_value, gmp], axis = 1)

    with tf.name_scope("score"):
      # 全连接层，后面接dropout以及relu激活
      W_fc1 = tf.Variable(tf.truncated_normal([256 + 16, 1024], stddev=0.1))
      b_fc1 = tf.Variable(tf.constant(0.1, shape=[1024]))
      h_fc1 = tf.nn.relu(tf.matmul(gmp, W_fc1) + b_fc1)
      h_fc1_drop = tf.nn.dropout(h_fc1, self.keep_prob)

      # 分类器
      W_fc2 = tf.Variable(tf.truncated_normal([1024, self.config.num_classes], stddev=0.1))
      b_fc2 = tf.Variable(tf.constant(0.1, shape=[self.config.num_classes]))
      y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2
      self.y_sore = tf.nn.softmax(y_conv)
      self.y_pred_cls = tf.argmax(y_conv, 1)  # 预测类别

    with tf.name_scope("optimize"):
      self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.label, logits=y_conv))
      self.optim = tf.train.AdamOptimizer(learning_rate=self.config.learning_rate).minimize(self.loss)

    with tf.name_scope("accuracy"):
    # 准确率
      correct_pred = tf.equal(tf.argmax(self.label, 1), self.y_pred_cls)
      self.acc = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
