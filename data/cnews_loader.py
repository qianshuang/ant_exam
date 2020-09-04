# -*- coding: utf-8 -*-

import re
import numpy as np
from sklearn.preprocessing import StandardScaler


def open_file(filename, mode='r'):
  return open(filename, mode, encoding='utf-8', errors='ignore')


def read_file(filename):
  """读取文件数据"""
  contents, labels = [], []
  with open_file(filename) as f:
    for line in f:
      try:
        line = line.replace("NA", "0")
        cols = line.strip().split('\t')
        contents.append(cols[1:])
        labels.append(cols[0])
      except:
        print(line)
  return contents, labels


def read_pred_file(filename):
  """读取文件数据"""
  contents = []
  with open_file(filename) as f:
    for line in f:
      try:
        line = line.replace("NA", "-1")
        contents.append(line.strip().split('\t'))
      except:
        print(line)
  return contents


def process_file(filename):
  contents, labels = read_file(filename)
  t_0 = []
  t_1 = []
  for i in range(len(contents)):
    if labels[i] == '0':
      t_0.append(contents[i])
    else:
      t_1.append(contents[i])
  return t_0, t_1


def to_categorical(y, num_classes=None):
  """Converts a class vector (integers) to binary class matrix.

  E.g. for use with categorical_crossentropy.

  Arguments:
      y: class vector to be converted into a matrix
          (integers from 0 to num_classes).
      num_classes: total number of classes.

  Returns:
      A binary matrix representation of the input.
  """
  y = np.array(y, dtype='int').ravel()
  if not num_classes:
    num_classes = np.max(y) + 1
  n = y.shape[0]
  categorical = np.zeros((n, num_classes))
  categorical[np.arange(n), y] = 1
  return categorical


def process_wd_file(filename, word_to_id):
  contents, labels = read_file(filename)

  feat_indexes = []
  feat_values = []
  numeric_values = []

  for content in contents:
    feat_index = []
    feat_value = []
    numeric_value = []
    for i in range(len(content)):
      if i == 0 or i == 9 or i == 10:
        feat_index.append(word_to_id[str(i) + '_' + content[i]])
        feat_value.append(1)
        if content[i] == '0':
          numeric_value = numeric_value + [0, 1]
        else:
          numeric_value = numeric_value + [1, 0]
      else:
        feat_index.append(word_to_id[str(i)])
        feat_value.append(content[i])
        numeric_value.append(content[i])

    feat_indexes.append(feat_index)
    feat_values.append(feat_value)
    numeric_values.append(numeric_value)

  labels = to_categorical(list(map(int, labels)), 2)  # 将标签转换为one-hot表示
  return feat_indexes, feat_values, numeric_values, labels


def read_cf_file(filename):
  """读取文件数据"""
  contents, labels = [], []
  with open_file(filename) as f:
    for line in f:
      cols = line.strip().split('\t')
      conti_cols = [cols[1], cols[10], cols[11]]

      del cols[1]
      del cols[1]  # 删除缺失值过多的列
      del cols[8]
      del cols[8]

      # 离散特征one-hot
      for i in conti_cols:
        if i == '0':
          cols = cols + [0, 1]
        else:
          cols = cols + [1, 0]

      contents.append(cols[1:])
      labels.append(cols[0])
  return contents, labels


def process_cl_file(filename):
  contents, labels = read_cf_file(filename)
  contents = np.array(contents)
  # 归一化
  stand_conts = []
  col_size = len(contents[0]) - 6
  for i in range(col_size):
    stand_conts.append(StandardScaler().fit_transform(contents[:, i].reshape(-1, 1)).reshape(-1))

  return np.concatenate((np.array(stand_conts).T, contents[:, col_size:]), axis=1).astype(float), np.array(labels)


def process_pred_file(filename):
  contents = read_pred_file(filename)
  return np.array(contents).astype(float)


def batch_iter(x1, batch_size=200):
  """生成批次数据"""
  data_len = len(x1)
  num_batch = int((data_len - 1) / batch_size) + 1

  indices = np.random.permutation(np.arange(data_len))
  x1_shuffle = []
  for i in range(len(indices)):
    x1_shuffle.append(x1[indices[i]])

  for i in range(num_batch):
    start_id = i * batch_size
    end_id = min((i + 1) * batch_size, data_len)
    yield x1_shuffle[start_id:end_id]


def batch_iter_wd(x1, x2, x3, y, batch_size=128):
  data_len = len(x1)
  num_batch = int((data_len - 1) / batch_size) + 1

  indices = np.random.permutation(np.arange(data_len))
  x1_shuffle = []
  x2_shuffle = []
  x3_shuffle = []
  y_shuffle = []
  for i in range(len(indices)):
    x1_shuffle.append(x1[indices[i]])
    x2_shuffle.append(x2[indices[i]])
    x3_shuffle.append(x3[indices[i]])
    y_shuffle.append(y[indices[i]])

  for i in range(num_batch):
    start_id = i * batch_size
    end_id = min((i + 1) * batch_size, data_len)
    yield x1_shuffle[start_id:end_id], x2_shuffle[start_id:end_id], x3_shuffle[start_id:end_id], y_shuffle[
                                                                                                 start_id:end_id]
