# -*- coding: utf-8 -*-

import os
from scipy import stats
import random
from data.cnews_loader import *

base_dir = 'data/cnews'
all_dir = os.path.join(base_dir, 'cnews.all.txt')
balance_train_dir = os.path.join(base_dir, 'cnews.balance_train.txt')
train_dir = os.path.join(base_dir, 'cnews.train.txt')
test_dir = os.path.join(base_dir, 'cnews.test.txt')
balance_test_dir = os.path.join(base_dir, 'cnews.balance_test.txt')


# 将原始数据按照9:1拆分为训练集与测试集
def split_data():
  with open_file(all_dir) as f:
    lines = f.readlines()
  random.shuffle(lines)
  len_test = int(len(lines) * 0.1)
  lines_test = lines[0:len_test]
  lines_train = lines[len_test:]
  train_w = open_file(train_dir, mode='w')
  test_w = open_file(test_dir, mode='w')
  for i in lines_train:
    train_w.write(i)
  for j in lines_test:
    test_w.write(j)


# 随机正采样
def balance_sample():
  v0 = []
  v = []
  with open_file(test_dir) as f:
    for line in f:
      cols = line.strip().split('\t')
      if cols[0] == '1':
        v.append(line)
      else:
        v0.append(line)

  v = list(set(v))
  l = len(v)
  # 补足样本数
  v = np.array(v)
  cnt = len(v0)
  v = v.repeat(int(cnt / l + 1))
  v = random.sample(list(v), cnt)
  v_all = v + v0

  train_w = open_file(balance_test_dir, mode='w')
  for i in v_all:
    train_w.write(i)


# balance_sample()
# split_data()

from sklearn.preprocessing import StandardScaler

a = [1,2,3,4]
a = np.array(a).reshape(-1, 1)
print(a)
print()
print(StandardScaler().fit_transform(a).reshape(-1))
