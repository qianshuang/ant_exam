# -*- coding: utf-8 -*-

import pickle
import os
from scipy import stats
from sklearn import metrics, svm, neural_network, linear_model, naive_bayes, neighbors, tree, ensemble

from data.cnews_loader import *

base_dir = 'data/cnews'
train_dir = os.path.join(base_dir, 'cnews.train.txt')
test_dir = os.path.join(base_dir, 'cnews.test.txt')
pred_dir = os.path.join(base_dir, 'cnews.pred.txt')


def train():
  train_0, train_1 = process_file(train_dir)
  # 模型训练
  cnt_ = 0
  for x0_batch in batch_iter(train_0):
    cnt_ += 1
    print("start training " + str(cnt_) + "...")
    model = ensemble.RandomForestClassifier()
    # model = linear_model.LogisticRegression()
    model.fit(x0_batch + train_1, ['0'] * len(x0_batch) + ['1'] * len(train_1))
    models.append(model)


def test():
  print("start testing...")
  # 处理测试数据
  test_0, test_1 = process_file(test_dir)

  all_pres = []
  for model in models:
    test_predict = model.predict(np.array(test_0 + test_1).astype(float))  # 返回预测类别
    all_pres.append(test_predict)
    # test_predict_proba = model.predict_proba(test_feature)  # 返回属于各个类别的概率
    # test_predict_true_proba = test_predict_proba[:, 1]  # true probability

  test_predict = []
  all_pres = np.array(all_pres)
  for i in range(len(test_0 + test_1)):
    test_predict.append(str(stats.mode(all_pres[:, i])[0][0]))

  # accuracy
  test_target = ['0'] * len(test_0) + ['1'] * len(test_1)
  true_false = (test_predict == test_target)
  accuracy = np.count_nonzero(true_false) / float(len(test_target))
  print()
  print("accuracy is %f" % accuracy)

  # precision    recall  f1-score
  print()
  print(metrics.classification_report(test_target, test_predict, target_names=['0', '1']))

  # 混淆矩阵
  print("Confusion Matrix...")
  print(metrics.confusion_matrix(test_target, test_predict))

  # AUC
  # test_target = list(map(float, test_target))
  # fpr, tpr, thresholds = metrics.roc_curve(test_target, test_predict_true_proba)
  # print("\nAUC...")
  # print(metrics.auc(fpr, tpr))


models = []
train()
test()
