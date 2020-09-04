# -*- coding: utf-8 -*-

import pickle
import os
from sklearn import metrics, svm, neural_network, linear_model, naive_bayes, neighbors, tree, ensemble

from data.cnews_loader import *

base_dir = 'data/cnews'
train_dir = os.path.join(base_dir, 'cnews.balance_train.txt')
balance_all_dir = os.path.join(base_dir, 'cnews.balance_all.txt')
test_dir = os.path.join(base_dir, 'cnews.test.txt')
all_dir = os.path.join(base_dir, 'cnews.all.txt')
pred_dir = os.path.join(base_dir, 'cnews.pred.txt')


def train():
  # 处理训练数据，如果矩阵过大，可以采用Python scipy库中对稀疏矩阵的优化算法：scipy.sparse.csr_matrix((dd, (row, col)), )
  train_feature, train_target = process_cl_file(balance_all_dir)

  # 模型训练
  print("start training...")
  model.fit(train_feature, train_target)

  # 模型的重要特征
  print("feature importance...")
  n = model.coef_
  print(list(np.array(n).reshape(-1)))

  # 模型导出
  # f = open(os.path.join('model', 'exam.pickle'), 'wb')
  # pickle.dump(model, f, True)


def test():
  print("start testing...")
  # 处理测试数据
  test_feature, test_target = process_cl_file(test_dir)
  test_predict = model.predict(test_feature)  # 返回预测类别
  test_predict_proba = model.predict_proba(test_feature)  # 返回属于各个类别的概率
  test_predict_true_proba = test_predict_proba[:, 1]  # true probability

  # accuracy
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
  test_target = list(map(float, test_target))
  fpr, tpr, thresholds = metrics.roc_curve(test_target, test_predict_true_proba)
  print("\nAUC...")
  print(metrics.auc(fpr, tpr))


def predict():
  print("\nstart predicting...")
  pred_feature = process_pred_file(pred_dir)
  result_txt = []

  # load model
  # f_l = open(os.path.join('model', 'MLP.pickle'), 'rb')
  # model = pickle.load(f_l)

  results = model.predict_proba(pred_feature)
  for result in results:
    result = list(result)
    max_index = result.index(max(result))
    result_txt.append(str(max_index) + '\t' + str(result[1]))
    open_file(os.path.join('data', 'result.txt'), mode='w').write('\n'.join(result_txt) + '\n')


# random forest
# model = ensemble.RandomForestClassifier()
# logistic regression
model = linear_model.LogisticRegression(penalty='l1')  # ovr
# SVM
# model = svm.LinearSVC()  # 线性，无概率结果
# model = svm.SVC(probability=True)  # 核函数，训练慢
# MLP
# model = neural_network.MLPClassifier(hidden_layer_sizes=(512, 128), max_iter=1000, verbose=True, early_stopping=True)  # 注意max_iter是epoch数

train()
test()
predict()
