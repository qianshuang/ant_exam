# -*- coding: utf-8 -*-

from data.cnews_loader import *
from wide_deep_model import *
from sklearn import metrics
import os

import time
from datetime import timedelta

base_dir = 'data/cnews'
train_dir = os.path.join(base_dir, 'cnews.balance_train.txt')
test_dir = os.path.join(base_dir, 'cnews.test.txt')
val_dir = os.path.join(base_dir, 'cnews.balance_test.txt')

save_dir = 'checkpoints/textcnn'
save_path = os.path.join(save_dir, 'best_validation')  # 最佳验证结果保存路径


def get_time_dif(start_time):
  """获取已使用时间"""
  end_time = time.time()
  time_dif = end_time - start_time
  return timedelta(seconds=int(round(time_dif)))


def feed_data(feat_indexes_batch, feat_values_batch, numeric_values_batch, y_batch, keep_prob):
  feed_dict = {
    model.feat_index: feat_indexes_batch,
    model.feat_value: feat_values_batch,
    model.numeric_value: numeric_values_batch,
    model.label: y_batch,
    model.keep_prob: keep_prob,
  }
  return feed_dict


def evaluate(sess, x1, x2, x3, y_):
  """评估在某一数据上的准确率和损失"""
  data_len = len(x1)
  batch_eval = batch_iter_wd(x1, x2, x3, y_, 64)
  total_loss = 0.0
  total_acc = 0.0
  for feat_indexes_batch, feat_values_batch, numeric_values_batch, y_batch in batch_eval:
    batch_len = len(feat_indexes_batch)
    feed_dict = feed_data(feat_indexes_batch, feat_values_batch, numeric_values_batch, y_batch, 1.0)
    loss, acc = sess.run([model.loss, model.acc], feed_dict=feed_dict)
    total_loss += loss * batch_len
    total_acc += acc * batch_len
  return total_loss / data_len, total_acc / data_len


def train():
  print("Configuring Saver...")
  # 配置 Saver
  saver = tf.train.Saver()
  if not os.path.exists(save_dir):
    os.makedirs(save_dir)

  # 载入训练集与验证集
  print("Loading training data...")
  feat_indexes_train, feat_values_train, numeric_values_train, y_train = process_wd_file(train_dir, word_to_id)
  feat_indexes_val, feat_values_val, numeric_values_val, y_val = process_wd_file(val_dir, word_to_id)

  # 创建session
  session = tf.Session()
  session.run(tf.global_variables_initializer())

  print('Training and evaluating...')
  start_time = time.time()
  total_batch = 0  # 总批次
  best_acc_val = 0.0  # 最佳验证集准确率
  last_improved = 0  # 记录上一次提升批次
  require_improvement = 2000  # 如果超过1000轮未提升，提前结束训练

  flag = False
  for epoch in range(config.num_epochs):
    print('Epoch:', epoch + 1)
    batch_train = batch_iter_wd(feat_indexes_train, feat_values_train, numeric_values_train, y_train, 128)
    for feat_indexes_batch, feat_values_batch, numeric_values_batch, y_batch in batch_train:
      feed_dict = feed_data(feat_indexes_batch, feat_values_batch, numeric_values_batch, y_batch,
                            config.dropout_keep_prob)

      if total_batch % config.print_per_batch == 0:
        # 每多少轮次输出在训练集和验证集上的性能
        feed_dict[model.keep_prob] = 1.0
        loss_train, acc_train = session.run([model.loss, model.acc], feed_dict=feed_dict)
        loss_val, acc_val = evaluate(session, feat_indexes_val, feat_values_val, numeric_values_val, y_val)

        if acc_val > best_acc_val:
          # 保存最好结果
          best_acc_val = acc_val
          last_improved = total_batch
          saver.save(sess=session, save_path=save_path)
          improved_str = '*'
        else:
          improved_str = ''

        time_dif = get_time_dif(start_time)
        msg = 'Iter: {0:>6}, Train Loss: {1:>6.2}, Train Acc: {2:>7.2%},' \
              + ' Val Loss: {3:>6.2}, Val Acc: {4:>7.2%}, Time: {5} {6}'
        print(msg.format(total_batch, loss_train, acc_train, loss_val, acc_val, time_dif, improved_str))

      feed_dict[model.keep_prob] = config.dropout_keep_prob
      session.run(model.optim, feed_dict=feed_dict)  # 运行优化
      total_batch += 1

      if total_batch - last_improved > require_improvement:
        # 验证集正确率长期不提升，提前结束训练
        print("No optimization for a long time, auto-stopping...")
        flag = True
        break  # 跳出循环
    if flag:  # 同上
      break


def test():
  print("Loading test data...")
  start_time = time.time()
  feat_indexes_test, feat_values_test, numeric_values_test, y_test = process_wd_file(test_dir, word_to_id)

  session = tf.Session()
  session.run(tf.global_variables_initializer())
  saver = tf.train.Saver()
  saver.restore(sess=session, save_path=save_path)  # 读取保存的模型

  print('Testing...')
  loss_test, acc_test = evaluate(session, feat_indexes_test, feat_values_test, numeric_values_test, y_test)
  msg = 'Test Loss: {0:>6.2}, Test Acc: {1:>7.2%}'
  print(msg.format(loss_test, acc_test))

  batch_size = 128
  data_len = len(feat_indexes_test)
  num_batch = int((data_len - 1) / batch_size) + 1

  y_test_cls = np.argmax(y_test, 1)
  y_pred_cls = np.zeros(shape=data_len, dtype=np.int32)  # 保存预测结果
  y_pred_score = np.zeros(shape=data_len, dtype=np.float32)  # 保存预测分数
  for i in range(num_batch):  # 逐批次处理
    start_id = i * batch_size
    end_id = min((i + 1) * batch_size, data_len)
    feed_dict = {
      model.feat_index: feat_indexes_test[start_id:end_id],
      model.feat_value: feat_values_test[start_id:end_id],
      model.numeric_value: numeric_values_test[start_id:end_id],
      model.label: y_test[start_id:end_id],
      model.keep_prob: 1.0,
    }
    pc, ys = session.run([model.y_pred_cls, model.y_sore], feed_dict=feed_dict)
    y_pred_cls[start_id:end_id] = pc
    y_pred_score[start_id:end_id] = ys[:, 1]

  # 评估
  print("Precision, Recall and F1-Score...")
  print(metrics.classification_report(y_test_cls, y_pred_cls, target_names=['0', '1']))

  # 混淆矩阵
  print("Confusion Matrix...")
  cm = metrics.confusion_matrix(y_test_cls, y_pred_cls)
  print(cm)

  # AUC
  # test_target = list(map(float, test_target))
  fpr, tpr, thresholds = metrics.roc_curve(y_test_cls, y_pred_score)
  print("\nAUC...")
  print(metrics.auc(fpr, tpr))

  time_dif = get_time_dif(start_time)
  print("\nTime usage:", time_dif)


if __name__ == '__main__':
  print('Configuring wide & deep model...')
  config = TCNNConfig()
  # 构建embedding字典
  words = ['0_0', '0_1', '1', '2', '3', '4', '5', '6', '7', '8', '9_0', '9_1', '10_0', '10_1', '11', '12']
  word_to_id = dict(zip(words, range(len(words))))

  config.cate_feature_size = len(words)
  config.field_size = 13
  model = TextCNN(config)
  train()
  test()
