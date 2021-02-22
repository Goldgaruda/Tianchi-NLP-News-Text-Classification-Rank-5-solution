#!/usr/bin/env python
# coding: utf-8

import numpy as np
import pandas as pd
import sys
import joblib

from sklearn.model_selection import KFold,StratifiedKFold,train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import f1_score
from lightgbm import LGBMClassifier
from scipy import sparse

from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = 'all'

train_df = pd.read_csv('../input/train_set.csv', sep='\t')
test_df = pd.read_csv('../input/test_a.csv', sep='\t', nrows=None)

# #load model to clf
# clf = joblib.load('../model/my_LGBMC_model.pkl')


# #load pre-stored test_word_features
train_tfidf = sparse.load_npz('../data/train_word_features_stpw.npz')
test_tfidf = sparse.load_npz('../data/test_word_features_stpw.npz')



skf = StratifiedKFold(n_splits=5, random_state=16)

# 存储测试集预测结果 行数：len(X_test) ,列数：14列 类别
test_pred = np.zeros((test_tfidf.shape[0], 14), dtype=np.float32)  


clf = LGBMClassifier(n_jobs=-1, min_child_samples=21, max_depth=100, subsample=0.7217, colsample_bytree=0.6, reg_alpha=0.001, reg_lambda=0.5, num_leaves=67, learning_rate=0.088, n_estimators=2400)
# 存储测试集预测结果 行数：len(X_test) ,列数：1列

for idx, (train_index, valid_index) in enumerate(skf.split(train_tfidf, train_df['label'].values)):
    print('第', idx+1, '折交叉验证开始...')
    
    # 训练集划分
    x_train_, x_valid_ =train_tfidf[train_index], train_tfidf[valid_index]
    y_train_, y_valid_ = train_df['label'].values[train_index], train_df['label'].values[valid_index]
    
    # 模型构建
    clf.fit(x_train_, y_train_, eval_set=(x_valid_, y_valid_)) #early_stopping_rounds=60
    
    # 模型预测
    val_pred = clf.predict(x_valid_)
    print("准确率为：",f1_score(y_valid_, val_pred, average='macro'))
    
    # 保存测试集预测结果
    test_pred += clf.predict_proba(test_tfidf) #这里预测的是概率
    
    #保存模型
    joblib.dump(clf, f'../model/Gai1_LGBMC_model_0809_{idx}.pkl', compress=3)


df = pd.DataFrame()
df['label'] = test_pred.argmax(1)
df.to_csv('../output/0809_submit_1.csv', index=None)



# df = pd.DataFrame()
# df['label'] = clf.predict(test_tfidf)
# df.to_csv('submit_cv.csv', index=None)

# submission = pd.read_csv('../data/test_a_sample_submit.csv')
# submission['label'] = preds
# submission.to_csv('../output/LGBMClassifier_submission.csv', index=False)





