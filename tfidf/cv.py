#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import f1_score
from lightgbm import LGBMClassifier

# from IPython.core.interactiveshell import InteractiveShell
# InteractiveShell.ast_node_interactivity = 'all'


# In[16]:


train_df = pd.read_csv('../input/train_set.csv', sep='\t')
test_df = pd.read_csv('../input/test_a.csv', sep='\t', nrows=None)


# In[17]:


tfidf = TfidfVectorizer(
    sublinear_tf=True,
    strip_accents='unicode',
    analyzer='word',
    token_pattern=r'\w{1,}',
    stop_words='english',
    ngram_range=(1,3),
    max_features=10000)


# In[20]:


# train_df['text'].shape
# test_df['text'].shape


# In[21]:


train_df.head()


# In[23]:


# train_df['text'].iloc[:].values.shape


# In[24]:


#特征合起来  
#tfidf.fit(pd.concat(train_df['text'],test_df['text']))
#拼接数组需要() You need to pass the arrays as an iterable (a tuple or list), thus the correct syntax is
tfidf.fit(np.concatenate((train_df['text'].iloc[:].values,test_df['text'].iloc[:].values),axis=0))


# In[25]:


train_word_features = tfidf.transform(train_df['text'].iloc[:].values)
test_word_features = tfidf.transform(test_df['text'].iloc[:].values)


# In[ ]:


X_train = train_word_features
y_train = train_df['label']
X_test = test_word_features

KF = KFold(n_splits=5, random_state=7) 

clf = LGBMClassifier(n_jobs=-1, feature_fraction=0.7, bagging_fraction=0.4, lambda_l1=0.001, lambda_l2=0.01, n_estimators=600)

# 存储测试集预测结果 行数：len(X_test) ,列数：1列
test_pred = np.zeros((X_test.shape[0], 1), int)  
for KF_index, (train_index,valid_index) in enumerate(KF.split(X_train)):
    print('第', KF_index+1, '折交叉验证开始...')
    # 训练集划分
    x_train_, x_valid_ = X_train[train_index], X_train[valid_index]
    y_train_, y_valid_ = y_train[train_index], y_train[valid_index]
    # 模型构建
    clf.fit(x_train_, y_train_)
    # 模型预测
    val_pred = clf.predict(x_valid_)
    print("准确率为：",f1_score(y_valid_, val_pred, average='macro'))
    
    # 保存测试集预测结果
    test_pred = np.column_stack((test_pred, clf.predict(X_test)))  # 将矩阵按列合并

# 取测试集中预测数量最多的数
preds = []
for i, test_list in enumerate(test_pred):
    preds.append(np.argmax(np.bincount(test_list)))
preds = np.array(preds)


# In[ ]:


# df = pd.DataFrame()
# df['label'] = clf.predict(test_tfidf)
# df.to_csv('submit_cv.csv', index=None)

submission = pd.read_csv('../data/test_a_sample_submit.csv')
submission['label'] = preds
submission.to_csv('../output/LGBMClassifier_submission.csv', index=False)


# In[ ]:


import joblib
joblib.dump(clf, '../model/my_LGBMC_model_5cv.pkl', compress=3)


# In[ ]:




