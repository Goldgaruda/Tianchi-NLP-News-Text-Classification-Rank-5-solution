#!/usr/bin/env python
# coding: utf-8

# get_ipython().run_line_magic('pylab', 'inline')
# get_ipython().run_line_magic('matplotlib', 'inline')
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import RidgeClassifier, LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMClassifier
import pandas as pd
# from multiprocessing import Pool, cpu_count

# from IPython.core.interactiveshell import InteractiveShell
# InteractiveShell.ast_node_interactivity = 'all'


# ![image.png](attachment:image.png)
# 
# scikit-learning是Python最通用的机器学习库：
# 
# - 官网：https://scikit-learn.org/stable/index.html
# - 十分钟入门：https://scikit-learn.org/stable/getting_started.html
# - 入门教程：https://scikit-learn.org/stable/tutorial/index.html
# - 用户指南：https://scikit-learn.org/stable/user_guide.html#
# - 专业术语表：https://scikit-learn.org/stable/glossary.html
# - API文档：https://scikit-learn.org/stable/modules/classes.html
# 
# LightGBM：https://lightgbm.readthedocs.io/en/latest/Python-API.html

# In[14]:


# print("CPU count:", cpu_count())


# In[8]:


#train_df = pd.read_csv('../input/train_set.csv', sep='\t', nrows=1000)
train_df = pd.read_csv('../input/train_set.csv', sep='\t')
test_df = pd.read_csv('../input/test_a.csv', sep='\t', nrows=None)


# In[9]:


# train_df.shape


# In[10]:


# train_df.head()


# In[16]:


# train_df['text_len'] = train_df['text'].apply(lambda x: len(x.split(' ')))


# In[17]:


# train_df


# In[18]:


# print(train_df['text_len'].describe())


# In[ ]:


# tfidf = TfidfVectorizer(ngram_range=(1, 3), max_features=10000).fit(train_df['text'].iloc[:].values)
# train_tfidf = tfidf.transform(train_df['text'].iloc[:].values)
# test_tfidf = tfidf.transform(test_df['text'].iloc[:].values)
#
# # Choose one from three methods to classify news
# # clf = RidgeClassifier()
# # clf = LogisticRegression()
# clf = LGBMClassifier()
# clf.fit(train_tfidf, train_df['label'].iloc[:].values)
#
# df = pd.DataFrame()
# df['label'] = clf.predict(test_tfidf)
# df.to_csv('submit.csv', index=None)

tfidf = TfidfVectorizer(ngram_range=(1, 4), max_features=4000).fit(train_df['text'].iloc[:].values)
train_tfidf = tfidf.transform(train_df['text'].iloc[:].values)
test_tfidf = tfidf.transform(test_df['text'].iloc[:].values)

# clf = RidgeClassifier()
#clf = LogisticRegression()
#clf = LGBMClassifier()
clf = RandomForestClassifier(n_estimators=1000, max_depth=None, max_features = 0.3, 
                             min_samples_split=2, random_state=32,min_samples_leaf = 1, verbose = 1, n_jobs=-1)

clf.fit(train_tfidf, train_df['label'].iloc[:].values)

df = pd.DataFrame()
df['label'] = clf.predict(test_tfidf)
df.to_csv('submit2.csv', index=None)

