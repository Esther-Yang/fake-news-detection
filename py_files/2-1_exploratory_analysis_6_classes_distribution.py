#!/usr/bin/env python
# coding: utf-8

# # 2-1 Exploratory Analysis - 6 classes - distribution
# 

# In[1]:


get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')

from import_file import*


# In[2]:


path = Path.cwd() / "processed_data" / "df_raw.pkl"
df_raw = pd.read_pickle(path)


# ## Distribution of LIAR-plus dataset (original 6 classes)

# In[3]:


sns.countplot(x='label', data = df_raw, palette = 'hls')

