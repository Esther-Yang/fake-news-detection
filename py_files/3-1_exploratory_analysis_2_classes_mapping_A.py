#!/usr/bin/env python
# coding: utf-8

# # 3-1 Exploratory Analysis (2 classes Mapping A)
# 

# In[1]:


get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')

from import_file import*


# In[2]:


path = Path.cwd() / "processed_data" / "df_raw.pkl"
df_raw = pd.read_pickle(path)


# ## Convert 6 classes to 2 classses
# `df_bi_A` 
#     
#     'true','mostly-true', 'half-true' : Truth
#     
#     'barely-true', 'false', 'pants-fire' : Fake

# In[3]:


df_bi_A = pd.DataFrame(df_raw)


# In[4]:


map_label_A = {
        'true': 'Truth',
        'mostly-true': 'Truth',
        'half-true': 'Truth',
        'barely-true': 'Fake',
        'false': 'Fake',
        'pants-fire': 'Fake'
    }
df_bi_A['label'] = df_bi_A['label'].replace(map_label_A)


# ## Distribution

# In[5]:


df_bi_A.shape


# In[6]:


df_bi_A.head()


# In[7]:


sns.countplot(x='label', data = df_bi_A, palette = 'hls')


# ## Save dataframe

# In[8]:


path = Path.cwd() / "processed_data" / "df_bi_A.pkl"
df_bi_A.to_pickle(path)

