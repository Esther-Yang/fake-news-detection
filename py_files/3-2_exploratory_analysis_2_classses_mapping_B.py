#!/usr/bin/env python
# coding: utf-8

# # 3-2 Exploratory Analysis (2 classes Mapping B)
# 

# In[1]:


get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')

from import_file import*


# In[2]:


path = Path.cwd() / "processed_data" / "df_raw.pkl"
df_raw = pd.read_pickle(path)


# ## Convert 6 classes to 2 classes
# `df_bi_B` 
#     
#     'true','mostly-true' : Truth
#     
#     'half-true', 'barely-true', 'false', 'pants-fire' : Fake

# In[3]:


df_bi_B = pd.DataFrame(df_raw)


# In[4]:


map_label_B = {
        'true': 'Truth',
        'mostly-true': 'Truth',
        'half-true': 'Fake',
        'barely-true': 'Fake',
        'false': 'Fake',
        'pants-fire': 'Fake'
    }
df_bi_B['label'] = df_bi_B['label'].replace(map_label_B)


# ## Distribution

# In[5]:


df_bi_B.shape


# In[6]:


df_bi_B.head()


# In[7]:


sns.countplot(x='label', data = df_bi_B, palette = 'hls')


# ## Save dataframe

# In[8]:


path = Path.cwd() / "processed_data" / "df_bi_B.pkl"
df_bi_B.to_pickle(path)

