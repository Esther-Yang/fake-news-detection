#!/usr/bin/env python
# coding: utf-8

# # 2-3 Exploratory analysis - 6 classes - statements

# In[1]:


get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')

from import_file import*


# In[2]:


path = Path.cwd() / "processed_data" / "df_raw.pkl"
df_raw = pd.read_pickle(path)


# ## Function to generate wordcloud

# In[3]:


def generate_wordcloud(df, labeltype):
    wordcloud = WordCloud(width=3000, height=2000, background_color='white',stopwords=STOPWORDS).generate(str(df))
    plt.imshow(wordcloud)
    plt.axis('off')
    plt.title("Word Cloud of %s Statements in LIAR-PLUS dataset" %labeltype)
    plt.show()
    


# ## Generate dataframes for statements

# In[4]:


# For all statements
df_all = df_raw.loc[:, ['statement']]
print(df_all.shape)
print(df_all.head())


# In[5]:


# For true statements
df_true = df_raw.loc[df_raw['label']=='true',['statement']]
print(df_true.shape)
print(df_true.head())


# In[6]:


# For mostly-true statments
df_mostlytrue = df_raw.loc[df_raw['label']=='mostly-true',['statement']]
print(df_mostlytrue.shape)
print(df_mostlytrue.head())


# In[7]:


# For half-true statments
df_halftrue = df_raw.loc[df_raw['label']=='half-true',['statement']]
print(df_halftrue.shape)
print(df_halftrue.head())


# In[8]:


# For barely-true statments
df_barelytrue = df_raw.loc[df_raw['label']=='barely-true',['statement']]
print(df_barelytrue.shape)
print(df_barelytrue.head())


# In[9]:


# For false statments
df_false = df_raw.loc[df_raw['label']=='false',['statement']]
print(df_false.shape)
print(df_false.head())


# In[10]:


# For pants-fire statments
df_pantsfire = df_raw.loc[df_raw['label']=='pants-fire',['statement']]
print(df_pantsfire.shape)
print(df_pantsfire.head())


# ## Generate wordclouds

# In[11]:


generate_wordcloud(df_all, 'all')

generate_wordcloud(df_true, 'true')
generate_wordcloud(df_mostlytrue, 'mostly-true')
generate_wordcloud(df_halftrue, 'half-true')
generate_wordcloud(df_barelytrue, 'barely-true')
generate_wordcloud(df_false, 'false')
generate_wordcloud(df_pantsfire, 'pants-fire')

