#!/usr/bin/env python
# coding: utf-8

# # 1-1 Load data
# ## Import libraries

# In[1]:


get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')

from import_file import*


# ## Load dataset

# * Column 1: the ID of the statement ([ID].json).
# * Column 2: the label.
# * Column 3: the statement.
# * Column 4: the subject(s).
# * Column 5: the speaker.
# * Column 6: the speaker's job title.
# * Column 7: the state info.
# * Column 8: the party affiliation.
# * Columns 9-13: the total credit history count, including the current statement.
#     * 9: barely true counts.
#     * 10: false counts.
#     * 11: half true counts.
#     * 12: mostly true counts.
#     * 13: pants on fire counts.
# * Column 14: the context (venue / location of the speech or statement).
# * Column 15: the extracted justification

# In[2]:


# Directory of datasets
uri_train = Path.cwd() / "liar_dataset" / "train2.tsv" 
uri_test = Path.cwd() / "liar_dataset" / "test2.tsv" 
uri_valid = Path.cwd() / "liar_dataset" / "valid2.tsv" 


# In[3]:


# Importing the datasets with column names
df_train = pd.read_csv(uri_train, delimiter='\t', quoting=3, header=None, names=["json ID", "label", "statement", "subject", "speaker", "job title", "state", "party",
                         "barely true", "false", "half true", "mostly true", "pants on fire", "context", "justification"])
df_test = pd.read_csv(uri_test, delimiter='\t', quoting=3, header=None, names=["json ID", "label", "statement", "subject", "speaker", "job title", "state", "party",
                         "barely true", "false", "half true", "mostly true", "pants on fire", "context", "justification"])
df_valid = pd.read_csv(uri_valid, delimiter='\t', quoting=3, header=None, names=["json ID", "label", "statement", "subject", "speaker", "job title", "state", "party",
                         "barely true", "false", "half true", "mostly true", "pants on fire", "context", "justification"])


# In[4]:


# Print the shape of files
print(df_train.shape)
print(df_test.shape)
print(df_valid.shape)


# In[5]:


# Concatenate the train, test and validation files
df_raw = pd.concat([df_train, df_test, df_valid], axis=0, sort=False)

# Print the shape of df_raw
df_raw.shape


# ## Check for missing labels

# In[6]:


n_missing = df_raw['label'].isnull().sum()
print('There are {} missing labels in LIAR-PLUS dataset.'.format(n_missing))


# In[7]:


df_raw_missing = df_raw.loc[df_raw['label'].isnull(), : ]
print(df_raw_missing.shape)


# In[8]:


# Show some of the rows with missing labels
print(df_raw_missing.head())


# ## Remove missing labels (if exists)

# In[9]:


df_raw = df_raw.loc[df_raw['label'].notna(), :]


# In[10]:


# Print the shape of the dataset after removing missing labels
df_raw.shape


# ## Save dataframe

# In[11]:


path = Path.cwd() / "processed_data" / "df_raw.pkl"
df_raw.to_pickle(path)

