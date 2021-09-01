#!/usr/bin/env python
# coding: utf-8

# # 4-1 Six classes - train model 

# In[1]:


get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')

from import_file import*


# In[2]:


path = Path.cwd() / "processed_data" / "df_raw.pkl"
df = pd.read_pickle(path)


# In[3]:


# Specifiying features and labels
X = df['statement']
y = df['label']


# In[4]:


# Specifying train and test split with ratio of 80:20
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state = 0, stratify=y)
print(len(X_train), len(X_test))
print(len(y_train), len(y_test))


# ## Defining functions

# In[5]:


stemmer = PorterStemmer()

def tokenizer(input_string):
    words = re.sub(r"[^A-Za-z0-9-]", " ", input_string) # Replace everything except a-zA-Z0-9- by a space
    words = words.lower() # Lower casing
    words = words.split() # Split a string into a list  
    words = [stemmer.stem(word) for word in words] #Stemming
    return words


# In[6]:


# Stopwords list set to english
stopwords_list = stopwords.words('english')


# ## Logistic Regression

# ### Creating `pipe1`

# In[7]:


pipe1 = Pipeline([
    ('vectorizer', CountVectorizer(analyzer='word',tokenizer=tokenizer,stop_words=stopwords_list)),
    ('transformer', TfidfTransformer()),
    ('clf', LogisticRegression(solver='liblinear', random_state=0))
])


# In[8]:


# Fit the model using training set
pipe1.fit(X_train, y_train)


# In[9]:


# Save model
path = Path.cwd() / "saved_model" / "pipe1_LR.pkl"
with path.open('wb') as f:
    pickle.dump(pipe1, f)


# ### Creating `pipe2`
# For normalisation, use binarizer instead of Tfidftransformer, i.e. scale to 0 for not present or 1 for present at any frequency)

# In[10]:


pipe2 = Pipeline([
    ('vectorizer', CountVectorizer(analyzer='word',tokenizer=tokenizer,stop_words=stopwords_list)),
    ('transformer', Binarizer()),
    ('clf', LogisticRegression(solver='liblinear', random_state=0))
])


# In[11]:


# Fit the model using training set
pipe2.fit(X_train, y_train)


# In[12]:


# Save model
path = Path.cwd() / "saved_model" / "pipe2_LR.pkl"
with path.open('wb') as f:
    pickle.dump(pipe2, f)


# ## Multinomial Naive Bayes classifier

# ### Using `pipe1`

# In[13]:


pipe1.set_params(clf=MultinomialNB())

# Fit the model using training set
pipe1.fit(X_train, y_train)


# In[14]:


# Save model
path = Path.cwd() / "saved_model" / "pipe1_MNB.pkl"
with path.open('wb') as f:
    pickle.dump(pipe1, f)


# ### Using `pipe2`

# In[15]:


pipe2.set_params(clf=MultinomialNB())

# Fit the model using training set
pipe2.fit(X_train, y_train)


# In[16]:


# Save model
path = Path.cwd() / "saved_model" / "pipe2_MNB.pkl"
with path.open('wb') as f:
    pickle.dump(pipe2, f)


# ## Support Vector Machine

# ### Using `pipe1`

# In[17]:


pipe1.set_params(clf=svm.SVC(C=1.0, kernel='linear', degree=3, gamma='auto',random_state=0))

# Fit the model using training set
pipe1.fit(X_train, y_train)


# In[18]:


# Save model
path = Path.cwd() / "saved_model" / "pipe1_SVC.pkl"
with path.open('wb') as f:
    pickle.dump(pipe1, f)


# ### Using `pipe2`

# In[19]:


pipe2.set_params(clf=svm.SVC(C=1.0, kernel='linear', degree=3, gamma='auto',random_state=0))

# Fit the model using training set
pipe2.fit(X_train, y_train)


# In[20]:


# Save model
path = Path.cwd() / "saved_model" / "pipe2_SVC.pkl"
with path.open('wb') as f:
    pickle.dump(pipe2, f)


# ## DecisionTree Classifier

# ### Using `pipe1`

# In[21]:


pipe1.set_params(clf=tree.DecisionTreeClassifier(random_state=0))

# Fit the model using training set
pipe1.fit(X_train, y_train)


# In[22]:


# Save model
path = Path.cwd() / "saved_model" / "pipe1_DTC.pkl"
with path.open('wb') as f:
    pickle.dump(pipe1, f)


# ### Using `pipe2`

# In[23]:


pipe2.set_params(clf=tree.DecisionTreeClassifier(random_state=0))

# Fit the model using training set
pipe2.fit(X_train, y_train)


# In[24]:


# Save model
path = Path.cwd() / "saved_model" / "pipe2_DTC.pkl"
with path.open('wb') as f:
    pickle.dump(pipe2, f)


# ## Random Forest Classifier

# ### Using `pipe1`

# In[25]:


pipe1.set_params(clf=RandomForestClassifier(random_state=0))

# Fit the model using training set
pipe1.fit(X_train, y_train)


# In[26]:


# Save model
path = Path.cwd() / "saved_model" / "pipe1_RFC.pkl"
with path.open('wb') as f:
    pickle.dump(pipe1, f)


# ### Using `pipe2`

# In[27]:


pipe2.set_params(clf=RandomForestClassifier(random_state=0))

# Fit the model using training set
pipe2.fit(X_train, y_train)


# In[28]:


# Save model
path = Path.cwd() / "saved_model" / "pipe2_RFC.pkl"
with path.open('wb') as f:
    pickle.dump(pipe2, f)

