#!/usr/bin/env python
# coding: utf-8

# # 4-2 Six class classification - test model

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


# ## Defining Functions

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


# In[7]:


model_dict = {}
model_accuracy = []


# In[8]:


def print_results():
    print("Accuracy: ", accuracy_score(y_test, y_pred))
    model_accuracy.append(accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred))
    print(confusion_matrix(y_test, y_pred))


# In[9]:


def display_confusion_matrix(cm, classes, title='Confusion Matrix'):
    df_cm = pd.DataFrame(cm, index = classes, columns = classes)
    size = (10, 8)
    fig, ax = plt.subplots(figsize = size)
    sns.heatmap(df_cm, annot=True, cmap="Blues", center=0, fmt='g')
    ax.set_ylabel('Actual')    
    ax.set_xlabel('Predicted')
    plt.title(title)


# ## Logistic Regression

# ### Using `pipe1`

# In[10]:


# Make predictions using testing set
path = Path.cwd() / "saved_model" / "pipe1_LR.pkl"
with open(path, 'rb') as f:
    loaded_model = pickle.load(f)
y_pred = loaded_model.predict(X_test)
model_dict[0]='pipe 1 Logistic Regression'
print_results() 


# In[11]:


display_confusion_matrix(confusion_matrix(y_test, y_pred), loaded_model.classes_, 'pipe 1 Logistic Regression')


# ### Using `pipe2`

# In[12]:


# Make predictions using testing set
path = Path.cwd() / "saved_model" / "pipe2_LR.pkl"
with open(path, 'rb') as f:
    loaded_model = pickle.load(f)
y_pred = loaded_model.predict(X_test)
model_dict[1]='pipe 2 Logistic Regression'
print_results() 


# In[13]:


display_confusion_matrix(confusion_matrix(y_test, y_pred), loaded_model.classes_, 'pipe 2 Logistic Regression')


# ## Multinomial Naive Bayes Classifier

# ### Using `pipe1`

# In[14]:


# Make predictions using testing set
path = Path.cwd() / "saved_model" / "pipe1_MNB.pkl"
with open(path, 'rb') as f:
    loaded_model = pickle.load(f)
y_pred = loaded_model.predict(X_test)
model_dict[2]='pipe 1 Multinomial Naive Bayes'
print_results() 


# In[15]:


display_confusion_matrix(confusion_matrix(y_test, y_pred), loaded_model.classes_, 'pipe 1 Multinomial Naive Bayes')


# ### Using `pipe2`

# In[16]:


# Make predictions using testing set
path = Path.cwd() / "saved_model" / "pipe2_MNB.pkl"
with open(path, 'rb') as f:
    loaded_model = pickle.load(f)
y_pred = loaded_model.predict(X_test)
model_dict[3]='pipe 2 Multinomial Naive Bayes'
print_results() 


# In[17]:


display_confusion_matrix(confusion_matrix(y_test, y_pred), loaded_model.classes_, 'pipe 2 Multinomial Naive Bayes')


# ## Support Vector Machine

# ### Using `pipe1`

# In[18]:


# Make predictions using testing set
path = Path.cwd() / "saved_model" / "pipe1_SVC.pkl"
with open(path, 'rb') as f:
    loaded_model = pickle.load(f)
y_pred = loaded_model.predict(X_test)
model_dict[4]='pipe 1 Support Vector Classifier'
print_results() 


# In[19]:


display_confusion_matrix(confusion_matrix(y_test, y_pred), loaded_model.classes_, 'pipe 1 Support Vector Classifier')


# ### Using `pipe2`

# In[20]:


# Make predictions using testing set
path = Path.cwd() / "saved_model" / "pipe2_SVC.pkl"
with open(path, 'rb') as f:
    loaded_model = pickle.load(f)
y_pred = loaded_model.predict(X_test)
model_dict[5]='pipe 2 Support Vector Classifier'
print_results() 


# In[21]:


display_confusion_matrix(confusion_matrix(y_test, y_pred), loaded_model.classes_, 'pipe 2 Support Vector Classifier')


# ## Decision Tree Classifier

# ### Using `pipe1`

# In[22]:


# Make predictions using testing set
path = Path.cwd() / "saved_model" / "pipe1_DTC.pkl"
with open(path, 'rb') as f:
    loaded_model = pickle.load(f)
y_pred = loaded_model.predict(X_test)
model_dict[6]='pipe 1 Decision Tree Classifier'
print_results() 


# In[23]:


display_confusion_matrix(confusion_matrix(y_test, y_pred), loaded_model.classes_, 'pipe 1 Decision Tree Classifier')


# ### Using `pipe2`

# In[24]:


# Make predictions using testing set
path = Path.cwd() / "saved_model" / "pipe2_DTC.pkl"
with open(path, 'rb') as f:
    loaded_model = pickle.load(f)
y_pred = loaded_model.predict(X_test)
model_dict[7]='pipe 2 Decision Tree Classifier'
print_results() 


# In[25]:


display_confusion_matrix(confusion_matrix(y_test, y_pred), loaded_model.classes_, 'pipe 2 Decision Tree Classifier')


# ## Random Forest Classifier

# ### Using `pipe1`

# In[26]:


# Make predictions using testing set
path = Path.cwd() / "saved_model" / "pipe1_RFC.pkl"
with open(path, 'rb') as f:
    loaded_model = pickle.load(f)
y_pred = loaded_model.predict(X_test)
model_dict[8]='pipe 1 Random Forest Classifier'
print_results() 


# In[27]:


display_confusion_matrix(confusion_matrix(y_test, y_pred), loaded_model.classes_, 'pipe 1 Random Forest Classifier')


# ### Using `pipe2`

# In[28]:


# Make predictions using testing set
path = Path.cwd() / "saved_model" / "pipe2_RFC.pkl"
with open(path, 'rb') as f:
    loaded_model = pickle.load(f)
y_pred = loaded_model.predict(X_test)
model_dict[9]='pipe 2 Random Forest Classifier'
print_results() 


# In[29]:


display_confusion_matrix(confusion_matrix(y_test, y_pred), loaded_model.classes_, 'pipe 2 Random Forest Classifier')


# ## Accuracy of the models

# In[30]:


for i in range(len(model_accuracy)):
    print("{} accuracy: {}".format(model_dict[i], model_accuracy[i]))

