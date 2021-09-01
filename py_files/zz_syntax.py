# Syntax of pipeline
pipe = Pipeline([
    ('named_step_1', object_1),
    ('named_step_2', object_2),
    ...
    ('named_step_n', object_n)
])

# Syntax to change clf
pipe.set_params(clf=<object_name>)

[MultinomialNB(),
svm.SVC(C=1.0, kernel='linear', degree=3, gamma='auto',random_state=1000),
tree.DecisionTreeClassifier(random_state=1000),
RandomForestClassifier(random_state=1000)]

# Make predictions using testing set
# Load the model
path = Path.cwd() / "saved_model" / "<filename>"
with open(path, 'rb') as f:
    loaded_model = pickle.load(f)

# Make predictions using testing set
y_pred = loaded_model.predict(X_test)

# Update the model dictionary for later use
model_dict[0]='<name of the model>'

# Call the function
print_results() 


# In[11]:


display_confusion_matrix(confusion_matrix(y_test, y_pred), loaded_model.classes_, 'Logistic Regression')