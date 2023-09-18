#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
from sklearn.preprocessing import RobustScaler

import warnings
warnings.filterwarnings('ignore')


# In[5]:


df_train = pd.read_csv('/home/shweta_chaudhari/personality_prediction/train.csv')
train_length = len(df_train)
df_test = pd.read_csv('/home/shweta_chaudhari/personality_prediction/test.csv')
df_train.rename(columns = {'Personality (Class label)':'Personality'}, inplace = True) 
df_test.rename(columns = {'Personality (class label)':'Personality'}, inplace = True) 
df = pd.concat([df_train, df_test])
df.head()


# In[6]:


df['Gender'] = df['Gender'].map({'Male': 0,
                                 'Female': 1})

df['Personality'] = df['Personality'].map({'dependable': 0,
                                           'extraverted': 1,
                                           'lively': 2,
                                           'responsible': 3,
                                           'serious': 4})


# In[ ]:


# Imputing values


# In[7]:


import random
def impute(df):
    persn=df['Personality']
    if persn==0:
        return int(random.choice([1, 2, 3,4,5,6]))
    if persn==1:
        return int(random.choice([6,7,8,9,10]))
    if persn==2:
        return int(random.choice([7,8,9,10]))
    if persn==3:
        return int(random.choice([5,6,7,8]))   
    if persn==4:
        return int(random.choice([1, 2, 3,4,5]))   
df['openness']=df.apply(lambda x: impute(x),axis=1)


# In[8]:


def impute(df):
    persn=df['Personality']
    if persn==0:
        return int(random.choice([2, 3,4,5,6]))
    if persn==1:
        return int(random.choice([1,2, 3,4]))
    if persn==2:
        return int(random.choice([1,2, 3,4]))
    if persn==3:
        return int(random.choice([3,4,5,6,7]))   
    if persn==4:
        return int(random.choice([6,7,8,9,10]))  
df['neuroticism']=df.apply(lambda x: impute(x),axis=1)


# In[9]:


def impute(df):
    persn=df['Personality']
    if persn==0:
        return int(random.choice([5,6,7,8]))
    if persn==1:
        return int(random.choice([6,7,8,9,10]))
    if persn==2:
        return int(random.choice([7,8,9,10]))
    if persn==3:
        return int(random.choice([7,8,9,10]))   
    if persn==4:
        return int(random.choice([3,4,5,6,7])) 
df['conscientiousness']=df.apply(lambda x: impute(x),axis=1)


# In[10]:


def impute(df):
    persn=df['Personality']
    if persn==0:
        return int(random.choice([7,8,9,10]))
    if persn==1:
        return int(random.choice([5,6,7,8]))
    if persn==2:
        return int(random.choice([3,4,5,6,7,8]))
    if persn==3:
        return int(random.choice([5,6,7,8,9]))   
    if persn==4:
        return int(random.choice([3,4,5,6,7])) 
df['agreeableness']=df.apply(lambda x: impute(x),axis=1)


# In[11]:


def impute(df):
    persn=df['Personality']
    if persn==0:
        return int(random.choice([5,6,7,8]))
    if persn==1:
        return int(random.choice([7,8,9,10]))
    if persn==2:
        return int(random.choice([7,8,9,10]))
    if persn==3:
        return int(random.choice([5,6,7,8,9]))   
    if persn==4:
        return int(random.choice([5,6,7,8])) 
df['extraversion']=df.apply(lambda x: impute(x),axis=1)


# Before I preprocess anything, I'll do some exploration. I will draw a box-and-whiskers plot for each Big Five trait, to roughly see the distribution of each trait within each of the personality label categories.

# In[12]:


sns.catplot(x="Personality", y="openness", kind="box", data=df)


# Each personality label has very similar distributions of 'openness', besides 'lively' which has a distribution which seems a bit more concentrated at above average values.

# In[13]:


sns.catplot(x="Personality", y="neuroticism", kind="box", data=df)


# 'neuroticism' looks a bit more useful for us, in as much as it has somewhat different distributions within each personality label.

# In[14]:


sns.catplot(x="Personality", y="conscientiousness", kind="box", data=df)


# 'conscientiousness' is distributed quite similarly within each personality label, although the 'extraverted' and 'responsible' labels have more individuals with a below average rating.

# In[15]:


sns.catplot(x="Personality", y="agreeableness", kind="box", data=df)


# 'agreeableness' has basically identical distributions for each personality label, and is thus unlikely to be very useful when making predictions.

# In[16]:


sns.catplot(x="Personality", y="extraversion", kind="box", data=df)


# In[ ]:


#checking nan values


# In[17]:


df.isnull().sum().sum()


# Only one row with a null value, in 'Gender'. I'll quickly impute it, in a somewhat ad-hoc manner. I'll group our dataframe by 'Personality' and 'Age', and see what the mean age value is for 'serious' 21-year-olds:

# In[18]:


df.groupby(by=['Personality', 'Age']).mean().loc[4]


# The mean is closer to 0 i.e. closer to the male numerical value. I'll therefore impute the missing value with 0.

# In[19]:


df.at[449, 'Gender'] = 0


# In[20]:


df.isnull().sum().sum()


# I'll now go back to do some more exploration: let's take a look at the correlation matrix of our data. This will allow us to see if there are any clear linear relationships between any of our values, and more importantly between any of our features and the label. We already have a rough idea from our box-and-whiskers plots that no Big Five trait seems to have a strong relationship with any of the personality labels, but looking at a correlation matrix may still be useful.

# In[21]:


import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(8,8)) 
sns.heatmap(df.corr(), annot=True,ax=ax)


# Alas, it's now clear that there aren't any strong linear relationships at play: the strongest correlations with regards to our label are with 'Gender' and 'neuroticism', and even those are quite weak. This does not discount the possibility of non-linear relationships that could allow a model to make good predictions, but it's not an encouraging start. Still, let us press on!
# 
# For our models to work well, we should make sure that our data isn't too skewed:

# 'openness' and 'agreeableness' have a negative skew below -0.5, so it's probably a good idea to unskew these columns. I'll use a square root transformation over a reflection of these columns:

# At this point, I will split the unified database back into the training and the test set, and then scale them separately:

# In[22]:


df_train = df[:train_length]
df_test = df[train_length:]
Y, X = df_train.values[:,-1], df_train.values[:,:-1]
Y_test, X_test = df_test.values[:,-1], df_test.values[:,:-1]


# In[23]:


X_test.shape


# In[24]:


from sklearn.preprocessing import LabelEncoder


# In[ ]:





# In[25]:


import numpy as np
from sklearn.preprocessing import LabelEncoder
from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import np_utils

# for modeling
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.callbacks import EarlyStopping


# In[26]:


# work with labels
# encode class values as integers
encoder = LabelEncoder()
encoder.fit(Y)
encoded_Y = encoder.transform(Y)
# convert integers to dummy variables (i.e. one hot encoded)
dummy_y = np_utils.to_categorical(encoded_Y)


# In[27]:


Y_test_dummy=np_utils.to_categorical(Y_test)


# In[52]:


Y_test_dummy


# In[28]:


# build a model
model = Sequential()
model.add(Dense(32, input_shape=(X.shape[1],), activation='relu')) # input shape is (features,)
model.add(Dense(16, activation='relu')) # input shape is (features,)
model.add(Dense(8, activation='relu')) # input shape is (features,)
model.add(Dense(5, activation='softmax'))
model.summary()

# compile the model
model.compile(optimizer='rmsprop', 
              loss='categorical_crossentropy', # this is different instead of binary_crossentropy (for regular classification)
              metrics=['accuracy'])


# In[29]:


import keras
from keras.callbacks import EarlyStopping

# early stopping callback
# This callback will stop the training when there is no improvement in  
# the validation loss for 10 consecutive epochs.  
es = keras.callbacks.EarlyStopping(monitor='val_loss', 
                                   mode='min',
                                   patience=10, 
                                   restore_best_weights=True) # important - otherwise you just return the last weigths...

# now we just update our model fit call
history = model.fit(X,
                    dummy_y,
                    callbacks=[es],
                    epochs=5000, # you can set this to a big number!
                    batch_size=10,
                    shuffle=True,
                    validation_split=0.2,
                    verbose=1)


# In[30]:


from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

preds = model.predict(X_test) # see how the model did!



# In[31]:



matrix = confusion_matrix(Y_test_dummy.argmax(axis=1), preds.argmax(axis=1))
matrix
## array([[50,  0,  0],
##        [ 0, 46,  4],
##        [ 0,  1, 49]])


# In[33]:


#checking accuracy
print(classification_report(Y_test_dummy.argmax(axis=1), preds.argmax(axis=1)))


# In[ ]:


#saving the final model weights


# In[34]:


model.save('/home/shweta_chaudhari/personality_prediction/model2')


# In[35]:


# loading weights
model1 = keras.models.load_model('/home/shweta_chaudhari/personality_prediction/model2')


# In[37]:


# testing on one sample
result=model1.predict([[ 0., 17.,  5.,  6.,  4.,  5.,  5.]]).argmax(axis=1)


# In[39]:


list(result)[0]

