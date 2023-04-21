#!/usr/bin/env python
# coding: utf-8

# # Importing Libraries

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Embedding
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import LSTM,Dense, Dropout, SpatialDropout1D


# # Importing Data

# In[2]:


df = pd.read_csv("D://Datasets//Tweets.csv")
df.head ()


# # EDA

# In[3]:


review_df = df[['tweet','label']]
print('Shape: ', review_df.shape)
display (review_df.head(5))


# In[4]:


print (review_df["label"].value_counts())


# In[5]:


sentiment_label = review_df.label.factorize()
sentiment_label


# In[6]:


tweet = review_df.tweet.values
print (tweet)


# # Model Building, Testing

# In[7]:


tokenizer = Tokenizer(num_words=5000)
tokenizer.fit_on_texts(tweet)
vocab_size = len(tokenizer.word_index) + 1
encoded_docs = tokenizer.texts_to_sequences(tweet)
padded_sequence = pad_sequences(encoded_docs, maxlen=100)


# In[8]:


embedding_vector_length = 16
model = Sequential()
model.add(Embedding(vocab_size, embedding_vector_length, input_length=100))
model.add(SpatialDropout1D(0.25))
model.add(LSTM(10, dropout=0.3, recurrent_dropout=0.4))
model.add(Dropout(0.25))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy',optimizer='adam', metrics=['accuracy'])

print(model.summary())


# In[9]:


history = model.fit(padded_sequence,sentiment_label[0],validation_split=0.3, epochs=3, batch_size=30)


# # Plots

# In[10]:


plt.title ('Accuracies')
plt.plot(history.history['accuracy'], label='acc')
plt.plot(history.history['val_accuracy'], label='val_acc')
plt.legend ()
plt.show ()


# In[11]:


plt.title ('Loss')
plt.plot(history.history['loss'], label='loss')
plt.plot(history.history['val_loss'], label='val_loss')
plt.legend()
plt.show()


# # Prediction

# #### Label {0: Negative, 1: Postive}

# In[15]:


def predict_sentiment(text):
    tw = tokenizer.texts_to_sequences([text])
    tw = pad_sequences(tw,maxlen=100)
    prediction = int(model.predict(tw).round().item())
    if sentiment_label[1][prediction] == 0:
        print("Predicted label: Negative")
    else:
        print ("Predicted label: Positive")


# In[16]:


predict_sentiment ('Hate this particular stuff')

