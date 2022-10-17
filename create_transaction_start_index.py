#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#performed once to generate a pickle file with the first transaction id
#this pickle file will be rewritten each time a new daily batch is generated so that each transation has a unique id


# In[ ]:


import pickle


# In[ ]:


with open('/home/ubuntu/Fraud-detection/transaction_id.pickle', 'wb') as f:
    pickle.dump(-1, f)

