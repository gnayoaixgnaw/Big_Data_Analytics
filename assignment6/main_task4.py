#!/usr/bin/env python
# coding: utf-8

# In[1]:


from operator import add
from pyspark import SparkConf, SparkContext


# In[2]:


text = sc.textFile(sys.argv[1])


# In[3]:


textandlabel = text.map(lambda x : (x[x.index('(')+1 : x.index('<doc')-3],x[x.index('20_newsgroups/')+14 : x.index('" url="')-6]))


# In[4]:


label_add = textandlabel.map(lambda x: (str(x[0]),1))


# In[5]:


result_label = label_add.reduceByKey(add).collect()


# In[6]:


top_3 = sorted(result_label, key = lambda x: x[1], reverse = True)[0:3]


# In[7]:


top_3


# In[8]:


top_18 = textandlabel.filter(lambda x: x[0] == '18')
top_18_3 = sorted(top_18.map(lambda x: (x[1], 1)).reduceByKey(add).collect(),key = lambda x: x[1], reverse = True)[0:3]
tab18 = []
for i in top_18_3:
    temp = int(i[1])/top_3[0][1]
    tab18.append(temp)
print('Result of top one categrory (18): ')
print(tab18)


# In[9]:


top_16 = textandlabel.filter(lambda x: x[0] == '16')
top_16_3 = sorted(top_16.map(lambda x: (x[1], 1)).reduceByKey(add).collect(),key = lambda x: x[1], reverse = True)[0:3]
tab16 = []
for i in top_16_3:
    temp = int(i[1])/top_3[1][1]
    tab16.append(temp)
print('Result of top one categrory (16): ')
print(tab16)


# In[10]:


top_10 = textandlabel.filter(lambda x: x[0] == '10')
top_10_3 = sorted(top_10.map(lambda x: (x[1], 1)).reduceByKey(add).collect(),key = lambda x: x[1], reverse = True)[0:3]
tab10 = []
for i in top_10_3:
    temp = int(i[1])/top_3[2][1]
    tab10.append(temp)
print('Result of top one categrory (10): ')
print(tab10)

    
