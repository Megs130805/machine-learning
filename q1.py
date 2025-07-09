#!/usr/bin/env python
# coding: utf-8

# In[1]:


str= input("enter string:")
vowels="aeiouAEIOU"
v_count=0
c_count=0
for ch in str:
    if ch.isalpha():
        if ch in vowels:
            v_count+=1

        else:
            c_count+=1
print("vowels:",v_count)
print("Consonants:",c_count)


# In[ ]:




