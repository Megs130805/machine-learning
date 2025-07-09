#!/usr/bin/env python
# coding: utf-8

# In[1]:


import random

num = [random.randint(100, 150) for i in range(100)]

mean = sum(num) / len(num)

num.sort()
n = len(num)

if n % 2 == 0:
    median = (num[n//2 - 1] + num[n//2]) / 2
else:
    median = num[n//2]

freq = {}
for nu in num:
    if nu in freq:
        freq[nu] += 1
    else:
        freq[nu] = 1

maxim = max(freq.values())
mode = [k for k, v in freq.items() if v == maxim]

print("mean:", mean)
print("median:", median)
print("mode:", mode)


# In[ ]:




