#!/usr/bin/env python
# coding: utf-8

# In[5]:


matrix = [[1, 2, 4], [5, 9, 0]]
transpose = []

for i in range(len(matrix[0])):
    row = []
    for j in range(len(matrix)):
        row.append(matrix[j][i])
    transpose.append(row)

print("Transpose:", transpose)


# In[ ]:




