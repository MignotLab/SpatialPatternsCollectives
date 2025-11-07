import numpy as np
f = open("algo1.txt", "r")
a = f.read()
a = a.split(" ")
a = np.array(a[:-1]).astype(float)
print(np.sum(a))

f = open("algo2.txt", "r")
b = f.read()
b = b.split(" ")
b = np.array(b[:-1]).astype(float)
print(np.sum(b))
