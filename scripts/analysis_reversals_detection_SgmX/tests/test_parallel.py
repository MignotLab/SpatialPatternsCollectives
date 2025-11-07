#%%
from joblib import Parallel
from joblib import delayed
import numpy as np
from time import time

def lol(i):
    return(i**2)

start1 = time()
Parallel(n_jobs = 2)(delayed(lol)(i) for i in range(100000))
end1 = time()

time1 = end1-start1
print(time1)

start2=time()
for i in range(1000000):
    lol(i)
end2 = time()

time2 = end2-start2
print(time2)

#%%
from joblib import Parallel, delayed
import time, math
def my_fun(i):
    """ We define a simple function here.
    """
    time.sleep(1)
    return math.sqrt(i**2)

num = 2
start = time.time()
for i in range(num):
    my_fun(i)
end = time.time()
print('{:.4f} s'.format(end-start))

start = time.time()
# n_jobs is the number of parallel jobs
Parallel(n_jobs=2)(delayed(my_fun)(i) for i in range(num))
end = time.time()
print('{:.4f} s'.format(end-start))