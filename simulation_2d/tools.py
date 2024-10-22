import numpy as np
import time
from functools import wraps
import os
import csv

class Tools:


    def __init__(self):
        pass

    def array_info(self,array):
        """
        Compute the size and the type of an array
        
        """
        print("Size of array :", array / 1000000, "Mo")
        print("Type of array :", array.dtype)


    def timer(f):
        """
        Decorator which compute the computation time of a function
        Put @timer before a function to compute its time
        The decorator @wraps allow to keep this metadata when you call help(f)
        where f has the @timer decorator
        
        """
        @wraps(f)
        def wrapper(*args, **dargs):
            start = time.time()
            res = f(*args, **dargs)
            print('{:.2} s'.format(time.time()-start))
            return res
        
        return wrapper
    

    def initialize_csv(self,filename,column_names):
        """
        Initialize csv file.
        In case the file exist first remove it and write the columns name.
        
        """
        if os.path.isfile(filename):
            os.remove(filename)  # Supprimer le fichier existant s'il existe
        with open(filename, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(column_names)

    def append_to_csv(self,filename,data):
        """
        Write in a csv file
        
        """
        with open(filename, 'a', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerows(data)
    

    def gen_coord_str(self,n,xy=True):
        """
        Generate tuple of string as (start_x,...,end_x,start_y,...,end_y) n times
        or two lists of sting as [start_x,...,end_x] and [start_y,...,end_y]

        """
        if xy:
            name = []
            for i in range(n):
                name.append('x'+str(i))
            for i in range(n):
                name.append('y'+str(i))
            return name
        else:
            name_x = []
            name_y = []
            for i in range(n):
                name_x.append('x'+str(i))
                name_y.append('y'+str(i))

            return name_x, name_y


    def rotation(self,theta):
        """
        Rotation matrix that we use for the changement of direction of the bacteria

        """
        tmp = np.array([[np.cos(theta),-np.sin(theta)], [np.sin(theta), np.cos(theta)]])

        return tmp

    def py_ang(self,v1,v2):
        """
        Compute the angle between the vectors v1 and v2 in radian between [-pi,pi]

            Parameters
            -----------
                v1: array or list
                    Vector 1

                v2: array or list
                    Vector 2

            Returns
            --------
                angle: float
                    The angle between u and v in radian

        """
        dot = v1[0]*v2[0] + v1[1]*v2[1]      # dot product
        det = v1[0]*v2[1] - v1[1]*v2[0]      # deter_a_minant
        angle = np.arctan2(det, dot)

        return angle


    def nunique_axis0_maskcount_app1(self,A):

        m,n = A.shape[1:]
        mask = np.zeros((A.max()+1,m,n),dtype=bool)
        mask[A,np.arange(m)[:,None],np.arange(n)] = 1

        return mask.sum(0)


    def nunique_axis0_maskcount_app2(self,A):

        m,n = A.shape[1:]
        A.shape = (-1,m*n)
        maxn = A.max()+1
        N = A.shape[1]
        mask = np.zeros((maxn,N),dtype=bool)
        mask[A,np.arange(N)] = 1
        A.shape = (-1,m,n)
        
        return mask.sum(0).reshape(m,n)