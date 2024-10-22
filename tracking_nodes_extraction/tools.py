import numpy as np
import matplotlib.pyplot as plt
import os


class Tools:


    def __init__(self) -> None:

        pass

    
    def save_fig(self, fig, name, dpi):
        """
        Save figure
        
        """
        fig.savefig(name, dpi=dpi)


    def rotation(self,theta):
        """
        Rotation matrix that we use for the changement of direction of the bacteria

        """
        tmp = np.array([[np.cos(theta),-np.sin(theta)], [np.sin(theta), np.cos(theta)]])

        return tmp


    def py_ang(self,v1,v2):
        """Compute the angle between the vectors v1 and v2 in radian between [-pi,pi]

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
        det = v1[0]*v2[1] - v1[1]*v2[0]      # determinant
        angle = np.arctan2(det, dot)

        return angle


    def gen_coord_str(self, n, xy=True):
        """
        Generate a list of string as [x0,y0,...,xn,yn] if xy = True
        else generate tuple of lists of string: ([x0,...,xn], [y0,...,yn])

        """
        if xy:
            name = []
            for i in range(n):
                name.append('x'+str(i))
                name.append('y'+str(i))

            return name

        else:
            name_x = []
            name_y = []
            for i in range(n):
                name_x.append('x'+str(i))
                name_y.append('y'+str(i))

            return name_x, name_y
        
    def gen_string_numbered(self, n, str_name):
        """
        Generate a list like [str_name0, ..., str_namen]
        
        """
        res = []
        
        for i in range(n):
            res += [str_name + str(i)]

        return res
    

    def save_df(self, df, path, file_name):
        """
        Save and create the save folder if needed
        
        """
        # Write dataframe into the csv file
        # Check whether the specified path exists or not
        isExist = os.path.exists(path)
        if not isExist:
            # Create a new directory because it does not exist
            os.makedirs(path)
            print("The new directory is created!")
        
        df.to_csv(path+file_name, index=False)


    def mean_angle(self,angles):
        """ 
        Compute the angle mean (different than the arithmetic mean)
        
            Parameters
            -----------
                angles: numpy.ndarray
                    Multi dimensional array of angle you want to mean along axis=1

            Returns
            --------
                mean_angle: array or float
                    The mean angle
        
        """
        tmp = angles.astype(float).T
        mean_angle = np.arctan2(np.sum(np.sin(tmp),axis=0), np.sum(np.cos(tmp),axis=0))
        
        return mean_angle


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
