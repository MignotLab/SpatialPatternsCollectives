import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import os
import pickle


class Tools:


    def __init__(self) -> None:

        pass


    def mean_angle(self, angles, axis):
        """
        Calculate the mean of a set of angles in radians.

        Parameters:
        - angles (list): List of angles in radians.

        Returns:
        - float: Mean of angles in radians.
        """

        # Convert angles to Cartesian coordinates (vectors)
        vectors = np.array([np.cos(angles), np.sin(angles)])

        # Calculate the mean of Cartesian coordinates
        mean_vector = np.mean(vectors, axis=int(axis+1))

        # Convert the mean coordinates to an angle
        mean_angle = np.arctan2(mean_vector[1], mean_vector[0])

        return mean_angle


    def initialize_directory_or_file(self, path, columns=None):
        """
        Create the parent directories specified in the path (if they don't already exist).
        If the path contains a file name, create the file or reset it if it already exists.
        If the path does not contain a file name, create only the parent directories.

        Parameters
        ----------
        path : str
            The full path of the file or directories to initialize.

        columns : list or None, optional
            A list of column names to write as the header if the path contains a file name.
            Default is None, which means no header will be written.

        Returns
        -------
        None

        Notes
        -----
        This function creates the parent directories for the specified path, and if the path
        includes a file name (with extension), it will create or reset the file.

        Examples
        --------
        >>> initialize_directory_or_file("data/images/image.jpg")
        # Creates the 'data/images' directory if it doesn't exist and creates/reset the 'image.jpg' file.

        >>> initialize_directory_or_file("data/results/")
        # Creates the 'data/results' directory if it doesn't exist.

        >>> initialize_directory_or_file("data/data.csv", columns=["Name", "Age", "City"])
        # Creates/reset the 'data.csv' file with the specified columns as header.
        """
        # Get the parent directory of the path
        folder_path = os.path.dirname(path)

        # Check if the parent directory exists, if not, create it
        if folder_path and not os.path.exists(folder_path):
            os.makedirs(folder_path)

        # Check if the path contains a file name
        if os.path.splitext(path)[1]:  # If the path has an extension (it's a file)
            # Open the file in write mode and write the columns as header if provided
            with open(path, 'w', newline='') as file:
                if columns:
                    header = ",".join(columns)
                    file.write(header + '\n')

                    
    def get_rgba_color(self, color_name, alpha):
        """
        Get the RGBA color code from a color name and alpha value.

        This function takes a color name and an alpha value as inputs and returns the RGBA color code.
        The color name should be a valid Matplotlib color name, e.g., 'blue', 'red', 'green', 'royalblue', etc.
        The alpha value specifies the transparency of the color, ranging from 0.0 (fully transparent) to 1.0 (fully opaque).

        Parameters:
            color_name (str): Name of the color.
            alpha (float): Alpha value, ranging from 0.0 (fully transparent) to 1.0 (fully opaque).

        Returns:
            tuple: Tuple representing the RGBA color code (Red, Green, Blue, Alpha).
                Each component is a float value between 0.0 and 1.0.

        Example:
            color_with_alpha = 'royalblue'
            alpha = 0.4
            original_color = get_rgba_color(color_with_alpha, alpha)
            print("Original RGBA color:", original_color)
        """
        rgba_color = mcolors.to_rgba(color_name, alpha=alpha)

        return rgba_color
    

    def save_or_load_object(obj, file_path, action='save'):
        """
        Save or load a Python object to/from a file using pickle.

        Parameters:
        - obj: Python object to be saved or loaded.
        - file_path: Path to the file where the object will be saved or loaded.
        - action: 'save' to save the object, 'load' to load the object. Default is 'save'.

        Returns:
        - If action is 'load', returns the loaded Python object.
        """
        if action == 'save':
            with open(file_path, 'wb') as file:
                pickle.dump(obj, file)
        elif action == 'load':
            with open(file_path, 'rb') as file:
                obj = pickle.load(file)
            return obj
        else:
            raise ValueError("Invalid action. Use 'save' or 'load'.")

    
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
        
    def gen_string_numbered(self,n,str_name):
        """
        Generate a list like [str_name0, ..., str_namen]
        
        """
        res = []
        
        for i in range(n):
            res += [str_name+str(i)]

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


    # def mean_angle(self,angles):
    #     """ 
    #     Compute the angle mean (different than the arithmetic mean)
        
    #         Parameters
    #         -----------
    #             angles: numpy.ndarray
    #                 Multi dimensional array of angle you want to mean along axis=1

    #         Returns
    #         --------
    #             mean_angle: array or float
    #                 The mean angle
        
    #     """
    #     tmp = angles.astype(float).T
    #     mean_angle = np.arctan2(np.sum(np.sin(tmp),axis=0), np.sum(np.cos(tmp),axis=0))
        
    #     return mean_angle


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
