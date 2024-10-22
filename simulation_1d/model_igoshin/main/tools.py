import os


class Tools:


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
