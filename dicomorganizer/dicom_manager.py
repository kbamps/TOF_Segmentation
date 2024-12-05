import os
import pandas as pd
import pydicom

from dicomorganizer.utils import parallel_tasks

class DicomManager:
    """
    A class to manage DICOM files, extract metadata, and handle operations on them.

    Attributes:
        directory (str): Path to the directory containing DICOM files.
        tags (list): List of DICOM tags to extract.
        DEFAULT_DICOM_TAGS (list): Default tags to extract if no custom tags are specified.
        CLEAR_TAGS (list): Tags that can be cleared to anonymize DICOM files.
    """

    DEFAULT_DICOM_TAGS = [
        "PatientName", "PatientID", "StudyID", "StudyDate", 
        "SOPInstanceUID", "SeriesInstanceUID", "Modality", 
        "BurnedInAnnotation", "SOPClassUID", "StudyInstanceUID"
    ]
    
    CLEAR_TAGS = [
        "PatientBirthDate", "PatientAge", "InstitutionName", 
        "StationName", "StudyID", "AccessionNumber", 
        "SeriesDescription", "StudyDescription"
    ]

    def __init__(self, directory, tags=None, group_by=None, num_workers=None, *args, **kwargs):
        """
        Initializes the DicomManager class, which is used to manage DICOM files, 
        extract metadata, and handle operations such as filtering and anonymization.

        Args:
            directory (str): 
                Path to the directory containing DICOM files. The directory will be 
                recursively searched for DICOM files (excluding 'DICOMDIR').
            
            tags (list, optional):
                A list of DICOM tags to extract from each DICOM file. Defaults to 
                `DEFAULT_DICOM_TAGS` if not provided. These tags define the metadata 
                to be extracted from each file.
            
            group_by (str or list, optional):
                Column name to group the resulting DataFrame by. This is an optional 
                argument that can be specified when calling methods that return 
                DataFrame results (e.g., `df_dicom`). If provided, the DataFrame will 
                be grouped by the specified column.
            
            num_workers (int, optional):
                The number of threads (workers) to use for parallel processing when 
                extracting DICOM metadata or performing tasks like anonymization. If 
                not provided, processing will be done sequentially.
            
            *args:
                Additional positional arguments to pass to any methods that need them.
            
            **kwargs:
                Additional keyword arguments to pass to any methods that need them. 
                This includes options like `group_by` which can be passed dynamically.
            
        """
         
        self.directory = directory
        self.tags = tags or self.DEFAULT_DICOM_TAGS
        self.num_workers = num_workers
        self.group_by = group_by
        self.args = args
        self.kwargs = kwargs
        self._df_dicom = None  # Lazy-loaded DataFrame to store DICOM metadata.
        
        
        
    @property
    def df_dicom(self):
        """
        Lazily loads and returns a pandas DataFrame containing DICOM metadata.

        Returns:
            pd.DataFrame: DataFrame containing DICOM metadata.
        """
        num_workers = self.num_workers
        group_by = self.group_by
        
        if self._df_dicom is None:
            self._df_dicom = self._get_dicom_info(num_workers=num_workers, group_by=group_by)
        return self._df_dicom
    
    
    def filter(self, filter_func, inplace=False):
        """
        Filters the DICOM DataFrame based on a provided filter function.
        
        Args:
            filter_func (function): A function that takes a row of the DataFrame and returns a boolean value.
        
            # Example filter function
            def filter_by_modality(row):
                return row['Modality'] == 'CT'
        
        Returns:
            pd.DataFrame: Filtered DataFrame containing only the rows for which the filter function returned True.
        """
        if not callable(filter_func):
            raise ValueError("The provided filter_func must be a callable function.")
        
        #first ungroup the DataFrame if it is grouped
        if isinstance(self.df_dicom, pd.core.groupby.DataFrameGroupBy):
            df = self._df_dicom.obj
        else:
            df = self._df_dicom
            
        # Apply the filter function to each row of the DataFrame
        df_filtered = df[df.apply(filter_func, axis=1)]
        
        self._df_dicom = df_filtered.groupby(self.group_by) if self.group_by else df_filtered
        return self._df_dicom
    
    
    def anonymize_dicom(self, output_directory, clear_tags=None, num_workers=None):
        """
        Anonymizes the DICOM files by clearing the provided tags which are in the self.df_dicom.

        Args:
            output_directory (str): Path to save the anonymized files.
            clear_tags (list): A list of tags to clear from the DICOM files. If None, uses the class's CLEAR_TAGS.
            num_workers (int): Number of workers for parallel processing. If None, runs sequentially.
            
        Returns:
            list: A list of file paths of the anonymized DICOM files.
        """
        clear_tags = clear_tags or self.CLEAR_TAGS  # Use the class's CLEAR_TAGS if no custom tags are provided
        
        assert os.path.exists(output_directory), f"Output directory '{output_directory}' does not exist."
        assert output_directory != self.directory, "Output directory cannot be the same as the input directory."

        # List to keep track of the anonymized files
        anonymized_files = []
        
        # Get all DICOM file paths from the self.df_dicom DataFrame
        dicom_paths = self.df_dicom['filename'].tolist()

        # Prepare arguments for parallel tasks
        args_list = [(path, clear_tags, output_directory) for path in dicom_paths]
        
        # Parallelize the anonymization task
        if num_workers is None:
            # Run sequentially if num_workers is not provided
            for args in args_list:
                result = self._anonymize_single_dicom(*args)
                if result:
                    anonymized_files.append(result)
        else:
            # Run in parallel if num_workers is specified
            results = parallel_tasks(self._anonymize_single_dicom, args_list, num_workers, description="Anonymizing DICOM files")
            anonymized_files.extend([r for r in results if r is not None])

        return anonymized_files
        

    def _get_dicom_info(self, group_by=None, num_workers=None):
        """
        Reads all DICOM files in the directory and returns their metadata as a pandas DataFrame.

        Args:
            group_by (str, list, optional): Column name to group the data by. Defaults to None.
            num_workers (int, optional): Number of threads for parallel processing. Defaults to None.

        Returns:
            pd.DataFrame or pd.core.groupby.DataFrameGroupBy: 
                DataFrame containing DICOM metadata, optionally grouped by the specified column.
        """
    
        
        dicom_info = self._get_dicom_info_parallel(self.tags, num_workers)
        df_dicom = pd.DataFrame(dicom_info)
        
        if group_by is not None:
            # Check if group_by columns are present in DICOM metadata
            group_by_list = group_by if isinstance(group_by, list) else list([group_by])
            col_check = [col in df_dicom.columns for col in group_by_list]
            if not all(col_check):
                error_message = ','.join([col_name for col_name, check in zip(group_by_list, col_check) if not check])
                raise ValueError(
                        f"Group by '{error_message}' not found in DICOM metadata. Available columns: {df_dicom.columns}"
                    )
                
            return df_dicom.groupby(group_by)
        
        return df_dicom

    def _get_dicom_info_parallel(self, tags, num_workers):
        """
        Retrieves DICOM metadata from multiple files in parallel.

        Args:
            tags (list): List of DICOM tags to extract from each file.
            num_workers (int): Number of threads to use for parallel processing. If None, processing is done sequentially.

        Returns:
            list[dict]: A list of dictionaries, each containing metadata for a single DICOM file. 
                        Returns an empty list if no valid DICOM files are found.
        """
        dicom_paths = self._get_dicom_file_paths()
        args_list = [(path, tags) for path in dicom_paths]
        
        if num_workers is None:
            return [self._get_single_dicom_info(*args) for args in args_list]
        
        results = parallel_tasks(self._get_single_dicom_info, args_list, num_workers, description="Reading DICOM files")
        return [r for r in results if r is not None]

    def _get_dicom_file_paths(self):
        """
        Recursively collects all DICOM file paths from the specified directory (self.directory).

        Returns:
            list[str]: A list of file paths pointing to DICOM files in the directory.
                       Excludes files named 'DICOMDIR'.
        """
        dicom_paths = []
        for root, _, files in os.walk(self.directory):
            for file in files:
                if file.upper() == "DICOMDIR":
                    continue
                dicom_paths.append(os.path.join(root, file))
        return dicom_paths

    def _get_single_dicom_info(self, filepath, tags, default_value=None):
        """
        Reads metadata for a single DICOM file.

        Args:
            filepath (str): Path to the DICOM file to read.
            tags (list): List of DICOM tags to extract.

        Returns:
            dict: A dictionary containing the extracted metadata for the specified DICOM file. 
                  Includes an 'error' key if an exception occurs or the file is invalid.
        """
        try:
            dicom_data = pydicom.dcmread(filepath, specific_tags=tags, stop_before_pixels=True)
        except pydicom.errors.InvalidDicomError:
            return None
        except Exception as e:
            return {'error': str(e)}

        dicom_info = {tag: dicom_data.get(tag, default_value) for tag in tags}
        dicom_info["filename"] = filepath
        return dicom_info

    
    def _anonymize_single_dicom(self, dicom_path, clear_tags, output_directory):
        """
        Anonymizes a single DICOM file by clearing specified tags.

        Args:
            dicom_path (str): Path to the DICOM file to anonymize.
            clear_tags (list): List of tags to clear.
            output_directory (str): Directory to save the anonymized file.

        Returns:
            str: Path to the anonymized DICOM file, or None if the process fails.
        """
        try:
            # Read the DICOM file
            dicom_data = pydicom.dcmread(dicom_path)

            # Clear specified tags
            for tag in clear_tags:
                if tag in dicom_data:
                    dicom_data[tag].value = ""
                
            # Determine the output file path
            output_path = output_directory
            
            # Save the anonymized DICOM file
            dicom_data.save_as(output_path)
            
            return output_path
        except Exception as e:
            print(f"Failed to anonymize {dicom_path}: {e}")
            return None
