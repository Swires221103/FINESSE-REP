"""
Import .sav radiance data for analysis.
"""
# Setup ===========================================================================

from glob import glob
import os
import shutil
import re
from pathlib import Path
import scipy as sp # Import Scipy.io to read .sav files
import numpy as np # Import numpy for array manipulation
np.set_printoptions(threshold=10)

# Functions =======================================================================

def split_trials_to_dict(arr, header, dic):
    """
    Given a 2d Array of the different spectra, split each
    trial and append to a dictionary for future use.

    Args:
        arr: The 2d array of spectra data for several trials
        header: Desired header for each run. Will be formatted as "[header] 1,2..." in dict
        dic: The dictionary to append each trial to

    Returns:
        dic: Appended dictionary with spectra sorted by trials
    """
    for i, value in enumerate(arr[0]):
        temp_list = []
        for j, value in enumerate(arr):
            temp_list.append(arr[j][i])
        dic[str(header + " " + str(i + 1))] = temp_list
    return dic

def import_sav_file(filename, dic = None):
    """
    Import the .sav file and save as a dictionary of each spectrum to be 
    used later in analysis.

    Args:
        filename: path of .sav file to import

    Returns:
        output_dict: The dictionary of all imported data formated with 
        Wavenumber as the first column, then each spectra sorted by 
    """
    if dic is None:
        dic = {}

    imported_dict = sp.io.readsav(filename) # import the data in unsplit formula
    headers = list(imported_dict.keys()) # Create a list of keys for appending data

    output_dict = {} # Create an output dic for new formatting
    output_dict["WVN"] = imported_dict.get(headers[0]) # Add Wavenumber

    for i in range(1, (len(headers))):
        if hasattr(imported_dict.get(headers[i])[0], "__len__") is True:
            split_trials_to_dict(imported_dict.get(headers[i]), headers[i], output_dict)
        else:
            output_dict[headers[i]] = imported_dict.get(headers[i])

    return output_dict

def import_spectra(file_path, label=None, dic=None, remove=None, indices=None):
    """
    Imports Individual Spectra from within a file and appends them
    all to a dictionary for use in later analysis.
    """
    if dic is None:
        dic = {}

    if remove is None:
        remove = []

    if indices is None:
        indices = [0,-1]
    start = indices[0]
    stop = indices[1]
    

    if isinstance(file_path, str) is False:
        raise ValueError("Filename: Expected String")
    
    text_filenames = glob(str(file_path+'*.txt'))

    # Custom sorting function
    def extract_number(file):
        match = re.search(r'(\d+)\.txt$', file)
        return int(match.group(1)) if match else float('inf')
    
    text_filenames = sorted(text_filenames, key=extract_number)

    # print(text_filenames)
    
    for i, file in enumerate(text_filenames):
        if label is None:
            basename = os.path.basename(file)
            label = os.path.splitext(basename)[0]

            if i == 0:
                temp_data = np.loadtxt(file)
                dic["WVN"] = temp_data[:,0][start:stop]
                dic[label] = temp_data[:,1][start:stop]
            elif i not in remove:
                temp_data = np.loadtxt(file)
                dic[label] = temp_data[:,1][start:stop]
            else:
                continue
            label = None

        if label is not None:
            if i == 0:
                temp_data = np.loadtxt(file)
                dic["WVN"] = temp_data[:,0][start:stop]
                dic[label+' '+str(i+1)] = temp_data[:,1][start:stop]
            elif i not in remove:
                temp_data = np.loadtxt(file)
                dic[label+' '+str(i+1)] = temp_data[:,1][start:stop]
            else:
                continue
        
        print("Imported",i+1,"of",len(text_filenames))


    keys = list(dic.keys())
    keys.sort()
    print(keys)
    return dic

def average_spectra_and_save(file_path, script):
    # Remove specified angles from the script
    script = [item for item in script if item not in {225, 270}]
    
    # Create the output directory if it doesn't exist
    Path(file_path + "averaged_spectra/").mkdir(parents=True, exist_ok=True)
    
    if not isinstance(file_path, str):
        raise ValueError("Filename: Expected String")
    
    # Find folders and extract numeric parts for sorting
    for ang in script:
        folders = glob(f"{file_path}{ang} */")
        print(folders)
        folders_with_numbers = []
    
        for folder in folders:
            match = re.search(rf'{ang} (\d+(?:\.\d+)?)', folder)
            if match:
                number = float(match.group(1))
                folders_with_numbers.append((number, folder))
            else:
                print(f"Warning: Could not extract number from {folder}")
    
        # Sort folders based on extracted number
        print(folders_with_numbers)
        folders_with_numbers.sort(key=lambda x: x[0])
        sorted_folders = [folder for _, folder in folders_with_numbers]

        print("Loading Spectra")

        for i, folder in enumerate(sorted_folders):
            files = glob(os.path.join(folder, '*.txt'))
            files.sort()

            print(f"Processing folder: {folder}")

            label = f"{ang} Degrees {i+1}"
            wvn = None
            measurements = []

            for j, file in enumerate(files):
                temp_data = np.loadtxt(file)
                if j == 0:
                    wvn = temp_data[:, 0]
                measurements.append(temp_data[:, 1])
        
            mean_spectrum = np.mean(measurements, axis=0)
            data = np.column_stack((wvn, mean_spectrum))
            output_filename = os.path.join(file_path, 'averaged_spectra', f"{label}.txt")
            np.savetxt(output_filename, data, delimiter='\t')

            print(f"Imported {i+1} of {len(sorted_folders)}")

def sort_files_into_folders(source_folder, folder_size):
    # List all files in the source folder
    files = [f for f in os.listdir(source_folder) if os.path.isfile(os.path.join(source_folder, f))]
    files.sort()  # Sort files by name (optional, depending on your needs)

    # Calculate the number of folders needed
    num_folders = len(files) // folder_size
    
    # Create and move files into subfolders
    for i in range(num_folders):
        folder_name = "Failed to name"
        if i % 2 == 0:
            folder_name = "30 "+str(i/2 + 1)
        else:
            folder_name = "150 "+str((i+1)/2)

        folder_path = os.path.join(source_folder, folder_name)
        os.makedirs(folder_path, exist_ok=True)
        
        # Move files to the new folder
        for j in range(folder_size):
            file_index = i * folder_size + j
            file_name = files[file_index]
            shutil.move(os.path.join(source_folder, file_name), os.path.join(folder_path, file_name))
        print(f'Moved files to {folder_name}')

def extract_mean_times(parent_folder, script):
    
    script = [x for x in script if x not in {270,225}]
    
    mean_times = []
    # List all subfolders in the parent folder and sort them
    subfolders = sorted([f.path for f in os.scandir(parent_folder) if f.is_dir()])

    for subfolder in subfolders:
        times = []
        # List all files in the current subfolder
        files = [f for f in os.listdir(subfolder) if os.path.isfile(os.path.join(subfolder, f))]

        for file in files:
            # Extract the numeric part of the filename (assuming it's before '.txt')
            match = re.match(r"(\d+)\.txt", file)
            if match:
                time = int(match.group(1))
                times.append(time)

        if times:
            # Calculate the mean time for the current subfolder
            mean_time = sum(times) / len(times)
            mean_times.append(mean_time)

    mean_times.sort()
    mtfea = []
    n_s = len(script)
    n_t = len(mean_times)
    for i in range(len(script)):
        step = 0
        temp_ts = []
        while step < n_t:
            temp_ts.append(mean_times[step+i])
            step += n_s
        mtfea.append(temp_ts)

    return mean_times, mtfea

def analysis_setup(folder, script):
    mt, mtfea = extract_mean_times(folder,script)
    data = load_pth("Data Files/20240729_no_shields/PTH_1436.txt")
    indices = []
    press = []
    temps = []
    humid = []
    for t in mtfea[0]:
        i = (np.abs(data[0] - t)).argmin()
        press.append(data[1][i])
        temps.append(data[2][i])
        humid.append(data[3][i])
    print("times","=",mtfea[0])
    print("press","=",press)
    print("a_temps","=",temps)
    print("humid","=",humid)

# Testing =========================================================================
