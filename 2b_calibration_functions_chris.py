"""More helper functions for spectrum calibration"""

import os
import shutil
import glob
import re
import calibration_functions_sanjee as cal
import numpy as np


def load_single_int(filename):
    """Load single interferogram produced using
        2_prepare_interferogram.py; adapted from load_average_int

        Args:
            filename (string): Location of interferogram file

        Returns:
            array: interferogram
            array: start time of interfergram (appears twice in the list for consistency with other methods)
            float: mirror angle for interferogram
        """
    interferogram = np.loadtxt(filename)
    with open(filename, "r") as f:
        for i, line in enumerate(f):
            if i == 2:
                times_raw = line
            if i == 3:
                angle_raw = line
            if i > 3:
                break
    times_split = times_raw.split(" ")[1:]
    time = np.array((times_split[0], times_split[0]), dtype=float)
    angle = float(angle_raw[2:-1])
    return interferogram, time, angle

def calculate_nesr_from_bb(bb_ints):
    """DEPRECATED but kept here as a record.

    Find the residual between one interferogram and the average of the other interferograms in a scan
    cycle in order to calculate the NESR. Can be applied to hot or cold blackbody.

    Args:
        bb_ints (np.array): list containing all the interferogram radiances of a blackbody from a single scan cycle

    Returns:
        np.array: NESR for this cycle
    """
    # separate first interferogram from others, and take average of all others
    separated_rad = bb_ints[0]
    rads_to_average = bb_ints[1:]
    average_rad = np.zeros(len(rads_to_average[0]))  # initialise list for averages
    for rads in rads_to_average:
        for j, rad in enumerate(rads):
            average_rad[j] += rad / len(rads_to_average)  # contribute to the average list
    # find difference of radiance at each wavenumber between first interferogram and average interferogram
    nesr = []
    for i in range(len(separated_rad)):
        nesr.append(average_rad[i] - separated_rad[i])
    return np.array(nesr)

def calculate_nesr(wns, rads):
    """Calculate the NESR.

    Args:
        wns (np.array): List of list of wavenumbers for all scans
        rads (np.array): List of list of radiances for all scans at the wavenumber of the same index in wn

    Returns:
        float: NESR
    """
    # get difference between each radiance
    rad_differences = []
    for i in range(len(rads) - 1):
        current_rad = rads[i]
        next_rad = rads[i + 1]
        rad_difference = []
        for j in range(len(current_rad)):
            rad_difference.append(next_rad[j] - current_rad[j])
        rad_differences.append(rad_difference)

    # get the RMS of the differences in rolling 5 cm^-1 bands
    # get indices of the rolling 5 cm^-1 bands
    first_wavenumber = 400
    last_wavenumber = 1605
    indices = []
    for i, wn in enumerate(wns[0]):  # all the wavenumbers for every scan should be the same
        if first_wavenumber < wn < last_wavenumber:
            indices.append([i, i + 100])        # wavenumber increases by 5 cm^-1 after 100 steps

    # get RMS for the radiances inside the bands
    nesr_values = []
    for i, index in enumerate(indices):
        start_index = index[0]
        end_index = index[1]
        # go through all the radiance difference lists, get RMS for each list, then take square root of mean RMS
        rms_for_each_scan = []
        for j, rad_difference_list in enumerate(rad_differences):
            # get RMS of radiance differences in this wavenumber range for this scan
            relevant_square_rad_differences = []
            for rad_difference in rad_difference_list[start_index:end_index]:
                relevant_square_rad_differences.append(rad_difference ** 2)     # square the difference to get RMS later
            # take mean and square root to get RMS
            rms_for_each_scan.append(np.sqrt(np.mean(relevant_square_rad_differences)))
        # take mean of RMS for each scan, then take square root to get NESR
        nesr = np.sqrt(np.mean(rms_for_each_scan))
        # return with the start index (which is the index for which this NESR is valid)
        nesr_values.append([start_index, nesr])

    # create a list with NESR values between the first and last wavenumber. Outside of those bounds, return nan
    nesr = []
    for i in range(len(wns[0])):
        if indices[0][0] <= i <= indices[-1][0]:
            nesr.append(nesr_values[i-indices[0][0]][1])
        else:
            nesr.append(np.nan)

    return nesr

def find_dist_to_bb(script, current_index, angle_to_find):

    """
    Given the current index of the data within the FINESSE Script, finds
    the distance to the HBB and CBB (given as 270 deg or 225 deg respectively)

    Args:
        script: The script that FINESSE uses to take measurements eg [270,225,30,250]
        current_index: where in the script the current file is
        angle_to_find: angle of the BB to find eg 270 for the HBB

    Returns:
        The distance to the nearest BB measurement requried
    """
    bb_min_dist = 1000
    actual_distance = 0
    for i, ang in enumerate(script):
        d=0
        if abs(ang-angle_to_find) < 2.5:
            d = abs(i - current_index)
            if d < bb_min_dist:
                bb_min_dist = d
                actual_distance = i-current_index

    return actual_distance

def sort_files_into_folders(source_folder, script, folder_size=None):

    if folder_size is None:
        folder_size = 40 # Default for FINESSE

    # Derive angle labels from script
    script = [str(x) for x in script if x not in {270,225}]
    n_ang = len(script)

    # List all files in the source folder
    files = [f for f in os.listdir(source_folder) if os.path.isfile(os.path.join(source_folder, f))]
    files.sort()  # Sort files by name (optional, depending on your needs)

    # Calculate the number of folders needed
    num_folders = len(files) // folder_size
    
    # Create and move files into subfolders
    cycle_counter = 0
    for i in range(num_folders):
        angle = script[i % n_ang]
        folder_name = str(int(angle)) + ' ' + str(int(cycle_counter + (i % n_ang)))

        folder_path = os.path.join(source_folder, folder_name)
        os.makedirs(folder_path, exist_ok=True)
        if i % len(script) == len(script)-1:
            cycle_counter += 1
        
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

def analysis_setup(data_folder, out_folder, script):
    mt, mtfea = extract_mean_times(out_folder,script)
    file_pth = glob.glob(data_folder+'PTH'+'*'+'.txt')
    data = cal.load_pth(file_pth[0])
    indices = []
    press = []
    temps = []
    humid = []
    for t in mtfea[0]:
        i = (np.abs(data[0] - t)).argmin()
        press.append(data[1][i])
        temps.append(data[2][i])
        humid.append(data[3][i])

    file_co2 = glob.glob(data_folder+'CO2'+'*'+'.txt')
    print(file_co2)
    data = cal.load_co2(file_co2[0])
    indices = []
    co2 = []
    for t in mtfea[0]:
        i = (np.abs(data[0] - t)).argmin()
        co2.append(data[1][i])
    print("\n*** Use these Parameters for analysis ***\n")
    print("times","=",mtfea[0])
    print("press","=",press)
    print("a_temps","=",temps)
    print("humid","=",humid)
    print("co2","=",co2,"\n")