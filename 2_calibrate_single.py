"""Calibrate individual interferograms to get a spectrum based on only that one"""

import calibration_functions_sanjee as cal
import calibration_functions_chris as cal2
import matplotlib.pyplot as plt
from glob import glob
from pathlib import Path
import numpy as np

# prepare interferograms from 1_prepare_individual_interferograms.py
# creates a folder with each interferogram from a measurement cycle

FINESSE_SCRIPT = [270,225,30,150]

# Location and names of interferograms
INT_LOCATION = "Output Files/20240729_both_shields/prepared_individual_ints/"
# RUN NAME is the string at the end of the folder
RUN_NAME = "Measurement"
GUI_DATA_LOCATION = "Data Files/20240729_both_shields/gui_no_shields.csv"
gui_data = cal.load_gui(GUI_DATA_LOCATION)

# The INDIVIDUAL_SPECTRUM_SAVE_LOCATION will be created if it does not already exist
INDIVIDUAL_SPECTRUM_SAVE_LOCATION = "Output Files/20240729_both_shields/calibrated_spectra/"
Path(INDIVIDUAL_SPECTRUM_SAVE_LOCATION).mkdir(parents=True, exist_ok=True)

# paths to average HBB and CBB
HBB_PATH = "Data Files/20240729_both_shields/"
CBB_PATH = "Data Files/20240729_both_shields/"

SAVE_SPECTRA_PNGS = False

FOLDERS = glob(INT_LOCATION + "*" + RUN_NAME + "/")
FOLDERS.sort()
# FOLDERS = FOLDERS

OPD = 1.21
OUTPUT_FREQUENCY = 0.0605 / OPD
CAL_OFFSET = 0.2  # K
STRETCH_FACTOR = 1.00016  # not sure the origin of these constants; pulled from 3a_calibrate_spectra_in_cycles

# placeholder lists for spectra
ints = []
HBB_temps = []
HBB_std = []
CBB_temps = []
CBB_std = []
angles = []
times = []
wn_calibrateds = []
rad_calibrateds = []
nesr_calibrateds = []
cal_errors = []
folders = []

for i, FOLDER in enumerate(FOLDERS):  # folders 2, 3 are 50º; folders 4, 5 are 130º; change numbers to whatever folder is desired
    print("processing folder " + str(FOLDER))
    # get folder name (to append later)
    slash_positions = [pos for pos, char in enumerate(FOLDER) if char == "/"]
    # Find all interferogram files
    int_list = glob(FOLDER + "*.txt")
    int_list.sort()
    total_ints = len(int_list)

    IS_BB_MEASUREMENT = False

    AVERAGE_SPECTRA = True

    # get HBB and CBB average interferograms (used for calibration)
    hbb_int = None
    cbb_int = None

    if ".txt" in HBB_PATH:
        # Load the defined single HBB and CBBS
        hbb_int, hbb_time, hbb_angle = cal.load_averaged_int(HBB_PATH)
        cbb_int, cbb_time, cbb_angle = cal.load_averaged_int(CBB_PATH)
    
    else:
        # Find where the nearest hot and cold black Body measurements are relative
        # to the current folder
        RAWFOLDERS = glob(HBB_PATH + "*" + RUN_NAME + "/")
        RAWFOLDERS.sort()

        current_index = i % len(FINESSE_SCRIPT)
        hbb_min_distance = 1e10
        cbb_min_distance = 1e10

        if len(FOLDERS) - (i+1) >= len(FINESSE_SCRIPT):
            temp_script = FINESSE_SCRIPT * 2
            hbb_min_distance = cal2.find_dist_to_bb(temp_script,current_index,270)
            cbb_min_distance = cal2.find_dist_to_bb(temp_script,current_index,225)

        else:
            hbb_min_distance = cal2.find_dist_to_bb(FINESSE_SCRIPT,current_index,270)
            cbb_min_distance = cal2.find_dist_to_bb(FINESSE_SCRIPT,current_index,225)

        if hbb_min_distance == 0 or cbb_min_distance == 0:
            IS_BB_MEASUREMENT = True
        
        hbb_int, start_end = cal.average_ints_in_folder(
        RAWFOLDERS[i+hbb_min_distance],
        len_int=57090,
        return_n=False,
        centre_place=False
        )
        cbb_int, start_end = cal.average_ints_in_folder(
        RAWFOLDERS[i+cbb_min_distance],
        len_int=57090,
        return_n=False,
        centre_place=False
        )

        print("HBB and CBB Found!")
            
    if IS_BB_MEASUREMENT is False:
        for i, name in enumerate(int_list):
            # as in 3a_calibrate_spectra_in_cycles, get interferogram data
            if i % 5 == 0:
                print("Loading %i of %i" % (i, total_ints))
            # append relevant folder to folders list
            folders.append(FOLDER[slash_positions[-2]:])
            inter_temp, times_temp, angle_temp = cal2.load_single_int(
                name)  # using the function for averaged interferograms, but passing a single (not averaged) interferogram into it
            HBB_temp, HBB_std_temp = cal.colocate_time_range_gui(
                gui_data, times_temp, "HBB", )
            CBB_temp, CBB_std_temp = cal.colocate_time_range_gui(
                gui_data, times_temp, "CBB", )
            ints.append(inter_temp)
            times.append(times_temp)
            angles.append(angle_temp)
            HBB_temps.append(HBB_temp)
            HBB_std.append(HBB_std_temp)
            CBB_temps.append(CBB_temp)
            CBB_std.append(CBB_std_temp)

            # get spectrum and add to lists (technically not all the lists are not used, but are here for completion)
            wn_calibrated, rad_calibrated, nesr_calibrated, (
                plus_cal_error, minus_cal_error) = cal.calibrate_spectrum_with_cal_error(
                inter_temp, hbb_int, cbb_int, HBB_temp, CBB_temp, HBB_std_temp, CBB_std_temp, fre_interval=OUTPUT_FREQUENCY)
            wn_calibrated *= STRETCH_FACTOR  # apply pre-determined stretch factor
            wn_calibrateds.append(wn_calibrated)
            rad_calibrateds.append(rad_calibrated)
            nesr_calibrateds.append(nesr_calibrated)
            cal_errors.append((plus_cal_error, minus_cal_error))
    else:
        print("Skipping blackbody measurement")

print("Calculating NESR")

# get NESR according to correct method
nesr_calibrateds = cal2.calculate_nesr(wn_calibrateds, rad_calibrateds)

print("Saving Data")

for i, calib in enumerate(wn_calibrateds):
    wn_calibrated = wn_calibrateds[i]
    rad_calibrated = rad_calibrateds[i]
    nesr_calibrated = nesr_calibrateds
    (plus_cal_error, minus_cal_error) = cal_errors[i]
    folder = folders[i]
    # output data (same way as in 3a_calibrate_spectra_in_cycles)
    data_out = np.column_stack(
        (wn_calibrated, rad_calibrated, nesr_calibrated, plus_cal_error, minus_cal_error))

    SCENE_NUMBER = i  # not sure about these, so just leaving them as the counter
    scene_index = i

    header = (
            "Spectrum %i of %i including wn stretch\n\n" % (i + 1, SCENE_NUMBER)
            + "Scene\nStart and end times (seconds since midnight)\n"
            # + "%.3f %.3f\n" % (times[scene_index][0], times[scene_index][1])
            + "Angle %.2f\n\n" % angles[scene_index]
            + "Hot black body\nStart and end times (seconds since midnight)\n"
            # + "%.3f %.3f\n" % (hbb_time[0], hbb_time[1])
            + "Temperature (C)\n%.3f +/- %.3f\n\n"
            % (HBB_temps[i], HBB_std[i])
            + "Cold black body\nStart and end times (seconds since midnight)\n"
            # + "%.3f %.3f\n" % (cbb_time[0], cbb_time[1])
            + "Temperature (C)\n%.3f +/- %.3f\n\n"
            % (CBB_temps[i], CBB_std[i])
            + "Wavenumber (cm-1), Radiance, NESR, "
            + "+ve Calibration error, -ve Calibration error"
    )

    # save each spectrum into a folder corresponding to its original
    SAVE_LOCATION = INDIVIDUAL_SPECTRUM_SAVE_LOCATION
    Path(SAVE_LOCATION).mkdir(parents=True, exist_ok=True)

    np.savetxt(
        SAVE_LOCATION + "%i.txt" % int(times[scene_index][0]),
        data_out,
        header=header,
    )
    if SAVE_SPECTRA_PNGS is True:
        fig, axs = plt.subplots()
        axs.plot(wn_calibrated, rad_calibrated)
        axs.set_ylim(0, 0.2)
        axs.set_xlim(400, 1600)
        axs.set_xlabel('WN (cm$^{-1}$)')
        axs.set_ylabel('Radiance \n (W/m$^2$/sr/cm$^{-1}$)')
        fig.suptitle('Angle ' + str(angles[scene_index]))
        fig.savefig(SAVE_LOCATION + "%i.jpg" % int(times[scene_index][0]), bbox_inches='tight')
        plt.close(fig)
    print("Saved %i of %i" % (i+1, len(wn_calibrateds)+1))

# Sort the spectra back into folders with simple naming based on angle
print("Sorting Folders")
cal2.sort_files_into_folders(INDIVIDUAL_SPECTRUM_SAVE_LOCATION,FINESSE_SCRIPT,40)

# Provide Parameters for easier analysis / RTM later

cal2.analysis_setup(HBB_PATH,INDIVIDUAL_SPECTRUM_SAVE_LOCATION,FINESSE_SCRIPT)

print("\nData Saved \nCallibration Complete!\n")