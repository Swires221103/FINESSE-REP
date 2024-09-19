"""
Prepare interferograms for use in calibration
"""

from glob import glob # Used to import all files of a particular type
from pathlib import Path # Lets you create folders and paths
from matplotlib import pyplot as plt # For plotting
import numpy as np # For Maths
import calibration_functions_sanjee as cal # Sanjee Callibration functions

# # =======================================================================
# # Location and names of files to combine
# # =======================================================================

INT_LOCATION = "Data Files/20240729_wo_sheet/" # File path for location of Interferograms
RUN_NAME = "Measurement" # String at the end of the folder name to search for
GUI_DATA_LOCATION = INT_LOCATION + "gui_no_shields.csv" # File path for the GUI Data

# The INDIVIDUAL_SAVE_LOCATION will be created if it does not already exist
INDIVIDUAL_SAVE_LOCATION = "Output Files/20240729_wo_sheet/prepared_individual_ints/"
Path(INDIVIDUAL_SAVE_LOCATION).mkdir(parents=True, exist_ok=True) # Create new folder if it doesn't exist

gui_data = cal.load_gui(GUI_DATA_LOCATION)

FOLDERS = glob(INT_LOCATION + "*" + RUN_NAME + "/") # Find all folders that are 'Measurements'
FOLDERS.sort() # Make sure they are sorted chronologically

SAVE_INT_PNGS = False

# Uncomment below if you want to only process certain folders
# FOLDERS=FOLDERS

ints: list = []
n: list = []
centre_place: list = []

for FOLDER in FOLDERS:
    Path(INDIVIDUAL_SAVE_LOCATION+FOLDER[len(INT_LOCATION):]).mkdir(parents=True, exist_ok=True)

    int_temp, start_end_temp, n_temp, centre_place_temp = cal.average_ints_in_folder_return_individuals(
        FOLDER,
        len_int=57090,
        return_n=True,
        centre_place=True
    )
    ints=int_temp

    times = []
    # times.append(start_end_temp)
    print('OFFSETTING TIME BY 5 SECONDS)')
    print(start_end_temp)
    for t in start_end_temp:
        times.append(t-5)
    n=n_temp
    centre_place=centre_place_temp

    angles = []

    for i, interferogram in enumerate(ints):

        gui_index_start = gui_data["time"].sub(times[i]-1).abs().idxmin()
        gui_index_end = gui_data["time"].sub(times[i]+1).abs().idxmin()
        variable = gui_data.loc[gui_index_start:gui_index_end, 'angle']
        angle, angle_std= np.mean(variable), np.std(variable)

        angles=angle

        header = (
            "Interferogram %i of %i\n" % (i + 1, len(ints))
            + "Start and end times (seconds since midnight)\n"
            + "%.1f " % (times[i])
            + "Mirror angle\n%.1f\n" % angles
        )
        print(header)
        np.savetxt(
            INDIVIDUAL_SAVE_LOCATION +FOLDER[len(INT_LOCATION):]+ "int_%.0f.txt" % times[i],
            interferogram,
            header=header,
        )
        if SAVE_INT_PNGS is True:
            cal.update_figure(1)
            fig1, ax1 = plt.subplots(1,1)
            ax1.plot(interferogram)
            ax1.set(
                title=f"Start time: {times[i]:.0f} Angle: {angle}",
                ylim=(-0.15, 0.15),
                xlim=(20000, 37000)
            )
            fig1.savefig(INDIVIDUAL_SAVE_LOCATION +FOLDER[len(INT_LOCATION):]+ "int_%.0f.png" % times[i])
            plt.close(fig1)
        else:
            continue
