from importdata import *
from analysistools import *
import calibration_functions_sanjee as cal
from calibration_functions_chris import analysis_setup
import matplotlib.cm as cm
import math

# # ================================================================
# Using Jon's Data
# # ================================================================

# data_dict = import_sav_file('Data Files/radiances_sims_obs.sav')
# # print(data_dict.keys())

# old_wn = data_dict.get("WVN")

# DWRADS40 = ['sky140_rad_stretched 1', 'sky140_rad_stretched 2', 'sky140_rad_stretched 3', 'sky140_rad_stretched 4', 'sky140_rad_stretched 5']
# UPRADS40 = ['surf40_rad_stretched 1', 'surf40_rad_stretched 2', 'surf40_rad_stretched 3', 'surf40_rad_stretched 4', 'surf40_rad_stretched 5']
# TRSMSVTY40 = ['run40_sims_trans 1', 'run40_sims_trans 2', 'run40_sims_trans 3', 'run40_sims_trans 4', 'run40_sims_trans 5']
# TEMPS40 = [17.33, 17.12, 17.0, 17.78, 18.48]
# SIMRADS40 = ['run40_sims_rad 1', 'run40_sims_rad 2', 'run40_sims_rad 3', 'run40_sims_rad 4', 'run40_sims_rad 5']

# DWRADS50 = ['sky130_rad_stretched 1', 'sky130_rad_stretched 2', 'sky130_rad_stretched 3', 'sky130_rad_stretched 4']
# UPRADS50 = ['surf50_rad_stretched 1', 'surf50_rad_stretched 2', 'surf50_rad_stretched 3', 'surf50_rad_stretched 4']
# TRSMSVTY50 = ['run50_sims_trans 1', 'run50_sims_trans 2', 'run50_sims_trans 3', 'run50_sims_trans 4']
# TEMPS50 = [16.61, 18.57, 17.91, 19.34]
# SIMRADS50 = ['run50_sims_rad 1', 'run50_sims_rad 2', 'run50_sims_rad 3', 'run50_sims_rad 4']

# Wavenumber_params = [730, 980, 20]
# Temp_params = [297, 337, 50]

# colors40 = ['salmon','lightcoral', 'indianred', 'firebrick', 'red']
# colors50 = ['skyblue', 'steelblue', 'dodgerblue', 'blue']

# def compare_angles_temp_retrieval():

#     temp_arr_40 = []
#     temp_arr_50 = []

#     for i, vals in enumerate(DWRADS40):
#         temp_dr = data_dict.get(DWRADS40[i])
#         temp_ur = data_dict.get(UPRADS40[i])
#         temp_trsmsvty = data_dict.get(TRSMSVTY40[i])
#         arrays = temperature_retrieval(wn, temp_ur, temp_dr, temp_trsmsvty, TEMPS40[i], Wavenumber_params, Temp_params)
#         temp_arr_40.append(arrays[1])
#         plt.scatter(arrays[2],arrays[1], marker = 'x', color = colors40[i])

#     for i, vals in enumerate(DWRADS50):
#         temp_dr = data_dict.get(DWRADS50[i])
#         temp_ur = data_dict.get(UPRADS50[i])
#         temp_trsmsvty = data_dict.get(TRSMSVTY50[i])
#         arrays = temperature_retrieval(wn, temp_ur, temp_dr, temp_trsmsvty, TEMPS50[i], Wavenumber_params, Temp_params)
#         temp_arr_50.append(arrays[1])
#         plt.scatter(arrays[2],arrays[1], marker = '.', color = colors50[i])

#     plt.xlabel(r'Wavenumber ($cm^-1$)')
#     plt.ylabel(r'Temperature ($\degree$C)')
#     plt.title('Temperature retrieval against wavenumber for 40 (red) and 50 (blue) degrees')
#     plt.savefig('Plots/Sing_Temp_retrieval_angle_comparison.png')
#     plt.show()

#     multiple_temperature_retrieval(data_dict,UPRADS40,DWRADS40,TRSMSVTY40,TEMPS40,Wavenumber_params,Temp_params)
#     multiple_temperature_retrieval(data_dict,UPRADS50,DWRADS50,TRSMSVTY50,TEMPS50,Wavenumber_params,Temp_params)

#     plt.show()

# def emis_with_temps():
#     mp,t_arrs,mt,Winds = multiple_temperature_retrieval(data_dict, UPRADS50, DWRADS50, TRSMSVTY50, TEMPS50, Wavenumber_params, Temp_params)
#     plt.show()
#     print("mp:",mp)
#     print("t_arrs:",t_arrs)
#     print("mt:",mt)
#     print("Winds:",Winds)
#     # wout_temps = multiple_emissivity_retrieval(data_dict, UPRADS50, DWRADS50, TRSMSVTY50, 303., TEMPS50)
#     # w_temps = multiple_emissivity_retrieval(data_dict, UPRADS50, DWRADS50, TRSMSVTY50, t_arrs, TEMPS50, Winds)

#     # for i,vals in enumerate(wout_temps):
#     #     filtwn, filttr, filtem = filter_emissivity(wn, data_dict.get(TRSMSVTY50[i]),vals,0.9)
#     #     print(len(filtwn))
#     #     plt.scatter(filtwn,filtem, marker = '.', s=1)
#     # plt.xlabel(r"Wavenumber ($cm^-1$)")
#     # plt.ylabel(r"Emissivity (No Units)")
#     # plt.xlim(400,1400)
#     # plt.ylim(0,1.2)
#     # plt.show()

#     # for i,vals in enumerate(w_temps):
#     #     filtwn, filttr, filtem = filter_emissivity(wn, data_dict.get(TRSMSVTY50[i]),vals,0.9)
#     #     plt.scatter(filtwn,filtem, marker = '.', s=1)
#     # plt.xlabel(r"Wavenumber ($cm^-1$)")
#     # plt.ylabel(r"Emissivity (No Units)")
#     # plt.xlim(400,1400)
#     # plt.ylim(0,1.2)
#     # plt.show()

# # plot_spectra_of_type(data_dict, UPRADS40)
# # plot_spectra_of_type(data_dict,DWRADS40)

# mp, Winds, temp_info = multiple_temperature_retrieval(data_dict,UPRADS40,DWRADS40,TRSMSVTY40,TEMPS40,Wavenumber_params,Temp_params)

# bgt = temp_info[0]
# sat = temp_info[1]
# wat = temp_info[2]
# raw_t = temp_info[3]

# modified_wat = []
# for t in wat:
#     modified_wat.append([t])

# wn,e = multiple_emissivity_retrieval(data_dict,UPRADS40,DWRADS40,TRSMSVTY40,modified_wat,TEMPS40,AirKeys=SIMRADS40)
# data_out = np.vstack([wn, e]).T
# np.savetxt(r"C:\Users\jacks\FINESSE Scripts\Output Files\Wandle_Retrieval.txt", data_out, delimiter='\t', fmt='%s')
# fes = []
# fw = None
# for i,arr in enumerate(e):
#     fw,ft,fe = filter_emissivity(wn,data_dict.get(TRSMSVTY40[i]),arr,0.98)
#     plt.scatter(fw,fe, s = 0.9, label = str(i+1))



# plt.xlabel(r'Wavenumber ($cm^-1$)')
# plt.ylabel(r'Emissivity (No Units)')
# plt.title('Emissivity Retrieval Using temperature retrieval')
# plt.legend()
# plt.xlim(400,1400)
# plt.ylim(0.6,1.2)
# plt.show()

# # ================================================================
# Wet Sand analysis
# # ================================================================
# def convert_seconds(seconds):
#     """
#     Convert a given number of seconds into hours, minutes, and seconds.

#     Args:
#         seconds (int): The number of seconds to convert.

#     Returns:
#         tuple: A tuple containing (hours, minutes, seconds).
#     """
#     hours = seconds // 3600
#     minutes = (seconds % 3600) // 60
#     seconds = seconds % 60
#     return hours, minutes, seconds

# def duplicate_values(arr, n):
#     return [item for item in arr for _ in range(n)]

# def calc_differences(dic,keys,times):
#     colormap = plt.get_cmap('cool', len(keys))
#     wn = np.array(dic.get("WVN"))
#     standard = np.array(dic.get(keys[0]))
#     for i,key in enumerate(keys[1:]):
#         temp_data = np.array(dic.get(key))
#         residuals = temp_data-standard
#         temp_time = times[i+1]
#         h,m,s = convert_seconds(temp_time)

#         if m-10 < 0:
#             label = str(h)+':0'+str(m)
#         else:
#             label = str(h)+':'+str(m)
#         plt.plot(wn,residuals,color = colormap(i), alpha = 0.8, label = label)
#     plt.xlabel(r'Wavenumber ($cm^-1$)')
#     plt.ylabel(r'Difference from first spectrum')
#     plt.xlim(400,1800)
#     plt.ylim(-0.06,0.06)
#     plt.legend()
#     plt.show()

# FINESSE_SCRIPT = [270,225,30]
# # average_spectra_and_save(r'C:\Users\jacks\FINESSE Scripts\Output Files\Wet_Sand_exp\calibrated_spectra/',FINESSE_SCRIPT)

# data_dict = import_spectra(r'C:\Users\jacks\FINESSE Scripts\Output Files\Wet_Sand_exp\calibrated_spectra\averaged_spectra_30/')
# data_dict = import_spectra(r'C:\Users\jacks\FINESSE Scripts\Output Files\Wet_Sand_exp\calibrated_spectra\averaged_spectra_150/',dic=data_dict)

# print(data_dict.keys())

# times30 = [54120.975, 54377.0, 54633.325, 54889.625, 55146.0, 55402.325, 55658.5, 55915.0, 56171.325, 56427.7, 56684.0, 56940.1]
# times150 = [54185.0, 54441.075, 54697.5, 54953.9, 55210.025, 55466.5, 55722.725, 55979.0, 56235.5, 56491.725, 56748.0, 57004.2]
# a_temps = [32.64,32.41,32.85,32.23,32.85,32.54,33.49,35.79,37.0,36.44,36.72,35.48]
# flir_temps = [28.8,26.2,30.6,33.5,35.4,35.4,37.5,37.5,42.2,42.2,45.3,46.0]
# peaks = [0.157,0.163,0.170,0.172,0.180,0.185,0.155,0.165,0.170,0.175,0.185,0.187]

# UPRADS = ['30 Degrees1 1', '30 Degrees2 2', '30 Degrees3 3', '30 Degrees4 4', '30 Degrees5 5', '30 Degrees6 6', '30 Degrees7 7', '30 Degrees8 8', '30 Degrees9 9', '30 Degrees10 10', '30 Degrees11 11', '30 Degrees12 12']
# DWRADS = ['150 Degrees1 1', '150 Degrees2 2', '150 Degrees3 3', '150 Degrees4 4', '150 Degrees5 5', '150 Degrees6 6', '150 Degrees7 7', '150 Degrees8 8', '150 Degrees9 9', '150 Degrees10 10', '150 Degrees11 11', '150 Degrees12 12']
# # # TRSMSVTY30 = ['run40_sims_trans 1'] * len(UPRADS30)
# # # TRSMSVTY20 = ['run40_sims_trans 1'] * len(UPRADS20)
# # plot_spectra_of_type(data_dict,UPRADS, labels=times30)
# # plot_spectra_of_type(data_dict,DWRADS, labels=times150)

# # calc_differences(data_dict,UPRADS,times30)
# def reformat_list(lst):
#     half = len(lst) // 2
#     reformatted_list = []
#     for i in range(half):
#         reformatted_list.append(lst[i])
#         reformatted_list.append(lst[i + half])
#     return reformatted_list

# def plot_spectra_and_bb(dic, keys, temps, times):
#     colormap = plt.get_cmap('cool', len(keys))
#     fig, axs = plt.subplots(2, int(np.round(len(keys)/2,0)),sharex=True,sharey=True)
#     effective_temps = []

#     for i in range(2):
#         for j in range(int(np.round(len(keys)/2,0))):
#             current_index = i + (j * 2)
#             temp_wn = dic.get("WVN")
#             temp_data = dic.get(keys[current_index])
#             planck_data =planck_function(temp_wn,(temps[current_index]+273.15)) 
#             planck_data_ub = planck_function(temp_wn,(temps[current_index]+273.15+0.7))
#             planck_data_lb = planck_function(temp_wn,(temps[current_index]+273.150-0.7))
#             current_time = times[current_index] 
#             h,m,s = convert_seconds(current_time)

#             start_temp = temps[current_index]+273.15
#             eff_temp = 0
#             best_so_far = 1e10
#             for t in np.linspace(start_temp,start_temp+5,100):
#                 target = peaks[current_index]
#                 # target = [x for x in target if not math.isnan(x)]
#                 peak = np.max(planck_function(temp_wn,t)[1:])
#                 res = abs(target - peak)
#                 if res < best_so_far:
#                     eff_temp = t -273.15
#             effective_temps.append(eff_temp)

#             axs[i,j].plot(temp_wn,planck_data,color = 'black')
#             axs[i,j].fill_between(temp_wn,planck_data_ub,planck_data_lb,color = 'red')
#             axs[i,j].scatter(temp_wn,temp_data,s=0.5,color=colormap(current_index))
#             if m > 9:
#                 axs[i,j].set_title(str(int(h))+':'+str(int(m)))
#             else:
#                 axs[i,j].set_title(str(int(h))+':0'+str(int(m)))
#             axs[i,j].set_xlim(400,2000)
#             axs[i,j].set_ylim(0,0.2)

#     plt.show()
#     return effective_temps

# ef_temps = plot_spectra_and_bb(data_dict,UPRADS,flir_temps,times30)
# ef_temps = reformat_list(ef_temps)
# plt.plot(times30,ef_temps,color = 'black',linestyle = '--',label = "Effective")
# plt.plot(times30,flir_temps, color = 'black', label = "Measured")
# plt.ylabel("Temperature (k)")
# plt.xlabel("Time (Seconds since midnight)")
# plt.legend()
# plt.show()

# plt.plot(times30,np.array(ef_temps)-np.array(flir_temps))
# plt.ylabel("Temperature Difference (k)")
# plt.xlabel("Time (Seconds since midnight)")
# plt.show()

# plot_spectra_and_bb(data_dict,UPRADS,ef_temps,times30)

# plot_spectra_and_bb(data_dict,DWRADS,a_temps,times150)

# start = 6000
# stop = 40000
# div = stop-start

# new_trans = data_dict.get(TRSMSVTY20[0])
# new_trans = duplicate_values(new_trans,4)
# data_dict["New_Trans"] = new_trans

# NEWTRSMSVTY30 = ["New_Trans"] * len(UPRADS30)
# NEWTRSMSVTY20 = ["New_Trans"] * len(UPRADS20)
# print(len(new_trans))


# # #55560
# rough_times = [53940,54300,54540,54720,55020,55560,56280,56520,56820,57060]
# rough_tims30 = [53940,54300,54540,54720,55020,56280,56520,56820,57060]
# rough_times20 = [53940,54300,54540,54720,55560,56280,56520,57060]
# rough_surf_t = [28.8,26.2,30.6,33.5,35.4,37.5,42.2,44.2,45.3,46.0]

# # plot_spectra_of_type(data_dict,UPRADS20,rough_times20,"Upwelling 20 degrees")
# # plot_spectra_of_type(data_dict,DWRADS20,rough_times20,"Downwelling 20 degrees")
# # plot_spectra_of_type(data_dict,UPRADS30,rough_times30,"Upwelling 30 degrees")
# # plot_spectra_of_type(data_dict,DWRADS30,rough_times30,"Downwelling 30 degrees")

# # calc_differences(data_dict,UPRADS20,rough_times20)
# # calc_differences(data_dict,UPRADS30,rough_times30)

# pth = cal.load_pth(r"C:\Users\jacks\FINESSE Scripts\Data Files\Wet sand experiment\PTH_1554.txt")
# pth_times = np.array(pth[0])
# pth_temps = np.array(pth[2])

# wavenumber_params = [730, 980, 20]
# temp_params = [297, 337, 50]

# A_Temps20 = []
# A_Temps30 = []

# for t in rough_times20:
#     index_of_closest_value = np.argmin(np.abs(pth_times - t))
#     A_Temps20.append(pth_temps[index_of_closest_value])

# for t in rough_times30:
#     index_of_closest_value = np.argmin(np.abs(pth_times - t))
#     A_Temps30.append(pth_temps[index_of_closest_value])

# print("Loaded All Data")

# mp, Winds, temp_info = multiple_temperature_retrieval(data_dict,UPRADS30,DWRADS30,1.,A_Temps30,wavenumber_params,temp_params)
# wat30 = temp_info[2]
# modified_wat30 = []
# for t in wat30:
#     modified_wat30.append([t])

# wn,e = multiple_emissivity_retrieval(data_dict,UPRADS30,DWRADS30,NEWTRSMSVTY30,modified_wat30,A_Temps30,WInd=[start,stop,div])

# for i,arr in enumerate(e):
#     fw,ft,fe = filter_emissivity(wn,data_dict.get(NEWTRSMSVTY30[i]),arr,0.99)
#     colour = colormap(i)
#     h,m,s = convert_seconds(rough_times30[i])
#     plt.scatter(wn,arr,s=0.8, color = colour, alpha = 0.8, label = str(h)+':'+str(m))

# plt.xlabel(r'Wavenumber ($cm^-1$)')
# plt.ylabel(r'Emissivity (No Units)')
# plt.title('Emissivity Retrieval Using temperature retrieval - 30 deg')
# plt.legend(fontsize=14, scatterpoints=1)
# plt.xlim(400,1400)
# plt.ylim(0.2,1.2)
# plt.show()


# mp, Winds, temp_info = multiple_temperature_retrieval(data_dict,UPRADS20,DWRADS20,1.,A_Temps20,wavenumber_params,temp_params)
# wat20 = temp_info[2]
# modified_wat20 = []
# for t in wat20:
#     modified_wat20.append([t])

# wn,e = multiple_emissivity_retrieval(data_dict,UPRADS20,DWRADS20,1.,modified_wat20,A_Temps20)
# # e = np.array(e)
# # mean_es = np.mean(e, axis=0)
# # print(len(mean_es),len(wn))
# # plt.scatter(wn,mean_es, s = 1)

# for i,arr in enumerate(e):
#     # tcolor = ((i*20)/300,0,0)
#     colour = colormap(i)
#     fw,ft,fe = filter_emissivity(wn,data_dict.get(TRSMSVTY20[i]),arr,0.98)
#     h,m,s = convert_seconds(rough_times20[i])
#     plt.scatter(fw,fe,s=0.8, color = colour, alpha = 0.8, label = str(h)+':'+str(m))

# plt.xlabel(r'Wavenumber ($cm^-1$)')
# plt.ylabel(r'Emissivity (No Units)')
# plt.title('Emissivity Retrieval Using temperature retrieval - 20 deg')
# plt.legend(fontsize=14)
# plt.xlim(400,1400)
# plt.ylim(0.2,1.2)
# plt.show()

# plt.plot(rough_times20,np.array(wat20)-273.15,color='red',linestyle=':',label='Retrieved Surface Temperatures - 20 deg')
# plt.plot(rough_times30,np.array(wat30)-273.15,color='red',linestyle='-',label='Retrieved Surface Temperatures - 30 deg')
# plt.plot(rough_times,rough_surf_t,color='orange',label='Recorded Surface Temperature (Flir)')
# plt.plot(rough_times30,A_Temps30,color='blue',label='Vaisala Air Temperature')

# plt.legend(fontsize=14)
# plt.xlabel('Seconds since Midnight (s)')
# plt.ylabel('Temperature (celsius)')
# plt.show()

# # ================================================================
# Wet Sand analysis 2
# # ================================================================
# FINESSE_SCRIPT = [270,225,150]
# # average_spectra_and_save(r'C:\Users\jacks\FINESSE Scripts\Output Files\wet_wandle\calibrated_spectra/',FINESSE_SCRIPT,'30 ')

# data_dict = import_sav_file(r"Data Files/20240729_wet_wandle_2/wet_wandle_17cycle_simulations.sav")
# data_dict = import_spectra(r'C:\Users\jacks\FINESSE Scripts\Output Files\wet_wandle\calibrated_spectra\averaged_spectra_30/',dic=data_dict,indices=[6000,40000])
# data_dict = import_spectra(r'C:\Users\jacks\FINESSE Scripts\Output Files\wet_wandle\calibrated_spectra\averaged_spectra_150/',dic=data_dict,indices=[6000,40000])

# print(data_dict.keys())

# times30 = [70956.5, 71212.5, 71468.5, 71724.4, 71980.5, 72236.325, 72492.4, 72748.45, 73004.5, 73260.5, 73516.5, 73772.5, 74028.5, 74284.5, 74540.4, 74796.375, 75052.5]
# times150 =  [71020.5, 71276.325, 71532.475, 71788.425, 72044.425, 72300.35, 72556.25, 72812.5, 73068.425, 73324.5, 73580.425, 73836.5, 74092.475, 74348.375, 74604.35, 74860.5, 75116.5]
# a_temps = [26.5,26.03,25.82,25.95,25.51,25.62,25.32,25.25,25.12,24.8,24.69,24.79,24.54,24.49,24.17,23.91,23.8] 
# press = [1013.78,1013.72,1013.65,1013.7,1013.69,1013.77,1013.76,1013.78,1013.84,1013.89,1013.95,1014.0,1014.01,1014.01,1014.06,1014.13,1014.05]
# humid = [43.08,43.9,44.64,44.39,45.31,45.50,46.02,46.46,46.92,47.50,48.15,47.79,48.28,48.37,48.67,49.27,50.41]
# co2 = [449.5,448.8,448.5,446.7,446.3,449.8,445.1,447.0,448.0,450.0,449.9,452.9,450.2,542.1,451.5,447.3,450.0]
# flir_temps = [31,29.3,28,27,28,29.3,29.8,29.9,32,33,35.1,38.2,38.2,38.3,39.4,38.3,39]


# UPRADS = ['30 Degrees1', '30 Degrees2', '30 Degrees3', '30 Degrees4', '30 Degrees5', '30 Degrees6', '30 Degrees7', '30 Degrees8', '30 Degrees9', '30 Degrees10', '30 Degrees11', '30 Degrees12', '30 Degrees13', '30 Degrees14', '30 Degrees15', '30 Degrees16', '30 Degrees17']
# DWRADS = ['150 Degrees1', '150 Degrees2', '150 Degrees3', '150 Degrees4', '150 Degrees5', '150 Degrees6', '150 Degrees7', '150 Degrees8', '150 Degrees9', '150 Degrees10', '150 Degrees11', '150 Degrees12', '150 Degrees13', '150 Degrees14', '150 Degrees15', '150 Degrees16', '150 Degrees17']
# SIMRADS = ['run20240729_30_sims_rad 1', 'run20240729_30_sims_rad 2', 'run20240729_30_sims_rad 3', 'run20240729_30_sims_rad 4', 'run20240729_30_sims_rad 5', 'run20240729_30_sims_rad 6', 'run20240729_30_sims_rad 7', 'run20240729_30_sims_rad 8', 'run20240729_30_sims_rad 9', 'run20240729_30_sims_rad 10', 'run20240729_30_sims_rad 11', 'run20240729_30_sims_rad 12', 'run20240729_30_sims_rad 13', 'run20240729_30_sims_rad 14', 'run20240729_30_sims_rad 15', 'run20240729_30_sims_rad 16', 'run20240729_30_sims_rad 17']
# TRSMSVTY = ['run20240729_30_sims_trans 1', 'run20240729_30_sims_trans 2', 'run20240729_30_sims_trans 3', 'run20240729_30_sims_trans 4', 'run20240729_30_sims_trans 5', 'run20240729_30_sims_trans 6', 'run20240729_30_sims_trans 7', 'run20240729_30_sims_trans 8', 'run20240729_30_sims_trans 9', 'run20240729_30_sims_trans 10', 'run20240729_30_sims_trans 11', 'run20240729_30_sims_trans 12', 'run20240729_30_sims_trans 13', 'run20240729_30_sims_trans 14', 'run20240729_30_sims_trans 15', 'run20240729_30_sims_trans 16', 'run20240729_30_sims_trans 17']

# wavenumber_params = [730, 980, 20]
# temp_params = [297, 337, 50]
# colormap = plt.get_cmap('cool', len(UPRADS))

# plot_spectra_of_type(data_dict,UPRADS, labels=times30)
# plot_spectra_of_type(data_dict,DWRADS, labels=times150)

# mp, Winds, temp_info = multiple_temperature_retrieval(data_dict,UPRADS,DWRADS,TRSMSVTY,a_temps,wavenumber_params,temp_params)
# wat20 = temp_info[2]
# modified_wat20 = []
# for t in wat20:
#     modified_wat20.append([t])

# wn,e = multiple_emissivity_retrieval(data_dict,UPRADS,DWRADS,TRSMSVTY,modified_wat20,a_temps,SIMRADS)
# # e = np.array(e)
# # mean_es = np.mean(e, axis=0)
# # print(len(mean_es),len(wn))
# # plt.scatter(wn,mean_es, s = 1)

# for i,arr in enumerate(e):
#     # tcolor = ((i*20)/300,0,0)
#     colour = colormap(i)
#     fw,ft,fe = filter_emissivity(wn,data_dict.get(TRSMSVTY[i]),arr,0.98)
#     h,m,s = convert_seconds(times30[i])
#     plt.scatter(fw,fe,s=0.8, color = colour, alpha = 0.8, label = str(int(h))+':'+str(int(m)))

# plt.xlabel(r'Wavenumber ($cm^-1$)')
# plt.ylabel(r'Emissivity (No Units)')
# plt.title('Emissivity Retrieval Using temperature retrieval - 30 deg wet wandle sand')
# plt.legend(fontsize=14)
# plt.xlim(400,1400)
# plt.ylim(0.2,1.2)
# plt.show()

# standard = e[0]
# for i,arr in enumerate(e):
#     temp_data = np.array(e[i])
#     residuals = temp_data-standard
#     temp_time = times[i]
#     h,m,s = convert_seconds(temp_time)
#     if m-10 < 0:
#         label = str(h)+':0'+str(m)
#     else:
#         label = str(h)+':'+str(m)
#     plt.plot(wn,residuals,color = colormap(i), alpha = 0.8, label = label)
# plt.xlabel(r'Wavenumber ($cm^-1$)')
# plt.ylabel(r'Difference from first spectrum')
# #plt.xlim(400,1800)
# #plt.ylim(-0.06,0.06)
# plt.legend()
# plt.show()

# plt.plot(times30, flir_temps, color = 'red', label = "Measured Surface Temp")
# plt.plot(times30, a_temps, color = 'blue', label = "Vaisala Air Temps")
# plt.plot(times30, np.array(modified_wat20)-273.15, color = 'orange', label = "Retrieved Surface Temp")

# plt.xlabel("Seconds since midnight (s)")
# plt.ylabel("Temperature (deg C)")
# plt.legend()
# plt.show()

# def reformat_list(lst):
#     half = len(lst) // 2
#     reformatted_list = []
#     for i in range(half):
#         reformatted_list.append(lst[i])
#         reformatted_list.append(lst[i + half])
#     return reformatted_list

# # ================================================================
# Building Shielding analysis
# # ================================================================
FINESSE_SCRIPT = [270,225,30,150]

wavenumber_params = [730, 980, 20]
temp_params = [297, 337, 50]

analysis_setup("Data Files/20240729_wandle_60/","Output Files/20240729_wandle_60/calibrated_spectra/",FINESSE_SCRIPT)
average_spectra_and_save(r"Output Files\20240729_wandle_60\calibrated_spectra/",FINESSE_SCRIPT)

times = [46745.5, 47001.5, 47257.5, 47513.5, 47769.5, 48025.5, 48281.65, 48537.85, 48793.725, 49049.775, 49305.825]
press = [1016.35, 1016.55, 1016.49, 1016.4, 1016.39, 1016.38, 1016.4, 1016.4, 1016.31, 1016.27, 1016.34]
a_temps = [26.65, 26.71, 27.02, 27.09, 27.17, 27.07, 27.3, 27.24, 26.96, 27.49, 27.62]
humid = [35.18, 35.79, 35.18, 35.75, 34.62, 34.39, 34.25, 34.12, 34.26, 32.53, 31.92]
co2 = [441.5, 446.0, 445.3, 444.4, 440.2, 442.6, 443.6, 451.3, 443.7, 443.5, 443.4]

# data_dict = import_sav_file(r"Data Files/20240729_wet_wandle_2/wet_wandle_17cycle_simulations.sav")
# data_dict = import_spectra(r"Output Files\20240729_wo_bld_shield\calibrated_spectra/averaged_spectra/",indices=[6000,40000],dic=data_dict)

# # print(data_dict.keys())

# UPRADS = ['30 Degrees 1', '30 Degrees 2', '30 Degrees 3', '30 Degrees 4', '30 Degrees 5']
# DWRADS = ['150 Degrees 1', '150 Degrees 2', '150 Degrees 3', '150 Degrees 4', '150 Degrees 5']
# TRSMSVTY = ['run20240729_30_sims_trans 1'] * len(UPRADS)
# SIMRADS = ['run20240729_30_sims_rad 1'] * len(UPRADS)

# complete_analysis_1(data_dict,UPRADS,DWRADS,TRSMSVTY,a_temps,times, wavenumber_params,temp_params,SIMRADS)