"""
Tools for use in analysing radiance spectra and retrieving emissivity and temperature
"""
# Setup ===========================================================================

import matplotlib.pyplot as plt
import numpy as np

# Functions =======================================================================

def convert_seconds(seconds):

    """
    Convert a given number of seconds into hours, minutes, and seconds.

    Args:
        seconds (int): The number of seconds to convert.

    Returns:
        tuple: A tuple containing (hours, minutes, seconds).
    """
    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    seconds = seconds % 60
    return hours, minutes, seconds

def duplicate_values(arr, n):
    return [item for item in arr for _ in range(n)]

def calc_differences(dic,keys,times):
    colormap = plt.get_cmap('cool', len(keys))
    wn = np.array(dic.get("WVN"))
    standard = np.array(dic.get(keys[0]))
    for i,key in enumerate(keys[1:]):
        temp_data = np.array(dic.get(key))
        residuals = temp_data-standard
        temp_time = times[i+1]
        h,m,s = convert_seconds(temp_time)

        if m-10 < 0:
            label = str(h)+':0'+str(m)
        else:
            label = str(h)+':'+str(m)
        plt.plot(wn,residuals,color = colormap(i), alpha = 0.8, label = label)
    plt.xlabel(r'Wavenumber ($cm^-1$)')
    plt.ylabel(r'Difference from first spectrum')
    plt.xlim(400,1800)
    plt.ylim(-0.06,0.06)
    plt.legend()
    plt.show()

def plot_spectra_of_type(dic, keys, labels = None, title = None):
    """
    Plot spectra from one header type on single axes.
    (E.g: plots all spectra from "sky140_rad")

    Args:
        dic: The dictionary from which the data will be pulled
        header: The string which defines the type of data (e.g "sky140_rad")
        trials: list of trials (e.g [1,2...5] will plot "sky140_rad 1...5")
    """
    label = None
    
    colormap = plt.get_cmap('cool', len(keys))
    for i,key in enumerate(keys):
        if labels is None:
            label = key
        else:
            time = labels[i]
            h,m,s = convert_seconds(time)
            
            if m-10 < 0:
                label = str(int(h))+':0'+str(int(m))
            else:
                label = str(int(h))+':'+str(int(m))
        
        plt.scatter(dic.get("WVN"),dic.get(key), s=1,color = colormap(i), label = label)
    plt.xlabel(r'Wavenumber ($cm^-1$)')
    plt.ylabel(r'Radiance ($wm^-2{\Omega^-1}$)')
    plt.xlim(400,2000)
    plt.ylim(0,0.2)
    plt.legend()
    if title is not None:
        plt.title(title)
    plt.show()

def planck_function(wn, temp):
    """
    Planck function in terms of wavenumber and temperature.

    Args:
        wn: Wave number in cm^-1
        temp: Temperature of the blackbody in k

    Returns:
        Theoretical radiance
    """
    c1 = 1.1911e-8
    c2 = 1.439
    return (c1 * wn**3) / ((np.exp((c2 * wn) / temp))-1) # Given by Jon Murray

def emissivity_model(wn, up_rad, dw_rad, tsvty, s_temp, a_temp, air_rad=None):
    """
    Calculates the emissivity of a surface being measured
    for a single wavenumber.

    Args:
        wn: The wavenumber of the sample
        up_rad: The Radiance from the sample
        dw_rad: The Radiance from the sky
        tsvty: Transmissivity
        s_temp: Surface temperature
        a_temp: Air Temperature

    Returns:
        Emissivity
    """
    output = None

    # Equation taken from Laura's Thesis.
    if air_rad is None:
        numerator = up_rad - (tsvty**2 * dw_rad) - (1 - tsvty**2) * planck_function(wn, a_temp)
        denom = tsvty * (planck_function(wn, s_temp) - tsvty * dw_rad - (1 - tsvty) * planck_function(wn, a_temp))
        output = numerator / denom

    # Equation taken from Jon's poster.
    else:
        numerator = up_rad - (tsvty**2 * dw_rad) - (1 + tsvty) * air_rad
        denom = tsvty * (planck_function(wn, s_temp) - tsvty * dw_rad) - tsvty * air_rad
        output = numerator/denom    

    return output

def emissivity_retrieval(wn, up_rad, dw_rad, tsvty, s_temp, a_temp, air_rad=None):
    """
    For a range of wavenumber, calculate the emissivities

    Args:
        wn: The wavenumber of the sample
        up_rad: The Radiance from the sample
        dw_rad: The Radiance from the sky
        tsvty: Transmissivity
        s_temp: Surface temperature

    Returns:
        Array of Emissivity
    """
    # Firstly ensure that this code works with both possible types of transmissivity inputs
    if isinstance(tsvty, float) is True: # Check whether a single value or range of values given for Transmissivity
        tsvty = tsvty * np.ones(len(wn))

    if isinstance(s_temp, float) is True: # Check whether a single value or range of values given for Temp
        s_temp = s_temp * np.ones(len(wn))

    # Create an output array to append the values of emissivity to
    
    e_vals = []
    if air_rad is None:
        for i, value in enumerate(wn): # Perform Emissivity Retrieval for each wavenumber
            e_vals.append(emissivity_model(wn[i], up_rad[i], dw_rad[i], tsvty[i], s_temp[i], a_temp))
    else:
        for i, value in enumerate(wn): # Perform Emissivity Retrieval for each wavenumber
            e_vals.append(emissivity_model(wn[i], up_rad[i], dw_rad[i], tsvty[i], s_temp[i], a_temp, air_rad[i]))
    
    return e_vals

def filter_emissivity(wn, transmissivity, emissivity, threshold):
    """
    Filter the emissivity retrieval to only data points where 
    transmissivity is > threshold.

    Args:
        wn: The array of wavenumbers
        transmissivity: The array of tranmissivity
        emissivity: The array of emmissivity
        threshold: The minimum value of tranmissivity

    Returns:
        filtered arrays

    """


    f_wn = []
    f_tr = []
    f_em = []

    for i,t in enumerate(transmissivity):
        if t > threshold:
            f_wn.append(wn[i])
            f_tr.append(transmissivity[i])
            f_em.append(emissivity[i])

    return f_wn, f_tr, f_em

def multiple_emissivity_retrieval(dic, UpKeys, DwKeys, Tsvty, s_temp, a_temps, AirKeys=None, WInd=None):
    """
    A script to run multiple emissivity retrievals for
    a number of spectra

    Args:
        dic: Dictionary of data
        UpKeys: A list of keys which provide the upwelling radiances
        DwKeys: A list of keys which provide the downwelling radiances
        Tsvty: A list of keys which provide the Transmissivity values
        s_temp: A value or list of values for the surface temperature
        a_temps: Vaisala air temperature readings
        WInd: Needed if using temperature retrieval to specify the 
              bounds and number of wavenumber divisions.
        
    Returns:
        wn: The array of wavenumbers (useful if modified when using
            temperature retrieval data)
        emissivity_arrs: A 2d array of all the emissivity values
    """
    n_arrays = len(UpKeys) # The number of spectra being analysed and 
    n_points = len(dic.get(UpKeys[0])) # the number of points per spectra
    
    if WInd is None: # If no specifics for where to analyse given, just analyse everything
        WInd = [0,n_points,n_points]

    # Get all the indices for the data
    start = WInd[0]
    stop = WInd[1]
    div = WInd[2]
    step = int(np.round(((stop - start) / div),0))
    corrected_stop = start+div*step # Needed to account for rounding errors
    
    # If constant t is given, create an array that replaces the s_temp array from temperature retrieval
    reformated_temps = None
    
    if isinstance(s_temp, int):
        s_temp = float(s_temp)
    
    if isinstance(s_temp, float) is True:
        reformated_temps = np.full((n_arrays,n_points),s_temp)

    # Create the temp arrays for each run.
    if hasattr(s_temp,'__len__') is True:
        reformated_temps = []
        if len(s_temp[0]) == 1:
            for row in s_temp:
                expanded_row = []
                for value in row:
                    expanded_row.extend([value] * n_points)
                reformated_temps.append(expanded_row)
        else:
            for row in s_temp:
                expanded_row = []
                for value in row:
                    expanded_row.extend([value] * step)
                reformated_temps.append(expanded_row)
    
    # Since Wavenumber is the same for all, define it here
    wn = dic.get("WVN")[start:corrected_stop]
    emissivity_arrs = []

    # Run through each spectra and do an emissivity retrieval
    for i,v in enumerate(UpKeys):
        temp_uprad = dic.get(UpKeys[i])[start:corrected_stop]
        temp_dwrad = dic.get(DwKeys[i])[start:corrected_stop]
        temp_tsvty = None
        if isinstance(Tsvty,float) is False:
            temp_tsvty = dic.get(Tsvty[i])#[start:corrected_stop]
        else:
            temp_tsvty = [Tsvty] * (corrected_stop - start)
        temp_s_temp = reformated_temps[i]
        temp_a_temp = a_temps[i]
        temp_a_rad = None

        if AirKeys is not None:
            temp_a_rad = dic.get(AirKeys[i])[start:corrected_stop]
        # print(len(temp_a_rad))
        e = emissivity_retrieval(wn, temp_uprad, temp_dwrad, temp_tsvty,
                             temp_s_temp, temp_a_temp, temp_a_rad)
        emissivity_arrs.append(e)

    return wn, emissivity_arrs

def temperature_retrieval(wn, up_rad, dw_rad, tsvty, a_temp, WNs, Temps):
    """
    Uses statistical analysis to retrive surface temperature over range
    of wavenumbers

    Args:
        wn: The wavenumber of the sample
        up_rad: The Radiance from the sample
        dw_rad: The Radiance from the sky
        tsvty: Transmissivity
        WNs: The parameters for the wavenumbers to iterate over [start, stop, num]
        Temps: The parameters for the temperature to iterate over [start, stop, num]

    Returns:
        std_arrays: 2d array of the standard deviation vs temp for each wavenumber subdivision
        std_minima: array of temperature values which minimise standard deviation
        midpoints: the midpoints of each wavenumber subdivision, for easier plotting
    """
    if isinstance(tsvty, float) is True: # Check whether a single value or range of values given for Transmissivity
        tsvty = tsvty * np.ones(len(wn))

    # First search for where the data is within WNstart-WNstop
    WNstart = WNs[0]
    WNstop = WNs[1]
    WNdivisions = WNs[2]

    Tstart = Temps[0]
    Tstop = Temps[1]
    Ttrials = Temps[2]

    WNstartx = np.argmin((np.abs(wn - WNstart))) # Find Start point
    WNstopx = np.argmin((np.abs(wn - WNstop))) # Find End point
    WNstep = int(np.round(((WNstop - WNstart) / WNdivisions),0)) # Find step in wn
    WNstepx = int(np.round(((WNstopx - WNstartx) / WNdivisions),0)) # Find step in index for wn

    WNIndices = [WNstartx,WNstopx,WNdivisions]

    # Run through the temperature range for each subdivision and find the std relation

    std_arrays = []
    std_minima = []
    midpoints = []

    for i in range(WNdivisions):
        Tvals = np.linspace(Tstart, Tstop, Ttrials)
        tp_WN = wn[WNstartx + (i * WNstepx):(WNstartx + (i + 1) * WNstepx)]
        tp_tsvty = tsvty[WNstartx + (i * WNstepx):(WNstartx + (i + 1) * WNstepx)]
        tp_UpRad = up_rad[WNstartx + (i * WNstepx):(WNstartx + (i + 1) * WNstepx)]
        tp_DwRad = dw_rad[WNstartx + (i * WNstepx):(WNstartx + (i + 1) * WNstepx)]

        std_array = []
        for T in Tvals:
            e = emissivity_retrieval(tp_WN, tp_UpRad, tp_DwRad, tp_tsvty, T, a_temp)
            std_array.append(np.std(e))

        std_arrays.append(std_array)
        index = np.argmin(np.abs(std_array - min(std_array)))
        std_minima.append(Tvals[index])
        midpoints.append(WNstart+(i+.5)*WNstep)
        
    return std_arrays, std_minima, midpoints, WNIndices

def polynomial_temp_retrieval(wn, up_rad, dw_rad, tsvty, a_temp, WNs, Temps, order):
    """
    Uses Polynomial fitting methods to retrive surface 
    temperature over range of wavenumbers

    Args:
        wn: The wavenumber of the sample
        up_rad: The Radiance from the sample
        dw_rad: The Radiance from the sky
        tsvty: Transmissivity
        WNs: The parameters for the wavenumbers to iterate over [start, stop, num]
        Temps: The parameters for the temperature to iterate over [start, stop, num]
        order: The order of polynomial to be fitted to the emissivity data

    Returns:
        std_arrays: 2d array of the standard deviation vs temp for each wavenumber subdivision
        std_minima: array of temperature values which minimise standard deviation
        midpoints: the midpoints of each wavenumber subdivision, for easier plotting
    """
    if isinstance(tsvty, float) is True: # Check whether a single value or range of values given for Transmissivity
        tsvty = tsvty * np.ones(len(wn))

    # First search for where the data is within WNstart-WNstop
    WNstart = WNs[0]
    WNstop = WNs[1]
    WNdivisions = WNs[2]

    Tstart = Temps[0]
    Tstop = Temps[1]
    Ttrials = Temps[2]

    WNstartx = np.argmin((np.abs(wn - WNstart))) # Find Start point
    WNstopx = np.argmin((np.abs(wn - WNstop))) # Find End point
    WNstep = int(np.round(((WNstop - WNstart) / WNdivisions),0)) # Find step in wn
    WNstepx = int(np.round(((WNstopx - WNstartx) / WNdivisions),0)) # Find step in index for wn

    WNIndices = [WNstartx,WNstopx,WNdivisions]

    # Run through the temperature range for each subdivision and find the std relation

    fit_errors = []
    minimised_temps = []
    midpoints = []

    for i in range(WNdivisions):
        Tvals = np.linspace(Tstart, Tstop, Ttrials)
        tp_WN = wn[WNstartx + (i * WNstepx):(WNstartx + (i + 1) * WNstepx)]
        tp_tsvty = tsvty[WNstartx + (i * WNstepx):(WNstartx + (i + 1) * WNstepx)]
        tp_UpRad = up_rad[WNstartx + (i * WNstepx):(WNstartx + (i + 1) * WNstepx)]
        tp_DwRad = dw_rad[WNstartx + (i * WNstepx):(WNstartx + (i + 1) * WNstepx)]

        unc_array = []
        for T in Tvals:
            e = emissivity_retrieval(tp_WN, tp_UpRad, tp_DwRad, tp_tsvty, T, a_temp)
            # plt.plot(tp_WN, e)
            # plt.show()
            fit, cov = np.polyfit(tp_WN, e, order, cov='unscaled')
            rms_unc = np.sqrt(abs(np.trace(cov)))
            unc_array.append(rms_unc)

        fit_errors.append(unc_array)
        index = np.argmin(np.abs(unc_array - min(unc_array)))
        minimised_temps.append(Tvals[index])
        midpoints.append(WNstart+(i+.5)*WNstep)
        
    return fit_errors, minimised_temps, midpoints, WNIndices

def multiple_temperature_retrieval(dic, UpKeys, DwKeys, Tsvty, a_temp, WNs, Temps, order=None):
    """
    Perform temperature retrieval for all data given dictionary keys for up and down radiance.

    Args:
        dic: Dictionary of data
        UpKeys: Keys for the upward radiances
        DwKeys: Keys for the downward radiances
        tsvty: Keys for the Transmissivity sims data
        a_temp: Array of air temp values for each spectra being analysed
        WNs: Wavenumber band parameters in the form [start, stop, number of bands]
        Temps: Temperature trial paramaters in the form [start, stop, number]
        order: Optional if wanting to do the polynomial retrieval method instead

    Returns:
        midpoints: The midpoints of each wavenumber band that was analsysed
        WNInd: The indices of that defined the start stop and size of each wn band
        Temp_info: A 2d array containing the Best guess of surface temperature (averaged
                   over wavenumber and spectrum), the temperatures averaged across all
                   spectra for each wavenumber, the temperatures averaged over wavenumber 
                   for each spectra, and the raw array of all retrieved temperatures. These
                   are all useful for different purposes, so are all returned.
    """
    # Create subplots

    fig1, axes1 = plt.subplots(len(UpKeys))
    fig1.suptitle('Standard deviation vs Temp for each wavenumber-band')

    fig2 = plt.figure()
    ax2 = fig2.add_subplot()
    
    WN = dic.get("WVN") # Get the wavenumber array from the dictionary

    temp_arrays = [] # Create empty arrays for data
    spectra_averaged_temps = []
    wn_averaged_temps = []
    if order is None:
        for i, value in enumerate(UpKeys): 
            temp_up_data = dic.get(UpKeys[i]) # Get each up and down radiance from dictionary
            temp_dw_data = dic.get(DwKeys[i])
            temp_tsvty_data = None
            if isinstance(Tsvty,float) is False:
                temp_tsvty_data = dic.get(Tsvty[i])
            else:
                temp_tsvty_data = [Tsvty] * len(temp_up_data)

            std_arrays, std_minima, midpoints, WNInd = temperature_retrieval(WN,  # do temp retrieval
                        temp_up_data, temp_dw_data, temp_tsvty_data, a_temp, WNs, Temps)
        
            for std in std_arrays:
                axes1[i].plot(np.linspace(Temps[0], Temps[1], Temps[2]), std)
        
            wn_averaged_temps.append(np.mean(std_minima))                
            temp_arrays.append(std_minima) # Write data
    else:
        for i, value in enumerate(UpKeys): 
            temp_up_data = dic.get(UpKeys[i]) # Get each up and down radiance from dictionary
            temp_dw_data = dic.get(DwKeys[i])
            temp_tsvty_data = dic.get(Tsvty[i])

            std_arrays, std_minima, midpoints, WNInd = polynomial_temp_retrieval(WN,  # do temp retrieval
                        temp_up_data, temp_dw_data, temp_tsvty_data, a_temp, WNs, Temps, order)
        
            for std in std_arrays:
                axes1[i].plot(np.linspace(Temps[0], Temps[1], Temps[2]), std)
        
            wn_averaged_temps.append(np.mean(std_minima))                
            temp_arrays.append(std_minima) # Write data 

    for i in range(len(temp_arrays[0])): # Run through each of the temperature arrays
        temp_temp = []
        for arr in temp_arrays:
            temp_temp.append(arr[i])

        spectra_averaged_temps.append(np.mean(temp_temp)) # Average the temp for each wn subdivision
    
    ax2.scatter(midpoints, spectra_averaged_temps, color="black", marker='x', label='Temperature for each wn band')
    ax2.set_title("Mean temperatures over wavenumber range")
    ax2.set_xlabel(r"Wavenumber ($cm^-1$)")
    ax2.set_ylabel("Temperature (k)")

    best_guess_temp = np.mean(spectra_averaged_temps)
    ax2.hlines(best_guess_temp,WNs[0],WNs[1], color = 'r', label="Mean retrieved Temperature")
    ax2.hlines(273.15+26.5,WNs[0],WNs[1],label="Max and Min Recorded Temp",color = 'blue')
    ax2.hlines(273.15+29,WNs[0],WNs[1],color = 'blue')
    ax2.legend()

    fig1.savefig("Plots/"+str("Std_vs_Temp "+UpKeys[0]+" "+DwKeys[0]))
    fig2.savefig("Plots/"+str("Mean_Temperatures "+UpKeys[0]+" "+DwKeys[0]))
    plt.show()
    return midpoints, WNInd, [best_guess_temp, spectra_averaged_temps, wn_averaged_temps, temp_arrays] # Return useful stuff

def complete_analysis_1(dic, UPRADS, DWRADS, TRSMSVTY, ATEMPS, TIMES, WNP, TMP, SIMRADS=None, STEMPS=None,save_files=False,location=None):
    
    plot_spectra_of_type(dic,UPRADS, labels=TIMES,title="Upwelling Radiances")
    plot_spectra_of_type(dic,DWRADS, labels=TIMES,title="Downwelling Radiances")

    colormap = plt.get_cmap('cool', len(UPRADS))
    
    mp, Winds, temp_info = multiple_temperature_retrieval(dic,UPRADS,DWRADS,TRSMSVTY,ATEMPS,WNP,TMP)
    wat = temp_info[2]
    modified_wat = []
    for t in wat:
        modified_wat.append([t])

    wn,e = multiple_emissivity_retrieval(dic,UPRADS,DWRADS,TRSMSVTY,modified_wat,ATEMPS,SIMRADS)

    if save_files is True:
        data = [wn]
        for arr in e:
            data.append(e)
        data = np.transpose(data)
        np.savetxt(location+"emissivity_data.txt",data,delimiter='\t')

    for i,arr in enumerate(e):
        colour = colormap(i)
        fw,ft,fe = filter_emissivity(wn,dic.get(TRSMSVTY[i]),arr,0.98)
        h,m,s = convert_seconds(TIMES[i])
        if m-10 < 0:
            label = str(int(h))+':0'+str(int(m))
        else:
            label = str(int(h))+':'+str(int(m))
        plt.scatter(fw,fe,s=0.8, color = colour, alpha = 0.8, label = label)

    plt.xlabel(r'Wavenumber ($cm^-1$)')
    plt.ylabel(r'Emissivity (No Units)')
    plt.title('Emissivity Retrieval Using temperature retrieval')
    plt.legend(fontsize=14)
    plt.xlim(400,1400)
    plt.ylim(0.2,1.2)
    plt.show()

    standard = e[0]
    for i,arr in enumerate(e):
        temp_data = np.array(e[i])
        residuals = temp_data-standard
        temp_time = TIMES[i]
        h,m,s = convert_seconds(temp_time)
        if m-10 < 0:
            label = str(h)+':0'+str(m)
        else:
            label = str(h)+':'+str(m)
        plt.plot(wn,residuals,color = colormap(i), alpha = 0.8, label = label)
    plt.xlabel(r'Wavenumber ($cm^-1$)')
    plt.ylabel(r'Difference from first spectrum')
    #plt.xlim(400,1800)
    #plt.ylim(-0.06,0.06)
    plt.legend()
    plt.show()

    if STEMPS is not None:
        plt.plot(TIMES, STEMPS, color = 'red', label = "Measured Surface Temp")
    plt.plot(TIMES, ATEMPS, color = 'blue', label = "Vaisala Air Temps")
    plt.plot(TIMES, np.array(modified_wat)-273.15, color = 'orange', label = "Retrieved Surface Temp")

    plt.xlabel("Seconds since midnight (s)")
    plt.ylabel("Temperature (deg C)")
    plt.legend()
    plt.show()

# Testing =========================================================================
