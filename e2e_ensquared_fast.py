"""

[FAST MODE] Ensquared Energy Analysis


Author: Alvaro Menduina
Date: July 2020
Latest version: December 2020

Description:
With this script we calculate the Ensquared Energy at the detector plane
for the E2E files

This is a FAST MODE version of the e2e_ensquared.py script.
Instead of looping over all wavelengths and configurations,
we loop through each configuration and generate a single instance
of the Raytrace for all wavelengths

[!] Updated in Dec 2020 to work for both the Nominal Design and the Monte Carlos

"""

import json
import os
import numpy as np
import e2e_analysis as e2e
import matplotlib.pyplot as plt
import matplotlib.tri as tri


def detector_ensquared_energy(zosapi, file_options, N_rays, box_size, files_path, results_path):
    """
    Calculate the Ensquared Energy at the detector plane for all 4 IFU channels
    The Ensquared Energy is calculated for the Field Point at the centre of the slice
    for all slices and all wavelengths

    The results are shown at the DETECTOR Plane for all 4 IFU channels

    :param zosapi: Zemax API
    :param sys_mode: [str] system mode, either 'HARMONI', 'ELT' or 'IFS'
    :param ao_modes: [list] containing the AO mode we want to analyze (if running 'HARMONI' system mode')
    :param spaxel_scale: [str] the spaxel scale, ex. '60x30'
    :param grating: [str] spectral band
    :param N_rays: how many rays to trace in the Pupil to calculate the Ensquared Energy (typically 500 - 1000)
    :param box_size: size in [spaxels] for the EE (default is 2 spaxels, HARMONI requirement)
    :param files_path: path to the E2E files we will load for the analysis
    :param results_path: path where we want to save the results
    :return:
    """

    spaxel_scale = file_options['SPAX_SCALE']
    grating = file_options['GRATING']
    monte_carlo = file_options['MONTE_CARLO']
    ao_mode = file_options['AO_MODE']

    # We will create a separate folder within results_path to save the results
    analysis_dir = os.path.join(results_path, 'ENSQUARED ENERGY')
    print("Analysis Results will be saved in folder: ", analysis_dir)
    if not os.path.exists(analysis_dir):
        os.mkdir(analysis_dir)

    analysis = e2e.EnsquaredEnergyFastAnalysis(zosapi=zosapi)       # The analysis object

    # For speed, we only use a fraction of the available wavelengths
    # at least, 3 wavelengths (min, central, max), typically 5 or 7 to get decent sampling at the detector plane
    waves = np.linspace(1, 23, 7).astype(int)

    ifu_sections = ['AB', 'CD', 'EF', 'GH']
    # ifu_sections = ['AB', 'CD']
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    for i, ifu in enumerate(ifu_sections):      # Loop over the IFU channels

        # change the IFU path option
        file_options['IFU_PATH'] = ifu

        if monte_carlo is True:
            # read the MC instance number of the specific IFU path
            ifu_mc = file_options['IFU_PATHS_MC'][ifu]
            file_options['IFU_MC'] = ifu_mc

        list_results = analysis.loop_over_files(files_dir=files_path, files_opt=file_options, results_path=results_path,
                                                wavelength_idx=waves, configuration_idx=None, N_rays=N_rays,
                                                box_size=box_size, monte_carlo=monte_carlo)

        # No Monte Carlo so the list_results only contains 1 entry, for the IFU channel
        energy, obj_xy, slicer_xy, detector_xy, wavelengths = list_results[0]

        # Separate the Odd / Even configurations to avoid triangulating over the gap on the detector plane
        ener_ogg, ener_even = energy[::2].flatten(), energy[1::2].flatten()
        x, y = detector_xy[:, :, 0].flatten(), detector_xy[:, :, 1].flatten()
        x_odd, y_odd = detector_xy[::2, :, 0].flatten(), detector_xy[::2, :, 1].flatten()
        x_even, y_even = detector_xy[1::2, :, 0].flatten(), detector_xy[1::2, :, 1].flatten()
        triang_odd = tri.Triangulation(x_odd, y_odd)
        triang_even = tri.Triangulation(x_even, y_even)

        # Remove the flat triangles at the detector edges that appear because of the field curvature
        min_circle_ratio = .05
        mask_odd = tri.TriAnalyzer(triang_odd).get_flat_tri_mask(min_circle_ratio)
        triang_odd.set_mask(mask_odd)
        mask_even = tri.TriAnalyzer(triang_even).get_flat_tri_mask(min_circle_ratio)
        triang_even.set_mask(mask_even)

        min_ener, max_ener = np.min(energy), np.max(energy)

        ax = axes.flatten()[i]
        tpc_odd = ax.tripcolor(triang_odd, ener_ogg, shading='flat', cmap='Blues', vmin=min_ener, vmax=1.0)
        tpc_odd.set_clim(vmin=min_ener, vmax=1.0)
        tpc_even = ax.tripcolor(triang_even, ener_even, shading='flat', cmap='Blues', vmin=min_ener, vmax=1.0)
        tpc_even.set_clim(vmin=min_ener, vmax=1.0)
        ax.scatter(x, y, s=2, color='black')
        axis_label = 'Detector'
        ax.set_xlabel(axis_label + r' X [mm]')
        ax.set_ylabel(axis_label + r' Y [mm]')
        ax.set_aspect('equal')
        plt.colorbar(tpc_odd, ax=ax, orientation='horizontal')
        title = r'IFU-%s | %s | %s | %s | EE min:%.2f max:%.2f' % (ifu, spaxel_scale, grating, ao_mode, min_ener, max_ener)
        ax.set_title(title)

    save_path = os.path.join(results_path, analysis_dir)
    fig_name = "ENSQ_ENERGY_DETECTOR_%s_SPEC_%s_MODE_%s" % (spaxel_scale, grating, ao_mode)
    if os.path.isfile(os.path.join(save_path, fig_name)):
        os.remove(os.path.join(save_path, fig_name))
    fig.savefig(os.path.join(save_path, fig_name))

    return


def ensquared_spaxel_size(zosapi, file_options, N_rays, files_path, results_path):
    """
    For the nominal HARMONI design, the performance is so close to perfect that there's almost no variation
    in Ensquared Energy. Therefore, the DETECTOR plane plots are not very informative

    To properly demonstrate that we have a lot of margin, we will calculate the Ensquared Energy
    as a function of the pixel box size. This will show that the EE reaches its maximum for values that are
    lower than

    :param zosapi:
    :param sys_mode:
    :param ao_modes:
    :param spaxel_scale:
    :param grating:
    :param N_rays:
    :param files_path:
    :param results_path:
    :return:
    """

    spaxel_scale = file_options['SPAX_SCALE']
    grating = file_options['GRATING']
    monte_carlo = file_options['MONTE_CARLO']
    ao_mode = file_options['AO_MODE']

    analysis_dir = os.path.join(results_path, 'ENSQUARED ENERGY')
    print("Analysis Results will be saved in folder: ", analysis_dir)
    if not os.path.exists(analysis_dir):
        os.mkdir(analysis_dir)

    analysis = e2e.EnsquaredEnergyFastAnalysis(zosapi=zosapi)
    waves = np.linspace(1, 23, 5).astype(int)

    # ifu_sections = ['AB', 'CD', 'EF', 'GH']
    ifu_sections = ['AB']

    sizes = np.linspace(0.0, 2.0, 50, endpoint=True)
    EE_min, EE_max = [], []
    EE_mean = []
    for box_size in sizes:
        print("\nBox size: %.3f detector pixels" % box_size)

        energy_all_ifus = []
        for i, ifu in enumerate(ifu_sections):

            # change the IFU path option
            file_options['IFU_PATH'] = ifu

            if monte_carlo is True:
                # read the MC instance number of the specific IFU path
                ifu_mc = file_options['IFU_PATHS_MC'][ifu]
                file_options['IFU_MC'] = ifu_mc

                # select the proper ISP MC instance, according to the IFU path
                isp_mc = file_options['IFU_ISP_MC'][ifu]
                file_options['ISP_MC'] = isp_mc

            list_results = analysis.loop_over_files(files_dir=files_path, files_opt=file_options, results_path=results_path,
                                                    wavelength_idx=waves, configuration_idx=None, N_rays=N_rays,
                                                    box_size=box_size, monte_carlo=monte_carlo)

            #
            # list_results = analysis.loop_over_files(files_dir=files_path, files_opt=options, results_path=results_path,
            #                                         wavelength_idx=waves, configuration_idx=[37, 38], N_rays=N_rays,
            #                                         box_size=box_size)

            # No Monte Carlo so the list_results only contains 1 entry, for the IFU channel
            energy, obj_xy, slicer_xy, detector_xy, wavelengths = list_results[0]
            energy_all_ifus.append(energy.flatten())
        energy_all_ifus = np.array(energy_all_ifus)
        EE_min.append(np.min(energy_all_ifus))
        EE_max.append(np.max(energy_all_ifus))
        EE_mean.append(np.mean(energy_all_ifus))

    fig, ax = plt.subplots(1, 1)
    ax.plot(sizes, EE_min, color='red', label='Min EE')
    ax.plot(sizes, EE_mean, color='green', label='Mean EE')
    ax.plot(sizes, EE_max, color='blue', label='Max EE')
    ax.set_xlabel(r'Box size [pixels]')
    ax.set_ylabel(r'Ensquared Energy [ ]')
    ax.set_xlim([0, sizes[-1]])
    ax.set_ylim([0, 1.0])
    ax.set_title(r"EE_%s_SPEC_%s_MODE_%s_%s" % (spaxel_scale, grating, sys_mode, ao_modes[0]))
    plt.legend()

    save_path = os.path.join(results_path, analysis_dir)
    fig_name = "PLOT_ENSQ_ENERGY_%s_SPEC_%s_MODE_%s_%s" % (spaxel_scale, grating, sys_mode, ao_modes[0])
    if os.path.isfile(os.path.join(save_path, fig_name)):
        os.remove(os.path.join(save_path, fig_name))
    fig.savefig(os.path.join(save_path, fig_name))

    return


if __name__ == """__main__""":

    plt.rc('font', family='serif')
    plt.rc('text', usetex=False)

    # Create a Python Standalone Application
    psa = e2e.PythonStandaloneApplication()

    # [*] Monte Carlo Instances [*]
    ao_mode = 'NOAO'
    spaxel_scale = '10x10'
    N_rays = 500       # How many rays to trace for the Ensquared  Energy calculation
    # gratings = ['VIS', 'Z_HIGH', 'IZ', 'J', 'IZJ', 'H', 'H_HIGH', 'HK', 'K', 'K_SHORT', 'K_LONG']
    gratings = ['H']
    ipo_mcs = {'10x10': '0064', '20x20': '0042'}

    file_options = {'MONTE_CARLO': True, 'AO_MODE': ao_mode, 'SPAX_SCALE': spaxel_scale,
                    'FPRS_MC': '0694', 'IPO_MC': ipo_mcs[spaxel_scale],
                    'IFU_PATHS_MC': {'AB': '0028', 'CD': '0068', 'EF': '0071', 'GH': '0095'},
                    'IFU_ISP_MC': {'AB': '0024', 'CD': '0009', 'EF': '0013', 'GH': '0073'}}


    files_path = os.path.abspath("D:/End to End Model/Monte_Carlo_Dec/Median")
    results_path = os.path.abspath("D:/End to End Model/Monte_Carlo_Dec/Median/Results/Mode_%s/Scale_%s" % (ao_mode, spaxel_scale))
    # [*] Monte Carlo Instances [*]

    # # [*] This is the bit we have to change when you run the analysis in your system [*]
    # sys_mode = 'HARMONI'
    # ao_modes = ['NOAO']
    # spaxel_scale = '20x20'
    # # gratings = ['VIS', 'Z_HIGH', 'IZ', 'J', 'IZJ', 'H', 'H_HIGH', 'HK', 'K', 'K_SHORT', 'K_LONG']
    # gratings = ['H']
    # N_rays = 500
    # files_path = os.path.abspath("D:\End to End Model\August_2020")
    # results_path = os.path.abspath("D:\End to End Model\Results_ReportAugust\Mode_%s\Scale_%s" % (ao_modes[0], spaxel_scale))
    # # [*] This is the bit we have to change when you run the analysis in your system [*]

    analysis_dir = os.path.join(results_path, 'ENSQUARED ENERGY')
    print("Analysis Results will be saved in folder: ", analysis_dir)
    if not os.path.exists(analysis_dir):
        os.mkdir(analysis_dir)

    # (1) First of all, we save a txt file with the MC Instances used to generate the E2E files (if running MC mode)
    # for each of the subsystems, as well as the slice sampling (field points per slice)
    if file_options['MONTE_CARLO'] is True:
        with open(os.path.join(analysis_dir, 'MC_settings.txt'), 'w') as file:
            file.write(json.dumps(file_options))

    # for grating in gratings:
    #     # change the grating option
    #     file_options['GRATING'] = grating
    #     detector_ensquared_energy(zosapi=psa, file_options=file_options, N_rays=N_rays, files_path=files_path, results_path=results_path,
    #                               box_size=2.0)
    # plt.show()

    for grating in gratings:
        file_options['GRATING'] = grating
        ensquared_spaxel_size(zosapi=psa, file_options=file_options, N_rays=N_rays,
                              files_path=files_path, results_path=results_path)

    plt.show()