"""

Geometric Ensquared Energy analysis


Author: Alvaro Menduina
Date: May 2020
Modified: June 2020

Description:
With this script we calculate the Geometric Ensquared Energy for the E2E files
at the Detector plane for a 2x2 spaxel box.

You can select:
    - The system MODE: IFS, HARMONI (with NOAO)
    -


"""

import os
import numpy as np
import e2e_analysis as e2e
import matplotlib.pyplot as plt
import matplotlib.tri as tri

def detector_ensquared_energy(zosapi, sys_mode, ao_modes, spaxel_scale, grating, N_rays, alpha, files_path, results_path):
    """
    Calculate the Ensquared Energy at the detector plane for all 4 IFU channels
    The Ensquared Energy is calculated for the Field Point at the centre of the slice
    for all slices and all wavelengths

    :param zosapi: the Python Standalone Application
    :param spaxel_scale: str, spaxel scale to analyze
    :param grating: str, grating to use
    :param N_rays: number of rays to sample the pupil with
    :param files_path: path to the Zemax E2E model files
    :param results_path: path to save the results
    :return:
    """

    analysis_dir = os.path.join(results_path, 'ENSQUARED ENERGY')
    print("Analysis Results will be saved in folder: ", analysis_dir)
    if not os.path.exists(analysis_dir):
        os.mkdir(analysis_dir)

    analysis = e2e.EnsquaredEnergyAnalysis(zosapi=zosapi)

    waves = np.linspace(1, 23, 5).astype(int)

    ifu_sections = ['AB', 'CD', 'EF', 'GH']
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    for i, ifu in enumerate(ifu_sections):

        options = {'which_system': sys_mode, 'AO_modes': ao_modes, 'scales': [spaxel_scale], 'IFUs': [ifu],
                   'grating': [grating]}
        list_results = analysis.loop_over_files(files_dir=files_path, files_opt=options, results_path=results_path,
                                                wavelength_idx=waves, configuration_idx=None, N_rays=N_rays,
                                                alpha=alpha)

        # No Monte Carlo so the list_results only contains 1 entry, for the IFU channel
        energy, obj_xy, slicer_xy, detector_xy, wavelengths = list_results[0]

        # Separate the Odd / Even configurations to avoid triangulating over the gap on the detector plane
        ener_ogg, ener_even = energy[:, ::2].flatten(), energy[:, 1::2].flatten()
        x, y = detector_xy[:, :, 0].flatten(), detector_xy[:, :, 1].flatten()
        x_odd, y_odd = detector_xy[:, ::2, 0].flatten(), detector_xy[:, ::2, 1].flatten()
        x_even, y_even = detector_xy[:, 1::2, 0].flatten(), detector_xy[:, 1::2, 1].flatten()
        triang_odd = tri.Triangulation(x_odd, y_odd)
        triang_even = tri.Triangulation(x_even, y_even)

        #Remove the flat triangles at the detector edges that appear because of the field curvature
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
        title = r'IFU-%s | %s mas | %s SPEC | EE min:%.2f max:%.2f' % (ifu, spaxel_scale, grating, min_ener, max_ener)
        ax.set_title(title)

    save_path = os.path.join(results_path, analysis_dir)
    fig_name = "ENSQ_ENERGY_DETECTOR_%s_SPEC_%s_MODE_%s" % (spaxel_scale, grating, sys_mode)
    if os.path.isfile(os.path.join(save_path, fig_name)):
        os.remove(os.path.join(save_path, fig_name))
    fig.savefig(os.path.join(save_path, fig_name))

    return


def ensquared_spaxel_size(zosapi, sys_mode, ao_modes, spaxel_scale, grating, N_rays, files_path, results_path):
    """
    Calculate the Ensquared Energy as a function of the pixel box
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

    analysis_dir = os.path.join(results_path, 'ENSQUARED ENERGY')
    print("Analysis Results will be saved in folder: ", analysis_dir)
    if not os.path.exists(analysis_dir):
        os.mkdir(analysis_dir)

    analysis = e2e.EnsquaredEnergyAnalysis(zosapi=zosapi)
    waves = np.linspace(1, 23, 5).astype(int)

    ifu_sections = ['AB', 'CD', 'EF', 'GH']

    sizes = np.linspace(0.0, 2.0, 50, endpoint=True)
    EE_min, EE_max = [], []
    EE_mean = []
    for alpha in sizes:
        print("\nBox size: %.3f detector pixels" % alpha)

        energy_all_ifus = []
        for i, ifu in enumerate(ifu_sections):

            options = {'which_system': sys_mode, 'AO_modes': ao_modes, 'scales': [spaxel_scale], 'IFUs': [ifu],
                       'grating': [grating]}
            list_results = analysis.loop_over_files(files_dir=files_path, files_opt=options, results_path=results_path,
                                                    wavelength_idx=waves, configuration_idx=[37, 38], N_rays=N_rays,
                                                    alpha=alpha)

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
    ax.set_title(r"EE_%s_SPEC_%s_MODE_%s" % (spaxel_scale, grating, sys_mode))
    plt.legend()

    save_path = os.path.join(results_path, analysis_dir)
    fig_name = "PLOT_ENSQ_ENERGY_%s_SPEC_%s_MODE_%s" % (spaxel_scale, grating, sys_mode)
    if os.path.isfile(os.path.join(save_path, fig_name)):
        os.remove(os.path.join(save_path, fig_name))
    fig.savefig(os.path.join(save_path, fig_name))

    return



if __name__ == """__main__""":

    plt.rc('font', family='serif')
    plt.rc('text', usetex=False)

    # Create a Python Standalone Application
    psa = e2e.PythonStandaloneApplication()

    # Don't forget to specify where to find the E2E files and where to save results
    # files_path = os.path.abspath("D:\End to End Model\June_2020")
    # results_path = os.path.abspath("D:\End to End Model\Results_June")

    files_path = os.path.abspath("D:\End to End Model\June_John2020")
    results_path = os.path.abspath("D:\End to End Model\Results_JuneJohn")

    # Adapt the parameters here: system mode, ao_modes, spaxel_scale, gratings, N_rays
    sys_mode = 'HARMONI'
    ao_modes = ['NOAO']
    spaxel_scale = '20x20'
    # gratings = ['Z_HIGH', 'IZ', 'J', 'IZJ', 'H', 'H_HIGH', 'HK', 'K', 'K_LONG', 'K_SHORT']
    gratings = ['H']
    N_rays = 500

    for grating in gratings:
        detector_ensquared_energy(zosapi=psa, sys_mode=sys_mode, ao_modes=ao_modes, spaxel_scale=spaxel_scale,
                                  grating=grating, N_rays=N_rays, files_path=files_path, results_path=results_path,
                                  alpha=0.5)

    # Some info on speed and number of rays:
    #   - 100 rays: ~6 sec per wavelength
    #   - 500 rays: ~8 sec per wavelength
    #   - 1000 rays: ~ 10 sec per wavelength

    plt.show()

    # Show the plot of the Ensquared Energy vs Pixel size

    # spaxel_scale = '10x10'
    # ensquared_spaxel_size(zosapi=psa, sys_mode=sys_mode, ao_modes=ao_modes, spaxel_scale=spaxel_scale,
    #                               grating='H', N_rays=N_rays, files_path=files_path, results_path=results_path)
    #
    # del psa
    # psa = None
    #
    #






