"""

Ensquared Energy analysis


Author: Alvaro Menduina
Date: May 2020

"""

import os
import numpy as np
import e2e_analysis as e2e
import matplotlib.pyplot as plt
import matplotlib.tri as tri

def detector_ensquared_energy(zosapi, mode, spaxel_scale, grating, N_rays, files_path, results_path):
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

    if mode == 'IFS':
        ao_modes = []
    elif mode == 'HARMONI':
        ao_modes = ['NOAO']

    analysis = e2e.EnsquaredEnergyAnalysis(zosapi=zosapi)

    ifu_sections = ['AB', 'CD', 'EF', 'GH']
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    for i, ifu in enumerate(ifu_sections):
        options = {'which_system': mode, 'AO_modes': ao_modes, 'scales': [spaxel_scale], 'IFUs': [ifu],
                   'grating': [grating]}
        list_results = analysis.loop_over_files(files_dir=files_path, files_opt=options, results_path=results_path,
                                                wavelength_idx=None, configuration_idx=None, N_rays=N_rays)

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

    fig_name = "ENSQUARED_ENERGY_DETECTOR_PO%s_SPEC_%s" % (spaxel_scale, grating)
    if os.path.isfile(os.path.join(results_path, fig_name)):
        os.remove(os.path.join(results_path, fig_name))
    fig.savefig(os.path.join(results_path, fig_name))

    # plt.show()
    return


if __name__ == """__main__""":

    plt.rc('font', family='serif')
    plt.rc('text', usetex=False)

    # Create a Python Standalone Application
    psa = e2e.PythonStandaloneApplication()

    files_path = os.path.abspath("D:\End to End Model\June_2020")
    results_path = os.path.abspath("D:\End to End Model\Results_June")

    mode = 'HARMONI'
    spaxel_scale = '10x10'
    gratings = ['Z_HIGH', 'IZ', 'J', 'IZJ', 'H', 'H_HIGH', 'HK', 'K', 'K_LONG', 'K_SHORT']
    N_rays = 1000

    for grating in gratings:
        detector_ensquared_energy(zosapi=psa, mode=mode, spaxel_scale=spaxel_scale, grating=grating, N_rays=N_rays,
                               files_path=files_path, results_path=results_path)

    # Some info on speed and number of rays:
    #   - 100 rays: ~6 sec per wavelength
    #   - 500 rays: ~8 sec per wavelength
    #   - 1000 rays: ~ 10 sec per wavelength

    del psa
    psa = None








