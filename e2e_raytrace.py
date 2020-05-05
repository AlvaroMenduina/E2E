"""
Example of how to use the RaytraceAnalysis to calculate the XY coordinates
at different surfaces, both in Local and Global reference frames

Author: Alvaro Menduina
Date: May 2020
"""


import os
import numpy as np
import e2e_analysis as e2e
import matplotlib.pyplot as plt
import matplotlib.tri as tri

def raytrace_global(zosapi, spaxel_scale, surface_code, grating, wavelength_idx, rays_per_slice=3):
    """


    Surface_code follows the convention used in e2e_analysis.py
    'FPRS': Focal Plane Relay System, 'PO': PreOptics Focal Plane, 'IS': Image Slicer, 'SL': IFU Slits, 'DET': Detector


    :param spaxel_scale:
    :param surface_code:
    :param grating:
    :param wavelength_idx:
    :param rays_per_slice:
    :return:
    """
    codenames = {'PO': 'PreOptics', 'IS': 'Image Slicer', 'SL': 'Slits', 'DET': 'Detector'}
    name = codenames[surface_code]

    analysis = e2e.RaytraceAnalysis(zosapi=psa)

    surface_number = e2e.focal_planes['IFS'][spaxel_scale][surface_code]

    fig, ax = plt.subplots(1, 1)
    for ifu, color in zip(['AB', 'CD', 'EF', 'GH'], ['black', 'red', 'blue', 'green']):
        options = {'which_system': 'IFS', 'AO_modes': [], 'scales': [spaxel_scale], 'IFUs': [ifu], 'grating': [grating]}
        list_results = analysis.loop_over_files(files_dir=files_path, files_opt=options,
                                                                    results_path=results_path, wavelength_idx=wavelength_idx,
                                                                    configuration_idx=None, surface=surface_number,
                                                                    rays_per_slice=rays_per_slice)
        # We should only have 1 list of results (no Monte Carlo)
        obj_xy, foc_xy, global_xy, waves = list_results[0]

        if wavelength_idx is None:
            ax.scatter(global_xy[:, :, :, 0], global_xy[:, :, :, 1], s=2, color=color, label=ifu)
            title = r'Focal Plane: %s | %s scale | %s grating' % (name, spaxel_scale, grating)
        elif len(wavelength_idx) == 1:
            wavelength = waves[0]
            ax.scatter(global_xy[0, :, :, 0], global_xy[0, :, :, 1], s=2, color=color, label=ifu)
            title = r'Focal Plane: %s | %s scale | %s grating %.3f $\mu$m' % (name, spaxel_scale, grating, wavelength)

        ax.set_aspect('equal')
    ax.set_title(title)
    ax.set_xlabel(r'Global X [mm]')
    ax.set_ylabel(r'Global Y [mm]')
    plt.legend(title='IFU Channel')


if __name__ == """__main__""":

    plt.rc('font', family='serif')
    plt.rc('text', usetex=False)

    # Create a Python Standalone Application
    psa = e2e.PythonStandaloneApplication()

    files_path = os.path.abspath("D:\End to End Model\April_2020")
    results_path = os.path.abspath("D:\End to End Model\Results_April")

    # PreOptics, Image Slicer, Slits
    for surface_code in ['PO', 'IS', 'SL']:
        raytrace_global(zosapi=psa, spaxel_scale='60x30', surface_code=surface_code, grating='H',
                        wavelength_idx=[1], rays_per_slice=3)

    # Detector
    raytrace_global(zosapi=psa, spaxel_scale='60x30', surface_code='DET', grating='H',
                    wavelength_idx=None, rays_per_slice=3)

    plt.show()