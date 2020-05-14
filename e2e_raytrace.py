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

    analysis = e2e.RaytraceAnalysis(zosapi=zosapi)



    fig, ax = plt.subplots(1, 1)
    for ifu, color in zip(['AB', 'CD', 'EF', 'GH'], ['black', 'red', 'blue', 'green']):
        options = {'which_system': 'IFS', 'AO_modes': [], 'scales': [spaxel_scale], 'IFUs': [ifu], 'grating': [grating]}
        surface_number = e2e.focal_planes['IFS'][spaxel_scale][ifu][surface_code]
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

        if surface_code == 'SL':
            circ = plt.Circle((0.0, 0.0), 480, color='b', fill=False)
            ax.add_artist(circ)

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

    scale = '60x30'

    # PreOptics, Image Slicer, Slits
    for surface_code in ['PO']:
        raytrace_global(zosapi=psa, spaxel_scale=scale, surface_code=surface_code, grating='H',
                        wavelength_idx=[1], rays_per_slice=3)

    # # Detector
    # raytrace_global(zosapi=psa, spaxel_scale=scale, surface_code='DET', grating='H',
    #                 wavelength_idx=None, rays_per_slice=3)



    ### Ensquared Detector

    analysis = e2e.EnsquaredDetector(zosapi=psa)

    N_rays = 20
    size = 0.25 # mm
    surface_number = e2e.focal_planes['IFS']['60x30']['AB']['DET']
    options = {'which_system': 'IFS', 'AO_modes': [], 'scales': ['60x30'], 'IFUs': ['AB'], 'grating': ['H']}
    list_results = analysis.loop_over_files(files_dir=files_path, files_opt=options,
                                            results_path=results_path, wavelength_idx=[1],
                                            configuration_idx=[5], surface=surface_number,
                                            N_rays=N_rays, size=size)

    obj_xy, foc_xy, global_xy, waves = list_results[0]

    colors = ['black', 'red', 'blue', 'green', 'orange']
    labels = ['low', 'right', 'top', 'left', 'mid']
    fig, (ax1, ax2) = plt.subplots(1, 2)
    for i in range(5):
        ox = obj_xy[:, :, i*N_rays:(i+1)*N_rays, 0]
        oy = obj_xy[:, :, i*N_rays:(i+1)*N_rays, 1]
        ax1.scatter(ox, oy, label=labels[i], color=colors[i])
        ax1.set_title(r'Object Space | Box %.f $\mu$m' % (1000 * size))


    # fig, ax = plt.subplots(1, 1)
    for i in range(5):
        ox = foc_xy[:, :, i*N_rays:(i+1)*N_rays, 0]
        oy = foc_xy[:, :, i*N_rays:(i+1)*N_rays, 1]
        ax2.scatter(ox, oy, label=labels[i], color=colors[i])
        ax2.set_title(r'Detector Space ')
    plt.legend()

    plt.show()

    #
    #
    # fig, ax = plt.subplots(1, 1)
    # ax.scatter(foc_xy[:, 0, :, 0], foc_xy[:, 0, :, 1], s=2, color='red')
    # ax.scatter(foc_xy[:, 1, :, 0], foc_xy[:, 1, :, 1], s=2, color='blue')
    # ax.scatter(foc_xy[:, 2, :, 0], foc_xy[:, 2, :, 1], s=2, color='green')
    #

    plt.show()