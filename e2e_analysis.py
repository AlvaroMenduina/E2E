"""

Performance Analysis for the End-to-End Models

This is the main module that contains all the stuff needed to
construct a performance analysis

Author: Alvaro Menduina
Date Created: March 2020
Latest version: December 2020

"""

import os
import numpy as np
from numpy.fft import ifft2, fft2, fftshift
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from time import time
import h5py
import datetime
from sklearn.neighbors import KernelDensity
from scipy.signal import convolve2d
from scipy.optimize import curve_fit
import functools
import re

from win32com.client.gencache import EnsureDispatch, EnsureModule
from win32com.client import constants
from win32com.client import CastTo

# Get the module version from Git
# We will use the commit's SHA hash to identify which version we used when running an analysis
import git
repo = git.Repo(search_parent_directories=True)
sha = repo.head.object.hexsha

# Taken from Zemax example code
class PythonStandaloneApplication(object):

    class LicenseException(Exception):
        pass

    class ConnectionException(Exception):
        pass

    class InitializationException(Exception):
        pass

    class SystemNotPresentException(Exception):
        pass

    def __init__(self, path=None):
        # make sure the Python wrappers are available for the COM client and
        # interfaces
        EnsureModule('{EA433010-2BAC-43C4-857C-7AEAC4A8CCE0}', 0, 1, 0)

        # EnsureModule('{F66684D7-AAFE-4A62-9156-FF7A7853F764}', 0, 1, 0)
        # Note - the above can also be accomplished using 'makepy.py' in the
        # following directory:
        #      {PythonEnv}\Lib\site-packages\wind32com\client\
        # Also note that the generate wrappers do not get refreshed when the
        # COM library changes.
        # To refresh the wrappers, you can manually delete everything in the
        # cache directory:
        #      {PythonEnv}\Lib\site-packages\win32com\gen_py\*.*

        self.TheConnection = EnsureDispatch("ZOSAPI.ZOSAPI_Connection")
        if self.TheConnection is None:
            raise PythonStandaloneApplication.ConnectionException(
                "Unable to intialize COM connection to ZOSAPI")

        self.TheApplication = self.TheConnection.CreateNewApplication()
        if self.TheApplication is None:
            raise PythonStandaloneApplication.InitializationException(
                "Unable to acquire ZOSAPI application")

        if self.TheApplication.IsValidLicenseForAPI == False:
            raise PythonStandaloneApplication.LicenseException(
                "License is not valid for ZOSAPI use")

        self.TheSystem = self.TheApplication.PrimarySystem
        if self.TheSystem is None:
            raise PythonStandaloneApplication.SystemNotPresentException(
                "Unable to acquire Primary system")

    def __del__(self):
        if self.TheApplication is not None:
            self.TheApplication.CloseApplication()
            self.TheApplication = None

        self.TheConnection = None

    def OpenFile(self, filepath, saveIfNeeded):
        if self.TheSystem is None:
            raise PythonStandaloneApplication.SystemNotPresentException(
                "Unable to acquire Primary system")
        self.TheSystem.LoadFile(filepath, saveIfNeeded)

    def CloseFile(self, save):
        if self.TheSystem is None:
            raise PythonStandaloneApplication.SystemNotPresentException(
                "Unable to acquire Primary system")
        self.TheSystem.Close(save)

    def SamplesDir(self):
        if self.TheApplication is None:
            raise PythonStandaloneApplication.InitializationException(
                "Unable to acquire ZOSAPI application")

        return self.TheApplication.SamplesDir

    def ExampleConstants(self):
        if (self.TheApplication.LicenseStatus is
                constants.LicenseStatusType_PremiumEdition):
            return "Premium"
        elif (self.TheApplication.LicenseStatus is
              constants.LicenseStatusType_ProfessionalEdition):
            return "Professional"
        elif (self.TheApplication.LicenseStatus is
              constants.LicenseStatusType_StandardEdition):
            return "Standard"
        else:
            return "Invalid"


MAX_WAVES = 23
@functools.lru_cache(maxsize=MAX_WAVES)
def airy_and_slicer(surface, wavelength, scale_mas, PSF_window, N_window):
    """
    Compute the Airy pattern at a given surface (slicer or detector), wavelength, and spaxel scale
    We then add the effect of Image Slicer

    Since the calculations here are rather time-consuming
    and this function will be called multiple times (once for each configuration / slice)
    we use LRU cache to recycle previous results

    :param surface: string, which surface to calculate the PSF at ('IS': Image Slicer, 'DET': Detector)
    :param wavelength: wavelength in [microns]
    :param scale_mas: spaxel scale in [mas]
    :param PSF_window: length of the PSF window [microns]
    :param N_window: number of points of the PSF window [pixels]
    :return:
    """

    # Print message to know we are updating the cache
    print('Recalculating Airy Pattern for %.3f microns' % wavelength)

    # Plate scales [Px, Py] for each spaxel scale in mm / arcsec, depending on the surface
    plate_scales = {'IS': {4.0: [125, 250], 10.0: [50, 100], 20.0: [25, 50], 60.0: [16.67, 16.67]},
                    'DET': {4.0: [3.75, 7.5], 10.0: [1.5, 3.0], 20.0: [0.75, 1.5], 60.0: [0.5, 0.5]}}
    plate_x = plate_scales[surface][scale_mas][0]
    plate_y = plate_scales[surface][scale_mas][1]

    # We know how many Microns the pixels of the Geometric PSF span [PSF_window / N_window]
    pix_sampling = PSF_window / N_window  # micron at the detector plane
    # Using the plate scale we calculate how many m.a.s each of those pixels have to span
    pix_scale_x = pix_sampling / plate_x  # milliarcsec / pixel
    pix_scale_y = pix_sampling / plate_y  # milliarcsec / pixel

    # Calculate the relative size of the pupil aperture needed to ensure the PSF is
    # sampled with the given pix_scale at the detector
    ELT_DIAM = 39
    MILIARCSECS_IN_A_RAD = 206265000
    pix_rad_x = pix_scale_x / MILIARCSECS_IN_A_RAD  # radians / pixel
    pix_rad_y = pix_scale_y / MILIARCSECS_IN_A_RAD
    RHO_APER_x = pix_rad_x * ELT_DIAM / (wavelength * 1e-6)
    RHO_APER_y = pix_rad_y * ELT_DIAM / (wavelength * 1e-6)
    RHO_OBSC_x = 0.30 * RHO_APER_x  # ELT central obscuration
    RHO_OBSC_y = 0.30 * RHO_APER_y  # ELT central obscuration

    # Sanity check
    PIX_RAD_x = RHO_APER_x * wavelength / ELT_DIAM * 1e-6
    PIX_RAD_y = RHO_APER_y * wavelength / ELT_DIAM * 1e-6
    PIX_MAS_x = PIX_RAD_x * MILIARCSECS_IN_A_RAD
    PIX_MAS_y = PIX_RAD_y * MILIARCSECS_IN_A_RAD

    # Define the ELT pupil mask
    N = 2048
    x = np.linspace(-1, 1, N)
    xx, yy = np.meshgrid(x, x)

    # To get the anamorphic scaling we define the equation for an ellipse
    rho = np.sqrt((xx / RHO_APER_x) ** 2 + (yy / RHO_APER_y) ** 2)

    # (1) Propagate to the Image Slicer Focal plane
    elt_mask = (RHO_OBSC_x / RHO_APER_x < rho) & (rho < 1.0)
    pupil = elt_mask * np.exp(1j * elt_mask)
    image_electric = fftshift(fft2(pupil))

    if surface == 'IS':
        # print("IS")
        # We are already at the Image Slicer, don't do anything else
        min_pix, max_pix = N // 2 - N_window // 2, N // 2 + N_window // 2
        final_psf = (np.abs(image_electric))**2
        final_psf /= np.max(final_psf)
        crop_psf = final_psf[min_pix:max_pix, min_pix:max_pix]

    elif surface == 'DET':
        # print("DET")
        # (1.1) Add slicer effect by masking
        # We mask the PSF covering a band of size 1x SPAXEL, depending on the scale
        # If we have 4x4 mas, then we cover a band of 4 mas over the PSF
        x_min, x_max = -N/2 * PIX_MAS_x, N/2 * PIX_MAS_x
        y_min, y_max = -N/2 * PIX_MAS_y, N/2 * PIX_MAS_y
        x_slice = np.linspace(x_min, x_max, N, endpoint=True)
        y_slice = np.linspace(y_min, y_max, N, endpoint=True)
        x_grid, y_grid = np.meshgrid(x_slice, y_slice)
        slicer_mask = np.abs(y_grid) < scale_mas / 2

        # ## Show the PSF both in [mas] space where it should be circular and in [pixel] space where it should be anamorphic
        # fig, ax = plt.subplots(1, 1)
        # img1 = ax.imshow((np.abs(image_electric))**2, extent=[x_min, x_max, y_min, y_max], cmap='bwr')
        # # plt.colorbar(img1, ax=ax)
        # ax.set_title(r'Airy Pattern | %.1f mas scale | Wavelength: %.3f $\mu$m' % (scale_mas, wavelength))
        # ax.set_xlabel(r'X [mas]')
        # ax.set_ylabel(r'Y [mas]')
        # ax.set_xlim([-10, 10])
        # ax.set_ylim([-10, 10])
        #
        # fig, ax = plt.subplots(1, 1)
        # img1 = ax.imshow((np.abs(image_electric))**2, extent=[-N/2, N/2, -N/2, N/2], cmap='bwr')
        # ax.set_title(r'Airy Pattern | %.1f mas scale | Wavelength: %.3f $\mu$m' % (scale_mas, wavelength))
        # ax.set_xlabel(r'Pixels [ ]')
        # ax.set_ylabel(r'Pixels [ ]')
        # ax.set_xlim([-100, 100])
        # ax.set_ylim([-100, 100])

        # plt.show()

        # (2) Propagate the masked electric field to Pupil Plane
        pup_grating = ifft2(fftshift(slicer_mask * image_electric))
        # (2.1) Add pupil mask, this time without the central obscuration
        aperture_mask = rho < 1.0

        # (3) Propagate back to Focal Plane
        final_focal = fftshift(fft2(aperture_mask * pup_grating))
        final_psf = (np.abs(final_focal))**2
        final_psf /= np.max(final_psf)

        # (4) Crop the PSF to fit to the necessary window to ease the convolutions
        min_pix, max_pix = N//2 - N_window//2, N//2 + N_window//2
        crop_psf = final_psf[min_pix:max_pix, min_pix:max_pix]

        # If we want to show the plots for Documentation

        # fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
        # psf_airy = (np.abs(image_electric))**2
        # img1 = ax1.imshow(psf_airy, extent=[x_min, x_max, y_min, y_max], cmap='bwr')
        # ax1.axhline(y=scale_mas/2, linestyle='--', color='black')
        # ax1.axhline(y=-scale_mas/2, linestyle='--', color='black')
        # ax1.set_xlabel(r'X [mas]')
        # ax1.set_ylabel(r'Y [mas]')
        # ax1.set_xlim([-15, 15])
        # ax1.set_ylim([-15, 15])
        # ax1.set_title(r'Airy Pattern | Slicer Mask %.1f mas' % scale_mas)
        #
        # img2 = ax2.imshow(aperture_mask * (np.abs(pup_grating)**2), extent=[-1, 1, -1, 1], cmap='bwr')
        # ax2.set_title(r'Pupil Plane | Aperture Mask')
        # ax2.set_xlim([-0.25, 0.25])
        # ax2.set_ylim([-0.25, 0.25])
        #
        # img3 = ax3.imshow(final_psf, extent=[x_min, x_max, y_min, y_max], cmap='bwr')
        # ax3.set_xlabel(r'X [mas]')
        # ax3.set_ylabel(r'Y [mas]')
        # ax3.set_xlim([-15, 15])
        # ax3.set_ylim([-15, 15])
        # ax3.set_title(r'Diffraction Effects')
        # plt.show()

    return crop_psf


class DiffractionEffects(object):
    """
    A class to take care of diffraction effects
    for the PSF FWHM calculations
    """
    def __init__(self):

        return

    def add_diffraction(self, surface, psf_geo, PSF_window, scale_mas, wavelength):
        """
        Calculate the diffraction effects which include:
            - ELT pupil diffraction
            - Image Slicer + Grating

        and convolve it with a Geometric PSF to produce an
        estimate of the Diffraction PSF

        :param psf_geo: geometric PSF
        :param PSF_window: length of the geometric PSF window [microns]
        :param scale_mas: spaxel scale to consider [mas]
        :param wavelength: wavelength [microns]
        :return:
        """

        # Calculate the diffraction effects
        N_window = psf_geo.shape[0]
        diffraction = airy_and_slicer(surface=surface, wavelength=wavelength, scale_mas=scale_mas,
                                      PSF_window=PSF_window, N_window=N_window)

        # Convolve the diffraction effects with the geometric PSF
        psf_convol = convolve2d(psf_geo, diffraction, mode='same')
        # Normalize to peak 1.0
        psf_convol /= np.max(psf_convol)

        return psf_convol

    def gaussian2d(self, xy_data, x0, y0, sigma_x, sigma_y, amplitude, offset):
        """
        A 2D Gaussian evaluated the points in xy_data with the following characteristics:
        Centred at (x0, y0), with sigma_x, sigma_y, rotated by theta radians
        with amplitude, and offset
        :param xy_data:
        :param x0:
        :param y0:
        :param sigma_x:
        :param sigma_y:
        :param theta:
        :param amplitude:
        :param offset:
        :return:
        """
        theta = 0.0
        (x, y) = xy_data
        x0 = float(x0)
        y0 = float(y0)
        a = (np.cos(theta) ** 2) / (2 * sigma_x ** 2) + (np.sin(theta) ** 2) / (2 * sigma_y ** 2)
        b = -(np.sin(2 * theta)) / (4 * sigma_x ** 2) + (np.sin(2 * theta)) / (4 * sigma_y ** 2)
        c = (np.sin(theta) ** 2) / (2 * sigma_x ** 2) + (np.cos(theta) ** 2) / (2 * sigma_y ** 2)
        g = offset + amplitude * np.exp(-(a * ((x - x0) ** 2) + 2 * b * (x - x0) * (y - y0) + c * ((y - y0) ** 2)))
        return g

    def fit_psf_to_gaussian(self, xx, yy, psf_data, x0, y0, sigmax0, sigmay0):
        """
        Fit a PSF array to a 2D Gaussian
        :param xx: grid of X values to evaluate the Gaussian
        :param yy: grid of Y values to evaluate the Gaussian
        :param psf_data: PSF array to fit
        :param x0: estimate of the X centre (typically geometric PSF centroid)
        :param y0: estimate of the Y centre (typically geometric PSF centroid)
        :return:
        """
        peak0, offset0 = 1.0, 0.0     # Guesses
        rotation0 = 0.0              # Guesses
        # # We guess that the FWHM will be around 2 detector pixels [30 microns]
        # # Using the formula for a Gaussian FWHM = 2 sqrt(2 ln(2)) Sigma
        # sigma_fwhm = 100e-3 / 2 * np.sqrt(2 * np.log(2))
        # sigmax0, sigmay0 = sigma_fwhm, sigma_fwhm

        guess_param = [x0, y0, sigmax0, sigmay0, peak0, offset0]
        bounds = ([-np.inf, -np.inf, 0.0, 0.0, 0, -np.inf],
                  [np.inf, np.inf, np.inf, np.inf, np.inf, np.inf])
        # we don't let the theta vary more than +-45 degrees because that would
        # flip the values of FWHM_X and FWHM_Y

        while True:

            try:
                pars, cov = curve_fit(self.gaussian2d, (xx.ravel(), yy.ravel()), psf_data.ravel(),
                                      p0=guess_param, bounds=bounds)
                # Results of the fit for the sigmaX, sigmaY and theta
                sigmaX, sigmaY, theta = pars[2], pars[3], pars[4]
                # print("Successful Guassian Fit")
                break

            except RuntimeError:  # If it fails to converge
                # sigmaX, sigmaY, theta = np.nan, np.nan, np.nan
                print("\n[WARNING] Gaussian Fit of the PSF failed!")

                # try with other guesses
                deltax = np.random.uniform(low=-1.0, high=1.0)
                deltay = np.random.uniform(low=-1.0, high=1.0)
                new_sigmax0 = (1 + deltax) * sigmax0
                new_sigmay0 = (1 + deltay) * sigmay0
                print("Trying with new Sigmas: X %.3f & Y %.3f microns" % (1e3 * new_sigmax0, 1e3 * new_sigmay0))
                guess_param = [x0, y0, new_sigmax0, new_sigmay0, peak0, offset0]

            # if sigmaX > 0.0 and sigmaY > 0.0:  # we found a reasonable value
            #     break

        # try:
        #     pars, cov = curve_fit(self.gaussian2d, (xx.ravel(), yy.ravel()), psf_data.ravel(),
        #                           p0=guess_param, bounds=bounds)
        #     # Results of the fit for the sigmaX, sigmaY and theta
        #     sigmaX, sigmaY, theta = pars[2], pars[3], pars[4]
        #
        # except RuntimeError:    # If it fails to converge
        #
        #     # sigmaX, sigmaY, theta = np.nan, np.nan, np.nan
        #     print("\n[WARNING] Gaussian Fit of the PSF failed!")
        #
        #     # try with other guesses
        #     deltax = np.random.uniform(low=-1.0, high=1.0)
        #     deltay = np.random.uniform(low=-1.0, high=1.0)
        #     new_sigmax0 = (1 + deltax) * sigmax0
        #     new_sigmay0 = (1 + deltay) * sigmay0
        #     print("Trying with new Sigmas: X %.3f & Y %.3f microns" % (1e3 * new_sigmax0, 1e3*new_sigmay0))
        #     self.fit_psf_to_gaussian(xx, yy, psf_data, x0, y0, new_sigmax0, new_sigmay0)
        #     # plt.figure()
        #     # plt.imshow(psf_data)
        #     # plt.colorbar()
        #     # plt.show()

        fwhm_x = 2 * np.sqrt(2 * np.log(2)) * sigmaX * 1000
        fwhm_y = 2 * np.sqrt(2 * np.log(2)) * sigmaY * 1000

        return fwhm_x, fwhm_y, theta


diffraction = DiffractionEffects()


def read_hdf5(path_to_file):
    """
    Read a HDF5 file containing some analysis results

    HDF5 files for the E2E analysis have the following 'Group' structure:

        (1) "Zemax Metadata": this Group is a dictionary containing metadata on the Zemax file used in the analysis
            this includes: file name, system mode, spaxel scale, IFU channel, grating, AO modes
            Plus the Git hash for the E2E Python code [for traceability]

        (2) Analysis Name 1: each analysis type (RMS_WFE, FWHM_PSF) will have an associated group. Within each group,
            there will be an arbitrary number of subgroups.
                - Subgroup #0: each subgroup represents an instance of that analysis type; for example an RMS WFE
                  ran at a specific surface, with a set of parameters. Each subgroup will containing a series of
                  datasets representing the results of the analysis, plus some associated metadata (surface, parameters)
                - Subgroup #N: another analysis ran at a different surface, or with different sampling
        (3) Analysis Name 2: another type of analysis
                - [...]

    TODO: at the moment this function only reads the metadata for Zemax and each Analysis.

    :param path_to_file:
    :return:
    """

    print("\nReading HDF5 file: ", path_to_file)
    file = h5py.File(path_to_file, 'r')

    # List the groups
    groups = list(file.keys())
    print("Groups available: ", groups)

    # Read Zemax Metadata
    zemax_metadata = {}
    print("\nZemax Metadata:")
    for key in file['Zemax Metadata'].attrs.keys():
        print('{} : {}'.format(key, file['Zemax Metadata'].attrs[key]))
        zemax_metadata[key] = file['Zemax Metadata'].attrs[key]

    # Read the analysis groups
    for group_name in groups:
        if group_name != 'Zemax Metadata':
            analysis_group = file[group_name]
            print('\nAnalysis: ', group_name)
            # For each Analysis Group we loop over subgroups
            for subgroup_key in analysis_group.keys():
                subgroup = analysis_group[subgroup_key]
                print('Subgroup #', subgroup_key)
                # List the metadata of the subgroup
                for att_key in subgroup.attrs.keys():
                    print('     {} : {}'.format(att_key, subgroup.attrs[att_key]))

    file.close()

    return zemax_metadata


def get_fields(system, info=False):
    """
    Read the Field Points for a given Optical System
    :param system: a system loaded with the Python Standalone Application
    :param info: boolen, whether to print the information
    :return: field_points, an array of size [N_fields, 2]
    """

    system_data = system.SystemData
    fields = system_data.Fields
    N_fields = fields.NumberOfFields

    if info is True:
        print("\nReading Field Points")
        print("Total Number of Field Points: %d" % N_fields)

    field_points = np.zeros((N_fields, 2))

    for k in np.arange(1, N_fields + 1):
        _field = fields.GetField(k)
        field_points[k - 1, 0] = _field.X
        field_points[k - 1, 1] = _field.Y

        if info is True:
            print("X = %.4f, Y = %.4f" % (_field.X, _field.Y))

    return field_points


def get_wavelengths(system, info=False):
    """
    Read the Wavelengths for a given Optical System
    :param system: a system loaded with the Python Standalone Application
    :param info: boolen, whether to print the information
    :return: field_points, an array of size [N_wavelengths]
    """

    system_data = system.SystemData
    wavelengths = system_data.Wavelengths
    N_wavelengths = wavelengths.NumberOfWavelengths

    if info is True:
        print("\nReading Wavelengths")
        print("Total Number of Wavelengths: %d" % N_wavelengths)

    wavelength_array = np.zeros(N_wavelengths)

    for k in np.arange(1, N_wavelengths + 1):
        _wave = wavelengths.GetWavelength(k).Wavelength
        wavelength_array[k - 1] = _wave

        if info is True:
            print("%.5f microns" % _wave)

    return wavelength_array


def define_pupil_sampling(r_obsc, N_rays, mode='random'):

    if mode == 'random':
        # Ideally we want to define a Uniform Distribution for the (X, Y) points in the unit disk
        # but sampling from the radial coordinates
        # If one defines a uniform distribution for (r, theta) the resulting (x, y) coordinates won't follow
        # a Uniform Distribution! there will be more points concentrated around de centre...

        # We can fix that by using the "Probability Density Under Transformation" as described in:
        # https://www.cs.cornell.edu/courses/cs6630/2015fa/notes/pdf-transform.pdf
        # Sampling as: r -> sqrt(eps_1) and theta -> 2 pi eps_2, with eps_1 and eps_2 uniform distributions [0, 1]
        # results in a (X, Y) which is uniform!

        r = np.sqrt(np.random.uniform(low=r_obsc**2, high=1.0, size=N_rays))
        theta = np.random.uniform(low=0.0, high=2 * np.pi, size=N_rays)
        px, py = r * np.cos(theta), r * np.sin(theta)

    else:
        raise NotImplementedError
        # TODO: Implement a different pupil sampling, if needed

    return px, py


def create_zemax_filename_MC(AO_mode, FPRS_MC_instance, scale, IPO_MC_instance, IFUpath, IFU_MC_instance,
                             grating, ISP_MC_instance):
    """
    Newest version that accommodates the new naming convention for the Monte Carlos
    which includes the instance number of each subsystem

    We also get rid of the "system" option that allowed us to choose between ELT, HARMONI and IFS
    since all the analyses are done for the HARMONI mode only.

    The Zemax E2E files follow the filename structure of:
        FPRS_'AO_mode'_MC_T#### + CRYO_PO'SpaxelScale'_MC_T#### + IFU-'IFUpath'_MC_T#### + SPEC-'Grating'_MC_T####.zmx

    Bear in mind that at the moment a list of a single item with the filename is returned,
    this is something undesirable that we inherited from the loop_over_files functionality
    """

    fprs = "FPRS_%s_MC_T%s" % (AO_mode, FPRS_MC_instance)
    ipo = "CRYO_PO%s_MC_T%s" % (scale, IPO_MC_instance)
    if scale != "4x4":
        ifu = "IFU" + IFUpath
    else:
        # For 4x4 the IFU is reversed
        ifu = "IFU-" + IFUpath
    ifu += "_MC_T%s" % IFU_MC_instance

    if IFUpath == "AB" or IFUpath == "EF":
        # For 4x4 the SPEC is also reversed, we add a '-' sign
        spec = "SPEC-" + grating if scale == "4x4" else "SPEC" + grating

    if IFUpath == "CD" or IFUpath == "GH":
        spec = "SPEC" + grating if scale == "4x4" else "SPEC-" + grating
    spec += "_MC_T%s" % ISP_MC_instance

    filename = fprs + "_" + ipo + "_" + ifu + "_" + spec + ".zmx"
    # Save settings:
    settings = {"AO_MODE": AO_mode, "SPAX_SCALE": scale, "IFU_PATH": IFUpath, "GRATING": grating,
                "FPRS_MC": FPRS_MC_instance, "IPO_MC": IPO_MC_instance,
                "IFU_MC": IFU_MC_instance, "ISP_MC": ISP_MC_instance}

    print("\nCreating Zemax Filename:")
    print("AO_MODE: ", AO_mode)
    print("SPAX_SCALE: ", scale)
    print("IFU_PATH: ", IFUpath)
    print("GRATING: ", grating)
    print("Subsystem Monte Carlo Instances:")
    print("FPRS_MC: ", FPRS_MC_instance)
    print("IPO_MC: ", IPO_MC_instance)
    print("IFU_MC: ", IFU_MC_instance)
    print("ISP_MC: ", ISP_MC_instance)
    print("Filename: ", filename)

    return [filename], [settings]


def create_zemax_file_list(which_system,
                           AO_modes=('NOAO', 'LTAO', 'SCAO', 'HCAO'),
                           scales=('4x4', '10x10', '20x20', '60x30'),
                           IFUs=('AB',),
                           grating=('VIS', 'IZJ', 'HK', 'IZ', 'J', 'H', 'K', 'Z_HIGH', 'H_HIGH', 'K_LONG', 'K_SHORT')):
    """
    Creates a list of Zemax file names using the naming convention that Matthias defined for the E2E model

    ['CRYO_PO4x4_IFU-AB_SPEC-VIS.zmx',
    'CRYO_PO4x4_IFU-AB_SPEC-IZJ.zmx',
    'CRYO_PO4x4_IFU-AB_SPEC-HK.zmx',
    ...,]

    Output: file_list, and settings_list

    settings_list is a list of dictionaries containing the setting (scale, IFU, spectral band) of each file
    we need that in the future to postprocess the results of each Zemax file

    :param which_system: which system do we want to analyze, can be 'ELT' or 'HARMONI' or 'IFS'
    :param AO_modes: a list of AO modes that can be ['NOAO', 'LTAO', 'SCAO', 'HCAO']
    :param scales: a list of Spaxel Scales that can be ['4x4', '10x10', '20x20', '60x30']
    :param IFUs: a list of IFU subfields that can be ['AB', ...]
    :param grating: a list of Gratings that can be ['VIS', 'IZJ', 'HK', 'IZ', 'J', 'H', 'K', 'Z_HIGH', 'H_HIGH', 'K_LONG', 'K_SHORT']
    :return:
    """

    if which_system not in ['ELT', 'HARMONI', 'IFS']:
        raise ValueError("System must be 'ELT' or 'HARMONI' or 'IFS'")

    if which_system == 'IFS':     # Only the IFS, starting at the CRYO

        file_list = []
        settings_list = []          # List of settings dictionaries
        for scale in scales:
            name_scale = 'CRYO_PO' + scale
            for ifu in IFUs:
                if scale != '4x4':
                    name_IFU = '_IFU' + ifu
                else:
                    name_IFU = '_IFU-' + ifu            # For 4x4 the IFU is reversed
                for spec in grating:

                    if ifu == 'AB' or ifu == 'EF':
                        # For 4x4 the SPEC is also reversed, we add a '-' sign
                        name_spec = '_SPEC-' + spec if scale == '4x4' else '_SPEC' + spec

                    if ifu == 'CD' or ifu == 'GH': # _SPEC-K_LONG
                        name_spec = '_SPEC' + spec if scale == '4x4' else '_SPEC-' + spec

                    filename = name_scale + name_IFU + name_spec + '.zmx'
                    file_list.append(filename)

                    # Save settings:
                    sett_dict = {'system': which_system, 'scale': scale, 'ifu': ifu, 'grating': spec}
                    settings_list.append(sett_dict)

        return file_list, settings_list

    if which_system == 'HARMONI':     # The same as 'IFS' but including the FPRS with the AO_modes

        file_list = []
        settings_list = []
        # Create a list of file names from the CRYO onwards
        ifs_file_list, ifs_sett_list = create_zemax_file_list(which_system='IFS', scales=scales, IFUs=IFUs, grating=grating)
        # Add FPRS_ + AO mode as prefix
        for AO_mode in AO_modes:
            name_fprs = 'FPRS_' + AO_mode + '_'
            for ifs_file, ifs_sett in zip(ifs_file_list, ifs_sett_list):
                filename = name_fprs + ifs_file
                file_list.append(filename)

                ifs_sett['AO_mode'] = AO_mode       # Add AO mode to the settings
                ifs_sett['system'] = which_system   # Override the system mode to HARMONI, not IFS
                settings_list.append(ifs_sett)

        return file_list, settings_list

    if which_system == 'ELT':         # The same as 'HARMONI' but including the ELT

        # Create a list of file names from HARMONI
        harmoni_files_list, harmoni_sett_list = create_zemax_file_list(which_system='HARMONI', AO_modes=AO_modes,
                                                                       scales=scales, IFUs=IFUs, grating=grating)
        # Add ELT as prefix
        file_list = ['ELT_' + harmoni_file for harmoni_file in harmoni_files_list]
        settings_list = []
        for harm_sett in harmoni_sett_list:
            harm_sett['system'] = which_system      # Override the system mode to ELT, not HARMONI
            settings_list.append(harm_sett)

        return file_list, settings_list

    return


def expand_slicer_aperture(system):
    """
    For the 4x4 spaxel scale, a significant fraction of the rays can sometimes get vignetted at the Image Slicer
    This introduces a bias in the RMS WFE calculation. To avoid this, we modify the Image Slicer aperture definition
    so that all rays get through. Consequently, enough pupil rays are traced to get an unbiased estimation of RMS WFE

    :param system: the Optical System
    :return:
    """

    # First of all, we need to find the Surface Number for the IMAGE SLICER
    N_surfaces = system.LDE.NumberOfSurfaces
    surface_names = {}  # A dictionary of surface number -> surface comment
    for k in np.arange(1, N_surfaces):
        surface_names[k] = system.LDE.GetSurfaceAt(k).Comment
    # find the Slicer surface number
    try:
        # The naming convention for this surface has changed. Not the same for Nominal Design as Monte Carlos
        slicer_num = list(surface_names.keys())[list(surface_names.values()).index('Slicer Mirror')]
    except ValueError:
        slicer_num = list(surface_names.keys())[list(surface_names.values()).index('IFU ISA')]
    slicer = system.LDE.GetSurfaceAt(slicer_num)

    # Read Current Aperture Settings
    apt_type = slicer.ApertureData.CurrentType
    # print("Aperture type: ", apt_type)
    if apt_type == 4:  # 4 is Rectangular aperture
        current_apt_sett = slicer.ApertureData.CurrentTypeSettings
        # print("Current Settings:")
        x0 = current_apt_sett._S_RectangularAperture.XHalfWidth
        y0 = current_apt_sett._S_RectangularAperture.YHalfWidth
        # If the Y aperture hasn't been changed already, we change it here to 999 mm to get all rays through
        if y0 != 999:
            # Change Settings
            aperture_settings = slicer.ApertureData.CreateApertureTypeSettings(
                constants.SurfaceApertureTypes_RectangularAperture)
            aperture_settings._S_RectangularAperture.XHalfWidth = x0
            aperture_settings._S_RectangularAperture.YHalfWidth = 999
            slicer.ApertureData.ChangeApertureTypeSettings(aperture_settings)

            current_apt_sett = slicer.ApertureData.CurrentTypeSettings
            # Notify that we have successfully modified the aperture
            print("Changing aperture of surface: ", slicer.Comment)
            print("New Settings:")
            print("X_HalfWidth = %.2f" % current_apt_sett._S_RectangularAperture.XHalfWidth)
            print("Y_HalfWidth = %.2f" % current_apt_sett._S_RectangularAperture.YHalfWidth)

    return


class AnalysisGeneric(object):

    """
    Object to perform fairly generic analyses on Zemax files created by the E2E model

    If you want to run your own custom analysis the best way is to use Class Inheritance
    Create a child class from this AnalysisGeneric object
    Define a method (function, typically static method) in the child class representing the 'analysis_function'
    that self.run_analysis will call. That method / function should perform all the operations that
    you are interested in for your analysis

    """

    def __init__(self, zosapi, *args, **kwargs):
        """
        We basically need a ZOS API Standalone Application to
        open and close the files we want to analyze
        :param zosapi:
        :param args:
        :param kwargs:
        """

        self.zosapi = zosapi

    def run_analysis(self, analysis_function, results_shapes, results_names, files_dir, zemax_file, results_path,
                     wavelength_idx=None, configuration_idx=None, surface=None, *args, **kwargs):
        """

        Run some analysis on the Optical System over a range of Wavelengths and Configurations
        For this, we need some Python function "analysis_function" that receives as input
        (Optical System, Wavelength Number, Configuration Number, Surface Number, *args, **kwargs)
        *args and **kwargs represent whatever other information our function might need for the
        calculations. This could be something like Number of Fields, Number of Rays, or some other
        Zemax analysis parameters

        The "run_analysis" function will loop over our chosen Wavelengths and Configurations (Slices)
        and apply the specified "analysis_function"

        ~~ Extremely IMPORTANT! ~~
        We are aware that the shape and type of results will vary for each analysis
        In order to accommodate this variability we use the following strategy:

            You can customize the output of 'run_analysis' by using a list 'results_names'
            that contains the names of the variables that your 'analysis_function' will provide
            as well as a list 'results_shapes' containing the shape of such variables

            This code 'run_analysis' will dynamically create such variables using the globals() dictionary
            such that any array of arbitrary shape [N_waves, N_configs, Nx, Ny, ..., Nz] can be
            accommodated.

            For example, if we want to run a Spot Diagram analysis we would call
            'run_analysis' with analaysis_function=SpotDiagram.analysis_function and provide as arguments
            results_names = ['xy'], results_shapes = [(N_fields, N_rays^2, 2)]

            'run_analysis' would then loop over those lists (in this case we only have 1 array)
            and create the global variable xy, with shape [N_waves, N_config, N_fields, N_rays^2, 2]
            and loop over the Wavelengths and Configurations applying the 'analysis_function'

        All of this is done for a given "zemax_file" in the "files_dir". This is a template so if you
        want to run the analysis over multiple files you can create a Child Class and add a method
        that loops over the files (see RMS_ XXX)

        :param analysis_function: arbitrary function that does the calculations needed for our analyses
        :param results_shapes: list of shapes for the arrays that 'analysis_function' produces
        :param results_shape: list of names of the arrays that 'analysis_function' produces
        :param files_dir: path to where you keep the Zemax files
        :param zemax_file: the name of the Zemax file you want to analyze
        :param results_path: path where you will save the results
        :param wavelength_idx: list of Zemax wavelength numbers. If left as None, it will analyze all Wavelengths
        :param configuration_idx: list of Zemax configurations. If left as None, it will analyze all Configurations
        :param surface: zemax surface number at which we will compute the analysis. If left as None, it will use the Image Plane
        :param args: whatever other arguments your "analysis_function" may need
        :param kwargs: whatever other keyword arguments your "analysis_function" may need
        :return:
        """

        # check that the file name is correct and the zemax file exists
        if os.path.exists(os.path.join(files_dir, zemax_file)) is False:
            raise FileExistsError("%s does NOT exist" % zemax_file)

        print("\nOpening Zemax File: ", zemax_file)
        self.zosapi.OpenFile(os.path.join(files_dir, zemax_file), False)
        file_name = zemax_file.split(".")[0]            # Remove the ".zmx" suffix

        # Check if the results directory already exists
        results_dir = os.path.join(results_path, file_name)
        print("Results will be saved in: ", results_dir)
        if not os.path.exists(results_dir):
            os.mkdir(results_dir)           # If not, create the directory to store results

        # Get some info on the system
        system = self.zosapi.TheSystem                      # The Optical System
        MCE = system.MCE                                    # Multi Configuration Editor
        LDE = system.LDE                                    # Lens Data Editor
        N_surfaces = LDE.NumberOfSurfaces
        N_configs = MCE.NumberOfConfigurations

        # print("\nSystem Data:")
        print("Number of Surfaces: ", N_surfaces)
        # print("Number of Configurations / Slices: ", N_configs)
        _wavelengths = get_wavelengths(system, info=False)
        N_waves = _wavelengths.shape[0]
        # field_points = get_fields(system, info=False)

        # Let's deal with the data format
        if wavelength_idx is None:         # We use all the available wavelengths
            wavelengths = _wavelengths
            wavelength_idx = np.arange(1, N_waves + 1)
        else:     # We have received a list of wavelength indices
            wavelengths = [_wavelengths[idx - 1] for idx in wavelength_idx]
            N_waves = len(wavelength_idx)

        if configuration_idx is None:
            configurations = np.arange(1, N_configs + 1)
        else:       # We have received a list of configurations
            configurations = configuration_idx
            N_configs = len(configuration_idx)
        N_slices = N_configs

        if surface is None:     # we go to the final image plane
            surface = system.LDE.NumberOfSurfaces - 1

        N_surfaces = system.LDE.NumberOfSurfaces
        if surface != (N_surfaces - 1):
            # The Surface is not the final Image Plane. We must force Zemax to IGNORE
            # all surfaces after our surface of interest so that the RWCE operands works
            surfaces_to_ignore = np.arange(surface + 1, N_surfaces)
            for surf_number in surfaces_to_ignore:
                _surf = system.LDE.GetSurfaceAt(surf_number)
                _surf.TypeData.IgnoreSurface = True

            last_surface = system.LDE.GetSurfaceAt(surface)
            thickness = last_surface.Thickness
            print("Thickness: ", last_surface.Thickness)
            # image = system.LDE.GetSurfaceAt(N_surfaces - 1)
            # print("\nImage Name: ", image.Comment)

            # Temporarily set the Thickness to 0
            last_surface.Thickness = 0.0
            print("Thickness: ", last_surface.Thickness)

        # Double check we are using the right surface
        print("\nSurface Name: ", system.LDE.GetSurfaceAt(surface).Comment)

        # Customize the initilization of the results array so that we can vary the shapes
        # some analysis may be [spaxels_per_slice, N_slices, N_waves] such as an RMS WFE map
        # others might be [N_fields, N_rays, N_slices, N_waves] such as a Spot Diagram

        # print("\nDynamically creating Global Variables to store results")
        for _name, _shape in zip(results_names, results_shapes):
            # print("Variable: %s with shape (N_waves, N_configs) + " % _name, _shape)
            globals()[_name] = np.empty((N_waves, N_slices) + _shape)
            # print(globals()[_name].shape)

        # print("\nApplying 'analysis_function': ", analysis_function.__name__)
        print("At Surface #%d | Image Plane is #%d" % (surface, N_surfaces - 1))
        print("For Wavelength Numbers: ", wavelength_idx)
        print("For Configurations %d -> %d" % (configurations[0], configurations[-1]))

        # Main part of the Analysis. Here we loop over the Wavelengths and Configurations
        start = time()
        print("\nAnalysis Started")
        for k, (wave_idx, wavelength) in enumerate(zip(wavelength_idx, wavelengths)):
            print("Wavelength #%d: %.3f" % (wave_idx, wavelength))

            for j, config in enumerate(configurations):

                # print("Wavelength #%d: %.3f | config: %d" %(wave_idx, wavelength, config))

                # Here is where the arbitrary "analysis_function" will be called for each Wavelength and Configuration
                results = analysis_function(system=system, wave_idx=wave_idx, config=config, surface=surface,
                                            *args, **kwargs)

                # Store the result values in the arrays:
                for _name, res in zip(results_names, results):
                    globals()[_name][k, j] = res

        # Unignore the surfaces
        if surface != (N_surfaces - 1):
            last_surface = system.LDE.GetSurfaceAt(surface)
            last_surface.Thickness = thickness
            for surf_number in surfaces_to_ignore:
                _surf = system.LDE.GetSurfaceAt(surf_number)
                _surf.TypeData.IgnoreSurface = False

        self.zosapi.CloseFile(save=False)
        analysis_time = time() - start
        print("\nAnalysis done in %.1f seconds" % analysis_time)
        time_per_wave = analysis_time / len(wavelengths)
        print("%.2f seconds per Wavelength" % time_per_wave)

        # local_keys = [key for key in globals().keys()]
        # print(local_keys)

        # We return a list containing the results, and the value of the wavelengths
        list_results = [globals()[_name] for _name in results_names]
        list_results.append(wavelengths)

        return list_results

    def template_analysis_function(self, system, wave_idx, config, surface, *args, **kwargs):
        """
        Template for the analysis function

        Typically we pass as arguments the 'system' so that any information about it can be accessed
        The Zemax Wavelength number 'wave_idx' and Configuration 'config' at which we will apply the analysis
        And the Zemax surface number 'surface' at which we will apply the analysis

        We also include *args and **kwargs to accommodate other parameters your analysis may need
        For instance, the number of rings to be used in RMS wavefront calculations

        :param system:
        :param wave_idx:
        :param config:
        :param surface:
        :param args:
        :param kwargs:
        :return:
        """

        # Write your function operations here

        # This will typically contain a lot of ZOS API commands such as:
        # - Setting the Current configuration:          system.MCE.SetCurrentConfiguration(config)
        # - Reading the Field Points:                   sysField = system.SystemData.Fields
        #                                               fx = sysField.GetField(1).X
        #                                               fy = sysField.GetField(1).Y
        # - Running some Ray Trace:                     raytrace = system.Tools.OpenBatchRayTrace()
        #                                               normUnPolData = raytrace.CreateNormUnpol(N_rays,
        #                                                                        constants.RaysType_Real, toSurface)

        # You should return a list of arrays

        return

    def save_hdf5(self, analysis_name, analysis_metadata, list_results, results_names, file_name, file_settings, results_dir):
        """
        Some analysis take a long time to run and, at some point, we might want to revisit the results.
        Rather than repeating the analysis, we'd like to save them in HDF5 format and access them in the future

        This function takes care of that. In the HDF5 files, we follow a structure of groups / subgroups
        with a group dedicated to the Zemax Metadata, followed by a group for each analysis type (RMS_WFE, EE, etc)

        If we run the same analysis, but under different parameters (or at a later date), the results are stored as
        separate subgroups within that analysis type group

        :param analysis_name: string to identify the type of analysis we ran (eg RMS_WFE)
        :param analysis_metadata: dictionary containing metadata of the analysis
        :param list_results: a list containing the arrays for the results of the analysis
        :param results_names: a list containing the labels for the results
        :param file_name: name for the HDF5 file
        :param file_settings: settings of the Zemax file used for the analysis
        :param results_dir: path to the results, a separate folder 'HDF5' will be created there
        :return:
        """

        # First thing is to create a separate folder within the results directory for this analysis
        hdf5_dir = os.path.join(results_dir, 'HDF5')
        print("Analysis Results will be saved in folder: ", hdf5_dir)
        if not os.path.exists(hdf5_dir):
            os.mkdir(hdf5_dir)           # If not, create the directory to store results

        hdf5_file = file_name + '.hdf5'
        # Check whether the file already exists
        if os.path.isfile(os.path.join(hdf5_dir, hdf5_file)):   # Overwrite it
            print("HDF5 file already exists. Adding analysis")
            with h5py.File(os.path.join(hdf5_dir, hdf5_file), 'r+') as f:

                # Here we have 2 options: either we are adding another analysis type or we are adding the same type
                # but with different settings (such as at a different surface)
                file_keys = list(f.keys())
                if analysis_name in file_keys:
                    print("Analysis type already exists")
                    analysis_group = f[analysis_name]
                    # we need to know how many analyses of the same type already exist
                    subgroup_keys = list(analysis_group.keys())
                    subgroup_number = len(subgroup_keys)        # if [0, 1] already exist we call it '2'
                    subgroup = analysis_group.create_group(str(subgroup_number))
                    # Save results datasets
                    for array, array_name in zip(list_results, results_names + ['WAVELENGTHS']):
                        data = subgroup.create_dataset(array_name, data=array)
                    # Save analysis metadata
                    subgroup.attrs['E2E Python Git Hash'] = sha
                    subgroup.attrs['Analysis Type'] = analysis_name
                    date_created = datetime.datetime.now().strftime("%c")
                    subgroup.attrs['Date'] = date_created
                    subgroup.attrs['At Surface #'] = file_settings['surface']
                    # Add whatever extra metadata we have:
                    for key, value in analysis_metadata.items():
                        subgroup.attrs[key] = value

                else:       # It's a new analysis type

                    # (2) Create a Group for this analysis
                    analysis_group = f.create_group(analysis_name)
                    # (3) Create a Sub-Group so that we can have the same analysis at multiple surfaces / wavelength ranges
                    subgroup = analysis_group.create_group('0')
                    # Save results datasets
                    for array, array_name in zip(list_results, results_names + ['WAVELENGTHS']):
                        data = subgroup.create_dataset(array_name, data=array)
                    # Save analysis metadata
                    subgroup.attrs['E2E Python Git Hash'] = sha
                    subgroup.attrs['Analysis Type'] = analysis_name
                    date_created = datetime.datetime.now().strftime("%c")
                    subgroup.attrs['Date'] = date_created
                    subgroup.attrs['At Surface #'] = file_settings['surface']
                    # Add whatever extra metadata we have:
                    for key, value in analysis_metadata.items():
                        subgroup.attrs[key] = value

        else:       # File does not exist, we create it now
            print("Creating HDF5 file: ", hdf5_file)
            with h5py.File(os.path.join(hdf5_dir, hdf5_file), 'w') as f:

                # (1) Save Zemax Metadata
                zemax_metadata = f.create_group('Zemax Metadata')
                zemax_metadata.attrs['(1) Zemax File'] = file_name
                zemax_metadata.attrs['(2) System Mode'] = file_settings['system']
                zemax_metadata.attrs['(3) Spaxel Scale'] = file_settings['scale']
                zemax_metadata.attrs['(4) IFU'] = file_settings['ifu']
                zemax_metadata.attrs['(5) Grating'] = file_settings['grating']
                AO = file_settings['AO_mode'] if 'AO_mode' in list(file_settings.keys()) else 'NA'
                zemax_metadata.attrs['(6) AO Mode'] = AO

                # (2) Create a Group for this analysis
                analysis_group = f.create_group(analysis_name)

                # (3) Create a Sub-Group so that we can have the same analysis at multiple surfaces / wavelength ranges
                subgroup = analysis_group.create_group('0')
                # Save results datasets
                for array, array_name in zip(list_results, results_names + ['WAVELENGTHS']):
                    data = subgroup.create_dataset(array_name, data=array)
                # Save analysis metadata
                subgroup.attrs['E2E Python Git Hash'] = sha
                subgroup.attrs['Analysis Type'] = analysis_name
                date_created = datetime.datetime.now().strftime("%c")
                subgroup.attrs['Date'] = date_created
                subgroup.attrs['At Surface #'] = file_settings['surface']
                # Add whatever extra metadata we have:
                for key, value in analysis_metadata.items():
                    subgroup.attrs[key] = value

        return

# {***} Rationale for the AnalysisFast {***}
# AnalysisGeneric works fine, but it's not very efficient / fast. It is based on looping through Wavelength index
# and Configuration and running a set of operations for just 1 wavelength / 1 config.
# At some point, we realized that it could be faster to only loop through configurations and run a set of operations
# for all available Wavelength indices. For example, calculating the RMS WFE, it's much faster to just add operands
# for all wavelengths and field points of a given Configuration, update the Merit Function and then read the operands

# For that reason, we created AnalysisFast, a faster version of AnalysisGeneric that only loops over Configurations
# a whole suite of inherited analysis classes is available. They are very similar to their AnalysisGeneric
# counterparts, except for the fact that AnalysisGeneric uses array shapes of [N_waves, N_configs, (shape)]
# while AnalysisFast uses shapes of [N_configs, (shape)], with (shape) typically starting with N_waves,
# so there's a "flip" one should be aware of


class AnalysisFast(object):
    """
    Faster version that doesn't look over the wavelengths
    """

    def __init__(self, zosapi, *args, **kwargs):
        """
        We basically need a ZOS API Standalone Application to
        open and close the files we want to analyze
        :param zosapi:
        :param args:
        :param kwargs:
        """

        self.zosapi = zosapi

    def run_analysis(self, analysis_function, results_shapes, results_names, files_dir, zemax_file, results_path,
                     wavelength_idx=None, configuration_idx=None, surface=None,
                     *args, **kwargs):

        print("\nOpening Zemax File: ", zemax_file)
        # check that the file name is correct and the zemax file exists
        if os.path.exists(os.path.join(files_dir, zemax_file)) is False:
            raise FileExistsError("%s does NOT exist" % zemax_file)


        self.zosapi.OpenFile(os.path.join(files_dir, zemax_file), False)
        file_name = zemax_file.split(".")[0]  # Remove the ".zmx" suffix

        # # Check if the results directory already exists
        # results_dir = os.path.join(results_path, file_name)
        # print("Results will be saved in: ", results_dir)
        # if not os.path.exists(results_dir):
        #     os.mkdir(results_dir)  # If not, create the directory to store results

        # Get some info on the system
        system = self.zosapi.TheSystem  # The Optical System
        MCE = system.MCE  # Multi Configuration Editor
        LDE = system.LDE  # Lens Data Editor
        N_surfaces = LDE.NumberOfSurfaces
        N_configs = MCE.NumberOfConfigurations

        # print("\nSystem Data:")
        print("Number of Surfaces: ", N_surfaces)
        # print("Number of Configurations / Slices: ", N_configs)
        _wavelengths = get_wavelengths(system, info=False)
        N_waves = _wavelengths.shape[0]
        # field_points = get_fields(system, info=False)

        # Let's deal with the data format
        if wavelength_idx is None:  # We use all the available wavelengths
            wavelengths = _wavelengths
            wavelength_idx = np.arange(1, N_waves + 1)
        else:  # We have received a list of wavelength indices
            wavelengths = [_wavelengths[idx - 1] for idx in wavelength_idx]
            N_waves = len(wavelength_idx)

        if configuration_idx is None:
            configurations = np.arange(1, N_configs + 1)
        else:  # We have received a list of configurations
            configurations = configuration_idx
            N_configs = len(configuration_idx)
        N_slices = N_configs

        if surface is None:  # we go to the final image plane
            surface = system.LDE.NumberOfSurfaces - 1

        N_surfaces = system.LDE.NumberOfSurfaces
        if surface != (N_surfaces - 1):
            # The Surface is not the final Image Plane. We must force Zemax to IGNORE
            # all surfaces after our surface of interest so that the RWCE operands works
            surfaces_to_ignore = np.arange(surface + 1, N_surfaces)
            for surf_number in surfaces_to_ignore:
                _surf = system.LDE.GetSurfaceAt(surf_number)
                _surf.TypeData.IgnoreSurface = True

            last_surface = system.LDE.GetSurfaceAt(surface)
            thickness = last_surface.Thickness
            print("Thickness: ", last_surface.Thickness)
            # image = system.LDE.GetSurfaceAt(N_surfaces - 1)
            # print("\nImage Name: ", image.Comment)

            # Temporarily set the Thickness to 0
            last_surface.Thickness = 0.0
            print("Thickness: ", last_surface.Thickness)

        # Double check we are using the right surface
        print("\nSurface Name: ", system.LDE.GetSurfaceAt(surface).Comment)

        # Customize the initilization of the results array so that we can vary the shapes
        # some analysis may be [spaxels_per_slice, N_slices, N_waves] such as an RMS WFE map
        # others might be [N_fields, N_rays, N_slices, N_waves] such as a Spot Diagram

        # print("\nDynamically creating Global Variables to store results")
        for _name, _shape in zip(results_names, results_shapes):
            # Notice the change with respect to AnalysisGeneric, here we have N_configs + (shape)
            globals()[_name] = np.empty((N_slices, ) + _shape)
            # print(globals()[_name].shape)

        # print("\nApplying 'analysis_function': ", analysis_function.__name__)
        print("At Surface #%d | Image Plane is #%d" % (surface, N_surfaces - 1))
        print("For Wavelength Numbers: ", wavelength_idx)
        print("For Configurations %d -> %d" % (configurations[0], configurations[-1]))

        start = time()
        print("\nAnalysis Started: [FAST MODE]")
        # We only loop through configurations, which is much faster!
        for j, config in enumerate(configurations):

            # print("Config #%d: " % (config))

            # Here is where the arbitrary "analysis_function" will be called for each Wavelength and Configuration
            results = analysis_function(system=system, wavelength_idx=wavelength_idx, config=config, surface=surface,
                                        *args, **kwargs)

            # Store the result values in the arrays:
            for _name, res in zip(results_names, results):
                globals()[_name][j] = res

        # Unignore the surfaces
        if surface != (N_surfaces - 1):
            last_surface = system.LDE.GetSurfaceAt(surface)
            last_surface.Thickness = thickness
            for surf_number in surfaces_to_ignore:
                _surf = system.LDE.GetSurfaceAt(surf_number)
                _surf.TypeData.IgnoreSurface = False

        self.zosapi.CloseFile(save=False)
        analysis_time = time() - start
        print("\nAnalysis done in %.1f seconds" % analysis_time)
        time_per_wave = analysis_time / len(wavelengths)
        print("%.2f seconds per Wavelength" % time_per_wave)

        # local_keys = [key for key in globals().keys()]
        # print(local_keys)

        # We return a list containing the results, and the value of the wavelengths
        list_results = [globals()[_name] for _name in results_names]
        list_results.append(wavelengths)

        return list_results

    def save_hdf5(self, analysis_name, analysis_metadata, list_results, results_names, file_name, file_settings, results_dir):
        """
        Some analysis take a long time to run and, at some point, we might want to revisit the results.
        Rather than repeating the analysis, we'd like to save them in HDF5 format and access them in the future

        This function takes care of that. In the HDF5 files, we follow a structure of groups / subgroups
        with a group dedicated to the Zemax Metadata, followed by a group for each analysis type (RMS_WFE, EE, etc)

        If we run the same analysis, but under different parameters (or at a later date), the results are stored as
        separate subgroups within that analysis type group

        :param analysis_name: string to identify the type of analysis we ran (eg RMS_WFE)
        :param analysis_metadata: dictionary containing metadata of the analysis
        :param list_results: a list containing the arrays for the results of the analysis
        :param results_names: a list containing the labels for the results
        :param file_name: name for the HDF5 file
        :param file_settings: settings of the Zemax file used for the analysis
        :param results_dir: path to the results, a separate folder 'HDF5' will be created there
        :return:
        """

        # First thing is to create a separate folder within the results directory for this analysis
        hdf5_dir = os.path.join(results_dir, 'HDF5')
        print("Analysis Results will be saved in folder: ", hdf5_dir)
        if not os.path.exists(hdf5_dir):
            os.mkdir(hdf5_dir)           # If not, create the directory to store results

        hdf5_file = file_name + '.hdf5'
        # Check whether the file already exists
        if os.path.isfile(os.path.join(hdf5_dir, hdf5_file)):   # Overwrite it
            print("HDF5 file already exists. Adding analysis")
            with h5py.File(os.path.join(hdf5_dir, hdf5_file), 'r+') as f:

                # Here we have 2 options: either we are adding another analysis type or we are adding the same type
                # but with different settings (such as at a different surface)
                file_keys = list(f.keys())
                if analysis_name in file_keys:
                    print("Analysis type already exists")
                    analysis_group = f[analysis_name]
                    # we need to know how many analyses of the same type already exist
                    subgroup_keys = list(analysis_group.keys())
                    subgroup_number = len(subgroup_keys)        # if [0, 1] already exist we call it '2'
                    subgroup = analysis_group.create_group(str(subgroup_number))
                    # Save results datasets
                    for array, array_name in zip(list_results, results_names + ['WAVELENGTHS']):
                        data = subgroup.create_dataset(array_name, data=array)
                    # Save analysis metadata
                    subgroup.attrs['E2E Python Git Hash'] = sha
                    subgroup.attrs['Analysis Type'] = analysis_name
                    date_created = datetime.datetime.now().strftime("%c")
                    subgroup.attrs['Date'] = date_created
                    subgroup.attrs['At Surface #'] = file_settings['surface']
                    # Add whatever extra metadata we have:
                    for key, value in analysis_metadata.items():
                        subgroup.attrs[key] = value

                else:       # It's a new analysis type

                    # (2) Create a Group for this analysis
                    analysis_group = f.create_group(analysis_name)
                    # (3) Create a Sub-Group so that we can have the same analysis at multiple surfaces / wavelength ranges
                    subgroup = analysis_group.create_group('0')
                    # Save results datasets
                    for array, array_name in zip(list_results, results_names + ['WAVELENGTHS']):
                        data = subgroup.create_dataset(array_name, data=array)
                    # Save analysis metadata
                    subgroup.attrs['E2E Python Git Hash'] = sha
                    subgroup.attrs['Analysis Type'] = analysis_name
                    date_created = datetime.datetime.now().strftime("%c")
                    subgroup.attrs['Date'] = date_created
                    subgroup.attrs['At Surface #'] = file_settings['surface']
                    # Add whatever extra metadata we have:
                    for key, value in analysis_metadata.items():
                        subgroup.attrs[key] = value

        else:       # File does not exist, we create it now
            print("Creating HDF5 file: ", hdf5_file)
            with h5py.File(os.path.join(hdf5_dir, hdf5_file), 'w') as f:

                # (1) Save Zemax Metadata
                zemax_metadata = f.create_group('Zemax Metadata')
                zemax_metadata.attrs['(1) Zemax File'] = file_name
                zemax_metadata.attrs['(2) System Mode'] = file_settings['system']
                zemax_metadata.attrs['(3) Spaxel Scale'] = file_settings['scale']
                zemax_metadata.attrs['(4) IFU'] = file_settings['ifu']
                zemax_metadata.attrs['(5) Grating'] = file_settings['grating']
                AO = file_settings['AO_mode'] if 'AO_mode' in list(file_settings.keys()) else 'NA'
                zemax_metadata.attrs['(6) AO Mode'] = AO

                # (2) Create a Group for this analysis
                analysis_group = f.create_group(analysis_name)

                # (3) Create a Sub-Group so that we can have the same analysis at multiple surfaces / wavelength ranges
                subgroup = analysis_group.create_group('0')
                # Save results datasets
                for array, array_name in zip(list_results, results_names + ['WAVELENGTHS']):
                    data = subgroup.create_dataset(array_name, data=array)
                # Save analysis metadata
                subgroup.attrs['E2E Python Git Hash'] = sha
                subgroup.attrs['Analysis Type'] = analysis_name
                date_created = datetime.datetime.now().strftime("%c")
                subgroup.attrs['Date'] = date_created
                subgroup.attrs['At Surface #'] = file_settings['surface']
                # Add whatever extra metadata we have:
                for key, value in analysis_metadata.items():
                    subgroup.attrs[key] = value

        return


### This are the relevant analyses for the E2E model:
#   - Ensquared Energy
#   - FWHM_PSF
#   - Raytrace
#   - RMS WFE

class RMS_WFE_Analysis(AnalysisGeneric):
    """
    [NOTE] AnalysisGeneric is almost deprecated in favour of the faster version AnalysisFast
    Think of this as an 'example' for inspiration to learn how to work with the E2E models

    Example of how to use Class Inheritance with the AnalysisGeneric object
    to calculate the RMS WFE error

    In this case the "analysis_function" is "analysis_function_rms_wfe"
    """

    @staticmethod
    def analysis_function_rms_wfe(system, wave_idx, config, spaxels_per_slice, surface):
        """
        Analysis function that calculates the RMS Wavefront Error [nm]
        for an Optical System, at a given Wavelength and Configuration / Slice
        at a specified Surface

        This function will return a list of arrays:
            - RMS_WFE: array of size [N_spaxels_per_slice] containing the RMS WFE for each spaxel of the given config
            - OBJ_XY: array of size [N_spaxels_per_slice, 2] containing the (XY) coordinates at the Object plane
            - FOC_XY: same as OBJ_XY but at the Focal plane defined by Surface

        :param system: Optical System that the ZOS API has loaded
        :param wave_idx: Zemax Wavelength Number to be used in the RMS calculation
        :param config: Zemax Configuration Number, the current Slice
        :param spaxels_per_slice: how many spaxels per slice to use for sampling the slice
        :param surface: Zemax surface number at which to evaluate the RMS WFE
        :param rings: Zemax parameter for the RMS WFE calculation
        :return:
        """

        # Set Current Configuration
        system.MCE.SetCurrentConfiguration(config)

        # Get the Field Points for that configuration
        sysField = system.SystemData.Fields

        N_fields = sysField.NumberOfFields
        wavelength = system.SystemData.Wavelengths.GetWavelength(wave_idx).Wavelength

        fx_min, fy_min = sysField.GetField(1).X, sysField.GetField(1).Y
        fx_max, fy_max = sysField.GetField(3).X, sysField.GetField(3).Y

        # This assumes Rectangular Normalization
        X_MAX = np.max([np.abs(sysField.GetField(i + 1).X) for i in range(3)])
        Y_MAX = np.max([np.abs(sysField.GetField(i + 1).Y) for i in range(3)])

        # Normalized field coordinates (hx, hy)
        hx_min, hx_max = fx_min / X_MAX, fx_max / X_MAX
        hy_min, hy_max = fy_min / Y_MAX, fy_max / Y_MAX

        # print("h_x : (%.3f, %.3f)" % (hx_min, hx_max))
        # print("h_y : (%.3f, %.3f)" % (hy_min, hy_max))

        hx = np.linspace(hx_min, hx_max, spaxels_per_slice)
        hy = np.linspace(hy_min, hy_max, spaxels_per_slice)

        # The Field coordinates for the Object
        obj_xy = np.array([X_MAX * hx, Y_MAX * hy]).T
        RMS_WFE = np.empty(spaxels_per_slice)
        foc_xy = np.empty((spaxels_per_slice, 2))
        global_xy = np.empty((spaxels_per_slice, 2))
        local_xyz = np.empty((spaxels_per_slice, 3))

        raytrace = system.Tools.OpenBatchRayTrace()
        normUnPolData = raytrace.CreateNormUnpol(spaxels_per_slice, constants.RaysType_Real, surface)

        theMFE = system.MFE
        nops = theMFE.NumberOfOperands
        theMFE.RemoveOperandsAt(1, nops)
        # build merit function
        op = theMFE.GetOperandAt(1)
        op.ChangeType(constants.MeritOperandType_CONF)
        op.GetOperandCell(constants.MeritColumn_Param1).Value = config

        # Pupil Sampling (4 means an 8x8 pupil grid)
        samp = 4
        wfe_op = constants.MeritOperandType_RWRE

        # Loop over all Spaxels in the Slice
        for i, (h_x, h_y) in enumerate(zip(hx, hy)):

            # operand = constants.MeritOperandType_RWRE
            # rms = system.MFE.GetOperandValue(operand, surface, wave_idx, h_x, h_y, 0.0, 0.0, 0.0, 0.0)
            # RMS_WFE[i] = wavelength * 1e3 * rms         # We assume the Wavelength comes in Microns

            op = theMFE.AddOperand()
            op.ChangeType(wfe_op)
            op.GetOperandCell(constants.MeritColumn_Param1).Value = int(samp)
            op.GetOperandCell(constants.MeritColumn_Param2).Value = int(wave_idx)
            op.GetOperandCell(constants.MeritColumn_Param3).Value = float(h_x)
            op.GetOperandCell(constants.MeritColumn_Param4).Value = float(h_y)
            op.GetOperandCell(constants.MeritColumn_Weight).Value = 0

            # Add the ray to the RayTrace
            normUnPolData.AddRay(wave_idx, h_x, h_y, 0, 0, constants.OPDMode_None)

        # update merit function
        theMFE.CalculateMeritFunction()
        # retrieve value of each RWRE operand
        # theMCE.SetCurrentConfiguration(i)
        system.MCE.SetCurrentConfiguration(config)
        # print("N operands:", nops)
        for irow in range(2, theMFE.NumberOfOperands + 1):
            op = theMFE.GetOperandAt(irow)
            rms = op.Value
            # print(irow)
            # print("RMS: %.2f nm" % (wavelength * 1e3 * rms))
            RMS_WFE[irow - 2] = wavelength * 1e3 * rms  # We assume the Wavelength comes in Microns

        # Run the RayTrace for the whole Slice
        CastTo(raytrace, 'ISystemTool').RunAndWaitForCompletion()
        normUnPolData.StartReadingResults()
        for i in range(spaxels_per_slice):
            output = normUnPolData.ReadNextResult()
            if output[2] == 0:
                local_xyz[i, 0] = output[4]
                local_xyz[i, 1] = output[5]
                local_xyz[i, 2] = output[6]

                # Local Focal X Y
                foc_xy[i, 0] = output[4]
                foc_xy[i, 1] = output[5]

                # print("\nConfiguration #%d" % config)
                # print("fx: %.5f fy: %.5f" % (X_MAX * hx[i], Y_MAX * hy[i]))
                # print("hx: %.5f hy: %.5f" % (hx[i], hy[i]))
                # x_det, y_det = output[4], output[5]
                # print("x_det: %.3f y_det: %.3f" % (x_det, y_det))

            # elif output[2] == 0 and output[3] != 0:
            #     vignetting = output[3]
            #     surf_name = system.LDE.GetSurfaceAt(vignetting).Comment
            #     print("\nWARNING: Vignetting at Surface #%d, Name %s" % (vignetting, surf_name))
            #     print("Wavelength: %.3f | Configuration #%d" % (wavelength, config))
            #     print("Field #%d | fx: %.5f fy: %.5f" % (i + 1, X_MAX * hx[i], Y_MAX * hy[i]))
            #     print("hx: %.5f hy: %.5f" % (hx[i], hy[i]))
            #     x_det, y_det = output[4], output[5]
            #     print("x_det: %.3f y_det: %.3f" % (x_det, y_det))
            #
            #     # Add the local coordinates despite the vignetting
            #     local_xyz[i, 0] = output[4]
            #     local_xyz[i, 1] = output[5]
            #     local_xyz[i, 2] = output[6]


        normUnPolData.ClearData()
        CastTo(raytrace, 'ISystemTool').Close()

        # # Get the transformation from Local to Global Coordinates
        # global_mat = system.LDE.GetGlobalMatrix(surface)
        # R11, R12, R13, R21, R22, R23, R31, R32, R33, X0, Y0, Z0 = global_mat[1:]
        # global_matrix = np.array([[R11, R12, R13],
        #                           [R21, R22, R23],
        #                           [R31, R32, R33]])
        # offset = np.array([X0, Y0, Z0])
        #
        # # Transform from Local to Global and only save X and Y
        # global_xyz = (np.dot(global_matrix, local_xyz.T)).T + offset
        # global_xy = global_xyz[:, :2]

        return [RMS_WFE, obj_xy, foc_xy, global_xy]

    def loop_over_files(self, files_dir, files_opt, results_path, wavelength_idx=None,
                        configuration_idx=None, surface=None, spaxels_per_slice=51):
        """
        Function that loops over a given set of E2E model Zemax files, running the analysis
        defined by self.analysis_function_rms_wfe


        :param files_dir: path where the E2E model files are stored
        :param files_opt: dictionary containing the info to create the list of zemax files we want to analyse
        :param results_path: path where we want to store the results
        :param wavelength_idx: list containing the Wavelength numbers we want to analyze. If None, we will use All
        :param configuration_idx: list containing the Configurations we want to analyze. If None, we will use All
        :param surface: Zemax Surface number at which the analysis will be computed
        :param spaxels_per_slice: number spaxels to sample each slice with
        :return:
        """

        # We want the result to produce as output: the RMS WFE array, and the RayTrace at both Object and Focal plane
        results_names = ['RMS_WFE', 'OBJ_XY', 'FOC_XY', 'GLOB_XY']
        # we need to give the shapes of each array to self.run_analysis
        results_shapes = [(spaxels_per_slice,), (spaxels_per_slice, 2), (spaxels_per_slice, 2), (spaxels_per_slice, 2)]

        metadata = {}
        metadata['Spaxels per slice'] = spaxels_per_slice
        metadata['Configurations'] = 'All' if configuration_idx is None else configuration_idx
        metadata['Wavelengths'] = 'All' if wavelength_idx is None else wavelength_idx


        # read the file options
        file_list, sett_list = create_zemax_file_list(which_system=files_opt['which_system'],
                                                      AO_modes=files_opt['AO_modes'], scales=files_opt['scales'],
                                                      IFUs=files_opt['IFUs'], grating=files_opt['grating'])

        # Loop over the Zemax files
        results = []
        for zemax_file, settings in zip(file_list, sett_list):

            list_results = self.run_analysis(analysis_function=self.analysis_function_rms_wfe,
                                             files_dir=files_dir, zemax_file=zemax_file, results_path=results_path,
                                             results_shapes=results_shapes, results_names=results_names,
                                             wavelength_idx=wavelength_idx, configuration_idx=configuration_idx,
                                             surface=surface, spaxels_per_slice=spaxels_per_slice)

            results.append(list_results)
            rms_wfe, obj_xy, foc_xy, global_xy, wavelengths = list_results

            # Post-Processing the results
            file_name = zemax_file.split('.')[0]
            results_dir = os.path.join(results_path, file_name)
            settings['surface'] = 'IMG' if surface is None else surface

            self.save_hdf5(analysis_name='RMS_WFE', analysis_metadata=metadata, list_results=list_results, results_names=results_names,
                           file_name=file_name, file_settings=settings, results_dir=results_path)


        return results




class RMS_WFE_FastAnalysis(AnalysisFast):
    """
    [Fast Version] of the RMS WFE calculation
    """

    @staticmethod
    def analysis_function_rms_wfe(system, wavelength_idx, config, spaxels_per_slice, surface, pupil_sampling,
                                  remove_slicer=False):
        """
        Calculate the RMS WFE for a given Configuration, for an arbitrary number of Field Points and Wavelengths
        This is how we do it:

        (1) Get the Min and Max field points to normalize the coordinates
        (2) Sample the Slice from the min and max field points, using spaxels_per_slice points
        (3) Define the Merit Function. Loop over the Wavelengths, add 1 operand per field points to calculate RMS WFE
        (4) At the same time, add Chief Rays to the raytrace to get the detector coordinates
        (5) Read the results both for the Merit Function operands and the Raytrace results

        :param system: the Zemax system
        :param wavelength_idx: list of the Wavelength indices we want to use
        :param config: current configuration number
        :param spaxels_per_slice: how many points per slice to use to calculate the RMS WFE (typically 3)
        :param surface: the surface number at which we will calculate the RMS WFE
        :param pupil_sampling: how many points N to use on an N x N grid per pupil quadrant. RWRE operand parameter
        :param remove_slicer: whether to increase the Aperture of the Image Slicer mirrors to avoid vignetting
        :return:
        """

        # Set Current Configuration
        system.MCE.SetCurrentConfiguration(config)

        # [WARNING]: for the 4x4 spaxel scale we noticed that a significant fraction of the rays get vignetted at the slicer
        # this introduces a bias in the RMS WFE calculation. To avoid this, we modify the Image Slicer aperture definition
        # so that all rays get through. Consequently, enough pupil rays are traced to get an unbiased estimation of RMS WFE
        if remove_slicer is True:

            expand_slicer_aperture(system)

            # # First of all, we need to find the Surface Number for the IMAGE SLICER
            # N_surfaces = system.LDE.NumberOfSurfaces
            # surface_names = {}      # A dictionary of surface number -> surface comment
            # for k in np.arange(1, N_surfaces):
            #     surface_names[k] = system.LDE.GetSurfaceAt(k).Comment
            # # find the Slicer surface number
            # try:
            #     # The naming convention for this surface has changed. Not the same for Nominal Design as Monte Carlos
            #     slicer_num = list(surface_names.keys())[list(surface_names.values()).index('Slicer Mirror')]
            # except ValueError:
            #     slicer_num = list(surface_names.keys())[list(surface_names.values()).index('IFU ISA')]
            # slicer = system.LDE.GetSurfaceAt(slicer_num)
            #
            # # Read Current Aperture Settings
            # apt_type = slicer.ApertureData.CurrentType
            # # print("Aperture type: ", apt_type)
            # if apt_type == 4:  # 4 is Rectangular aperture
            #     current_apt_sett = slicer.ApertureData.CurrentTypeSettings
            #     # print("Current Settings:")
            #     x0 = current_apt_sett._S_RectangularAperture.XHalfWidth
            #     y0 = current_apt_sett._S_RectangularAperture.YHalfWidth
            #     # If the Y aperture hasn't been changed already, we change it here to 999 mm to get all rays through
            #     if y0 != 999:
            #         # Change Settings
            #         aperture_settings = slicer.ApertureData.CreateApertureTypeSettings(constants.SurfaceApertureTypes_RectangularAperture)
            #         aperture_settings._S_RectangularAperture.XHalfWidth = x0
            #         aperture_settings._S_RectangularAperture.YHalfWidth = 999
            #         slicer.ApertureData.ChangeApertureTypeSettings(aperture_settings)
            #
            #         current_apt_sett = slicer.ApertureData.CurrentTypeSettings
            #         # Notify that we have successfully modified the aperture
            #         print("Changing aperture of surface: ", slicer.Comment)
            #         print("New Settings:")
            #         print("X_HalfWidth = %.2f" % current_apt_sett._S_RectangularAperture.XHalfWidth)
            #         print("Y_HalfWidth = %.2f" % current_apt_sett._S_RectangularAperture.YHalfWidth)

        # [1] Some housekeeping and pre-processing operations
        # Get the Field Points for that configuration
        sysField = system.SystemData.Fields
        # Problem with the MC files. Before, all the E2E files had only 3 fields, now there's more, some spurious ones
        # So N_fields is no longer 3. Let's just hardcode the value to 3 temporarily
        # N_fields = sysField.NumberOfFields
        N_fields = 3
        N_waves = len(wavelength_idx)
        N_rays = N_waves * spaxels_per_slice

        # The only valid Field Points we should care about are 1-3 as defined by Matthias
        fx_min, fy_min = sysField.GetField(1).X, sysField.GetField(1).Y
        fx_max, fy_max = sysField.GetField(3).X, sysField.GetField(3).Y

        # Note that this assumes Rectangular Normalization, the default in the E2E files.
        X_MAX = np.max([np.abs(sysField.GetField(i + 1).X) for i in range(N_fields)])
        Y_MAX = np.max([np.abs(sysField.GetField(i + 1).Y) for i in range(N_fields)])

        # Normalized field coordinates (hx, hy)
        hx_min, hx_max = fx_min / X_MAX, fx_max / X_MAX
        hy_min, hy_max = fy_min / Y_MAX, fy_max / Y_MAX

        # Sample between the edges of the slice as given by "spaxels_per_slice" to include as many points as we want
        hx = np.linspace(hx_min, hx_max, spaxels_per_slice)
        hy = np.linspace(hy_min, hy_max, spaxels_per_slice)
        #
        # if config == 1:
        #     print("Field Information")
        #     print(hx)
        #     print(hy)
        #     raise ValueError
        # hx = np.array([sysField.GetField(i + 1).X / X_MAX for i in range(N_fields)])
        # hy = np.array([sysField.GetField(i + 1).Y / Y_MAX for i in range(N_fields)])

        # print("Field Definitions")
        # print(fx_min, fx_max)
        # print(fy_min, fy_max)
        # print(X_MAX)
        # print(Y_MAX)
        # print(hx, hy)

        # Some useful data that we'll store
        obj_xy = np.array([X_MAX * hx, Y_MAX * hy]).T           # The Field coordinates for the Object plane
        RMS_WFE = np.empty((N_waves, spaxels_per_slice))        # The RMS WFE results
        foc_xy = np.empty((N_waves, spaxels_per_slice, 2))      # The Chief Ray coordinates at the Detector

        # [2] This is where the core of the RMS WFE calculation takes place
        # First, we begin by defining the Raytrace
        raytrace = system.Tools.OpenBatchRayTrace()
        normUnPolData = raytrace.CreateNormUnpol(N_rays, constants.RaysType_Real, surface)

        # Start creating the Merit Function
        theMFE = system.MFE

        # Clear any operands that could be left from the E2E files
        nops = theMFE.NumberOfOperands
        theMFE.RemoveOperandsAt(1, nops)

        # Build the Merit Function
        # Set first operand to current configuration
        op = theMFE.GetOperandAt(1)
        op.ChangeType(constants.MeritOperandType_CONF)
        op.GetOperandCell(constants.MeritColumn_Param1).Value = config
        wfe_op = constants.MeritOperandType_RWRE            # The Type of RMS WFE Operand: RWRE rectangular

        # Populate the Merit Function with RMS WFE Operands
        # Loop over the wavelengths
        for i_wave, wave_idx in enumerate(wavelength_idx):

            # Loop over all Spaxels in the Slice
            for j_field, (h_x, h_y) in enumerate(zip(hx, hy)):

                op = theMFE.AddOperand()
                op.ChangeType(wfe_op)
                op.GetOperandCell(constants.MeritColumn_Param1).Value = int(pupil_sampling)
                op.GetOperandCell(constants.MeritColumn_Param2).Value = int(wave_idx)

                # if config == 1:
                #     print("\nField Information")
                #     print(float(h_x), float(h_y))

                op.GetOperandCell(constants.MeritColumn_Param3).Value = float(h_x)
                op.GetOperandCell(constants.MeritColumn_Param4).Value = float(h_y)
                op.GetOperandCell(constants.MeritColumn_Weight).Value = 0

                # Take advantage of the loop to simultaneously add the ray to the RayTrace
                normUnPolData.AddRay(wave_idx, h_x, h_y, 0, 0, constants.OPDMode_None)

        # time_1 = time() - start0
        # print("\nTime spent setting up MF and Raytrace: %.3f sec" % time_1)
        # start = time()

        # update the Merit Function
        theMFE.CalculateMeritFunction()
        # time_mf = time() - start
        # print("Time spent updating MF: %.3f sec" % time_mf)

        # start = time()
        # Run the RayTrace for the whole Slice
        CastTo(raytrace, 'ISystemTool').RunAndWaitForCompletion()
        # time_ray = time() - start
        # print("Time spent running Raytrace: %.3f sec" % time_ray)

        # start = time()
        # [3] Time to start reading the results of the RMS WFE Operands + Raytrace coordinates
        normUnPolData.StartReadingResults()
        # Loop over the wavelengths
        for i_wave, wave_idx in enumerate(wavelength_idx):
            # Loop over all Spaxels in the Slice
            for j_field, (h_x, h_y) in enumerate(zip(hx, hy)):

                # Calculate the Row index we need to get the Operand
                irow = 2 + i_wave * spaxels_per_slice + j_field
                # print(irow)

                op = theMFE.GetOperandAt(irow)

                # print(op.GetOperandCell(constants.MeritColumn_Param1).Value)
                # print(op.GetOperandCell(constants.MeritColumn_Param2).Value)
                # print(op.GetOperandCell(constants.MeritColumn_Param3).Value)
                # print(op.GetOperandCell(constants.MeritColumn_Param4).Value)
                rms = op.Value
                #
                # if i_wave == 0:
                #     print("\nField #%d" % (j_field + 1))
                #     print()
                #     print(float(h_x), float(h_y), rms)

                wavelength = system.SystemData.Wavelengths.GetWavelength(wave_idx).Wavelength

                RMS_WFE[i_wave, j_field] = wavelength * 1e3 * rms  # We assume the Wavelength comes in Microns

                # If we get an RMS value of 0.0, print the data so we can double check the Zemax file
                # This is bad news and it mean the Rays are being vignetted somewhere
                if RMS_WFE[i_wave, j_field] == 0.0:
                    print("\nConfig #%d | Wave #%d | Field #%d" % (config, wave_idx, j_field + 1))
                    # raise ValueError

                output = normUnPolData.ReadNextResult()
                if output[2] == 0:
                    x, y = output[4], output[5]
                    foc_xy[i_wave, j_field, 0] = x
                    foc_xy[i_wave, j_field, 1] = y

                    vignetting_code = output[3]
                    if vignetting_code != 0:
                        vignetting_surface = system.LDE.GetSurfaceAt(vignetting_code).Comment
                        # print("\nConfig #%d" % (config))
                        # print("Vignetting at surface #%d: %s" % (vignetting_code, vignetting_surface))
            # if config == 1:
            #     raise ValueError

        normUnPolData.ClearData()
        CastTo(raytrace, 'ISystemTool').Close()
        # time_res = time() - start
        # print("Time spent reading results: %.3f sec" % time_res)

        # time_total = time() - start0
        # print("TOTAL Time: %.3f sec" % time_total)
        # sec_per_wave = time_total / N_waves * 1000
        # print("%3.f millisec per Wavelength" % sec_per_wave)

        return [RMS_WFE, obj_xy, foc_xy]

    def loop_over_files(self, files_dir, files_opt, results_path, wavelength_idx=None,
                        configuration_idx=None, surface=None, spaxels_per_slice=3, pupil_sampling=4,
                        save_hdf5=True, remove_slicer_aperture=False,
                        monte_carlo=False):
        """
        Function that loops over a given set of E2E model Zemax files, running the analysis
        defined by self.analysis_function_rms_wfe


        :param files_dir: path where the E2E model files are stored
        :param files_opt: dictionary containing the info to create the list of zemax files we want to analyse
        :param results_path: path where we want to store the results
        :param wavelength_idx: list containing the Wavelength numbers we want to analyze. If None, we will use All
        :param configuration_idx: list containing the Configurations we want to analyze. If None, we will use All
        :param surface: Zemax Surface number at which the analysis will be computed
        :param spaxels_per_slice: number spaxels to sample each slice with
        :return:
        """

        # We want the result to produce as output: the RMS WFE array, and the RayTrace at both Object and Focal plane
        results_names = ['RMS_WFE', 'OBJ_XY', 'FOC_XY']
        N_waves = 23 if wavelength_idx is None else len(wavelength_idx)
        # we need to give the shapes of each array to self.run_analysis
        results_shapes = [(N_waves, spaxels_per_slice,), (spaxels_per_slice, 2), (N_waves, spaxels_per_slice, 2)]

        metadata = {}
        metadata['Spaxels per slice'] = spaxels_per_slice
        metadata['Configurations'] = 'All' if configuration_idx is None else configuration_idx
        metadata['Wavelengths'] = 'All' if wavelength_idx is None else wavelength_idx

        # read the file options. We have to make a distinction between the Nominal Design and the Monte Carlo files
        if monte_carlo is False:
            file_list, sett_list = create_zemax_file_list(which_system=files_opt['which_system'],
                                                          AO_modes=files_opt['AO_modes'], scales=[files_opt['SPAX_SCALE']],
                                                          IFUs=[files_opt['IFU_PATH']], grating=[files_opt['GRATING']])
        elif monte_carlo is True:
            file_list, sett_list = create_zemax_filename_MC(AO_mode=files_opt['AO_MODE'], scale=files_opt['SPAX_SCALE'],
                                                          IFUpath=files_opt['IFU_PATH'], grating=files_opt['GRATING'],
                                                          FPRS_MC_instance=files_opt['FPRS_MC'],
                                                          IPO_MC_instance=files_opt['IPO_MC'],
                                                          IFU_MC_instance=files_opt['IFU_MC'],
                                                          ISP_MC_instance=files_opt['ISP_MC'])

        # Loop over the Zemax files | We tend to run the analysis for single files so this is kind of pointless
        results = []
        for zemax_file, settings in zip(file_list, sett_list):

            list_results = self.run_analysis(analysis_function=self.analysis_function_rms_wfe,
                                             files_dir=files_dir, zemax_file=zemax_file, results_path=results_path,
                                             results_shapes=results_shapes, results_names=results_names,
                                             wavelength_idx=wavelength_idx, configuration_idx=configuration_idx,
                                             surface=surface, spaxels_per_slice=spaxels_per_slice,
                                             pupil_sampling=pupil_sampling, remove_slicer=remove_slicer_aperture)

            results.append(list_results)

            # if save_hdf5 == True:
            #     # Post-Processing the results
            #     file_name = zemax_file.split('.')[0]
            #     settings['surface'] = 'DETECTOR' if surface is None else surface
            #     self.save_hdf5(analysis_name='RMS_WFE', analysis_metadata=metadata, list_results=list_results, results_names=results_names,
            #                    file_name=file_name, file_settings=settings, results_dir=results_path)

        return results


class EnsquaredEnergyFastAnalysis(AnalysisFast):
    """
    Faster version of the Ensquared Energy calculation
    The outside loop is only for the configurations
    For each configuration, we construct a single Raytrace object
    that covers all the specified wavelengths
    """

    @staticmethod
    def analysis_function_ensquared(system, wavelength_idx, surface, config, px, py, box_size):
        """
        Calculation of the Geometric Ensquared Energy

        :param system: the Zemax optical system
        :param wavelength_idx: list of the Zemax Wavelength indices we want to use
        :param config: the Zemax Configuration (Slice) we are going to use
        :param surface: [not used] we always do the calculation at the last surface (Detector)
        :param slicer_surface: Zemax surface number for the Image Slicer
        :param px: Pupil X coordinates of the rays we will trace
        :param py: Pupil Y coordinates of the rays we will trace
        :param box_size: size of the box in [detector pixels] that we'll use for the calculation
        :return:
        """

        det_pix = 15e-3         # Size of the detector pixel [mm]

        # Set Current Configuration
        system.MCE.SetCurrentConfiguration(config)

        # First of all, we need to find the Surface Number for the IMAGE SLICER "Image Plane"
        N_surfaces = system.LDE.NumberOfSurfaces
        surface_names = {}  # A dictionary of surface number -> surface comment
        for k in np.arange(1, N_surfaces):
            surface_names[k] = system.LDE.GetSurfaceAt(k).Comment
        # find the Slicer surface number
        try:
            slicer_num = list(surface_names.keys())[list(surface_names.values()).index('Image Plane')]
        except ValueError:
            slicer_num = list(surface_names.keys())[list(surface_names.values()).index('Image plane')]
        slicer_surface = slicer_num
        # slicer = system.LDE.GetSurfaceAt(slicer_num)

        # Double check that the Image Slicer surface number is correct
        slicer = system.LDE.GetSurfaceAt(slicer_surface).Comment
        if slicer != 'Image Plane' and slicer != 'Image plane':
            print(slicer)
            raise ValueError("Surface #%d is not the Image Slicer. Please check the Zemax file" % slicer_surface)

        # Get the Field Points for that configuration
        sysField = system.SystemData.Fields
        N_fields = sysField.NumberOfFields
        N_waves = len(wavelength_idx)

        X_MAX = np.max([np.abs(sysField.GetField(i + 1).X) for i in range(N_fields)])
        Y_MAX = np.max([np.abs(sysField.GetField(i + 1).Y) for i in range(N_fields)])

        # Use the Field Point at the centre of the Slice
        fx, fy = sysField.GetField(2).X, sysField.GetField(2).Y
        hx, hy = fx / X_MAX, fy / Y_MAX  # Normalized field coordinates (hx, hy)
        obj_xy = np.array([fx, fy])

        N_pupil = px.shape[0]   # Number of rays in the Pupil for a given field point and wavelength
        N_rays = N_waves * N_pupil

        EE = np.empty(N_waves)
        sli_foc_xy = np.empty((N_waves, 2))
        det_foc_xy = np.empty((N_waves, 2))

        slicer_xy = np.empty((N_waves, N_pupil, 2))
        slicer_xy[:] = np.nan
        detector_xy = np.empty((N_waves, N_pupil, 2))
        detector_xy[:] = np.nan

        # (1) Run the raytrace up to the IMAGE SLICER
        raytrace = system.Tools.OpenBatchRayTrace()
        # remember to specify the surface to which you are tracing!
        rays_slicer = raytrace.CreateNormUnpol(N_rays, constants.RaysType_Real, slicer_surface)

        # Loop over all wavelengths
        for i_wave, wave_idx in enumerate(wavelength_idx):

            for (p_x, p_y) in zip(px, py):  # Add the ray to the RayTrace
                rays_slicer.AddRay(wave_idx, hx, hy, p_x, p_y, constants.OPDMode_None)

        CastTo(raytrace, 'ISystemTool').RunAndWaitForCompletion()
        rays_slicer.StartReadingResults()
        checksum_slicer = 0
        for k in range(N_rays):  # Get Raytrace results at the Image Slicer
            i_wave = k // N_pupil
            j_pupil = k % N_pupil
            # print(i_wave, j_pupil)
            output = rays_slicer.ReadNextResult()
            if output[2] == 0 and output[3] == 0:
                slicer_xy[i_wave, j_pupil, 0] = output[4]
                slicer_xy[i_wave, j_pupil, 1] = output[5]
                checksum_slicer += 1
        if checksum_slicer < N_rays:
            raise ValueError('Some rays were lost before the Image Slicer')

        rays_slicer.ClearData()

        # Count how many rays fall inside a +- 1 mm window in Y, wrt the centroid
        slicer_cent_x = np.nanmean(slicer_xy[:, :, 0], axis=1)
        slicer_cent_y = np.nanmean(slicer_xy[:, :, 1], axis=1)
        sli_foc_xy[:, 0] = slicer_cent_x
        sli_foc_xy[:, 1] = slicer_cent_y

        # print(slicer_cent_y)
        below_slicer = slicer_xy[:, :, 1] < slicer_cent_y[:, np.newaxis] + 1.0 * box_size / 2
        above_slicer = slicer_xy[:, :, 1] > slicer_cent_y[:, np.newaxis] - 1.0 * box_size / 2
        inside_slicer = (np.logical_and(below_slicer, above_slicer))
        # print(inside_slicer[0, :10])

        # Now, for each wavelength, we calculate which rays fulfil the Image Slicer conditions
        index_valid_slicer = [np.argwhere(inside_slicer[i, :] == True)[:, 0] for i in range(N_waves)]
        # print(index_valid_slicer[1][:10])
        # print(index_valid_slicer[2][:10])

        # (2) Run the raytrace up to the DETECTOR
        # For speed, we re-use the same Raytrace, just define new rays!
        # raytrace_det = system.Tools.OpenBatchRayTrace()
        # Detector is always the last surface
        detector_surface = system.LDE.NumberOfSurfaces - 1
        rays_detector = raytrace.CreateNormUnpol(N_rays, constants.RaysType_Real, detector_surface)
        # Loop over all wavelengths
        for i_wave, wave_idx in enumerate(wavelength_idx):
            for (p_x, p_y) in zip(px, py):
                rays_detector.AddRay(wave_idx, hx, hy, p_x, p_y, constants.OPDMode_None)

        CastTo(raytrace, 'ISystemTool').RunAndWaitForCompletion()

        rays_detector.StartReadingResults()
        checksum_detector = 0
        # index_valid_detector = []       # Valid means they make it to the detector even if vignetted at the Slicer
        vignetted = []
        index_vignetted = []
        index_valid_detector = np.empty((N_waves, N_pupil))
        index_valid_detector[:] = np.nan
        for k in range(N_rays):  # Get Raytrace results at the Detector
            i_wave = k // N_pupil
            j_pupil = k % N_pupil
            output = rays_detector.ReadNextResult()
            if output[2] == 0 and output[3] == 0:       # ErrorCode & VignetteCode
                detector_xy[i_wave, j_pupil, 0] = output[4]
                detector_xy[i_wave, j_pupil, 1] = output[5]
                checksum_detector += 1
                index_valid_detector[i_wave, j_pupil] = j_pupil

            elif output[2] == 0 and output[3] != 0:
                # Some rays are vignetted
                vignetted.append([output[4],  output[5]])
                detector_xy[i_wave, j_pupil, 0] = output[4]
                detector_xy[i_wave, j_pupil, 1] = output[5]
                checksum_detector += 1
                index_valid_detector[i_wave, j_pupil] = j_pupil
                index_vignetted.append(k)

        # index_valid_detector = np.array(index_valid_detector)
        # # print(index_valid_detector.shape)
        # # print(index_valid_detector)
        # index_valid_detector = index_valid_detector.reshape((N_waves, N_pupil))
        # # print(index_valid_detector.shape)

        rays_detector.ClearData()
        CastTo(raytrace, 'ISystemTool').Close()

        # (3) Calculate the ENSQUARED ENERGY
        # We only count the rays that where inside the slicer to begin with and the ones that make it to the detector
        for i_wave in range(N_waves):
            valid_both = []
            for k in range(N_pupil):
                # print(index_valid_detector[i_wave])
                if k in index_valid_slicer[i_wave] and k in index_valid_detector[i_wave]:
                    valid_both.append(k)

            valid_det_x = detector_xy[i_wave, :, 0][valid_both]
            valid_det_y = detector_xy[i_wave, :, 1][valid_both]

            # Now, out of the VALID rays, we calculate which detector rays fall inside a 2x pixel box along X
            dcx = np.mean(valid_det_x)  # Detector Centroid X
            dcy = np.mean(valid_det_y)
            det_foc_xy[i_wave] = [dcx, dcy]

            left_detector = valid_det_x < dcx + det_pix * box_size / 2
            right_detector = valid_det_x > dcx - det_pix * box_size / 2
            inside_detector = (np.logical_and(left_detector, right_detector))
            total_detector = np.sum(inside_detector)
            ensq = total_detector / N_pupil
            # print(ensq)
            EE[i_wave] = ensq * 0.98

        # SHOW THIS in the methodology

        # fig, axes = plt.subplots(2, N_waves)
        # colors = cm.Reds(np.linspace(0.5, 1, N_waves))
        # for j in range(N_waves):
        #     ax1 = axes[0][j]
        #     scy = sli_foc_xy[j, 1]
        #     scx = sli_foc_xy[j, 0]
        #     ax1.axhline(y=scy + 1.0 * box_size / 2, color='black', linestyle='--')
        #     ax1.axhline(y=scy - 1.0 * box_size / 2, color='black', linestyle='--')
        #     ax1.scatter(slicer_xy[j, :, 0], slicer_xy[j, :, 1], s=3, color=colors[j])
        #     ax1.scatter(sli_foc_xy[j, 0], sli_foc_xy[j, 1], s=3, color='black')
        #     wavelength = system.SystemData.Wavelengths.GetWavelength(wavelength_idx[j]).Wavelength
        #     ax1.set_title("IMG SLI | %.3f $\mu$m" % wavelength)
        #     ax1.set_aspect('equal')
        #     ax1.get_yaxis().set_visible(False)
        #     ax1.get_xaxis().set_visible(False)
        #
        #     p = 1.2
        #     ax1.set_xlim([scx - p * box_size / 2, scx + p * box_size / 2])
        #     ax1.set_ylim([scy - p * box_size / 2, scy + p * box_size / 2])
        #
        #     ax2 = axes[1][j]
        #     dcx = det_foc_xy[j, 0]
        #     dcy = det_foc_xy[j, 1]
        #     ax2.scatter(detector_xy[j, :, 0], detector_xy[j, :, 1], s=3, color=colors[j])
        #     ax2.scatter(det_foc_xy[j, 0], det_foc_xy[j, 1], s=3, color='black')
        #     ax2.axvline(x=dcx + det_pix * box_size / 2, color='black', linestyle='--')
        #     ax2.axvline(x=dcx - det_pix * box_size / 2, color='black', linestyle='--')
        #     ax2.set_title("DET | %.3f $\mu$m" % wavelength)
        #     ax2.set_aspect('equal')
        #     ax2.get_yaxis().set_visible(False)
        #     ax2.get_xaxis().set_visible(False)
        #     ax2.set_xlim([dcx - p * det_pix * box_size / 2, dcx + p * det_pix * box_size / 2])
        #     ax2.set_ylim([dcy - p * det_pix * box_size / 2, dcy + p * det_pix * box_size / 2])
        #
        #
        # plt.show()

        return EE, obj_xy, sli_foc_xy, det_foc_xy

    def loop_over_files(self, files_dir, files_opt, results_path, wavelength_idx=None,
                        configuration_idx=None, N_rays=500, box_size=2, monte_carlo=False):
        """

        """

        # We want the result to produce as output: Ensquared Energy, Object coords, Image Slicer and Detector centroids
        results_names = ['EE', 'OBJ_XY', 'SLI_XY', 'DET_XY']
        N_waves = 23 if wavelength_idx is None else len(wavelength_idx)
        # we need to give the shapes of each array to self.run_analysis
        results_shapes = [(N_waves,), (2,), (N_waves, 2,), (N_waves, 2,)]

        metadata = {}
        metadata['N_rays'] = N_rays
        metadata['Box Size Spaxels'] = box_size
        metadata['Configurations'] = 'All' if configuration_idx is None else configuration_idx
        metadata['Wavelengths'] = 'All' if wavelength_idx is None else wavelength_idx

        # read the file options
        if monte_carlo is False:
            file_list, sett_list = create_zemax_file_list(which_system=files_opt['which_system'],
                                                          AO_modes=files_opt['AO_modes'], scales=[files_opt['SPAX_SCALE']],
                                                          IFUs=[files_opt['IFU_PATH']], grating=[files_opt['GRATING']])
        elif monte_carlo is True:
            file_list, sett_list = create_zemax_filename_MC(AO_mode=files_opt['AO_MODE'], scale=files_opt['SPAX_SCALE'],
                                                            IFUpath=files_opt['IFU_PATH'], grating=files_opt['GRATING'],
                                                            FPRS_MC_instance=files_opt['FPRS_MC'],
                                                            IPO_MC_instance=files_opt['IPO_MC'],
                                                            IFU_MC_instance=files_opt['IFU_MC'],
                                                            ISP_MC_instance=files_opt['ISP_MC'])

        results = []
        for zemax_file, settings in zip(file_list, sett_list):

            # Generate a set of random pupil rays
            px, py = define_pupil_sampling(r_obsc=0.2841, N_rays=N_rays, mode='random')
            print("Using %d rays" % N_rays)

            list_results = self.run_analysis(analysis_function=self.analysis_function_ensquared,
                                             files_dir=files_dir, zemax_file=zemax_file, results_path=results_path,
                                             results_shapes=results_shapes, results_names=results_names,
                                             wavelength_idx=wavelength_idx, configuration_idx=configuration_idx,
                                             px=px, py=py, box_size=box_size)

            results.append(list_results)

            # Post-Processing the results
            file_name = zemax_file.split('.')[0]
            settings['surface'] = 'DETECTOR'
            # self.save_hdf5(analysis_name='ENSQ_ENERG', analysis_metadata=metadata, list_results=list_results,
            #                results_names=results_names, file_name=file_name, file_settings=settings, results_dir=results_path)

        return results




class FWHM_PSF_FastAnalysis(AnalysisFast):
    """

    """

    def calculate_fwhm(self, surface, xy_data, PSF_window, N_points, spaxel_scale, wavelength):

        start = time()
        # Calculate the Geometric PSF
        x, y = xy_data[:, 0], xy_data[:, 1]
        cent_x, cent_y = np.mean(x), np.mean(y)

        std_x, std_y = np.std(x), np.std(y)
        bandwidth = min(std_x, std_y)
        kde = KernelDensity(kernel='gaussian', bandwidth=bandwidth).fit(xy_data)

        # define a grid to compute the PSF
        xmin, xmax = cent_x - PSF_window/2/1000, cent_x + PSF_window/2/1000
        ymin, ymax = cent_y - PSF_window/2/1000, cent_y + PSF_window/2/1000
        x_grid = np.linspace(xmin, xmax, N_points)
        y_grid = np.linspace(ymin, ymax, N_points)
        xx_grid, yy_grid = np.meshgrid(x_grid, y_grid)
        xy_grid = np.vstack([xx_grid.ravel(), yy_grid.ravel()]).T
        log_scores = kde.score_samples(xy_grid)

        psf_geo = np.exp(log_scores)
        psf_geo /= np.max(psf_geo)
        psf_geo = psf_geo.reshape(xx_grid.shape)

        time_geopsf = time() - start
        # print("Time to estimate GeoPSF: %.3f sec" % time_geo)
        start = time()

        psf_diffr = diffraction.add_diffraction(surface=surface, psf_geo=psf_geo, PSF_window=PSF_window,
                                                scale_mas=spaxel_scale, wavelength=wavelength)
        time_diffpsf = time() - start
        # print("Time to add Diffraction: %.3f sec" % time_diffpsf)
        start = time()

        # guesses = {'IS': {4.0: }}

        # Fit the PSF to a 2D Gaussian
        guess_x = PSF_window / 2 / 1000
        fwhm_x, fwhm_y, theta = diffraction.fit_psf_to_gaussian(xx=xx_grid, yy=yy_grid, psf_data=psf_diffr,
                                                                x0=cent_x, y0=cent_y, sigmax0=guess_x, sigmay0=guess_x)

        time_gauss = time() - start

        # print('FWHM time: %.3f sec for GeoPSF estimate:' % time_geopsf)
        # print('FWHM time: %.3f sec for DiffPSF convolution:' % time_diffpsf)
        # print('FWHM time: %.3f sec for Gaussian fit:' % time_gauss)

        # fig, (ax1, ax2) = plt.subplots(1, 2)
        # img1 = ax1.imshow(psf_geo, extent=[xmin, xmax, ymin, ymax], cmap='plasma', origin='lower')
        # ax1.scatter(x, y, s=1, color='white', alpha=0.5)
        # plt.colorbar(img1, ax=ax1, orientation='horizontal')
        # ax1.set_xlabel(r'X [mm]')
        # ax1.set_ylabel(r'Y [mm]')
        # ax1.set_title(r'Geometric PSF estimate | Surface: %s' % surface)
        #
        # img2 = ax2.imshow(psf_diffr, extent=[xmin, xmax, ymin, ymax], cmap='plasma', origin='lower')
        # plt.colorbar(img2, ax=ax2, orientation='horizontal')
        # ax2.set_xlabel(r'X [mm]')
        # ax2.set_ylabel(r'Y [mm]')
        # if surface == 'DET':
        #     ax2.set_title(r'Diffr. PSF | %.3f microns | %.1f mas | FWHM_x: %.1f $\mu$m' % (wavelength, spaxel_scale, fwhm_x))
        # elif surface == 'IS':
        #     ax2.set_title(r'Diffr. PSF | %.3f microns | %.1f mas | FWHM_y: %.1f $\mu$m' % (wavelength, spaxel_scale, fwhm_y))

        return fwhm_x, fwhm_y

    def analysis_function_fwhm_psf(self, system, wavelength_idx, surface, config, px, py, spaxel_scale, N_points):

        start0 = time()
        # Set Current Configuration
        system.MCE.SetCurrentConfiguration(config)

        # First of all, we need to find the Surface Number for the IMAGE SLICER "Image Plane"
        N_surfaces = system.LDE.NumberOfSurfaces
        surface_names = {}  # A dictionary of surface number -> surface comment
        for k in np.arange(1, N_surfaces):
            surface_names[k] = system.LDE.GetSurfaceAt(k).Comment
        # find the Slicer surface number
        try:
            slicer_num = list(surface_names.keys())[list(surface_names.values()).index('Image Plane')]
        except ValueError:
            slicer_num = list(surface_names.keys())[list(surface_names.values()).index('Image plane')]
        slicer_surface = slicer_num

        # Get the Field Points for that configuration
        sysField = system.SystemData.Fields
        N_fields = sysField.NumberOfFields
        N_waves = len(wavelength_idx)

        X_MAX = np.max([np.abs(sysField.GetField(i + 1).X) for i in range(N_fields)])
        Y_MAX = np.max([np.abs(sysField.GetField(i + 1).Y) for i in range(N_fields)])

        # Use the Field Point at the centre of the Slice
        fx, fy = sysField.GetField(2).X, sysField.GetField(2).Y
        hx, hy = fx / X_MAX, fy / Y_MAX  # Normalized field coordinates (hx, hy)
        obj_xy = np.array([fx, fy])

        N_pupil = px.shape[0]   # Number of rays in the Pupil for a given field point and wavelength
        N_rays = N_waves * N_pupil

        FWHM = np.zeros((N_waves, 2))
        foc_xy = np.zeros((N_waves, 2))

        slicer_xy = np.empty((N_waves, N_pupil, 2))
        slicer_xy[:] = np.nan
        detector_xy = np.empty((N_waves, N_pupil, 2))
        detector_xy[:] = np.nan

        # (1) Run the raytrace up to the IMAGE SLICER
        raytrace = system.Tools.OpenBatchRayTrace()
        # remember to specify the surface to which you are tracing!
        rays_slicer = raytrace.CreateNormUnpol(N_rays, constants.RaysType_Real, slicer_surface)

        # Loop over all wavelengths
        for i_wave, wave_idx in enumerate(wavelength_idx):

            for (p_x, p_y) in zip(px, py):  # Add the ray to the RayTrace
                rays_slicer.AddRay(wave_idx, hx, hy, p_x, p_y, constants.OPDMode_None)

        CastTo(raytrace, 'ISystemTool').RunAndWaitForCompletion()
        rays_slicer.StartReadingResults()
        checksum_slicer = 0
        for k in range(N_rays):  # Get Raytrace results at the Image Slicer
            i_wave = k // N_pupil
            j_pupil = k % N_pupil
            # print(i_wave, j_pupil)
            output = rays_slicer.ReadNextResult()
            if output[2] == 0:
                slicer_xy[i_wave, j_pupil, 0] = output[4]
                slicer_xy[i_wave, j_pupil, 1] = output[5]
                checksum_slicer += 1
        if checksum_slicer < N_rays:
            raise ValueError('Some rays were lost before the Image Slicer')

        rays_slicer.ClearData()

        # (2) Run the raytrace up to the DETECTOR
        # For speed, we re-use the same Raytrace, just define new rays!
        # raytrace_det = system.Tools.OpenBatchRayTrace()
        # Detector is always the last surface
        detector_surface = system.LDE.NumberOfSurfaces - 1
        rays_detector = raytrace.CreateNormUnpol(N_rays, constants.RaysType_Real, detector_surface)
        # Loop over all wavelengths
        for i_wave, wave_idx in enumerate(wavelength_idx):
            for (p_x, p_y) in zip(px, py):
                rays_detector.AddRay(wave_idx, hx, hy, p_x, p_y, constants.OPDMode_None)

        CastTo(raytrace, 'ISystemTool').RunAndWaitForCompletion()

        rays_detector.StartReadingResults()
        checksum_detector = 0
        index_valid_detector = np.empty((N_waves, N_pupil))
        index_valid_detector[:] = np.nan
        for k in range(N_rays):  # Get Raytrace results at the Detector
            i_wave = k // N_pupil
            j_pupil = k % N_pupil
            output = rays_detector.ReadNextResult()
            if output[2] == 0:       # ErrorCode & VignetteCode
                detector_xy[i_wave, j_pupil, 0] = output[4]
                detector_xy[i_wave, j_pupil, 1] = output[5]
                checksum_detector += 1
                index_valid_detector[i_wave, j_pupil] = j_pupil
            #
            # elif output[2] == 0 and output[3] != 0:
            #     # Some rays are vignetted
            #     code = output[3]
            #     vignetting_surface = system.LDE.GetSurfaceAt(code).Comment
            #     # print("Config #%d | Wave #%d || Vignetting at surface #%d: %s" % (config, wavelength_idx[i_wave], code, vignetting_surface))

        rays_detector.ClearData()
        CastTo(raytrace, 'ISystemTool').Close()

        # time_rays = time() - start0
        # print("Time for Raytrace Slicer and Detector: %.3f sec" % time_rays)

        # fig, axes = plt.subplots(2, N_waves)
        # colors = cm.Reds(np.linspace(0.5, 1, N_waves))
        # for j in range(N_waves):
        #     ax1 = axes[0][j]
        #     ax1.scatter(slicer_xy[j, :, 0], slicer_xy[j, :, 1], s=3, color=colors[j])
        #     # scx =
        #     # ax1.scatter(sli_foc_xy[j, 0], sli_foc_xy[j, 1], s=3, color='black')
        #     # scy = sli_foc_xy[j, 1]
        #     # ax1.axhline(y=scy + 1.0 * box_size / 2, color='black', linestyle='--')
        #     # ax1.axhline(y=scy - 1.0 * box_size / 2, color='black', linestyle='--')
        #     wavelength = system.SystemData.Wavelengths.GetWavelength(wavelength_idx[j]).Wavelength
        #     ax1.set_title("%.3f $\mu$m" % wavelength)
        #
        #     ax2 = axes[1][j]
        #     # dcx = det_foc_xy[j, 0]
        #     ax2.scatter(detector_xy[j, :, 0], detector_xy[j, :, 1], s=3, color=colors[j])
        #     # ax2.scatter(det_foc_xy[j, 0], det_foc_xy[j, 1], s=3, color='black')
        #     # ax2.axvline(x=dcx + det_pix * box_size / 2, color='black', linestyle='--')
        #     # ax2.axvline(x=dcx - det_pix * box_size / 2, color='black', linestyle='--')
        #

        # FWHM

        windows = {4.0: [5000, 200], 60.0: [500, 50]}
        win_slicer = windows[spaxel_scale][0]
        win_detect = windows[spaxel_scale][1]
        if config == 1:
            print("Sampling the Image Slicer plane with %d points: %.3f microns / point" % (N_points, (win_slicer / N_points)))
            print("Sampling the Detector plane with %d points: %.3f microns / point" % (N_points, (win_detect / N_points)))

        for i_wave, wave_idx in enumerate(wavelength_idx):

            xy_data_slicer = slicer_xy[i_wave]
            wavelength = system.SystemData.Wavelengths.GetWavelength(wave_idx).Wavelength
            fwhm_x_slicer, fwhm_y_slicer = self.calculate_fwhm(surface='IS', xy_data=xy_data_slicer, PSF_window=win_slicer,
                                                               N_points=N_points, spaxel_scale=spaxel_scale,
                                                               wavelength=wavelength)

            xy_data_detector = detector_xy[i_wave]
            fwhm_x_det, fwhm_y_det = self.calculate_fwhm(surface='DET', xy_data=xy_data_detector, PSF_window=win_detect,
                                                         N_points=N_points, spaxel_scale=spaxel_scale,
                                                         wavelength=wavelength)

            # plt.show()

            foc_xy[i_wave] = [np.mean(xy_data_detector[:, 0]), np.mean(xy_data_detector[:, 1])]

            FWHM[i_wave] = [fwhm_x_det, fwhm_y_slicer]

            if config == 1:
                print("%.3f microns" % wavelength)
                print("FWHM in X [Detector]: %.1f microns | in Y [Image Slicer]: %.2f microns " % (fwhm_x_det, fwhm_y_slicer))

        # plt.show()

        return FWHM, obj_xy, foc_xy


    def loop_over_files(self, files_dir, files_opt, results_path, wavelength_idx=None,
                        configuration_idx=None, N_rays=500, N_points=50):
        """

        """

        # We want the result to produce as output: the RMS WFE array, and the RayTrace at both Object and Focal plane
        results_names = ['FWHM', 'OBJ_XY', 'FOC_XY']
        # we need to give the shapes of each array to self.run_analysis
        N_waves = 23 if wavelength_idx is None else len(wavelength_idx)
        results_shapes = [(N_waves, 2,), (2,), (N_waves, 2,)]

        px, py = define_pupil_sampling(r_obsc=0.2841, N_rays=N_rays, mode='random')
        print("Using %d rays" % N_rays)

        # read the file options
        file_list, sett_list = create_zemax_file_list(which_system=files_opt['which_system'], AO_modes=files_opt['AO_modes'],
                                           scales=files_opt['scales'], IFUs=files_opt['IFUs'], grating=files_opt['grating'])

        results = []
        for zemax_file, settings in zip(file_list, sett_list):

            # Clear the Cache for the Airy Pattern function
            airy_and_slicer.cache_clear()

            mas_dict = {'4x4': 4.0, '10x10': 10.0, '20x20': 20.0, '60x30': 60.0}
            spaxel_scale = mas_dict[settings['scale']]

            list_results = self.run_analysis(analysis_function=self.analysis_function_fwhm_psf,
                                             files_dir=files_dir, zemax_file=zemax_file, results_path=results_path,
                                             results_shapes=results_shapes, results_names=results_names,
                                             wavelength_idx=wavelength_idx, configuration_idx=configuration_idx,
                                             px=px, py=py, spaxel_scale=spaxel_scale, N_points=N_points)
            results.append(list_results)

        return results


class CommonWavelengthRange(AnalysisFast):
    """
    Calculate the Common Wavelength Range

    At some point we saw that, in some cases, the rays fall outside the active area
    of the detector (specially for short wavelengths). This means that the wavelengths
    defined in Zemax are "optimistic" and that we have to re-calculate the
    Common Wavelength Range
    """

    @staticmethod
    def analysis_function_cwr(system, wavelength_idx, config, surface):
        """
        To calculate the Common Wavelength Range we will perform a raytrace for
        all configurations and wavelengths in the E2E Model Zemax file

        We will use the results of the raytrace at the detector plane to interpolate
        and find the value of the shortest and longest wavelengths for which all slices (configurations)
        fall within the active area of the detector.

        Here we only run the Raytrace and store the values, we will do the post-processing
        in another script e2e_common_wave_range.py

        :param system: the Zemax optical system
        :param wavelength_idx: list of Zemax wavelength indices
        :param config: current configuration
        :param surface: Zemax surface number
        :return:
        """

        # Set Current Configuration
        system.MCE.SetCurrentConfiguration(config)

        # Get the Field Points for that configuration
        sysField = system.SystemData.Fields
        N_fields = sysField.NumberOfFields
        N_waves = len(wavelength_idx)

        # We will trace all 3 available Field Points (the centre and both edges of the slice)
        X_MAX = np.max([np.abs(sysField.GetField(i + 1).X) for i in range(N_fields)])
        Y_MAX = np.max([np.abs(sysField.GetField(i + 1).Y) for i in range(N_fields)])

        hx = np.array([sysField.GetField(i + 1).X / X_MAX for i in range(N_fields)])
        hy = np.array([sysField.GetField(i + 1).Y / Y_MAX for i in range(N_fields)])

        # The Field coordinates for the Object
        # obj_xy = np.array([X_MAX * hx, Y_MAX * hy]).T
        foc_xy = np.empty((N_waves, N_fields, 2))

        N_rays = N_waves * N_fields

        raytrace = system.Tools.OpenBatchRayTrace()
        normUnPolData = raytrace.CreateNormUnpol(N_rays, constants.RaysType_Real, surface)

        # Loop over the wavelengths
        for i_wave, wave_idx in enumerate(wavelength_idx):

            # Loop over all Spaxels in the Slice
            for j_field, (h_x, h_y) in enumerate(zip(hx, hy)):

                # Add the ray to the RayTrace
                normUnPolData.AddRay(wave_idx, h_x, h_y, 0, 0, constants.OPDMode_None)

        # Run the RayTrace for the whole Slice
        CastTo(raytrace, 'ISystemTool').RunAndWaitForCompletion()
        # time_ray = time() - start
        # print("Time spent running Raytrace: %.3f sec" % time_ray)

        # start = time()
        normUnPolData.StartReadingResults()
        # Loop over the wavelengths
        for i_wave, wave_idx in enumerate(wavelength_idx):
            # Loop over all Spaxels in the Slice
            for j_field, (h_x, h_y) in enumerate(zip(hx, hy)):

                output = normUnPolData.ReadNextResult()
                if output[2] == 0:      # ignore the vignetting
                    x, y = output[4], output[5]
                    foc_xy[i_wave, j_field, 0] = x
                    foc_xy[i_wave, j_field, 1] = y

        normUnPolData.ClearData()
        CastTo(raytrace, 'ISystemTool').Close()

        return [foc_xy]

    def loop_over_files(self, files_dir, files_opt, results_path, wavelength_idx=None,
                        configuration_idx=None, surface=None):

        results_names = ['FOC_XY', ]
        N_waves = 23 if wavelength_idx is None else len(wavelength_idx)
        # we need to give the shapes of each array to self.run_analysis
        results_shapes = [(N_waves, 3, 2), ]

        metadata = {}
        metadata['Configurations'] = 'All' if configuration_idx is None else configuration_idx
        metadata['Wavelengths'] = 'All' if wavelength_idx is None else wavelength_idx

        # read the file options
        file_list, sett_list = create_zemax_file_list(which_system=files_opt['which_system'],
                                                      AO_modes=files_opt['AO_modes'], scales=files_opt['scales'],
                                                      IFUs=files_opt['IFUs'], grating=files_opt['grating'])

        # Loop over the Zemax files
        results = []
        for zemax_file, settings in zip(file_list, sett_list):

            list_results = self.run_analysis(analysis_function=self.analysis_function_cwr,
                                             files_dir=files_dir, zemax_file=zemax_file, results_path=results_path,
                                             results_shapes=results_shapes, results_names=results_names,
                                             wavelength_idx=wavelength_idx, configuration_idx=configuration_idx,
                                             surface=surface)

            results.append(list_results)

            # Post-Processing the results
            file_name = zemax_file.split('.')[0]
            settings['surface'] = 'DETECTOR' if surface is None else surface
            self.save_hdf5(analysis_name='COMMON_WAVELENGTH_RANGE', analysis_metadata=metadata,
                           list_results=list_results, results_names=results_names,
                           file_name=file_name, file_settings=settings, results_dir=results_path)

        return results


class Raytrace_FastAnalysis(AnalysisFast):
    """
    Raytrace Analysis [Fast]
    Traces a set of Chief Rays to a given surface (typically the Detector plane)
    """

    @staticmethod
    def analysis_function_raytrace(system, wavelength_idx, config, spaxels_per_slice, surface, ignore_vignetting):
        """
        For a given configuration, trace chief rays for chosen wavelengths and field points (spaxels_per_slice)
        using a single Raytrace

        :param system: Zemax optical system
        :param wavelength_idx: list of Zemax wavelength indices we want
        :param config: current Zemax configuration number
        :param spaxels_per_slice: how many field points per slice we want to trace
        :param surface: Zemax surface number we want to trace the rays to
        :param ignore_vignetting: boolean, whether or not to ignore vignetting
        :return:
        """
        start0 = time()

        # Set Current Configuration
        system.MCE.SetCurrentConfiguration(config)

        # Get the Field Points for that configuration
        sysField = system.SystemData.Fields
        N_fields = sysField.NumberOfFields
        N_waves = len(wavelength_idx)
        N_rays = N_waves * spaxels_per_slice

        fx_min, fy_min = sysField.GetField(1).X, sysField.GetField(1).Y
        fx_max, fy_max = sysField.GetField(N_fields).X, sysField.GetField(N_fields).Y

        # Watch Out! here we are assuming Rectangular Normalization
        X_MAX = np.max([np.abs(sysField.GetField(i + 1).X) for i in range(N_fields)])
        Y_MAX = np.max([np.abs(sysField.GetField(i + 1).Y) for i in range(N_fields)])

        # Normalized field coordinates (hx, hy)
        hx_min, hx_max = fx_min / X_MAX, fx_max / X_MAX
        hy_min, hy_max = fy_min / Y_MAX, fy_max / Y_MAX

        hx = np.linspace(hx_min, hx_max, spaxels_per_slice)
        hy = np.linspace(hy_min, hy_max, spaxels_per_slice)

        # The Field coordinates for the Object
        obj_xy = np.array([X_MAX * hx, Y_MAX * hy]).T
        foc_xy = np.empty((N_waves, spaxels_per_slice, 2))

        raytrace = system.Tools.OpenBatchRayTrace()
        normUnPolData = raytrace.CreateNormUnpol(N_rays, constants.RaysType_Real, surface)

        # Loop over the wavelengths
        for i_wave, wave_idx in enumerate(wavelength_idx):

            # Loop over all Spaxels in the Slice
            for j_field, (h_x, h_y) in enumerate(zip(hx, hy)):

                # Add the ray to the RayTrace
                normUnPolData.AddRay(wave_idx, h_x, h_y, 0, 0, constants.OPDMode_None)

        # Run the RayTrace for the whole Slice
        CastTo(raytrace, 'ISystemTool').RunAndWaitForCompletion()
        # time_ray = time() - start
        # print("Time spent running Raytrace: %.3f sec" % time_ray)

        # start = time()
        normUnPolData.StartReadingResults()

        # Retrieve the results for the operands and raytrace
        # Loop over the wavelengths
        for i_wave, wave_idx in enumerate(wavelength_idx):
            # Loop over all Spaxels in the Slice
            for j_field, (h_x, h_y) in enumerate(zip(hx, hy)):

                output = normUnPolData.ReadNextResult()
                if ignore_vignetting == False:
                    # We do care about vignetting
                    if output[2] == 0 and output[3] == 0:
                        x, y = output[4], output[5]
                        foc_xy[i_wave, j_field, 0] = x
                        foc_xy[i_wave, j_field, 1] = y

                    elif output[2] == 0 and output[3] != 0:
                        vignet_code = output[3]
                        vignetting_surface = system.LDE.GetSurfaceAt(vignet_code).Comment
                        print("\nConfig #%d | Wavelength idx #%d" % (config, wave_idx))
                        fx, fy = h_x * X_MAX, h_y * Y_MAX
                        print("Field point #%d : hx=%.4f hy=%.4f | fx=%.4f, fy=%.4f" % (j_field + 1, h_x, h_y, fx, fy))
                        print("Vignetting at surface #%d: %s" % (vignet_code, vignetting_surface))
                else:
                    # If we don't care about vignetting (rays falling outside the active area of the detector, for example)
                    # we add the Raytrace results to the focal coordinates array no matter what
                    if output[2] == 0:
                        x, y = output[4], output[5]
                        foc_xy[i_wave, j_field, 0] = x
                        foc_xy[i_wave, j_field, 1] = y

        normUnPolData.ClearData()
        CastTo(raytrace, 'ISystemTool').Close()
        # time_res = time() - start
        # print("Time spent reading results: %.3f sec" % time_res)

        return [obj_xy, foc_xy]

    def loop_over_files(self, files_dir, files_opt, results_path, wavelength_idx=None,
                        configuration_idx=None, surface=None, spaxels_per_slice=3, ignore_vignetting=False,
                        save_hdf5=False):
        """
        We loop over the Zemax files, calculating the Raytrace

        The results will be (1) an OBJECT XY coordinates array of size [N_configs, spaxels_per_slice, 2], and
        (2) a FOCAL XY coordinates array of size [N_configs, N_waves, spaxels_per_slice, 2], notice the wavelengths!

        """

        results_names = ['OBJ_XY', 'FOC_XY']
        N_waves = 23 if wavelength_idx is None else len(wavelength_idx)
        # we need to give the shapes of each array to self.run_analysis
        results_shapes = [(spaxels_per_slice, 2), (N_waves, spaxels_per_slice, 2)]

        metadata = {}
        metadata['Spaxels per slice'] = spaxels_per_slice
        metadata['Configurations'] = 'All' if configuration_idx is None else configuration_idx
        metadata['Wavelengths'] = 'All' if wavelength_idx is None else wavelength_idx

        # read the file options
        file_list, sett_list = create_zemax_file_list(which_system=files_opt['which_system'],
                                                      AO_modes=files_opt['AO_modes'], scales=files_opt['scales'],
                                                      IFUs=files_opt['IFUs'], grating=files_opt['grating'])

        # Loop over the Zemax files
        results = []
        for zemax_file, settings in zip(file_list, sett_list):

            list_results = self.run_analysis(analysis_function=self.analysis_function_raytrace,
                                             files_dir=files_dir, zemax_file=zemax_file, results_path=results_path,
                                             results_shapes=results_shapes, results_names=results_names,
                                             wavelength_idx=wavelength_idx, configuration_idx=configuration_idx,
                                             ignore_vignetting=ignore_vignetting,
                                             surface=surface, spaxels_per_slice=spaxels_per_slice)

            results.append(list_results)

            if save_hdf5 == True:
                # We save the arrays in HDF5 format alongside the analysis metadata in case we want to look at it later
                file_name = zemax_file.split('.')[0]
                settings['surface'] = 'DETECTOR' if surface is None else surface
                self.save_hdf5(analysis_name='RAYTRACE', analysis_metadata=metadata, list_results=list_results,
                               results_names=results_names, file_name=file_name, file_settings=settings, results_dir=results_path)

        return results


class WavefrontsAnalysisMC(AnalysisFast):
    """
    Getting the Wavefront Maps for the Monte Carlo files
    """
    def __init__(self, zosapi):
        super().__init__(zosapi)
        self.samps = {'32x32': constants.SampleSizes_S_32x32,
                     '64x64': constants.SampleSizes_S_64x64,
                     '128x128': constants.SampleSizes_S_128x128,
                     '256x256': constants.SampleSizes_S_256x256,
                     '512x512': constants.SampleSizes_S_512x512,
                     '1024x1024': constants.SampleSizes_S_1024x1024,
                     '2048x2048': constants.SampleSizes_S_2048x2048,
                     '4096x4096': constants.SampleSizes_S_4096x4096,
                     '8192x8192': constants.SampleSizes_S_8192x8192,
                     '16384x16384': constants.SampleSizes_S_16384x16384}

    def analysis_function_wavefronts(self, system, wavelength_idx, config, surface, sampling, remove_slicer=False):

        # Set Current Configuration
        system.MCE.SetCurrentConfiguration(config)
        if config % 20 == 0:
            print("Config: ", config)


        # [WARNING]: for the 4x4 spaxel scale we noticed that a significant fraction of the rays get vignetted at the slicer
        # this introduces a bias in the RMS WFE calculation. To avoid this, we modify the Image Slicer aperture definition
        # so that all rays get through.
        if remove_slicer is True:
            expand_slicer_aperture(system)

        # Get the Field Points for that configuration
        sysField = system.SystemData.Fields
        N_fields = sysField.NumberOfFields
        sysWave = system.SystemData.Wavelengths
        N_waves = len(wavelength_idx)

        if N_fields != 3:       # doublec check that we only have 3 field points
            raise ValueError("Number of Field Points doesn't match. N=%d" % N_fields)

        # Get the Field Point at the centre of the Slice
        fx, fy = sysField.GetField(2).X, sysField.GetField(2).Y

        # Note that this assumes Rectangular Normalization, the default in the E2E files.
        X_MAX = np.max([np.abs(sysField.GetField(i + 1).X) for i in range(N_fields)])
        Y_MAX = np.max([np.abs(sysField.GetField(i + 1).Y) for i in range(N_fields)])

        # Normalized field coordinates (hx, hy)
        hx, hy = fx / X_MAX, fy / Y_MAX

        # The Field coordinates for the Object
        obj_xy = np.array([fx, fy])
        # foc_xy = np.empty((N_waves, 2))
        N_pix = int(sampling.split('x')[0])             # Get the integer value of the sampling by splitting the string
        wavefront_maps = np.zeros((N_waves, N_pix, N_pix))
        rms_wfe = np.zeros(N_waves)

        foc_xy = np.empty((N_waves, 2))      # The Chief Ray coordinates at the Detector
        N_rays = N_waves
        # [2] This is where the core of the RMS WFE calculation takes place
        # First, we begin by defining the Raytrace
        detector_surface = system.LDE.NumberOfSurfaces - 1
        # print(N_rays)
        # print(detector_surface)
        #
        # raytrace = system.Tools.OpenBatchRayTrace()
        # normUnPolData = raytrace.CreateNormUnpol(N_rays, constants.RaysType_Real, detector_surface)

        N_surfaces = system.LDE.NumberOfSurfaces

        # check pupil sampling
        if sampling not in self.samps:
            ValueError('sampling must be one of the following: ' + ', '.join(self.samps.keys()))

        # open Wavefront Map analysis
        awfe = system.Analyses.New_Analysis(constants.AnalysisIDM_WavefrontMap)
        # iterate through wavelengths
        for j, wave_idx in enumerate(wavelength_idx):

            # Take advantage of the loop to simultaneously add the ray to the RayTrace
            # normUnPolData.AddRay(wave_idx, hx, hy, 0, 0, constants.OPDMode_None)

            # set next wavelength
            wavelength = system.SystemData.Wavelengths.GetWavelength(wave_idx).Wavelength
            settings = awfe.GetSettings()
            csettings = CastTo(settings, 'IAS_WavefrontMap')

            # setup analysis
            csettings.Surface.SetSurfaceNumber(N_surfaces)
            csettings.Wavelength.SetWavelengthNumber(wave_idx)  # always changing wave 1
            csettings.Rotation = constants.Rotations_Rotate_0
            csettings.Sampling = self.samps[sampling]
            csettings.Polarization = constants.Polarizations_None
            csettings.ReferenceToPrimary = False    # True --> lateral color
            csettings.UseExitPupil = False          # Use the entrace pupil as reference, not the exit pupil
            csettings.RemoveTilt = True             # True --> ref. to centroid
            csettings.Subaperture_X = 0.
            csettings.Subaperture_Y = 0.
            csettings.Subaperture_R = 1.
            csettings.Field.SetFieldNumber(2)       # change field position

            # run analysis
            awfe.ApplyAndWaitForCompletion()

            # get results
            results = awfe.GetResults()
            cresults = CastTo(results, 'IAR_')
            data = np.array(cresults.GetDataGrid(0).Values)
            # The Wavefront Map comes in waves, we should rescale it to physical units to use it later
            wavefront_maps[j] = data * wavelength * 1e3

            # ptv_zos = float(cresults.HeaderData.Lines[2].split('=')[1].split(' ')[1])
            # rms_zos = float(cresults.HeaderData.Lines[2].split('=')[-1].split(' ')[1])
            # Also store the RMS WFE value so that we can get the distribution later on
            mdata = data - np.nanmean(data)
            rms = np.sqrt(np.nanmean(mdata * mdata))
            rms_wfe[j] = rms * wavelength * 1e3
            # print(ptv_zos, rms_zos)

            # plt.figure()
            # plt.imshow(data, cmap='jet', origin='lower')
            # plt.title(r'Wave: %.3f $\mu$m, Slice #%d | RMS: %.3f $\lambda$ (%.1f nm)' % (wavelength, config, rms_wfe, rms_wfe_nm))
            # plt.colorbar()
            # plt.show()

        # # Run the RayTrace
        # CastTo(raytrace, 'ISystemTool').RunAndWaitForCompletion()
        # normUnPolData.StartReadingResults()
        # for j, wave_idx in enumerate(wavelength_idx):
        #
        #     output = normUnPolData.ReadNextResult()
        #     if output[2] == 0:
        #         x, y = output[4], output[5]
        #         foc_xy[j, 0] = x
        #         foc_xy[j, 1] = y
        #
        #         vignetting_code = output[3]
        #         if vignetting_code != 0:
        #             vignetting_surface = system.LDE.GetSurfaceAt(vignetting_code).Comment
        #             print("\nConfig #%d" % (config))
        #             print("Vignetting at surface #%d: %s" % (vignetting_code, vignetting_surface))
        #
        # normUnPolData.ClearData()
        # CastTo(raytrace, 'ISystemTool').Close()

        # close analysis
        awfe.Close()

        return [wavefront_maps, rms_wfe, obj_xy]

    def loop_over_files(self, files_dir, files_opt, results_path, wavelength_idx=None,
                        configuration_idx=None, surface=None, sampling=1024, remove_slicer_aperture=True):

        results_names = ['WAVEFRONT', 'RMS_WFE', 'OBJ_XY']

        N_waves = 23 if wavelength_idx is None else len(wavelength_idx)
        # we need to give the shapes of each array to self.run_analysis
        results_shapes = [(N_waves, sampling, sampling,), (N_waves,), (2, )]
        sampling_str = '%sx%s' % (sampling, sampling)

        # metadata = {}
        # metadata['Sampling'] = sampling
        # metadata['Configurations'] = 'All' if configuration_idx is None else configuration_idx
        # metadata['Wavelengths'] = 'All' if wavelength_idx is None else wavelength_idx


        # # read the file options
        # file_list, sett_list = create_zemax_file_list(which_system=files_opt['which_system'],
        #                                               AO_modes=files_opt['AO_modes'], scales=files_opt['scales'],
        #                                               IFUs=files_opt['IFUs'], grating=files_opt['grating'])

        # elif monte_carlo is True:
        file_list, sett_list = create_zemax_filename_MC(AO_mode=files_opt['AO_MODE'], scale=files_opt['SPAX_SCALE'],
                                                      IFUpath=files_opt['IFU_PATH'], grating=files_opt['GRATING'],
                                                      FPRS_MC_instance=files_opt['FPRS_MC'],
                                                      IPO_MC_instance=files_opt['IPO_MC'],
                                                      IFU_MC_instance=files_opt['IFU_MC'],
                                                      ISP_MC_instance=files_opt['ISP_MC'])


        # Loop over the Zemax files
        results = []
        for zemax_file, settings in zip(file_list, sett_list):

            list_results = self.run_analysis(analysis_function=self.analysis_function_wavefronts,
                                             files_dir=files_dir, zemax_file=zemax_file, results_path=results_path,
                                             results_shapes=results_shapes, results_names=results_names,
                                             wavelength_idx=wavelength_idx, configuration_idx=configuration_idx,
                                             surface=surface, sampling=sampling_str, remove_slicer=remove_slicer_aperture)

            results.append(list_results)

        return results

### ================================================================================================================ ###
#
### ================================================================================================================ ###




# TODO: implement the PSF Dynamic Effects analysis
class PSFDynamic(AnalysisFast):
    """

    """

    def calculate_fwhm(self, surface, xy_data, PSF_window, N_points, spaxel_scale, wavelength):

        # (1) Calculate the Geometric PSF
        x, y = xy_data[:, 0], xy_data[:, 1]
        cent_x, cent_y = np.mean(x), np.mean(y)

        std_x, std_y = np.std(x), np.std(y)
        bandwidth = min(std_x, std_y)
        kde = KernelDensity(kernel='gaussian', bandwidth=bandwidth).fit(xy_data)

        # define a grid to compute the PSF
        xmin, xmax = cent_x - PSF_window/2/1000, cent_x + PSF_window/2/1000
        ymin, ymax = cent_y - PSF_window/2/1000, cent_y + PSF_window/2/1000
        x_grid = np.linspace(xmin, xmax, N_points)
        y_grid = np.linspace(ymin, ymax, N_points)
        xx_grid, yy_grid = np.meshgrid(x_grid, y_grid)
        xy_grid = np.vstack([xx_grid.ravel(), yy_grid.ravel()]).T
        log_scores = kde.score_samples(xy_grid)

        psf_geo = np.exp(log_scores)
        psf_geo /= np.max(psf_geo)
        psf_geo = psf_geo.reshape(xx_grid.shape)

        # (2) Calculate the Diffraction PSF
        psf_diffr = diffraction.add_diffraction(surface=surface, psf_geo=psf_geo, PSF_window=PSF_window,
                                                scale_mas=spaxel_scale, wavelength=wavelength)

        # collapse the PSF in X to get a Y profile
        psf_y = np.sum(psf_diffr, axis=1)
        psf_y /= np.max(psf_y)

        plt.figure()
        plt.imshow(psf_diffr, cmap='plasma', extent=[xmin, xmax, ymin, ymax])
        plt.colorbar()

        # the X coordinate at which the PSF has the peak
        i_peak = np.argwhere(psf_diffr == 1.0)[0, 0]
        y_profile = psf_diffr[i_peak, :]

        plt.figure()
        plt.plot(psf_y, y_grid)
        # plt.plot(y_grid, y_profile)
        plt.show()

        return

    def analysis_function_dynamic(self, system, wavelength_idx, surface, config, px, py, spaxel_scale, N_points):

        # Set Current Configuration
        system.MCE.SetCurrentConfiguration(config)

        # First of all, we need to find the Surface Number for the IMAGE SLICER "Image Plane"
        N_surfaces = system.LDE.NumberOfSurfaces
        surface_names = {}  # A dictionary of surface number -> surface comment
        for k in np.arange(1, N_surfaces):
            surface_names[k] = system.LDE.GetSurfaceAt(k).Comment
        # find the Slicer surface number
        try:
            slicer_num = list(surface_names.keys())[list(surface_names.values()).index('Image Plane')]
        except ValueError:
            slicer_num = list(surface_names.keys())[list(surface_names.values()).index('Image plane')]
        slicer_surface = slicer_num

        # Get the Field Points for that configuration
        sysField = system.SystemData.Fields
        N_fields = sysField.NumberOfFields
        N_waves = len(wavelength_idx)

        X_MAX = np.max([np.abs(sysField.GetField(i + 1).X) for i in range(N_fields)])
        Y_MAX = np.max([np.abs(sysField.GetField(i + 1).Y) for i in range(N_fields)])

        # Use the Field Point at the centre of the Slice
        fx, fy = sysField.GetField(2).X, sysField.GetField(2).Y
        hx, hy = fx / X_MAX, fy / Y_MAX  # Normalized field coordinates (hx, hy)
        obj_xy = np.array([fx, fy])

        N_pupil = px.shape[0]   # Number of rays in the Pupil for a given field point and wavelength
        N_rays = N_waves * N_pupil

        FWHM = np.zeros((N_waves, 2))
        foc_xy = np.zeros((N_waves, 2))

        slicer_xy = np.empty((N_waves, N_pupil, 2))
        slicer_xy[:] = np.nan
        detector_xy = np.empty((N_waves, N_pupil, 2))
        detector_xy[:] = np.nan

        # (1) Run the raytrace up to the IMAGE SLICER
        raytrace = system.Tools.OpenBatchRayTrace()
        # remember to specify the surface to which you are tracing!
        rays_slicer = raytrace.CreateNormUnpol(N_rays, constants.RaysType_Real, slicer_surface)

        # Loop over all wavelengths
        for i_wave, wave_idx in enumerate(wavelength_idx):

            for (p_x, p_y) in zip(px, py):  # Add the ray to the RayTrace
                rays_slicer.AddRay(wave_idx, hx, hy, p_x, p_y, constants.OPDMode_None)

        CastTo(raytrace, 'ISystemTool').RunAndWaitForCompletion()
        rays_slicer.StartReadingResults()
        checksum_slicer = 0
        for k in range(N_rays):  # Get Raytrace results at the Image Slicer
            i_wave = k // N_pupil
            j_pupil = k % N_pupil
            # print(i_wave, j_pupil)
            output = rays_slicer.ReadNextResult()
            if output[2] == 0:
                slicer_xy[i_wave, j_pupil, 0] = output[4]
                slicer_xy[i_wave, j_pupil, 1] = output[5]
                checksum_slicer += 1
        if checksum_slicer < N_rays:
            raise ValueError('Some rays were lost before the Image Slicer')

        rays_slicer.ClearData()

        # FWHM

        windows = {4.0: [5000, 200], 60.0: [500, 50]}
        win_slicer = windows[spaxel_scale][0]
        win_detect = windows[spaxel_scale][1]
        if config == 1:
            print("Sampling the Image Slicer plane with %d points: %.3f microns / point" % (N_points, (win_slicer / N_points)))


        for i_wave, wave_idx in enumerate(wavelength_idx):

            xy_data_slicer = slicer_xy[i_wave]
            wavelength = system.SystemData.Wavelengths.GetWavelength(wave_idx).Wavelength
            self.calculate_fwhm(surface='IS', xy_data=xy_data_slicer, PSF_window=win_slicer,
                                                               N_points=N_points, spaxel_scale=spaxel_scale,
                                                               wavelength=wavelength)

            # xy_data_detector = detector_xy[i_wave]
            # fwhm_x_det, fwhm_y_det = self.calculate_fwhm(surface='DET', xy_data=xy_data_detector, PSF_window=win_detect,
            #                                              N_points=N_points, spaxel_scale=spaxel_scale,
            #                                              wavelength=wavelength)
            #
            # # plt.show()
            #
            # foc_xy[i_wave] = [np.mean(xy_data_detector[:, 0]), np.mean(xy_data_detector[:, 1])]
            #
            # FWHM[i_wave] = [fwhm_x_det, fwhm_y_slicer]
            #
            # if config == 1:
            #     print("%.3f microns" % wavelength)
            #     print("FWHM in X [Detector]: %.1f microns | in Y [Image Slicer]: %.2f microns " % (fwhm_x_det, fwhm_y_slicer))
            #
        return []

    def loop_over_files(self, files_dir, files_opt, results_path, wavelength_idx=None,
                        configuration_idx=None, N_rays=500, N_points=50):
        """

        """

        # We want the result to produce as output: the RMS WFE array, and the RayTrace at both Object and Focal plane
        results_names = ['FWHM', 'OBJ_XY', 'FOC_XY']
        # we need to give the shapes of each array to self.run_analysis
        N_waves = 23 if wavelength_idx is None else len(wavelength_idx)
        results_shapes = [(N_waves, 2,), (2,), (N_waves, 2,)]

        px, py = define_pupil_sampling(r_obsc=0.2841, N_rays=N_rays, mode='random')
        print("Using %d rays" % N_rays)

        # read the file options
        file_list, sett_list = create_zemax_file_list(which_system=files_opt['which_system'], AO_modes=files_opt['AO_modes'],
                                           scales=files_opt['scales'], IFUs=files_opt['IFUs'], grating=files_opt['grating'])

        results = []
        for zemax_file, settings in zip(file_list, sett_list):

            # Clear the Cache for the Airy Pattern function
            airy_and_slicer.cache_clear()

            mas_dict = {'4x4': 4.0, '10x10': 10.0, '20x20': 20.0, '60x30': 60.0}
            spaxel_scale = mas_dict[settings['scale']]

            list_results = self.run_analysis(analysis_function=self.analysis_function_dynamic,
                                             files_dir=files_dir, zemax_file=zemax_file, results_path=results_path,
                                             results_shapes=results_shapes, results_names=results_names,
                                             wavelength_idx=wavelength_idx, configuration_idx=configuration_idx,
                                             px=px, py=py, spaxel_scale=spaxel_scale, N_points=N_points)
            results.append(list_results)

        return results


if __name__ == """__main__""":

    pass
