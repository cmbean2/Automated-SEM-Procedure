"""
This script performs the automated acquistion of SEM Images using the FEI Autoscript Functionality 
to capture a high resolution grid of images based on user specified settings.

The script is designed to have the user intialize the microscope and find the desired starting location. 
Then upon starting the script will complete the acquistion in a snake like pattern.
It will perform Autofocusing before every image, and then at the beginning and every 10 images 
it will also adjust the brightness and contract as well.

Similarly, the script will display a message on the screen indicating that a automated session 
is in progress, show an estimated end time, and the name and contact of the individual running the session. 

"""
from autoscript_sdb_microscope_client import SdbMicroscopeClient
from autoscript_sdb_microscope_client.enumerations import *
from autoscript_sdb_microscope_client.structures import *

import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv
import os
from scipy.optimize import curve_fit
import warnings
import time
#import em_tasklib2 as em
from datetime import datetime
import scipy
import matplotlib.cm as cm
import imsis as ims
import subprocess
import math
from pathlib import Path
import time
#New Imports
from tqdm import tqdm
import tkinter as tk
from tkinter import ttk
from skimage.feature import match_template
from skimage import io

'''
USER INPUT AREA
'''
#Contact Information
Name = "Name"         #<----------Enter Name Here"
email = "email@email.com"        #<----------Enter Email Here"

#MAKE SURE TO CHANGE THIS DIRECTORY FOR EVERY NEW SAMPLE - It will overwrite previous images if not changed. Also move your images off of the Microscope Computer as normal.
#  On microscope machine
step0Dir = "/path to your folder/"

dic_dwell = 10e-6 #in seconds, actual dwell of collected image
fov = 92e-6 #in meters (field of view of DIC image)

xDim = 2 #number of tiles in x
yDim = 1 #number of tiles in y

x_overlap = 0.15 #a fraction
y_overlap = 0.15

stage_delay = 0.25 #in s for stage movements

# Autofocus settings
autofocus_hfw = 8e-6
autofocus_range = 500e-6 #range of focus sweep
autofocus_dwell = 500e-9 #dwell during autofocus sweep

# Settings to restart in middle of step - 
start_in_middle = False
x_index = 0
y_index = 0


'''
The following classes and functions are from ThermoFisher Scientific and 
are utilized in the script to perform autofocusing and other methods.
'''

#------------------------------------------------------
#!/usr/bin/env python

# Copyright (c) 2022 ThermoFisher Scientific
# All rights reserved.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
Thermo Scientific em_tasklib2 2022
This module contains custom autofunctions.
"""

class AutoFunctions(object):

    @staticmethod
    def autosourcetilt_simple(microscope, stepx=0.05, stepy=0.05, cntmax=9, contrast=0.92, brightness=0.1):
        """ Auto source tilt - iterative search for optimal electron source tilt (gun-tilt)

        :Parameters: microscope, stepx=0.05, stepy=0.05, cntmax=9, contrast=0.92,brightness=0.1
        """
        st = AutoFunctions.__SourceTilt(microscope)
        st.searchoptimumtilt(STEPX=stepx, STEPY=stepy, CNTMAX=cntmax, contrast=contrast, brightness=brightness)

    '''
    @staticmethod
    def autofocus_simple(microscope, wdrange=500e-6, dwelltime=500e-9, scanresolution="768x512", save_images=True,
                         s_center=[0.5, 0.5], s_size=[0.5, 0.5], save_plot=False):
        """ Autofocus simple - Iterative search autofocus, fast for low and high magnifications, with medium to high accuracy.

        :Parameters: microscope,stepsize=0.0001,steps=10, dwell_time=300e-9, s_center=[0.5, 0.5], s_size=[0.5, 0.5]

        """
        af = AutoFunctions.__autofocus_simple(microscope)
        af.autofocus(microscope, wdrange=wdrange, dwelltime=dwelltime, scanresolution=scanresolution,
                     save_images=save_images, s_center=s_center, s_size=s_size, save_plot=save_plot)
    '''

    @staticmethod
    def autofocus_smart(microscope, wdrange=500e-6, dwelltime=500e-9, scanresolution="768x512", save_images=True,
                        s_center=[0.5, 0.5], s_size=[0.5, 0.5], save_plot=True, verbose=False,path = "./images/AFsmart/"):
        """ Autofocus smart - Iterative search autofocus, automatically sets wdrange, dwelltime and scanresolution for SE and BSE images.

        :Parameters: microscope

        """
        af = AutoFunctions.__autofocus_smart(microscope)
        af.autofocus(microscope, wdrange=wdrange, dwelltime=dwelltime, scanresolution=scanresolution,
                     save_images=save_images, s_center=s_center, s_size=s_size, save_plot=save_plot,verbose=verbose,path =path)

    @staticmethod
    def autofocus_sweep(microscope, wdrange=500e-6, dwelltime=500e-9, scanresolution="1536x1024", save_images=True,
                        s_center=[0.5, 0.5], s_size=[0.5, 0.5], save_plot=False, path= "./images/AFsweep/"):
        """ Sweep autofocus - search with curve fitting, slower, reproducible for high magnifications.

        :Parameters: microscope,range=500e-6, dwell=500e-9, s_center=[0.5, 0.5], s_size=[0.5, 0.5]

        .. code-block:: python

            em.AutoFunctions.autofocus_sweep(microscope=microscope)

        """

        af = AutoFunctions.__autofocus_sweep(microscope)
        af.reducedHoriz = s_size[0]
        af.reducedVert = s_size[1]
        af.save_images = save_images
        af.scanResolution = scanresolution
        af.saveGraph = save_plot
        af.sweepAutoFocus(wdrange=wdrange, dwelltime=dwelltime)
        af.savePath=path

    class __autofocus_simple:
        def __init__(self, microscopeClient):
            self.range = 500e-6
            self.dwell = 500e-9
            self.microscope = microscopeClient

        def autofocus(self, microscope, wdrange=500e-6, dwelltime=500e-9, scanresolution="768x512", save_images=True,
                      s_center=[0.5, 0.5], s_size=[0.5, 0.5], save_plot=False, path = "./images/AFsimple/"):
            os.makedirs(os.path.dirname(path), exist_ok=True)

            W0 = self.microscope.beams.electron_beam.working_distance.value
            print(W0)

            # Loop
            sharpnessOpt = 0
            WDopt = 0
            s_width = s_size[0]
            s_height = s_size[1]
            s_left = s_center[0] - s_width * 0.5
            s_top = s_center[1] - s_height * 0.5
            red_area = Rectangle(s_left, s_top, s_width, s_height)
            Wsteps = 21
            Wstepsize = wdrange / Wsteps

            s = GrabFrameSettings(scanresolution, dwell_time=dwelltime, reduced_area=red_area)
            print(s_center, s_size)

            starttime = datetime.now()
            xdata = []
            ydata = []

            for i in range(0, Wsteps):
                wd = W0 + ((i - int(Wsteps / 2)) * Wstepsize)
                self.microscope.beams.electron_beam.working_distance.set_value_no_degauss(wd)
                # microscope.beams.electron_beam.working_distance.value = wd
                img = self.microscope.imaging.grab_frame(s)
                fn = path + "energy-" + str(i)
                if (save_images == True):
                    img.save(fn)  # save to disk
                # LAPV = self.normalizedGraylevelVariance(img.data)
                img0 = img.data

                # LAPV = self.tenengrad(img0)
                autofocusalgorithms = ims.Analyze.SharpnessDetection()
                # sharpness = autofocusalgorithms.tenengrad(img0) #ALGORITHM!
                sharpness = autofocusalgorithms.normalizedGraylevelVariance(img0)  # ALGORITHM!
                xdata.append(wd * 1000.)
                ydata.append(sharpness)
                # GLVN = varianceOfLaplacian(img.data)
                print("AF WD {0:.4f}mm Sharpness {1:.2f}".format(wd * 1000., sharpness))
                if (sharpness > sharpnessOpt):
                    WDOpt = wd
                    sharpnessOpt = sharpness

            # set optimal focus
            # self.microscope.beams.electron_beam.working_distance.value = WDOpt
            self.microscope.beams.electron_beam.working_distance.set_value_no_degauss(WDOpt)
            endtime = datetime.now() - starttime

            # s = GrabFrameSettings(resolution="1536x1024", dwell_time=3000e-9)
            # self.microscope.imaging.grab_frame(s)
            print("Simple AF Found. Focus: WD = {0:.4f} mm Time: {1}".format(WDOpt * 1000., endtime))

            if (save_plot == True):
                plt.clf()
                plt.plot(xdata, ydata, 'o')
                plt.grid()
                plt.xlabel('WD [mm]')
                plt.ylabel('Sharpness')
                plt.title("AFSimple wd found: {0:.4f}".format(WDOpt * 1000.))
                fn = path + "AFSimple_Plot.png"
                plt.savefig(fn)
                plt.close()

    ####

    """
    ITERATIVE SWEEP AUTOFOCUS usable for electron or ion beam
    (Image sharpness calculation function taken from AF script by Remco Geurts)
    Version 1.1
    Lukas Kral, 2017-2018
    """

    class __autofocus_sweep:
        # CLASS CONSTANTS

        def __init__(self, microscopeClient):
            # DEFAULT SETTINGS
            self.save_images = False  # save autofocus sweep images
            self.enableACB = False  # auto contrast brightness at the beginning of each AF sweep
            self.showGraph = False  # show plot with sharpness values and gaussian fit
            self.saveGraph = False  # save sharpness plot(s) to image file(s)
            self.debugInfo = False  # print debug information for troubleshooting purposes
            self.numSteps = 21  # default number of steps of the sweep
            self.scanResolution = "1536x1024"  # full frame scan resolution for acquisition
            self.reducedHoriz = 0.4  # percentage of reduced area horizontal size vs. HFW
            self.reducedVert = 0.4  # percentage of reduced area vertical size vs. VFW
            self.WDrange = 200e-6  # sweep WD (full) range
            self.dwellTime = 1e-6  # dwell time used for acquisition
            self.maxIterations = 10  # maximum number of autofocus iterations
            self.maxCenterOff = 0.1  # iterative AF success criterion - max. WD offset from sweep center
            self.fitMaxErrSpread = 0.7  # criterion for gaussian fit validity assessment
            self.WDChangeTime1st = 0.5  # time after 1st (big) WD change till image gets stable
            self.WDChangeTime = 0.2  # time after consequent (small) WD changes till image gets stable
            self.savePath = "./images/AFsweep/"  # path for saving images and graphs

            # VARIABLES AND SUB-OBJECTS
            self.microscope = microscopeClient
            # plotting objects
            self.fig = None
            self.ax = None
            self.plot1 = None
            self.plot2 = None
            # miscellaneous
            self.nIter = 0  # number of current AF iteration
            self.WDfoc_sigma = 0  # 1-sigma error of focus WD Measurement (estimation from gaussian fitting)

            a = self.microscope.imaging.get_active_device()
            self.beamType = a  # SEM=1, FIB=2

        def getBeam(self):
            # return beam object for current beamType setting
            if self.beamType == 1:
                return self.microscope.beams.electron_beam
            elif self.beamType == 2:
                return self.microscope.beams.ion_beam
            else:
                return None  # error

        def getBeamName(self):
            # return beam name for current beamType setting
            if self.beamType == 1:
                return "SEM"
            elif self.beamType == 2:
                return "FIB"
            else:
                return ""  # error

        def micrWD(self, WD=None):
            # Return WD currently set in microscope for SEM or FIB beam acc. to self.beamType setting
            # If WD parameter is set, new WD will be set to microscope (without degauss!)
            if WD is not None:
                if self.beamType == 1:
                    self.getBeam().working_distance.set_value_no_degauss(WD)  # SEM ... set WD without degauss
                else:
                    self.getBeam().working_distance.value = WD  # FIB ... set WD directly (FIB has no degauss)
            return self.getBeam().working_distance.value

        def micrWD_getLimits(self):
            # get WD limits for current beam
            lims = self.getBeam().working_distance.limits
            return lims.min, lims.max

        def crop_image(self, imgdata, xfact, yfact):
            # Crop image around its center
            # imgdata ... image data (numpy array)
            # xfact, yfact ... relative size of the cropped area w.r.t. original image (< 1.0)
            # Cropped area will be centered around original image center
            ysize, xsize = imgdata.shape  # image dimensions in pixels
            cropx = int(xsize * xfact)  # crop region X size [pix]
            cropy = int(ysize * yfact)  # crop region Y size [pix]
            # crop start coordinates
            startx = xsize // 2 - (cropx // 2)
            starty = ysize // 2 - (cropy // 2)
            # return cropped image array
            return imgdata[starty:starty + cropy, startx:startx + cropx]

        def resolution2xy(self):
            # convert resolution string (e.g. "1024x768") to x and y pixel values
            xs, ys = self.scanResolution.split("x")
            return int(xs), int(ys)

        """
        def deleteAll(self, path, ext):
            # delete all files of given extension in a folder (given by path)
            filelist = [f for f in os.listdir(path) if f.endswith(ext)]
            for f in filelist:
                os.remove(path + f)
        """

        def grabImagesSweep(self):
            """
            Grab series of images - WD sweep
            The WD sweep will be: currently set WD +/- 0.5*WDrange
            WD range is truncated to the range allowed by the microscope SW
            Return: array of WDs and list of images
            """
            if self.save_images:
                os.makedirs(os.path.dirname(self.savePath), exist_ok=True)  # make sure folder exists

            # create array of WDs for the sweep; make sure they are within current instrument limits
            WD0 = self.micrWD()
            WDmin, WDmax = self.micrWD_getLimits()
            WD1 = max(WD0 - 0.5 * self.WDrange, WDmin)
            WD2 = min(WD0 + 0.5 * self.WDrange, WDmax)
            WDs = np.linspace(WD1, WD2, self.numSteps)

            if self.enableACB:
                self.microscope.auto_functions.run_auto_cb()

            # Image resolution x, y
            resx, resy = self.resolution2xy()
            # Time to complete one complete scan of the reduced area
            scantime = 1.1 * resx * self.reducedHoriz * resy * self.reducedVert * self.dwellTime

            # set imaging parameters
            self.getBeam().scanning.resolution.value = self.scanResolution
            self.getBeam().scanning.mode.set_reduced_area(0.5 - self.reducedHoriz / 2.0, 0.5 - self.reducedVert / 2.0,
                                                          self.reducedHoriz, self.reducedVert)
            self.getBeam().scanning.dwell_time.value = self.dwellTime

            i = 0
            images = []

            print("Image acquisition: ", end="", flush=True)

            for wd in WDs:
                i += 1
                print("*", end="", flush=True)

                # set working distance without degaussing
                self.micrWD(wd)

                if i == 1:
                    # In 1st grab cycle, start live scanning in reduced area (takes long time)
                    # and wait longer time for lens to settle (big WD change)
                    self.microscope.imaging.start_acquisition()
                    time.sleep(self.WDChangeTime1st + scantime)
                else:
                    time.sleep(self.WDChangeTime + scantime)

                # Get recently scanned image
                img = self.microscope.imaging.get_image()
                # Crop image to reduced area only

                cropped_img = AdornedImage(self.crop_image(img.data, self.reducedHoriz, self.reducedVert))
                cropped_img.metadata = img.metadata  # copy metadata into cropped image, cropped for fast IO and testing
                if self.save_images:
                    filepath = "{0}AFimage_{1:02d}_{2:02d}".format(self.savePath, self.nIter, i)
                    cropped_img.save(filepath)  # save to disk
                images += [cropped_img]

            # stop live scanning, set full frame again
            self.microscope.imaging.stop_acquisition()
            self.getBeam().scanning.mode.set_full_frame()

            print(" [Done]", flush=True)

            return WDs, images

        def analyzeSharpness(self, images):
            # Analyze sharpness for the set of images
            # Return: array of sharpness values
            sharps = np.array([])  # sharpness values array
            for im in images:
                # sharps = np.append(sharps, self.normalizedGraylevelVariance(im.data))
                autofocusalgorithms = ims.Analyze.SharpnessDetection()
                # sharps = np.append(sharps, autofocusalgorithms.tenengrad(im.data))
                sharps = np.append(sharps, autofocusalgorithms.normalizedGraylevelVariance(im.data))  # ALGORITHM!

            return sharps

        def gauss(self, x, A, mu, sigma, C):
            # Gaussian function to be fitted to sharpness data
            return A * np.exp(-(x - mu) ** 2 / (2. * sigma ** 2)) + C

        def fitGaussian(self, xdata, ydata):
            """
            Fit input x/y data with gaussian bell function
            If optimization fails, exception is raised!
            Warnings are ignored
            Output:
                params ... fit parameters (A, mu, sigma, C) - see gauss() function
                perr ... 1-sigma estimated errors of the fit parameters
            """
            # estimate initial values of parameters:
            initA = max(ydata) - min(ydata)
            initmu = (max(xdata) + min(xdata)) / 2.0
            initsigma = (max(xdata) - min(xdata)) / 2.0
            initC = min(ydata)
            # perform the fit
            with warnings.catch_warnings():  # ignore optimization warnings
                warnings.simplefilter("ignore")
                params, covar_matrix = curve_fit(self.gauss, xdata, ydata, p0=[initA, initmu, initsigma, initC])
                perr = np.sqrt(np.diag(covar_matrix))
            return params, perr

        def fitGaussAndCheck(self, WDs, sharps):
            """
            Fit data with gaussian and check validity of the fit for autofocus
            Input: array of working distances and image sharpnesses
            Output: fit parameters, parameters 1-sigma error estimates, fit success flag
                    (if fit not successful, the parameters and errors are not valid)
            """
            # perform the fit optimization
            try:
                params, perr = self.fitGaussian(WDs, sharps)
                fitSuccess = True
            except:
                if self.debugInfo: print("AF fit error: fit optimization failed")
                params = np.array([0, 0, 0, 0])
                perr = np.array([0, 0, 0, 0])
                fitSuccess = False

            if fitSuccess:
                WDfoc = params[1]  # focused WD
                gaussAmpl = params[0]  # gaussian fit amplitude - should be positive!
                WDfoc_sigma = perr[1]  # 1-sigma estimated error
                # rough check if the fit resulting parameters make any sense
                if (WDfoc_sigma < 1.0) and (WDfoc_sigma > 1e-7) and (gaussAmpl > 0):
                    # check the data spread around the fit - is the fit a reasonable approximation?
                    fitVals = self.gauss(WDs, *params)  # fit evaluation for input WDs
                    deltas = sharps - fitVals  # differences between real data and fit values
                    delta_range = max(deltas) - min(deltas)  # max. spread of differences (errors)
                    fitRelErrSpread = delta_range / (max(fitVals) - min(fitVals))  # relative error spread
                    if fitRelErrSpread > self.fitMaxErrSpread:
                        if self.debugInfo: print(
                            "AF fit error: data spread too large ({0:.2f})".format(fitRelErrSpread))
                        fitSuccess = False  # fit is not a good approximation of data
                    else:
                        if self.debugInfo: print("AF fit success; rel. data spread = {0:.2f}".format(fitRelErrSpread))
                else:  # fit parameters do not make sense
                    if self.debugInfo: print("AF fit error: incorrect result (sig={0:.3f}mm, A={1:.3f})".
                                             format(1e3 * WDfoc_sigma, gaussAmpl))
                    fitSuccess = False

            return params, perr, fitSuccess

        def rebin_factor(self, a, newshape):
            """Rebin an array to a new shape.
            newshape must be a factor of a.shape.
            """
            assert len(a.shape) == len(newshape)
            assert not np.sometrue(np.mod(a.shape, newshape))

            slices = [slice(None, None, old / new) for old, new in zip(a.shape, newshape)]
            return a[slices]

        def getArrow(self, centerOff):
            # get appropriate arrow symbol string for different centerOff values (for user messages only)
            if centerOff > 0.499: return ">>>"
            if 0.25 < centerOff <= 0.499: return "-->"
            if 0 <= centerOff <= 0.25: return "->"
            if -0.25 < centerOff < 0: return "<-"
            if -0.499 <= centerOff <= -0.25: return "<--"
            if centerOff < -0.499: return "<<<"

        def createPlot(self):
            # create plot objects and apply basic settings
            # warnings.filterwarnings("ignore", ".*GUI is implemented.*") #REMOVED CAUSES LOCKUPS
            # plt.ion()   # plotting set to interactive mode - not blocking further script execution #REMOVED CAUSES LOCKUPS
            self.fig = plt.figure()
            self.ax = self.fig.add_subplot(111)
            self.plot1, = self.ax.plot([], [], 'o')  # plot of sharpness values
            self.plot2, = self.ax.plot([], [], 'r-')  # plot of gaussian fit
            plt.xlabel('WD [mm]')
            plt.ylabel('Sharpness')

            plt.title("{0} Autofocus".format(self.getBeamName()))
            plt.grid(True)

        def plotResults(self, WDs, sharps, wdsfit=None, gfit=None):
            # plot sweep results and fit curve
            if self.fig is None:
                self.createPlot()

            self.plot1.set_xdata(1.0e3 * WDs)
            self.plot1.set_ydata(sharps)

            if wdsfit is not None:
                self.plot2.set_xdata(1.0e3 * wdsfit)
                self.plot2.set_ydata(gfit)
            else:
                self.plot2.set_xdata([])
                self.plot2.set_ydata([])

            self.ax.relim()
            self.ax.autoscale_view()

            if self.saveGraph:
                fn = "{0}AFplot_{1:02d}.png".format(self.savePath, self.nIter)
                self.fig.savefig(fn)

        def singleAF(self):
            """
            Single run of sweep autofocus (to be used as an internal function only!)
            Sets WD to optimal focus at the end
            Output:
                WDfoc ... focused WD
                WDfoc_sigma ... estimated standard deviation
                fitSuccess ... success flag boolean
                outOfRange ... boolean - was focus out of the sweep WD range?
            """
            # Perform the sweep
            WDs, images = self.grabImagesSweep()
            # Calculate sharpness for individual images
            sharps = self.analyzeSharpness(images)

            # Gaussian fitting
            params, perr, fitSuccess = self.fitGaussAndCheck(WDs, sharps)
            WDfoc = params[1]  # focused WD
            WDfoc_sigma = perr[1]  # 1-sigma estimated error

            outOfRange = False  # default
            # truncate the focused WD to the sweep WD range
            if WDfoc <= min(WDs):
                WDfoc = min(WDs)
                outOfRange = True
            if WDfoc >= max(WDs):
                WDfoc = max(WDs)
                outOfRange = True

            if not fitSuccess:
                # gaussian fit not successful => assume focus is out of sweep range
                # => use linear fit to find out if we should move the sweep window right or left
                #print(WDs,sharps)
                linpars = np.polyfit(WDs, sharps, 1)  # linear fit
                if linpars[0] >= 0:  # positive slope => go right
                    WDfoc = max(WDs)
                else:  # negative slope => go left
                    WDfoc = min(WDs)
                outOfRange = True
                WDfoc_sigma = 0  # error flag

            # set WD to focused value
            self.micrWD(WDfoc)

            if self.showGraph or self.saveGraph:
                wdsfit = np.linspace(min(WDs), max(WDs), 100)  # x values for fit curve evaluation
                if fitSuccess:
                    valsfit = self.gauss(wdsfit, *params)  # fit curve
                else:
                    p = np.poly1d(linpars)
                    valsfit = p(wdsfit)  # fit curve
                self.plotResults(WDs, sharps, wdsfit, valsfit)

            if self.debugInfo:
                print("AF sweep {0:.4f}...{1:.4f} mm, F={2:.4f}mm, sig={3:.4f}mm, fitsucc={4}, ooR={5}".
                      format(min(WDs) * 1000.0, max(WDs) * 1000.0, WDfoc * 1000.0, WDfoc_sigma * 1000.0, fitSuccess,
                             outOfRange))

            return WDfoc, WDfoc_sigma, fitSuccess, outOfRange

        def sweepAutoFocus(self, wdrange=None, dwelltime=None, singleSweep=False):
            """
            Perform autofocus, fit the sharpness values with gaussian function => focused WD
            Focused WD is set to the microscope at the end
            Input and output parameters:
            :param beam: SEM of FIB beam to be focused (see class constants); if omitted, last setting is used
            :param range: Autofocus sweep range [m]; if omitted, last setting is used
            :param dwell: Dwell time for AF images [s]; if omitted, last setting is used
            :param singleSweep: Perform single AF sweep only and finish (no iterative mode)
            :return: focused WD [m]; if AF not successful, WDfoc is set to -999
            Also, estimate of focused WD error (1-sigma) is available through AutoFocus.WDfoc_sigma property
            """

            self.nIter = 0

            # update class properties according to input parameters
            if wdrange is not None: self.WDrange = wdrange
            if dwelltime is not None: self.dwellTime = dwelltime
            # set active view and select beam
            if self.beamType == 1:
                self.microscope.imaging.set_active_view(1)
                self.microscope.imaging.set_active_device(ImagingDevice.ELECTRON_BEAM)
            elif self.beamType == 2:
                self.microscope.imaging.set_active_view(2)
                self.microscope.imaging.set_active_device(ImagingDevice.ION_BEAM)
            else:
                print("AF ERROR: Invalid beam type!")
                return

            if singleSweep:
                print(self.getBeamName() + " SWEEP AUTOFOCUS - single sweep")
            else:
                print(self.getBeamName() + " SWEEP AUTOFOCUS - iterative")

            if self.saveGraph:
                os.makedirs(os.path.dirname(self.savePath), exist_ok=True)  # make sure folder exists
                # self.deleteAll(self.savePath, ".png") # delete old graphs
                # self.deleteAll(self.savePath, ".tif")  # delete old images

            self.getBeam().scanning.mode.set_reduced_area()

            # perform first sweep AF
            WDorig = self.micrWD()
            WDcenter = WDorig
            WDfoc = WDcenter
            centerOff = 0  # relative offset of focused WD from sweep center
            AFsuccess = False
            # iterative autofocus - repeat sweeps till AF successful and focus is near sweep center (=> max. precision)
            starttime = datetime.now()

            while (not AFsuccess) or (abs(centerOff) > self.maxCenterOff):
                self.nIter += 1
                if not singleSweep:
                    msg = "AF iteration #{0}".format(self.nIter)
                    if self.nIter > 1:
                        msg += "    Sweep range shifted " + self.getArrow(centerOff)
                    print(msg)
                WDcenter = self.micrWD()
                # perform sweep autofocus
                WDfoc, self.WDfoc_sigma, fitSuccess, outOfRange = self.singleAF()
                AFsuccess = fitSuccess and not outOfRange
                centerOff = (WDfoc - WDcenter) / self.WDrange  # update center offset - is focus close to range center?
                if not AFsuccess and (self.nIter > self.maxIterations):
                    print("AF ERROR: Maximum number of iterations exceeded!")
                    break
                if singleSweep: break
            endtime = datetime.now() - starttime
            if AFsuccess:
                print("Sweep AF SUCCESS. Focus: WD = {0:.4f} +/- {1:.5f} mm, Time: {2}".format(1.0e3 * WDfoc,
                                                                                               1.0e3 * 3.0 * self.WDfoc_sigma,
                                                                                               endtime))
            else:
                # set WD to original value
                self.micrWD(WDorig)
                print("AF ERROR: Autofocus failed! Original WD restored.")
                WDfoc = -999  # error flag

            self.getBeam().scanning.mode.set_full_frame()
            return WDfoc

    class __SourceTilt:
        def __init__(self, microscopeClient):
            self.microscope = microscopeClient

        def tiltoffset(self):
            im = self.microscope.imaging.get_image()
            img2 = ims.Image.Adjust.threshold(im.data, 128)
            img3, tiltx, tilty = ims.Analyze.find_image_center_of_mass(img2)
            if (tiltx == 0) and (tilty == 0):
                print('tiltx, tilty {0},{1}'.format(tiltx, tilty))
            # ims.Image.plot(img3)

            imwidth = im.data.shape[1]
            imheight = im.data.shape[0]
            dx = int((imwidth * 0.5 - tiltx)) * -1
            dy = int((imheight * 0.5 - tilty)) * -1
            return dx, dy

        def searchoptimumtilt(self, STEPX=0.05, STEPY=0.05, CNTMAX=5, contrast=0.92, brightness=0.1):
            self.microscope.imaging.set_active_view(1)
            self.microscope.imaging.set_active_device(ImagingDevice.ELECTRON_BEAM)
            # self.microscope.detector.set_type_mode(DetectorType.ETD, DetectorMode.SECONDARY_ELECTRONS)
            self.microscope.beams.electron_beam.scanning.mode.set_crossover()

            self.microscope.detector.brightness.value = brightness
            self.microscope.detector.contrast.value = contrast

            # determine vector
            self.microscope.imaging.start_acquisition()
            # get start position
            time.sleep(1)
            dxmin, dymin = self.tiltoffset()
            p0 = self.microscope.beams.electron_beam.source_tilt.value
            p0min = p0

            FINISHED = False
            CNT = 0

            vectlist = []

            if ((np.abs(dxmin) > 1) or (np.abs(dymin) > 1)):
                while (FINISHED == False):
                    p0 = self.microscope.beams.electron_beam.source_tilt.value
                    p0.x = p0.x + STEPX
                    p0.y = p0.y + STEPY
                    self.microscope.beams.electron_beam.source_tilt.value = p0
                    time.sleep(0.25)
                    dx, dy = self.tiltoffset()

                    if (np.abs(dx) < np.abs(dxmin)):
                        dxmin = dx
                        STEPX = STEPX * 0.5
                        p0min.x = p0.x
                    else:
                        STEPX = STEPX * -1.0 * 0.7
                        self.microscope.beams.electron_beam.source_tilt.value = p0min

                    if (np.abs(dy) < np.abs(dymin)):
                        dymin = dy
                        p0min.y = p0.y
                        STEPY = STEPY * 0.5
                    else:
                        STEPY = STEPY * -1.0 * 0.7
                        self.microscope.beams.electron_beam.source_tilt.value = p0min

                    if (np.abs(dx) < 2):
                        STEPX = 0
                    if (np.abs(dy) < 2):
                        STEPY = 0

                    if (STEPX == 0) and (STEPY == 0):
                        FINISHED = True

                    if (CNT > CNTMAX):
                        FINISHED = True
                    print('step {0} pixels {1},{2} tilt {3},{4}'.format(CNT, dxmin, dymin, STEPX, STEPY))
                    CNT = CNT + 1
            else:
                print('already centered, pixels {0},{1}'.format(dxmin, dymin))
            self.microscope.beams.electron_beam.scanning.mode.set_full_frame()

    class __autofocus_smart:
        def __init__(self, microscopeClient):
            self.range = 500e-6  # not used
            self.dwell = 500e-9  # not used
            self.microscope = microscopeClient
            # range, dwelltime and resolution automatically set.

        def autofocus(self, microscope, wdrange=500e-6, dwelltime=500e-9, scanresolution="768x512", save_images=True,
                      s_center=[0.5, 0.5], s_size=[0.5, 0.5], save_plot=False,verbose=False, path = "./images/AFsmart/"):
            os.makedirs(os.path.dirname(path), exist_ok=True)
            print("SMART AUTOFOCUS")
            W0 = self.microscope.beams.electron_beam.working_distance.value

            # Loop
            s_width = s_size[0]
            s_height = s_size[1]
            s_left = s_center[0] - s_width * 0.5
            s_top = s_center[1] - s_height * 0.5
            red_area = Rectangle(s_left, s_top, s_width, s_height)
            Wsteps = 21

            starttime = datetime.now()
            hfwcurr = self.microscope.beams.electron_beam.horizontal_field_width.value
            hfwlist = [500 * 1e-6, 100 * 1e-6, 50 * 1e-6, 20*1e-6]
            wdrangelist = [1000 * 1e-6, 500 * 1e-6, 400 * 1e-6, 300*1e-6]

            try:
                if (microscope.beams.electron_beam.optical_mode.value == OpticalMode.IMMERSION):
                    hfwlist = [500 * 1e-6, 100 * 1e-6, 50 * 1e-6, 10*1e-6]
                    wdrangelist = [1000 * 1e-6, 500 * 1e-6, 200 * 1e-6, 50*1e-6]
            except:
                print("error: optical immersion most likely does not exist on this microscope, skipping")

            k = 0
            if (hfwcurr <= hfwlist[1]):
                k = 1
            if (hfwcurr <= hfwlist[2]):
                k = 2
            if (hfwcurr <= hfwlist[3]):
                k = 3

            if (verbose==True):
                print("selected conditions, ", k)
            scanresolutionlist = [ScanningResolution.PRESET_768X512, ScanningResolution.PRESET_768X512,
                                  ScanningResolution.PRESET_768X512, ScanningResolution.PRESET_768X512]
            dwelltimelist = [500 * 1e-9, 500 * 1e-9, 300 * 1e-9, 300 * 1e-9]

            wdrange = wdrangelist[k]
            scanresolution = scanresolutionlist[k]
            dwelltime = dwelltimelist[k]

            s = GrabFrameSettings(scanresolution, dwell_time=dwelltime, reduced_area=red_area)
            self.microscope.beams.electron_beam.scanning.mode.set_reduced_area(s_left, s_top, s_width, s_height)

            self.microscope.beams.electron_beam.scanning.resolution.value = scanresolution
            self.microscope.beams.electron_beam.scanning.dwell_time.value = dwelltime

            self.microscope.imaging.start_acquisition()
            time.sleep(0.2)
            img0 = self.microscope.imaging.get_image().data
            if (verbose==True):
                print(s_left,s_top,s_left+s_width,s_top+s_height)
            x0 = int(img0.shape[1]*s_left)
            y0 = int(img0.shape[0]*s_top)
            x1 = int(img0.shape[1]*(s_left+s_width))
            y1 = int(img0.shape[0]*(s_top+s_height))
            img0 = ims.Image.crop(img0, x0,y0,x1,y1)
            WDChangeTime = 0.2
            xdata = []
            ydata = []

            Wsteps=21
            Wstepsize = wdrange / Wsteps
            if verbose==True:
                print('range_mm {0:.2f} wsteps {1:.2f} wstepsize_mm {2:.2f}, currentwd_mm {3:.2f}'.format(wdrange*1000,Wsteps,Wstepsize*1000,W0*1000))

            scantime = img0.shape[0]*img0.shape[1]*dwelltime
            #acquire images with same time interval
            wdimage_list =[]
            for i in range(0, Wsteps):
                wd = W0 + ((i - int(Wsteps / 2)) * Wstepsize)
                self.microscope.beams.electron_beam.working_distance.set_value_no_degauss(wd)
                time.sleep(WDChangeTime+scantime)
                img = self.microscope.imaging.get_image()
                img0 = AdornedImage(ims.Image.crop(img.data, x0, y0, x1, y1))
                img0.metadata=img.metadata
                wdimage_list.append(img0)
                xdata.append(wd * 1000.)

            autofocusalgorithms = ims.Analyze.SharpnessDetection()
            i=0
            for img in wdimage_list:
                fn = path + "energy-" + str(i)
                if (save_images == True):
                    img.save(fn)  # save to disk

                # LAPV = self.normalizedGraylevelVariance(img.data)
                img0 = img.data
                sharpness = autofocusalgorithms.normalizedGraylevelVariance(img0)  # ALGORITHM!
                ydata.append(sharpness)
                # GLVN = varianceOfLaplacian(img.data)
                if (verbose==True):
                    print("AF_WD {0:.4f}mm Sharpness {1:.2f}".format(xdata[i], sharpness))
                i=i+1

            # set optimal focus
            xdata0 = np.asarray(xdata)
            ydata0 = np.asarray(ydata)
            WDOpt = xdata0[np.argmax(ydata0)]/1000.
            print("WD found: ",WDOpt)
            self.microscope.beams.electron_beam.working_distance.set_value_no_degauss(WDOpt)
            self.microscope.imaging.stop_acquisition()
            self.microscope.beams.electron_beam.scanning.mode.set_full_frame()
            #endtime = datetime.now() - starttime

            # s = GrabFrameSettings(resolution="1536x1024", dwell_time=3000e-9)
            # self.microscope.imaging.grab_frame(s)
            if (save_plot == True):
                plt.clf()
                plt.plot(xdata, ydata, 'o')
                plt.grid()
                plt.xlabel('WD [mm]')
                plt.ylabel('Sharpness')
                plt.title("AFSmart wd found: {0:.4f}".format(WDOpt * 1000.))
                fn = path + "AFSmart_Plot.png"
                plt.savefig(fn)

#-------------------------------------------------------

'''
Developed Code
'''

def setETD(quad):
    """
    Sets etd to quad of user's choice
    :param quad: which imaging view to select
        1 is upper left
        2 is upper right
        3 is lower left
        4 is lower right
    :return: none
    """
    microscope.imaging.set_active_view(quad)
    microscope.imaging.set_active_device(1)  # electron beam
    microscope.detector.type.value = "ETD"

#Checks to see if save folder exists and if not creates it.
if not os.path.exists(step0Dir):
    os.makedirs(step0Dir)

#opens a windows explorer for the save directory.
subprocess.Popen(['explorer', step0Dir])

# Calculate the total number of images to capture
total_images = xDim * yDim

# Calculate the estimated total time
HR_resolution_width = 6144
HR_resolution_height = 4096
focus_routine_time = 20 #seconds per image focus time
estimated_total_time = total_images*(dic_dwell * (HR_resolution_width * HR_resolution_height) + focus_routine_time)

def update_progress(progress_var, total_images, current_image, estimated_time_remaining):
    progress = int((current_image / total_images) * 100)
    progress_var.set(progress)
    time_label.config(text=f"Time Remaining: {estimated_time_remaining:.2f} seconds")
    root.update()

def update_time_label(remaining_time):
    hours, remainder = divmod(remaining_time, 3600)
    minutes, _ = divmod(remainder, 60)
    time_label.config(text=f"Time Remaining: {int(hours)} hours {int(minutes)} minutes")

# Create the main GUI window
root = tk.Tk()
root.title("Experiment in Progress: Please do not disturb.")
root.geometry("1500x400")  # Adjust the window size for the larger text

# Create a label for the title
title_label = tk.Label(root, text="Experiment in Progress: Do not disturb.", font=("Helvetica", 40))
title_label.pack(pady=10)

# Create labels to display the entered name and email
name_label = tk.Label(root, text=f"If needed, contact {Name} at {email} for help.", font=("Helvetica", 24))
name_label.pack()

# Create a progress bar
progress_var = tk.DoubleVar()
progress_bar = ttk.Progressbar(root, variable=progress_var, maximum=100)
progress_bar.pack(pady=10)

# Create a label to display the estimated time remaining
time_label = tk.Label(root, text="Time Remaining: Calculating...", font=("Helvetica", 24))
time_label.pack()

# Store the start time
start_time = time.time()


#############################################################################################################
#Begin Autoscript Code
microscope = SdbMicroscopeClient()
microscope.connect()
##Make sure we're working in RAW stage coordinates
microscope.specimen.stage.set_default_coordinate_system(CoordinateSystem.RAW)

#Turns off the Ion Beam
if microscope.beams.ion_beam.is_on:
    microscope.beams.ion_beam.turn_off()
    print("Ion Beam has been turned off.")
else:
    print("Ion Beam was already turned off.")
    
# Reset beam shift to zero
microscope.beams.electron_beam.beam_shift.value = Point(0, 0)

##Read current position. x,y,z in meters; t,r in radians
rawXstart = microscope.specimen.stage.current_position.x
rawYstart = microscope.specimen.stage.current_position.y
rawZstart = microscope.specimen.stage.current_position.z
rawTstart = microscope.specimen.stage.current_position.t
rawRstart = microscope.specimen.stage.current_position.r

print(microscope.specimen.stage.current_position.x, microscope.specimen.stage.current_position.y)

if not start_in_middle:
    start_x_index = 0
    start_y_index = 0
    microscope.specimen.stage.absolute_move(StagePosition(x=rawXstart))
    time.sleep(stage_delay)
    microscope.specimen.stage.absolute_move(StagePosition(y=rawYstart))
    time.sleep(stage_delay)
    with open("start_position.txt", "w") as initialPos:
        initialPos.write('xstartPosition: {} mm\ystartPosition: {} '
                         'mm'.format(rawXstart*1000, rawYstart*1000))

else:
    start_y_index = y_index
    if y_index % 2 != 0:
        start_x_index = xDim - x_index - 1
    else:
        start_x_index = x_index

    with open("start_position.txt", "r+") as initialPos:
        lines = initialPos.readlines()
    for i, line in enumerate(lines):
        if "xstartPosition" in line:
            equal_idx = line.find(":")
            if equal_idx != -1:
                sval = line[line.find(":") + 1:-3]
                x_start_position = float("".join(sval.split()))
                print(x_start_position)
        if "ystartPosition" in line:
            equal_idx = line.find(":")
            if equal_idx != -1:
                sval = line[line.find(":") + 1:-3]
                y_start_position = float("".join(sval.split()))
    microscope.specimen.stage.absolute_move(StagePosition(x=x_start_position/1000))
    time.sleep(stage_delay)
    microscope.specimen.stage.absolute_move(StagePosition(y=y_start_position/1000))
    time.sleep(stage_delay)

print("Start x position: {} mm\nStart y position: {} mm".format(microscope.specimen.stage.current_position.x*1000, microscope.specimen.stage.current_position.y*1000))

rawX = microscope.specimen.stage.current_position.x
rawY = microscope.specimen.stage.current_position.y

hfw = fov * 1.5 # from 4:3 ratio of images in microscope
xShift = (hfw - hfw*x_overlap) # This is horizontal shift from hfw
yShift = (fov - fov*y_overlap) # This is just FOV as the vertical resolution is 2/3 of hfw

microscope.beams.electron_beam.horizontal_field_width.value = hfw

if not start_in_middle:
    start_x_index = 0
    start_y_index = 0

# Initialize a counter variable to keep track of the number of images processed
image_counter = 0

with tqdm(total=total_images, desc="Capturing Images") as pbar:
    for j in range(start_y_index, yDim): #y shifts
        for i in range(start_x_index, xDim): #xshifts
            # Stage movement in serpentine pattern
            # Update the progress bar
            pbar.update(1)

            # Update the progress bar and time in the GUI window
            elapsed_time = time.time() - start_time
            remaining_time = estimated_total_time - elapsed_time
            update_progress(progress_var, total_images, pbar.n, remaining_time)
            update_time_label(remaining_time)

            # This line allows the GUI to update without becoming unresponsive
            root.update_idletasks()

            if j % 2 == 0:
                microscope.specimen.stage.absolute_move(
                    StagePosition(x=rawX + (xShift * i)))
                col = i
            else:
                microscope.specimen.stage.absolute_move(
                    StagePosition(x=rawX + (xShift * (xDim - i - 1))))
                col = xDim - i - 1
            time.sleep(stage_delay)
            microscope.specimen.stage.absolute_move(StagePosition(y=rawY - (yShift*j)))
            time.sleep(stage_delay)
            print(microscope.specimen.stage.current_position.x*1e6, microscope.specimen.stage.current_position.y*1e6)

            ################ BEGINNING OF IMAGE TAKING #################################
            # Run initial autofocus
            microscope.beams.electron_beam.horizontal_field_width.value = autofocus_hfw
            # Set inital resolution
            microscope.beams.electron_beam.scanning.resolution.value = "1024x884"  # or "768x512" , "3072x2048", "6144x4096"
            # Run smart autofocus for initial focus
            AutoFunctions.autofocus_smart(microscope,
                                wdrange=autofocus_range,
                                dwelltime=autofocus_dwell,
                                scanresolution="768x512",
                                save_images=False,
                                s_center=[0.5, 0.5],
                                s_size=[0.5, 0.5],
                                save_plot=False,
                                verbose=False,
                                path="./images/AFsmart/")

            ######################## Auto Astigmatism ################################
            #Run initial Autostigmator - the "OngEtAl" auto stigmator routine
            settings = RunAutoStigmatorSettings(method="OngEtAl")
            microscope.auto_functions.run_auto_stigmator(settings)
            print("Autostigmatism Set")

            ########################## Auto Contrast and Brightness ##############################

            # Update the image counter
            image_counter += 1

            # Check if the counter is a multiple of 10 to run auto_cb
            if image_counter == 1 or (image_counter % 10 == 0):
                #Default Contrast and Brightness function 
                #microscope.auto_functions.run_auto_cb()

                #Alternate Contrast/AutoBrightness function
                settings = RunAutoCbSettings(method="MaxContrast", resolution="768x512", dwell_time = autofocus_dwell, calibrate_detector=True)
                #Additional Setting options: "line_integration = int" , "max_black_clipping = float [0,1]", "max_white_clipping = float [0,1]", "number_of_frames = int", "brightness_target = float [0,1]", "method = str"
                microscope.auto_functions.run_auto_cb(settings)
   
            #Run Autofocus_smart after stigmator to refocus the sample.
            AutoFunctions.autofocus_smart(microscope,
                                wdrange=autofocus_range,
                                dwelltime=autofocus_dwell,
                                scanresolution="768x512",
                                save_images=False,
                                s_center=[0.5, 0.5],
                                s_size=[0.5, 0.5],
                                save_plot=False,
                                verbose=False,
                                path="./images/AFsmart/")
            
            ################ BEGINNING OF HR IMAGE TAKING #################################
            # run high-resolution images
            microscope.beams.electron_beam.horizontal_field_width.value = hfw
            microscope.beams.electron_beam.scanning.resolution.value = "6144x4096"
            
            # Collect HR image
            imgSettings = GrabFrameSettings(resolution = "6144x4096", dwell_time = dic_dwell)
            image = microscope.imaging.grab_frame(imgSettings)
            image.save("{}//{}_{}.tif".format(step0Dir, j+1, col+1))

            print("Image {}_{} Saved".format(j+1, col+1))
            
    microscope.beams.electron_beam.scanning.resolution.value = "1536x1024"
    microscope.beams.electron_beam.turn_off()

    #Turns off the Ion Beam
    if microscope.beams.ion_beam.is_on:
        microscope.beams.ion_beam.turn_off()
        print("Ion Beam has been turned off.")
    else:
        print("Ion Beam was already turned off.")
    print('Done!')