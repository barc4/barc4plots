#!/usr/bin/python
# coding: utf-8
###################################################################################
# BARC for data plotting
# Authors/Contributors: Rafael Celestre
# Rafael.Celestre@synchrotron-soleil.fr
# creation: 25.07.2018
# previously updated: 10.04.2024 (v.06)
# last update: 19.07.2024 (v.07)
###################################################################################

import warnings
from copy import deepcopy

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rcParamsDefault, ticker
from matplotlib.colors import LogNorm, PowerNorm
from matplotlib.widgets import EllipseSelector, RectangleSelector
from scipy.interpolate import RegularGridInterpolator
from scipy.stats import gaussian_kde, moment


class PlotManager:
    """
    Class for handling numpy arrays and plotting them.
    """

    def __init__(self, image=None, axis_x=None, axis_y=None):
        """
        Initializes the PlotManager object.

        Args:
            image (numpy.ndarray, optional): 1D or 2D numpy array representing the image data. Defaults to None.
            axis_x (numpy.ndarray, optional): Horizontal axis/events (for histograms). Defaults to None.
            axis_y (numpy.ndarray, optional): Vertical axis/events (for histograms). Defaults to None.
        """
        # ------------------------------------------------
        # basic elements of the class
        self.image = image  # 1D or 2D numpy array
        self.x = axis_x     # horizontal axis
        self.y = axis_y     # vertical axis
        # ------------------------------------------------
        # additional info for plotting
        self.AxLimits = [None, None, None, None]  # [xmin, xmax, ymin, ymax]
        self.AxLegends = [' ', 'x-axis', 'y-axis']
        # ------------------------------------------------
        # aesthetics
        self.dpi = 400
        self.LaTex = False  # Boolean - LaTex fonts when compiling
        self.AspectRatio = True
        self.ColorScheme = 0  # see _color_palette_1d and _color_palette_2d
        self.PlotScale = 0  
                            # 1D plot: 0 - linear; 1 - semilogY; 2 - semilogX; 3 - loglog
                            # 2D plot: 0 - linear; 1 - log10; 2: Gamma = 0.25 (default)
        self.FontsSize = 1 
        self.grid = False
        self.nbins = 4  # number of max of bins (Maximum number of intervals; one less than max number of ticks)
        # ------------------------------------------------
        # 1D plots
        self.Label = None  # curve label
        self.LabelPos = 1  # 'best' 0; 'upper right' 1; 'upper left' 2; 'lower left' 3; 'lower right' 4;
                           # 'right' 5; 'center left' 6; 'center right' 7; 'lower center' 8; 'upper center' 9;
                           # 'center' 10
        self.LineStyle = '-'  # '-', '--', '-.', ':'
                              # "o" circle; "v" triangle_down; "^" triangle_up; ">" triangle_right; "<" triangle_left;  "8" octagon;
                              # "s" square; "p" pentagon; "P" plus (filled); "*" star; "h" hexagon1; "H" hexagon2; "+" plus; "x" x;
                              # "X" x (filled); "D" diamond;  "d" thin_diamond
        self.FillBetween = False  # fill in between the curve and FillBetweenValue with alpha 50%
        self.FillBetweenValue = 0
        self.alpha = 1
        self.xticks = None
        self.xticksaxis = None
        # twin axis
        self.twinAxis = False
        self.twinImage = None
        self.twinX = None
        self.twinColorScheme = 1  # see _color_palette_1d
        self.twinLabel = None
        self.twinAlpha = 1        # see _color_palette_1d
        self.twinPlotScale = 0
        self.twinAxLegends='twin y-axis'
        self.twinLineStyle='-'
        self.twinAxLimits = [None, None]  # [twinYmin, twinYmin]
        # ------------------------------------------------
        # 2D plots
        self.MinMax = [None, None]                # [min, max] values on a 2D graph - self.old plt_limits
        self.Colorbar = True
        self.ColorbarExt = 'neither'              # [ 'neither' | 'both' | 'min' | 'max' ]
        self.Gamma = 0.25
        self.IsPhase = False
        self.showXhist = True
        self.showYhist = True
        # ------------------------------------------------
        # scatter
        self.s=None   # marker size in points**2
        self.edgeColors='none' # The edge color of the marker. 
        self.monochrome=True
        self.nSigma = 5   # std/nSigma bin width
        # ------------------------------------------------
        # histograms
        self.nbinsHist = None
        self.wbinsHist = None         
        self.rule = -1
        self.discardZeros = False
        self.norm = False
        self.bold = True
        # ------------------------------------------------
        # quiver plot 
        self.fld_X = None   # horizontal 2D field
        self.fld_Y = None   # vertical 2D field
        self.kk = 50
        # ------------------------------------------------
        # 3D plots
        self.Style3D = 'surf'  # contour, wire, surf

        if self.image is not None:
            if self.x is None and self.y is None:
                self.sort_axes()

    def get_attributes(self):
        """
        Prints all attribute of object
        """

        for i in (vars(self)):
            print("{0:10}: {1}".format(i, vars(self)[i]))

    def additional_info(self, title='', x_axis='', y_axis='', 
                        xmin=None, xmax=None, 
                        ymin=None, ymax=None, 
                        min=None, max=None, 
                        sort_ax=True, sort_ax_lim=True):

        self.AxLimits = [xmin, xmax, ymin, ymax]
        self.AxLegends = [title, x_axis, y_axis]
        self.MinMax = [min, max]

        if sort_ax:
            self.sort_axes()
        if sort_ax_lim:
            self.sort_axes_limits()
        return self

    def aesthetics(self, dpi=None, LaTex=None, AspectRatio=None, PlotScale=None, 
                   FontsSize=None, grid=None, nbins=None):

        if dpi is not None:
            self.dpi = dpi
        if LaTex is not None:
            self.LaTex = LaTex
        if AspectRatio is not None:
            self.AspectRatio = AspectRatio
        if FontsSize is not None:
            self.FontsSize = FontsSize
        if PlotScale is not None:
            self.PlotScale = PlotScale
        if grid is not None:
            self.grid = grid
        if nbins is not None:
            self.nbins = nbins
        return self

    def info_1d_plot(self, ColorScheme=None, Label=None, LabelPos=None, LineStyle=None, 
                     FillBetween=None, FillBetweenValue=None, alpha=None, xticks=None,
                     xticksaxis=None):

        if ColorScheme is not None:
            self.ColorScheme = ColorScheme
        if Label is not None:
            self.Label = Label
        if LabelPos is not None:
            self.LabelPos = LabelPos
        if LineStyle is not None:
            self.LineStyle = LineStyle
        if FillBetween is not None:
            self.FillBetween = FillBetween
        if FillBetweenValue is not None:
            self.FillBetweenValue = FillBetweenValue
        if alpha is not None:
            self.alpha = alpha
        if xticks is not None:
            self.xticks = xticks
        if xticksaxis is not None:
            self.xticksaxis = xticksaxis           
        return self

    def info_1d_twin(self, twinImage=None, twinX=None, twinAxLegends=None, twinYmin=None, 
                     twinYmax=None, twinColorScheme=None, twinLabel=None, twinLineStyle=None,
                     twinAlpha=None, twinPlotScale=None):

        self.twinAxis = True
        self.twinImage = twinImage
        self.twinX = twinX
        self.twinAxLimits = [twinYmin, twinYmax]
        if twinColorScheme is not None:
            self.twinColorScheme = twinColorScheme
        if twinPlotScale is None:
            self.twinPlotScale = self.PlotScale
        else:
            self.twinPlotScale = twinPlotScale
        if twinAxLegends is not None:
            self.twinAxLegends = twinAxLegends
        if twinLineStyle is not None:
            self.LineStyle = twinLineStyle
        if twinAlpha is not None:
            self.twinAlpha = twinAlpha
        if twinLabel is not None:
            self.twinLabel = twinLabel
        return self

    def info_2d_plot(self, ColorScheme=None, Colorbar=None, ColorbarExt=None, 
                     Gamma=None, IsPhase=None, showXhist=None, showYhist=None):
        if ColorScheme is not None:
            self.ColorScheme = ColorScheme
        if Colorbar is not None:
            self.Colorbar = Colorbar
        if ColorbarExt is not None:
            self.ColorbarExt = ColorbarExt
        if Gamma is not None:
            self.Gamma = Gamma
        if IsPhase is not None:
            self.IsPhase = IsPhase
        if showXhist is not None:
            self.showXhist = showXhist
        if showYhist is not None:
            self.showYhist = showYhist
        return self

    def info_scatter(self, ColorScheme=None, LineStyle=None, alpha=None, s=None, 
                     edgeColors=None, monochrome=None, sort_axes=False,
                     showXhist=None, showYhist=None, **kwargs):

        if ColorScheme is not None:
            self.ColorScheme = ColorScheme
        if LineStyle is not None:
            self.LineStyle = LineStyle
        if s is not None:
            self.s = s
        if edgeColors is not None:
            self.edgeColors = edgeColors
        if alpha is not None:
            self.alpha = alpha       
        if monochrome is not None:
            self.monochrome = monochrome 
        # if sort_axes is True:
        #     self.sort_scatter_limits(**kwargs)
        if showXhist is not None:
            self.showXhist = showXhist
        if showYhist is not None:
            self.showYhist = showYhist
        return self
    
    def info_histogram(self, ColorScheme=None, Colorbar=None, ColorbarExt=None,
                       nbinsHistX=None, nbinsHistY=None, wbinsHist=None, rule=None,
                       discardZeros=None, norm=None, bold=None):

        if ColorScheme is not None:
            self.ColorScheme = ColorScheme
        if Colorbar is not None:
            self.Colorbar = Colorbar
        if ColorbarExt is not None:
            self.ColorbarExt = ColorbarExt    
        if nbinsHistX is not None:
            if nbinsHistY is not None:
                self.nbinsHist = [nbinsHistX, nbinsHistY] 
            else:
                self.nbinsHist = [nbinsHistX, None] 
        if wbinsHist is not None:
            self.wbinsHist = wbinsHist
        if rule is not None:
            self.rule = self.rule
        if discardZeros is not None:
            self.discardZeros = discardZeros
        if norm is not None:
            self.norm = norm
        if bold is not None:
            self.bold = bold
        # for more info see: https://en.wikipedia.org/wiki/Histogram#Number_of_bins_and_width
        if self.nbinsHist is None:
            # self.nbinsHist = [None, None]
            n = len(self.x)
            if self.wbinsHist is not None: 
                self.nbinsHist = [int((np.amax(self.x)-np.amin(self.x))/self.wbinsHist),
                                  int((np.amax(self.y)-np.amin(self.y))/self.wbinsHist)]
            elif rule == -1:
                self.nbinsHist = np.min((np.std(self.x), np.std(self.y)))/self.nSigma
                self.nbinsHist = [int((np.amax(self.x)-np.amin(self.x))/self.wbinsHist),
                                  int((np.amax(self.y)-np.amin(self.y))/self.wbinsHist)]
            elif rule == 0:    # sqrt
                self.nbinsHist = [int(np.sqrt(n)), int(np.sqrt(n))]
            elif rule == 1:    # Sturge
                self.nbinsHist = [int(np.log2(n))+1, int(np.log2(n))+1]
            elif rule == 2:    # Rice
                self.nbinsHist = [int(2*n**(1/3)), int(2*n**(1/3))]
            elif rule == 3:    # Doane's
                sigma_g1 = np.sqrt(6*(n-2)/((n+1)*(n+3)))
                self.nbinsHist = [int(1+np.log2(n)*(1+moment(self.x, order=3)/sigma_g1)),
                                  int(1+np.log2(n)*(1+moment(self.y, order=3)/sigma_g1))]
            elif rule == 3:    # Scott's
                wbins = 3.49*np.amin([np.std(self.x), np.std(self.y)])*n**(-1/3)
                self.nbinsHist = [int((np.amax(self.x)-np.amin(self.x))/wbins),
                                  int((np.amax(self.y)-np.amin(self.y))/wbins)]
            elif rule == 4:    # Terrell–Scott
                self.nbinsHist = [(2*n)**(1/3), (2*n)**(1/3)]
            return self
        
    # def info_3d_plot(self):
        # return self

    def info_quiver(self, ColorScheme=None, Colorbar=None, ColorbarExt=None, fld_X=None,
                    fld_Y=None, kk=None):
        if ColorScheme is not None:
            self.ColorScheme = ColorScheme
        if Colorbar is not None:
            self.Colorbar = Colorbar
        if ColorbarExt is not None:
            self.ColorbarExt = ColorbarExt
        if fld_X is not None:
            self.fld_X = fld_X
        if fld_Y is not None:
            self.fld_Y = fld_Y
        if kk is not None:
            self.kk = kk            
        return self
    
    def sort_axes(self):
        if self.x is None:
            if self.image.ndim == 2:
                self.x = np.linspace(-self.image.shape[1] / 2, self.image.shape[1] / 2, self.image.shape[1])
            else:  # 1D array
                self.x = np.linspace(-self.image.shape[0] / 2, self.image.shape[0] / 2, self.image.shape[0])
        if self.y is None:
            if self.image.ndim == 2:
                self.y = np.linspace(-self.image.shape[0] / 2, self.image.shape[0] / 2, self.image.shape[0])

    # def sort_scatter_limits(self, mode='independent'):
    #     """
    #     Calculate the range for plotting based on the given vectors.

    #     mode: str, optional
    #         The mode for calculating the range. Can be 'independent' or 'same'.
    #         'independent' calculates ranges independently for X and Y.
    #         'same' calculates a single range based on the maximum of X and Y.

    #     """
    #     # Calculate the max absolute values for X and Y vectors
    #     max_abs_X = max(self.x)*1.05
    #     max_abs_Y = max(self.y)*1.05

    #     min_abs_X = min(self.x)*1.05
    #     min_abs_Y = min(self.y)*1.05

    #     if mode == 'independent':
    #         # Independent ranges for X and Y
    #         xmin, xmax = -max_abs_X, max_abs_X
    #         ymin, ymax = -max_abs_Y, max_abs_Y
    #     elif mode == 'same':
    #         # Same range for both X and Y based on the max of max_abs_X and max_abs_Y
    #         max_abs = max(max_abs_X, max_abs_Y)
    #         xmin, xmax = -max_abs, max_abs
    #         ymin, ymax = -max_abs, max_abs
    #     else:
    #         raise ValueError("Invalid mode. Use 'independent' or 'same'.")


    def sort_axes_limits(self):
        if self.AxLimits[0] is None:
            self.AxLimits[0] = np.amin(self.x)
        if self.AxLimits[1] is None:
            self.AxLimits[1] = np.amax(self.x)
        try:
            if self.image.ndim == 2:
                if self.AxLimits[2] is None:
                    self.AxLimits[2] = np.amin(self.y)
                if self.AxLimits[3] is None:
                    self.AxLimits[3] = np.amax(self.y)
        except:  # patch for quiver plot
            if self.AxLimits[2] is None:
                self.AxLimits[2] = np.amin(self.y)
            if self.AxLimits[3] is None:
                self.AxLimits[3] = np.amax(self.y)

    def get_roi_coords(self, coords='p', roi='r'):
        """

        :param coords: 'p' for pixel, 'a' for using x- and y-axis,
        :param roi: 'r' rectangular and 'c' for circular,
        :return:
        """
        self.Colorbar = False
        self.sort_axes()
        self.sort_axes_limits()

        (Xi, Xf, Yi, Yf) = self.plot_2d(self, get_roi=True, roi=roi)

        if coords == 'p':
            Xi = int(np.argmin(np.abs(self.x-Xi)))
            Xf = int(np.argmin(np.abs(self.x-Xf)))
            Yi = int(np.argmin(np.abs(self.y-Yi)))
            Yf = int(np.argmin(np.abs(self.y-Yf)))

        return Xi, Xf, Yi, Yf

    # ****************************************************************************
    # ********************** 1D plots
    # ****************************************************************************

    def plot_1d(self, file_name=None, hold=False, enable=True, silent=False, m=6.4 * 1.2, n=4.8):
        """
        Plots a 1D line graph with optional twin axis.

        Parameters:
            file_name (str): The name of the file to save the plot.
            hold (bool): If True, multiple plots can be overlaid on the same axes. If False, each plot replaces the previous one.
            enable (bool): If True, displays the plot. If False, the plot is not displayed.
            silent (bool): If True, suppresses the display of the plot. If False, the plot is displayed.
            m (float): The horizontal size of the figure.
            n (float): The vertical size of the figure.
        """
        def _setTwinAxLimits(twplt):
            if self.twinAxLimits[1] is None:
                if self.twinAxLimits[0] is None:
                    pass
                else:
                    twplt.axis(ymin=self.twinAxLimits[0])
            else:
                if self.twinAxLimits[0] is None:
                    twplt.axis(ymax=self.twinAxLimits[1])
                else:
                    twplt.axis(ymin=self.twinAxLimits[0],ymax=self.twinAxLimits[1])
            return twplt

        self._plt_settings(self.FontsSize, self.LaTex, _hold=hold, _silent=silent, m=m, n=n)

        if self.grid:
            if self.twinAxis:    
                plt.grid(which='major', linestyle='--', linewidth=0.5, color='dimgrey', axis='x')
                plt.grid(which='minor', linestyle='--', linewidth=0.5, color='lightgrey', axis='x')
            else:
                plt.grid(which='major', linestyle='--', linewidth=0.5, color='dimgrey')
                plt.grid(which='minor', linestyle='--', linewidth=0.5, color='lightgrey')

        if self.AxLegends[0] is not None:
            plt.title(self.AxLegends[0])
        if self.AxLegends[1] is not None:
            plt.xlabel(self.AxLegends[1])
        if self.AxLegends[2] is not None:
            plt.ylabel(self.AxLegends[2])

        alpha = 1
        fb_alpha = 1

        if self.alpha != 1 and self.FillBetween:
            alpha = 1
            fb_alpha = self.alpha
        else:
            alpha = self.alpha
            fb_alpha = self.alpha

        plt.tick_params(direction='in', which='both')
        plt.tick_params(axis='x', pad=8)

        if self.AxLimits[1] is None:
            if self.AxLimits[0] is None:
                pass
            else:
                plt.xlim(xmin=self.AxLimits[0])
        else:
            if self.AxLimits[0] is None:
                plt.xlim(xmax=self.AxLimits[1])
            else:
                plt.xlim((self.AxLimits[0], self.AxLimits[1]))

        if self.AxLimits[3] is None:
            if self.AxLimits[2] is None:
                pass
            else:
                plt.ylim(ymin=self.AxLimits[2])
        else:
            if self.AxLimits[2] is None:
                plt.ylim(ymax=self.AxLimits[3])
            else:
                plt.ylim((self.AxLimits[2], self.AxLimits[3]))     

        if self.Label is not None:
            if self.PlotScale == 0:
                im = plt.plot(self.x, self.image, self.LineStyle, label=self.Label)
                plt.setp(im, color=self._color_palette_1d(self.ColorScheme), alpha=alpha)
            elif self.PlotScale == 1:
                im = plt.semilogy(self.x, self.image, self.LineStyle, label=self.Label)
                plt.setp(im, color=self._color_palette_1d(self.ColorScheme), alpha=alpha)
            elif self.PlotScale == 2:
                im = plt.semilogx(self.x, self.image, self.LineStyle, label=self.Label)
                plt.setp(im, color=self._color_palette_1d(self.ColorScheme), alpha=alpha)
            elif self.PlotScale == 3:
                im = plt.loglog(self.x, self.image, self.LineStyle, label=self.Label)
                plt.setp(im, color=self._color_palette_1d(self.ColorScheme), alpha=alpha)
            plt.legend(loc=self.LabelPos)
        else:   # twin axis
            if self.PlotScale == 0:
                im = plt.plot(self.x, self.image, self.LineStyle)
                plt.setp(im, color=self._color_palette_1d(self.ColorScheme), alpha=alpha)
            elif self.PlotScale == 1:
                im = plt.semilogy(self.x, self.image, self.LineStyle)
                plt.setp(im, color=self._color_palette_1d(self.ColorScheme), alpha=alpha)
            elif self.PlotScale == 2:
                im = plt.semilogx(self.x, self.image, self.LineStyle)
                plt.setp(im, color=self._color_palette_1d(self.ColorScheme), alpha=alpha)
            elif self.PlotScale == 3:
                im = plt.loglog(self.x, self.image, self.LineStyle)
                plt.setp(im, color=self._color_palette_1d(self.ColorScheme), alpha=alpha)
            if self.twinAxis:        
                plt.tick_params(axis='y', labelcolor=self._color_palette_1d(self.ColorScheme))
                ax2 = plt.twinx()
                ax2 = _setTwinAxLimits(ax2)
                twinPlotScale = None
                if self.twinPlotScale is None:
                    twinPlotScale = self.PlotScale
                else:
                    twinPlotScale = self.twinPlotScale

                if twinPlotScale == 0:
                    ax2.plot(self.twinX, self.twinImage, self.twinLineStyle, 
                             color=self._color_palette_1d(self.twinColorScheme), 
                             alpha=self.twinAlpha)
                elif twinPlotScale == 1:
                    ax2.semilogy(self.twinX, self.twinImage, self.twinLineStyle, 
                                 color=self._color_palette_1d(self.twinColorScheme), 
                                 alpha=self.twinAlpha)
                elif twinPlotScale == 2:
                    ax2.semilogx(self.twinX, self.twinImage, self.twinLineStyle,
                                 color=self._color_palette_1d(self.twinColorScheme), 
                                 alpha=self.twinAlpha)
                elif twinPlotScale == 3:
                    ax2.loglog(self.twinX, self.twinImage, self.twinLineStyle,
                               color=self._color_palette_1d(self.twinColorScheme),
                               alpha=self.twinAlpha)

                ax2.set_ylabel(self.twinAxLegends) #, color=self._color_palette_1d(self.twinColorScheme))
                ax2.tick_params(axis='y', labelcolor=self._color_palette_1d(self.twinColorScheme))
                ax2.tick_params(direction='in', which='both')

        if self.FillBetween is True and self.twinAxis is False:
            plt.fill_between(self.x, self.FillBetweenValue, self.image, color=self._color_palette_1d(self.ColorScheme),
                             alpha=fb_alpha)

        if self.PlotScale != 0 or hold is False:
            plt.locator_params(tight=True)  # , nbins=self.nbins)

        if self.xticks is not None:
            plt.xticks(self.xticksaxis, self.xticks, rotation=0, horizontalalignment='center')

        self._save_and_show(file_name, silent, enable)

    # ****************************************************************************
    # ********************** 2D plots
    # ****************************************************************************

    def plot_2d(self, file_name=None, roi='r', get_roi=False, enable=True, silent=False, m=6.4, n=4.8):
        """
        Plots a 2D image with optional region of interest (ROI) selection.

        Parameters:
            file_name (str): The name of the file to save the plot.
            roi (str): Specifies the type of region of interest (ROI) selection. Use 'r' for rectangular ROI or 'e' for elliptical ROI.
            get_roi (bool): If True, enables interactive selection of the ROI.
            enable (bool): If True, displays the plot. If False, the plot is not displayed.
            silent (bool): If True, suppresses the display of the plot. If False, the plot is displayed.
            m (float): The horizontal size of the figure.
            n (float): The vertical size of the figure.

        Returns:
            extents (tuple): A tuple containing the extents of the selected region of interest (ROI) if get_roi is True. 
                Otherwise, None is returned.
        """
    
        self._plt_settings(self.FontsSize, self.LaTex, _hold=False, _silent=silent, m=m, n=n)
    
        if self.AxLegends[0] is not None:
            plt.title(self.AxLegends[0])
        if self.AxLegends[1] is not None:
            plt.xlabel(self.AxLegends[1])
        if self.AxLegends[2] is not None:
            plt.ylabel(self.AxLegends[2])

        if self.ColorbarExt is None:
            self.ColorbarExt = 'neither'
    
        plt.tick_params(direction='in', which='both')
        plt.tick_params(axis='x', pad=8)
    
        if self.AxLimits[1] is None:
            if self.AxLimits[0] is None:
                pass
            else:
                plt.xlim(xmin=self.AxLimits[0])
        else:
            if self.AxLimits[0] is None:
                plt.xlim(xmax=self.AxLimits[1])
            else:
                plt.xlim((self.AxLimits[0], self.AxLimits[1]))
    
        if self.AxLimits[3] is None:
            if self.AxLimits[2] is None:
                pass
            else:
                plt.ylim(ymin=self.AxLimits[2])
        else:
            if self.AxLimits[2] is None:
                plt.ylim(ymax=self.AxLimits[3])
            else:
                plt.ylim((self.AxLimits[2], self.AxLimits[3]))
    
        if self.PlotScale == 0:
            im = plt.imshow(self.image, cmap=self._color_palette_2d(self.ColorScheme),
                            extent=[self.x[0], self.x[-1], self.y[0], self.y[-1]], origin='lower',
                            vmax=self.MinMax[1], vmin=self.MinMax[0])
        elif self.PlotScale == 1:
            im = plt.imshow(self.image, cmap=self._color_palette_2d(self.ColorScheme),
                            extent=[self.x[0], self.x[-1], self.y[0], self.y[-1]], origin='lower',
                            norm=LogNorm(vmax=self.MinMax[1], vmin=self.MinMax[0]))
        else:
            im = plt.imshow(self.image, cmap=self._color_palette_2d(self.ColorScheme),
                            extent=[self.x[0], self.x[-1], self.y[0], self.y[-1]], origin='lower',
                            norm=PowerNorm(self.Gamma, vmax=self.MinMax[1], vmin=self.MinMax[0]))
    
        def format_func(x, pos):
            x = '%.2f' % x
            pad = '' if x.startswith('-') else ' '
            return '{}{}'.format(pad, x)
    
        if self.Colorbar:
            if self.PlotScale == 1:
                if self.AspectRatio:
                    im_ratio = (self.y[-1] - self.y[0]) / (self.x[-1] - self.x[0])
                else:
                    im_ratio = 1
                cb = plt.colorbar(im, fraction=0.046 * im_ratio, pad=0.04, extend=self.ColorbarExt, format='%.0e')
    
            else:
    
                if self.AspectRatio:
                    im_ratio = (self.y[-1] - self.y[0]) / (self.x[-1] - self.x[0])
                else:
                    im_ratio = 1
                if self.IsPhase:
                    cb = plt.colorbar(im, fraction=0.046 * im_ratio, pad=0.04, extend=self.ColorbarExt,
                                      spacing='uniform', ticks=[-2*np.pi, -np.pi, 0, np.pi, 2*np.pi])
                    cb.ax.set_yticklabels(['$-2\pi$', '$-\pi$', '0', '$\pi$', '$2\pi$'])
                else:
                    cb = plt.colorbar(im, fraction=0.046 * im_ratio, pad=0.04, extend=self.ColorbarExt,
                                      spacing='uniform')
    
                    # if self.PlotScale != 2:
                    #     def format_func(x, pos):
                    #         x = '%.2f' % x
                    #         pad = '' if x.startswith('-') else ' '
                    #         return '{}{}'.format(pad, x)
    
                    tick_locator = ticker.MaxNLocator(nbins=4)
                    cb.locator = tick_locator
                    cb.update_ticks()
                    if self.PlotScale != 2:
                        cb.ax.yaxis.major.formatter = ticker.FuncFormatter(format_func)
    
        plt.locator_params(tight=True, nbins=self.nbins)
        # plt.xscale("log")
        if self.grid:
            plt.grid(linestyle='--', linewidth=0.4, color='dimgrey')
    
        if self.AspectRatio:
            im._axes._axes.set_aspect('equal')
        else:
            im._axes._axes.set_aspect('auto')

        if get_roi is True:
            current_ax = im._axes  # make new plotting ranges

            def line_select_callback(eclick, erelease):
                'eclick and erelease are the press and release events'
                x1, y1 = eclick.xdata, eclick.ydata
                x2, y2 = erelease.xdata, erelease.ydata
                print("(X1, Y1, X2, Y2) = (%3.2f, %3.2f, %3.2f, %3.2f)" % (x1, y1, x2, y2))

            if roi == 'r':
                RS = RectangleSelector(current_ax, line_select_callback, drawtype='box', useblit=True, button=[1, 3],
                                       minspanx=5, minspany=5, spancoords='pixels', interactive=True)
            else:
                RS = EllipseSelector(current_ax, line_select_callback, drawtype='box', useblit=True, button=[1, 3],
                                       minspanx=5, minspany=5, spancoords='pixels', interactive=True)

            plt.show()
    
            return RS.extents

        self._save_and_show(file_name, silent, enable)


    def plot_2d_cuts(self, file_name=None, enable=True, silent=False, m=6.4, n=4.8, x=None, y=None):
        """
        Plot 2D image and its vertical and horizontal cuts.

        Parameters:
            file_name (str, optional): File path to save the plot. Defaults to None.
            enable (bool, optional): Whether to display the plot. Defaults to True.
            silent (bool, optional): Whether to suppress plot output. Defaults to False.
            m (float, optional): Width of the figure in inches. Defaults to 6.4.
            n (float, optional): Height of the figure in inches. Defaults to 4.8.
            x (array-like, optional): X coordinates for the horizontal cut. If not provided, cuts are made around the center of the image.
            y (array-like, optional): Y coordinates for the vertical cut. If not provided, cuts are made around the center of the image.
        """
        def get_slice(_image, _x, _y, _coords_x, _coords_y):
            """
            Extracts a vertical cut if _coords_x is ':', or a horizontal cut if _coords_y is ':'.

            Parameters:
                _image (array-like): The 2D image from which to extract the cut.
                _x (array-like): The x-coordinates associated with the image.
                _y (array-like): The y-coordinates associated with the image.
                _coords_x (str): Specifies the coordinates for the horizontal cut. Use ':' to indicate that the cut should be made along the y-axis.
                _coords_y (str): Specifies the coordinates for the vertical cut. Use ':' to indicate that the cut should be made along the x-axis.

            Returns:
                tuple: A tuple containing the extracted cut and the corresponding axis values.
            """
            image_for_fit = np.nan_to_num(_image, True)
    
            # f = interp2d(_x, _y, image_for_fit, kind='linear')
            if _coords_x == ':':
                # cut = f(_x, _coords_y)
                cut = image_for_fit[int(image_for_fit.shape[0] / 2), :]
                axis = _x
            if _coords_y == ':':
                # cut = f(_coords_x, _y)
                cut = image_for_fit[:, int(image_for_fit.shape[1] / 2)]
                axis = _y
            return np.asarray(cut), axis
    
        if x is None:
            x = (self.x[-1]+self.x[1])/2
        if y is None:
            y = (self.y[-1]+self.y[1])/2
    
        edges = [0, 0, 0, 0]
    
        if self.AxLimits[1] is None:
            if self.AxLimits[0] is None:
                edges[0] = self.x[0]
                edges[1] = self.x[-1]
            else:
                edges[0] = self.AxLimits[0]
        else:
            if self.AxLimits[0] is None:
                edges[1] = self.AxLimits[1]
            else:
                edges[0] = self.AxLimits[0]
                edges[1] = self.AxLimits[1]
    
        if self.AxLimits[3] is None:
            if self.AxLimits[2] is None:
                edges[2] = self.y[0]
                edges[3] = self.y[-1]
            else:
                edges[2] = self.AxLimits[2]
        else:
            if self.AxLimits[2] is None:
                edges[3] = self.AxLimits[3]
            else:
                edges[2] = self.AxLimits[2]
                edges[3] = self.AxLimits[3]
    
        if self.AspectRatio is True:
            dx = edges[1] - edges[0]
            dy = edges[3] - edges[2]
        else:
            dx = m
            dy = n
    
        left, bottom = 0.2, 0.10
        spacing = 0.02
        spacing_x = spacing
        spacing_y = spacing
        k = 0.25
        kx = k
        ky = k
    
        if dx >= dy:
            width = 0.50
            height = width * dy / dx
            spacing_y = spacing * dy / dx
            ky = k * dy / dx
        else:
            height = 0.50
            width = height * dx / dy
            spacing_x = spacing * dx / dy
            kx = k * dx / dy
    
        rect_image = [left, bottom, width, height]
        # rect_histx = [left, bottom + height + spacing_x + 0.02, width, kx]
        # rect_histy = [left + width + spacing_y + 0.02, bottom, ky, height]
        rect_histx = [left, bottom + height + spacing_x + 0.02, width, kx*.9]
        rect_histy = [left + width + spacing_x + 0.02, bottom, kx*.9, height]
        
        if self.AspectRatio is True:
            m = 6.4
            n = 6.4
    
        self._plt_settings(self.FontsSize, self.LaTex, _hold=False, _silent=silent, m=m, n=n)
    
        ax_image = plt.axes(rect_image)
        ax_image.tick_params(top=False, right=False)
    
        plt.xlabel(self.AxLegends[1])
        plt.ylabel(self.AxLegends[2])
    
        ax_histx = plt.axes(rect_histx, sharex=ax_image)
        ax_histx.tick_params(direction='in', which='both', labelbottom=False, top=True, right=True, colors='black')
        ax_histx.spines['bottom'].set_color('black')
        ax_histx.spines['top'].set_color('black')
        ax_histx.spines['right'].set_color('black')
        ax_histx.spines['left'].set_color('black')
    
        if self.AxLegends[0] is not None:
            plt.title(self.AxLegends[0])
    
        ax_histy = plt.axes(rect_histy, sharey=ax_image)
        ax_histy.tick_params(direction='in', which='both', labelleft=False, top=True, right=True, colors='black')
        ax_histy.spines['bottom'].set_color('black')
        ax_histy.spines['top'].set_color('black')
        ax_histy.spines['right'].set_color('black')
        ax_histy.spines['left'].set_color('black')
    
        if self.PlotScale == 0:
            ax_image.imshow(self.image, cmap=self._color_palette_2d(self.ColorScheme),
                            extent=[self.x[0], self.x[-1], self.y[0], self.y[-1]], origin='lower',
                            vmax=self.MinMax[1], vmin=self.MinMax[0])
    
        if self.PlotScale == 1:
            ax_image.imshow(self.image, cmap=self._color_palette_2d(self.ColorScheme),
                            extent=[self.x[0], self.x[-1], self.y[0], self.y[-1]], origin='lower',
                            norm=LogNorm(vmax=self.MinMax[1], vmin=self.MinMax[0]))
    
        ax_image.set_xlim((edges[0], edges[1]))
        ax_image.set_ylim((edges[2], edges[3]))
    
        if self.AspectRatio:
            ax_image._axes._axes.set_aspect('equal')
        else:
            ax_image._axes._axes.set_aspect('auto')
    
        ax_image.locator_params(tight=True, nbins=3)
    
        # vertical cut
        cut_x, axis_x = get_slice(self.image, self.x, self.y, _coords_x=':', _coords_y=y)
    
        if self.PlotScale == 0:
            ax_histx.plot(self.x, cut_x, color=self._color_palette_1d(0))

        else:
            ax_histx.semilogy(self.x, cut_x, color=self._color_palette_1d(0))

        ax_histx.set_xlim((edges[0], edges[1]))
        ax_histx.set_ylim((self.MinMax[0], self.MinMax[1]))
        if self.PlotScale == 0:
            ax_histx.locator_params(tight=True, nbins=3)
    
        if self.grid:
            ax_histx.grid(which='major', linestyle='--', linewidth=0.5, color='dimgrey')
            ax_histx.grid(which='minor', linestyle='--', linewidth=0.5, color='lightgrey')
    
        # horizontal cut
        cut_y, axis_y = get_slice(self.image, self.x, self.y, _coords_x=x, _coords_y=':')
    
        if self.PlotScale == 0:
            ax_histy.plot(cut_y, self.y, color=self._color_palette_1d(0))

        else:
            ax_histy.semilogx(cut_y, axis_y, color=self._color_palette_1d(0))

        ax_histy.set_ylim((edges[2], edges[3]))
        ax_histy.set_xlim((self.MinMax[0], self.MinMax[1]))
        if self.PlotScale == 0:
            ax_histy.locator_params(tight=True, nbins=3)
    
        if self.grid:
            ax_histy.grid(which='major', linestyle='--', linewidth=0.5, color='dimgrey')
            ax_histy.grid(which='minor', linestyle='--', linewidth=0.5, color='lightgrey')

        self._save_and_show(file_name, silent, enable)

    # ****************************************************************************
    # ********************** 3D plots
    # ****************************************************************************

    # def plot_3d(self):
    #     pass

    # ****************************************************************************
    # ********************** Wavefront coefficients
    # ****************************************************************************

    def plot_wft_coeffs(self, file_name=None, enable=True, silent=False, pol=0, noll=False, m=10, n=3):
        """
        Plot wavefront coefficients.

        Parameters:
            file_name (str, optional): File path to save the plot. Defaults to None.
            enable (bool, optional): Whether to display the plot. Defaults to True.
            silent (bool, optional): Whether to suppress plot output. Defaults to False.
            pol (int, optional): Type of wavefront polynomial. Defaults to 0.
            noll (bool, optional): Whether to use Noll indices. Defaults to False.
            m (float, optional): Width of the figure in inches. Defaults to 10.
            n (float, optional): Height of the figure in inches. Defaults to 3.

        Note:
            The `pol` parameter specifies the type of wavefront polynomial:
            - 0: Circular Zernike Polynomials
            - 1: Rectangular Zernike Polynomials
            - 2: Legendre Polynomials
        """
        self._plt_settings(self.FontsSize, self.LaTex, _hold=False, _silent=silent, m=m, n=n)

        if self.AxLegends[0] is not None:
            plt.title(self.AxLegends[0])
        if self.AxLegends[1] is not None:
            plt.xlabel(self.AxLegends[1])
        if self.AxLegends[2] is not None:
            plt.ylabel(self.AxLegends[2])

        anti_symmetric = deepcopy(self.image)
        symmetric = deepcopy(self.image) * np.nan

        if pol == 0:  # Circular Zernike Polynomials
            N = 37
            if noll:
                Zern_x_ticks = ['Z$_{1}$',
                                '$\cdot$', '$\cdot$',
                                '$\cdot$', 'Z$_{5}$', '$\cdot$',
                                '$\cdot$', '$\cdot$', '$\cdot$', 'Z$_{10}$',
                                '$\cdot$', '$\cdot$', '$\cdot$', '$\cdot$', 'Z$_{15}$',
                                '$\cdot$', '$\cdot$', '$\cdot$', '$\cdot$', 'Z$_{20}$', '$\cdot$',
                                '$\cdot$', '$\cdot$', '$\cdot$', 'Z$_{25}$', '$\cdot$', '$\cdot$', '$\cdot$',
                                '$\cdot$', 'Z$_{30}$', '$\cdot$', '$\cdot$', '$\cdot$', '$\cdot$', 'Z$_{35}$', '$\cdot$',
                                '$\cdot$'
                                ]
            else:
                Zern_x_ticks = ['Z$_{0}^{0}$',
                                'Z$_{1}^{1}$', 'Z$_{1}^{-1}$',
                                'Z$_{2}^{0}$', 'Z$_{2}^{-2}$', 'Z$_{2}^{2}$',
                                'Z$_{3}^{-1}$', 'Z$_{3}^{1}$', 'Z$_{3}^{-3}$', 'Z$_{3}^{3}$',
                                'Z$_{4}^{0}$', 'Z$_{4}^{2}$', 'Z$_{4}^{-2}$', 'Z$_{4}^{4}$', 'Z$_{4}^{-4}$',
                                'Z$_{5}^{1}$', 'Z$_{5}^{-1}$', 'Z$_{5}^{3}$', 'Z$_{5}^{-3}$', 'Z$_{5}^{5}$', 'Z$_{5}^{-5}$',
                                'Z$_{6}^{0}$', 'Z$_{6}^{-2}$', 'Z$_{6}^{2}$', 'Z$_{6}^{-4}$', 'Z$_{6}^{4}$', 'Z$_{6}^{-6}$', 'Z$_{6}^{6}$',
                                'Z$_{7}^{-1}$', 'Z$_{7}^{1}$', 'Z$_{7}^{-3}$', 'Z$_{7}^{3}$', 'Z$_{7}^{-5}$', 'Z$_{7}^{5}$', 'Z$_{7}^{-7}$', 'Z$_{7}^{7}$',
                                'Z$_{8}^{0}$',
                                ]

            symmetric[0] = anti_symmetric[0]  # bias
            symmetric[3] = anti_symmetric[3]  # defocus
            symmetric[10] = anti_symmetric[10]  # primary spherical
            symmetric[21] = anti_symmetric[21]  # secondary spherical
            symmetric[36] = anti_symmetric[36]  # tertiary spherical

        elif pol == 1:  # Rectangular Zernike Polynomials
            N = 15
            Zern_x_ticks = ['Rz$_{1}$', '$\cdot$', '$\cdot$', '$\cdot$', 'Rz$_{5}$',
                            '$\cdot$', '$\cdot$', '$\cdot$', '$\cdot$', 'Rz$_{10}$',
                            '$\cdot$', '$\cdot$', '$\cdot$', '$\cdot$', 'Rz$_{15}$',
                            ]

        elif pol == 2:  # Legendre Polynomials
            N = 44
            Zern_x_ticks = ['L$_{1}$', '$\cdot$', '$\cdot$', '$\cdot$', 'L$_{5}$',
                            '$\cdot$', '$\cdot$', '$\cdot$', '$\cdot$', 'L$_{10}$',
                            '$\cdot$', '$\cdot$', '$\cdot$', '$\cdot$', 'L$_{15}$',
                            '$\cdot$', '$\cdot$', '$\cdot$', '$\cdot$', 'L$_{20}$',
                            '$\cdot$', '$\cdot$', '$\cdot$', '$\cdot$', 'L$_{25}$',
                            '$\cdot$', '$\cdot$', '$\cdot$', '$\cdot$', 'L$_{30}$',
                            '$\cdot$', '$\cdot$', '$\cdot$', '$\cdot$', 'L$_{35}$',
                            '$\cdot$', '$\cdot$', '$\cdot$', '$\cdot$', 'L$_{40}$',
                            '$\cdot$', '$\cdot$', '$\cdot$', '$\cdot$',

                            ]
            symmetric[0] = anti_symmetric[0]  # piston
            symmetric[1] = anti_symmetric[1]  # x-tilt
            symmetric[2] = anti_symmetric[2]  # y-tilt
            symmetric[3] = anti_symmetric[3]  # x-defocus
            symmetric[5] = anti_symmetric[5]  # y-defocus
            symmetric[6] = anti_symmetric[6]  # primary x-coma
            symmetric[9] = anti_symmetric[9]  # primary y-coma
            symmetric[10] = anti_symmetric[10]  # primary x-spherical
            symmetric[14] = anti_symmetric[14]  # primary y-spherical
            symmetric[15] = anti_symmetric[15]  # secondary x-coma
            symmetric[20] = anti_symmetric[20]  # secondary y-coma
            symmetric[21] = anti_symmetric[21]  # secondary x-spherical
            symmetric[27] = anti_symmetric[27]  # secondary y-spherical
            symmetric[28] = anti_symmetric[28]  # tertiary x-coma
            symmetric[35] = anti_symmetric[35]  # tertiary y-coma
            symmetric[36] = anti_symmetric[36]  # tertiary x-spherical
            symmetric[43] = anti_symmetric[43]  # tertiary y-spherical

        ind = np.arange(N)  # the x locations for the groups

        # TODO: separate p1 in p1_a and p1_b to enable color code for polynomial pairs

        p1 = plt.bar(ind, anti_symmetric)
        p2 = plt.bar(ind, symmetric)

        # plt.xticks(ind, Zern_x_ticks, rotation=60, horizontalalignment= 'right')
        plt.xticks(ind, Zern_x_ticks, rotation=0, horizontalalignment='center')

        plt.setp(p1, color=self._color_palette_1d(4))
        plt.setp(p2, color=self._color_palette_1d(5))

        if self.label is not None:
            plt.legend((p1[0], p2[0]), ('Non-symmetric', 'Radially-symmetric'))

        plt.ylim((self.MinMax[0], self.MinMax[1]))
        plt.locator_params(tight=True)
        plt.locator_params(tight=True, nbins=self.nbins, axis='y')

        if self.grid:
            plt.grid(linestyle='--', linewidth=0.5, color='black')
        # else:
        plt.axhline(y=0, linewidth=0.5, color='black', alpha=0.75)

        self._save_and_show(file_name, silent, enable)

    # ****************************************************************************
    # ********************** Other 2D plots
    # ****************************************************************************

    def plot_scatter(self, file_name=None, hold=False, enable=True, silent=False, m=6.4, n=4.8):
        """
        Plot a 2D scatter plot.

        Parameters:
            file_name (str, optional): File path to save the plot. Defaults to None.
            hold (bool, optional): Whether to hold the plot. Defaults to False.
            enable (bool, optional): Whether to display the plot. Defaults to True.
            silent (bool, optional): Whether to suppress plot output. Defaults to False.
            m (float, optional): Width of the figure in inches. Defaults to 6.4.
            n (float, optional): Height of the figure in inches. Defaults to 4.8.
        """

        self._plt_settings(self.FontsSize, self.LaTex, _hold=hold, _silent=silent, m=m, n=n)

        if self.AxLegends[0] is not None:
            plt.title(self.AxLegends[0])
        if self.AxLegends[1] is not None:
            plt.xlabel(self.AxLegends[1])
        if self.AxLegends[2] is not None:
            plt.ylabel(self.AxLegends[2])

        plt.tick_params(direction='in', which='both')
        plt.tick_params(axis='x', pad=8)

        if self.AxLimits[1] is None:
            if self.AxLimits[0] is None:
                pass
            else:
                plt.xlim(xmin=self.AxLimits[0])
        else:
            if self.AxLimits[0] is None:
                plt.xlim(xmax=self.AxLimits[1])
            else:
                plt.xlim((self.AxLimits[0], self.AxLimits[1]))

        if self.AxLimits[3] is None:
            if self.AxLimits[2] is None:
                pass
            else:
                plt.ylim(ymin=self.AxLimits[2])
        else:
            if self.AxLimits[2] is None:
                plt.ylim(ymax=self.AxLimits[3])
            else:
                plt.ylim((self.AxLimits[2], self.AxLimits[3]))

        # RC20240424 - small bug for LineStyle = '+'
        if '-' in self.LineStyle:
            self.LineStyle = '.'
            self.edgeColors = 'none'

        if self.monochrome:
            im = plt.scatter(self.x, self.y, color=self._color_palette_1d(self.ColorScheme),
                alpha=self.alpha, edgecolors=self.edgeColors, s=self.s, marker=self.LineStyle) 
        else:
            xy = np.vstack([self.x, self.y])
            z = gaussian_kde(xy)(xy)
            z = z / z.max()

            cmap = self._color_palette_2d(self.ColorScheme)
            clr = cmap(z)

            im = plt.scatter(self.x, self.y, color=clr, alpha=self.alpha, 
                             edgecolors=self.edgeColors, s=self.s, marker=self.LineStyle) 
            # if self.Colorbar:
            #     def format_func(x, pos):
            #         x = '%.2f' % x
            #         pad = '' if x.startswith('-') else ' '
            #         return '{}{}'.format(pad, x)
                
            #     if self.AspectRatio:
            #         im_ratio = (self.y[-1] - self.y[0]) / (self.x[-1] - self.x[0])
            #     else:
            #         im_ratio = 1
            #     cb = plt.colorbar(im, fraction=0.046 * im_ratio, pad=0.04, extend=self.ColorbarExt,
            #                                         spacing='uniform')
            #     tick_locator = ticker.MaxNLocator(nbins=4)
            #     cb.locator = tick_locator
            #     cb.update_ticks()
            #     if self.PlotScale != 2:
            #         cb.ax.yaxis.major.formatter = ticker.FuncFormatter(format_func)

        plt.locator_params(tight=True, nbins=self.nbins)

        if self.grid:
            plt.grid(linestyle='--', linewidth=0.4, color='dimgrey')

        if self.AspectRatio:
            im._axes._axes.set_aspect('equal')
        else:
            im._axes._axes.set_aspect('auto')

        self._save_and_show(file_name, silent, enable)


    def plot_scatter_hist(self, file_name=None, enable=True, silent=False, m=6.4, n=4.8):
        """
        Plot a 2D scatter plot with histograms.

        Parameters:
            file_name (str, optional): File path to save the plot. Defaults to None.
            enable (bool, optional): Whether to display the plot. Defaults to True.
            silent (bool, optional): Whether to suppress plot output. Defaults to False.
            m (float, optional): Width of the figure in inches. Defaults to 6.4.
            n (float, optional): Height of the figure in inches. Defaults to 4.8.
        """

        edges = [0, 0, 0, 0]
    
        if self.AxLimits[1] is None:
            if self.AxLimits[0] is None:
                edges[0] = np.amin(self.x)
                edges[1] = np.amax(self.x)
            else:
                edges[0] = self.AxLimits[0]
        else:
            if self.AxLimits[0] is None:
                edges[1] = self.AxLimits[1]
            else:
                edges[0] = self.AxLimits[0]
                edges[1] = self.AxLimits[1]
    
        if self.AxLimits[3] is None:
            if self.AxLimits[2] is None:
                edges[2] = np.amin(self.y)
                edges[3] = np.amax(self.y)
            else:
                edges[2] = self.AxLimits[2]
        else:
            if self.AxLimits[2] is None:
                edges[3] = self.AxLimits[3]
            else:
                edges[2] = self.AxLimits[2]
                edges[3] = self.AxLimits[3]
    
        if self.AspectRatio is True:
            dx = edges[1] - edges[0]
            dy = edges[3] - edges[2]
        else:
            dx = m
            dy = n
    
        left, bottom = 0.2, 0.10
        spacing = 0.02
        spacing_x = spacing
        spacing_y = spacing
        k = 0.25
        kx = k
        ky = k
    
        if dx >= dy:
            width = 0.50
            height = width * dy / dx
            spacing_y = spacing * dy / dx
            ky = k * dy / dx
        else:
            height = 0.50
            width = height * dx / dy
            spacing_x = spacing * dx / dy
            kx = k * dx / dy
    
        rect_image = [left, bottom, width, height]
        # rect_histx = [left, bottom + height + spacing_x + 0.02, width, kx]
        # rect_histy = [left + width + spacing_y + 0.02, bottom, ky, height]
        rect_histx = [left, bottom + height + spacing_x + 0.02, width, kx*.9]
        rect_histy = [left + width + spacing_x + 0.02, bottom, kx*.9, height]
        
        if self.AspectRatio is True:
            m = 6.4
            n = 6.4
    
        self._plt_settings(self.FontsSize, self.LaTex, _hold=False, _silent=silent, m=m, n=n)

        ax_image = plt.axes(rect_image)
        ax_image.tick_params(top=False, right=False)
    
        plt.xlabel(self.AxLegends[1])
        plt.ylabel(self.AxLegends[2])
        if self.showXhist:
            ax_histx = plt.axes(rect_histx, sharex=ax_image)
            ax_histx.tick_params(direction='in', which='both', labelbottom=False, top=True, right=True, colors='black')
            ax_histx.spines['bottom'].set_color('black')
            ax_histx.spines['top'].set_color('black')
            ax_histx.spines['right'].set_color('black')
            ax_histx.spines['left'].set_color('black')
    
        if self.AxLegends[0] is not None:
            plt.title(self.AxLegends[0])
            
        if self.showYhist:
            ax_histy = plt.axes(rect_histy, sharey=ax_image)
            ax_histy.tick_params(direction='in', which='both', labelleft=False, top=True, right=True, colors='black')
            ax_histy.spines['bottom'].set_color('black')
            ax_histy.spines['top'].set_color('black')
            ax_histy.spines['right'].set_color('black')
            ax_histy.spines['left'].set_color('black')

        if self.AxLimits[1] is None:
            if self.AxLimits[0] is None:
                pass
            else:
                plt.xlim(xmin=self.AxLimits[0])
        else:
            if self.AxLimits[0] is None:
                plt.xlim(xmax=self.AxLimits[1])
            else:
                plt.xlim((self.AxLimits[0], self.AxLimits[1]))

        if self.AxLimits[3] is None:
            if self.AxLimits[2] is None:
                pass
            else:
                plt.ylim(ymin=self.AxLimits[2])
        else:
            if self.AxLimits[2] is None:
                plt.ylim(ymax=self.AxLimits[3])
            else:
                plt.ylim((self.AxLimits[2], self.AxLimits[3]))

        if '-' in self.LineStyle:
            self.LineStyle = '.'
            self.edgeColors = 'none'
        if '+' in self.LineStyle:
            self.LineStyle = 'P'
            self.edgeColors = 'none'

        if self.monochrome:
            ax_image.scatter(self.x, self.y, color=self._color_palette_1d(self.ColorScheme),
                alpha=self.alpha, edgecolors=self.edgeColors, s=self.s, marker=self.LineStyle) 
        else:
            xy = np.vstack([self.x, self.y])
            z = gaussian_kde(xy)(xy)
            z = z / z.max()

            cmap = self._color_palette_2d(self.ColorScheme)
            clr = cmap(z)

            ax_image.scatter(self.x, self.y, color=clr, alpha=self.alpha, 
                             edgecolors=self.edgeColors, s=self.s, marker=self.LineStyle) 
            
        ax_image.set_xlim((edges[0], edges[1]))
        ax_image.set_ylim((edges[2], edges[3]))
    
        if self.AspectRatio:
            ax_image._axes._axes.set_aspect('equal')
        else:
            ax_image._axes._axes.set_aspect('auto')
    
        ax_image.locator_params(tight=True, nbins=3)

        if self.grid:
            ax_image.grid(linestyle='--', linewidth=0.4, color='dimgrey')

        # histograms
        wbinsx = np.std(self.x)/self.nSigma
        wbinsy = np.std(self.y)/self.nSigma

        dbinsx = int((np.amax(self.x)-np.amin(self.x))/wbinsx)
        dbinsy = int((np.amax(self.y)-np.amin(self.y))/wbinsy)

        hx, bx = np.histogram(self.x, dbinsx)
        hy, by = np.histogram(self.y, dbinsy)
        hxM = np.amax(hx)
        hxm = np.amin(hx)
        hyM = np.amax(hy)
        hym = np.amin(hy)

        if self.MinMax[1] is None:
            self.MinMax[1] = np.amax((hxM, hyM))*1.075
        if self.MinMax[0] is None:
            if np.amin((hxm, hym))>0.1*np.amax((hxM, hyM)):
                self.MinMax[0] = np.amin((hxm, hym))*0.05
            else:
                self.MinMax[0] = -np.amax((hxM, hyM))*0.05

        if self.showXhist:
            ax_histx.hist(self.x, bins=dbinsx, color=self._color_palette_1d(0), linewidth=1, 
                        edgecolor=self._color_palette_1d(0), histtype="step", alpha=1)
            ax_histx.set_xlim((edges[0], edges[1]))



            ax_histx.set_ylim((self.MinMax[0], self.MinMax[1]))
            ax_histx.locator_params(tight=True, nbins=3)
            if self.grid:
                ax_histx.grid(which='major', linestyle='--', linewidth=0.5, color='dimgrey')
                ax_histx.grid(which='minor', linestyle='--', linewidth=0.5, color='lightgrey')
            ax_histx.set_ylabel('[counts]', fontsize='medium')

        if self.showYhist:
            ax_histy.hist(self.y, bins=dbinsy, color=self._color_palette_1d(0), linewidth=1, 
                        edgecolor=self._color_palette_1d(0), orientation='horizontal', histtype="step")
            ax_histy.set_ylim((edges[2], edges[3]))
            ax_histy.set_xlim((self.MinMax[0], self.MinMax[1]))
            ax_histy.locator_params(tight=True, nbins=3)
            if self.grid:
                ax_histy.grid(which='major', linestyle='--', linewidth=0.5, color='dimgrey')
                ax_histy.grid(which='minor', linestyle='--', linewidth=0.5, color='lightgrey')
            ax_histy.set_xlabel('[counts]', fontsize='medium')
            # ax_histx.set_ylabel('[counts]')
            # ax_histy.set_xlabel('[counts]')

        self._save_and_show(file_name, silent, enable)


    def plot_quiver(self, file_name=None, enable=True, silent=False, m=6.4, n=4.8):
        """
        Plots a quiver plot for gradient data.

        Parameters:
            file_name (str, optional): File path to save the plot. Defaults to None.
            enable (bool, optional): Whether to display the plot. Defaults to True.
            silent (bool, optional): Whether to suppress plot output. Defaults to False.
            m (float, optional): Width of the figure in inches. Defaults to 6.4.
            n (float, optional): Height of the figure in inches. Defaults to 4.8
        """
        self._plt_settings(self.FontsSize, self.LaTex, _hold=False, _silent=silent, m=m, n=n)
    
        if self.AxLegends[0] is not None:
            plt.title(self.AxLegends[0])
        if self.AxLegends[1] is not None:
            plt.xlabel(self.AxLegends[1])
        if self.AxLegends[2] is not None:
            plt.ylabel(self.AxLegends[2])
    
        plt.tick_params(direction='in', which='both')
        plt.tick_params(axis='x', pad=8)
    
        if self.AxLimits[1] is None:
            if self.AxLimits[0] is None:
                pass
            else:
                plt.xlim(xmin=self.AxLimits[0])
        else:
            if self.AxLimits[0] is None:
                plt.xlim(xmax=self.AxLimits[1])
            else:
                plt.xlim((self.AxLimits[0], self.AxLimits[1]))
    
        if self.AxLimits[3] is None:
            if self.AxLimits[2] is None:
                pass
            else:
                plt.ylim(ymin=self.AxLimits[2])
        else:
            if self.AxLimits[2] is None:
                plt.ylim(ymax=self.AxLimits[3])
            else:
                plt.ylim((self.AxLimits[2], self.AxLimits[3]))
    
        X, Y = np.meshgrid(self.x, self.y)
        kx = int((len(self.x)%self.kk)/2)
        ky = int((len(self.y)%self.kk)/2)
    
        if self.Colorbar:
            C = np.sqrt(self.fld_X ** 2 + self.fld_Y ** 2)
            im = plt.quiver(X[kx::self.kk, ky::self.kk], Y[kx::self.kk, ky::self.kk], 
                            self.fld_X[kx::self.kk, ky::self.kk], self.fld_Y[kx::self.kk, ky::self.kk],
                            C[kx::self.kk, ky::self.kk],
                            cmap=self._color_palette_2d(self.ColorScheme), angles='xy', scale_units='xy')

            im.set_clim(vmin=self.MinMax[1], vmax=self.MinMax[0])
            if self.PlotScale == 1:
                plt.colorbar(format='%.0e')
            else:
    
                if self.AspectRatio:
                    im_ratio = (self.y[-1] - self.y[0]) / (self.x[-1] - self.x[0])
                else:
                    im_ratio = 1
    
                cb = plt.colorbar(im, fraction=0.046 * im_ratio, pad=0.04, extend=self.ColorbarExt,
                                  spacing='uniform')
    
                if self.PlotScale != 2:
                    def format_func(x, pos):
                        x = '%.2f' % x
                        pad = '' if x.startswith('-') else ' '
                        return '{}{}'.format(pad, x)
    
                    tick_locator = ticker.MaxNLocator(nbins=4)
                    cb.locator = tick_locator
                    cb.update_ticks()
                    if self.PlotScale != 2:
                        cb.ax.yaxis.major.formatter = ticker.FuncFormatter(format_func)
        else:
            im = plt.quiver(X[kx::self.kk, ky::self.kk], Y[kx::self.kk, ky::self.kk], 
                            self.fld_X[kx::self.kk, ky::self.kk], self.fld_Y[kx::self.kk, ky::self.kk],
                            cmap=self._color_palette_2d(self.ColorScheme), angles='xy', scale_units='xy')
    
        im._axes.set(xlim=(self.x[0], self.x[-1]), ylim=(self.y[0], self.y[-1]))
        plt.locator_params(tight=True, nbins=self.nbins)
    
        if self.grid:
            plt.grid(linestyle='--', linewidth=0.4, color='dimgrey')
    
        if self.AspectRatio:
            im._axes._axes.set_aspect('equal')
        else:
            im._axes._axes.set_aspect('auto')

        self._save_and_show(file_name, silent, enable)


    def plot_contour(self, fld_X, fld_Y, file_name=None, enable=True, silent=False, m=6.4, n=4.8, dpi=500, kk=50):
        print('Place holder: not implemented yet...')
        # https://matplotlib.org/stable/plot_types/arrays/contour.html#sphx-glr-plot-types-arrays-contour-py


    # ****************************************************************************
    # ********************** Histograms
    # ****************************************************************************

    def plot_hist_1d(self, file_name=None, enable=True, silent=False, hold=False, m=6.4, n=4.8):
        """
        Plots a 1D histogram.

        Args:
            file_name (str, optional): File path to save the plot. Defaults to None.
            enable (bool, optional): Whether to display the plot. Defaults to True.
            silent (bool, optional): Whether to suppress plot output. Defaults to False.
            hold (bool, optional): Whether to hold the current plot. Defaults to False.
            m (float, optional): Width of the figure in inches. Defaults to 6.4.
            n (float, optional): Height of the figure in inches. Defaults to 4.8.

        Returns:
            Tuple[numpy.ndarray, numpy.ndarray]: Tuple containing the bin edges and the number of counts in each bin.
        """

        self._plt_settings(self.FontsSize, self.LaTex, _hold=hold, _silent=silent, m=m, n=n)
    
        if self.AxLegends[0] is not None:
            plt.title(self.AxLegends[0])
        if self.AxLegends[1] is not None:
            plt.xlabel(self.AxLegends[1])
        if self.AxLegends[2] is not None:
            plt.ylabel(self.AxLegends[2])
    
        plt.tick_params(direction='in', which='both')
        plt.tick_params(axis='x', pad=8)
    
        if self.AxLimits[1] is None:
            if self.AxLimits[0] is None:
                pass
            else:
                plt.xlim(xmin=self.AxLimits[0])
        else:
            if self.AxLimits[0] is None:
                plt.xlim(xmax=self.AxLimits[1])
            else:
                plt.xlim((self.AxLimits[0], self.AxLimits[1]))
    
        if self.label is not None:
            N, bins, patches = plt.hist(self.x, bins=self.nbins, color=self._color_palette_1d(self.ColorScheme),
                                        alpha=self.alpha, label=self.nbins, density=self.norm, stacked=self.norm,
                                        histtype='bar')
            plt.legend(loc=self.LabelPos)
    
        else:
            N, bins, patches = plt.hist(self.x, bins=self.nbins, color=self._color_palette_1d(self.ColorScheme),
                                        alpha=self.alpha, density=self.norm, stacked=self.norm, histtype='bar')
    
        if self.bold:
            plt.hist(self.x, bins=bins, color=self._color_palette_1d(0), alpha=self.alpha, density=self.norm,
                     stacked=self.norm, histtype='step')
    
        if self.AxLimits[3] is None:
            if self.AxLimits[2] is None:
                pass
            else:
                plt.ylim(ymin=self.AxLimits[2])
        else:
            if self.AxLimits[2] is None:
                plt.ylim(ymax=self.AxLimits[3])
            else:
                plt.ylim((self.AxLimits[2], self.AxLimits[3]))
    
        if self.grid:
            plt.grid(which='major', linestyle='--', linewidth=0.5, color='dimgrey')
            plt.grid(which='minor', linestyle='--', linewidth=0.5, color='lightgrey')
    
        # if self.FillBetween:
        #     plt.fill_between(self.x, self.FillBetweenValue, self.image,
        #                      color=self._color_palette_1d(self.ColorScheme), alpha=self.alpha)
    
        if self.PlotScale != 0 or hold is False:
            plt.locator_params(tight=True)#, nbins=self.nbins)
    
        if self.xticks is not None:
            plt.xticks(self.xticksaxis, self.xticks, rotation=0, horizontalalignment='center')

        self._save_and_show(file_name, silent, enable)

        return bins, N
    

    def plot_2d_hist(self, file_name=None, enable=True, silent=False, m=6.4, n=4.8):
        """
        Plots a 2D histogram.

        Args:
            file_name (str, optional): File path to save the plot. Defaults to None.
            enable (bool, optional): Whether to display the plot. Defaults to True.
            silent (bool, optional): Whether to suppress plot output. Defaults to False.
            m (float, optional): Width of the figure in inches. Defaults to 6.4.
            n (float, optional): Height of the figure in inches. Defaults to 4.8.

        """
        self._plt_settings(self.FontsSize, self.LaTex, _hold=False, _silent=silent, m=m, n=n)
    
        if self.AxLegends[0] is not None:
            plt.title(self.AxLegends[0])
        if self.AxLegends[1] is not None:
            plt.xlabel(self.AxLegends[1])
        if self.AxLegends[2] is not None:
            plt.ylabel(self.AxLegends[2])

        if self.ColorbarExt is None:
            self.ColorbarExt = 'neither'
    
        plt.tick_params(direction='in', which='both')
        plt.tick_params(axis='x', pad=8)
    
        if self.AxLimits[1] is None:
            if self.AxLimits[0] is None:
                pass
            else:
                plt.xlim(xmin=self.AxLimits[0])
        else:
            if self.AxLimits[0] is None:
                plt.xlim(xmax=self.AxLimits[1])
            else:
                plt.xlim((self.AxLimits[0], self.AxLimits[1]))
    
        if self.AxLimits[3] is None:
            if self.AxLimits[2] is None:
                pass
            else:
                plt.ylim(ymin=self.AxLimits[2])
        else:
            if self.AxLimits[2] is None:
                plt.ylim(ymax=self.AxLimits[3])
            else:
                plt.ylim((self.AxLimits[2], self.AxLimits[3]))

        hist, xbins, ybins = np.histogram2d(self.x, self.y, bins=self.nbinsHist)

        if self.discardZeros:
            hist[hist==0] = np.nan
            
        im = plt.imshow(hist, cmap=self._color_palette_2d(self.ColorScheme),
                        extent=[xbins[0], xbins[-1], ybins[0], ybins[-1]], origin='lower',
                        vmax=self.MinMax[1], vmin=self.MinMax[0])
    
        def format_func(x, pos):
            x = '%.2f' % x
            pad = '' if x.startswith('-') else ' '
            return '{}{}'.format(pad, x)
    
        if self.Colorbar:
            if self.AspectRatio:
                im_ratio = (ybins[-1] - ybins[0]) / (xbins[-1] - xbins[0])
            else:
                im_ratio = 1
            if self.IsPhase:
                cb = plt.colorbar(im, fraction=0.046 * im_ratio, pad=0.04, extend=self.ColorbarExt,
                                    spacing='uniform', ticks=[-2*np.pi, -np.pi, 0, np.pi, 2*np.pi])
                cb.ax.set_yticklabels(['$-2\pi$', '$-\pi$', '0', '$\pi$', '$2\pi$'])
            else:
                cb = plt.colorbar(im, fraction=0.046 * im_ratio, pad=0.04, extend=self.ColorbarExt,
                                    spacing='uniform')

                tick_locator = ticker.MaxNLocator(nbins=4)
                cb.locator = tick_locator
                cb.update_ticks()
                cb.ax.yaxis.major.formatter = ticker.FuncFormatter(format_func)
    
        plt.locator_params(tight=True, nbins=self.nbins)

        if self.grid:
            plt.grid(linestyle='--', linewidth=0.4, color='dimgrey')
    
        if self.AspectRatio:
            im._axes._axes.set_aspect('equal')
        else:
            im._axes._axes.set_aspect('auto')

        self._save_and_show(file_name, silent, enable)

    # ****************************************************************************
    # ********************** settings
    # ****************************************************************************

    @staticmethod
    def _color_palette_1d(_scheme):
        """
        Color palette for 1D plots.

        Parameters:
            _scheme (int or ndarray): Color number or RGB value.

        Returns:
            tuple: RGB values for the chosen color.
        """
        if type(_scheme) == np.ndarray:
            return _scheme
        elif type(_scheme) == int:
            if _scheme == -2:  # red
                color = (255./255, 000./255, 000./255)
            elif _scheme == -1:  # black
                color = (000./255, 000./255, 000./255)
            elif _scheme == 0:  # SOLEIL blue
                color = (000./255, 077./255, 157./255)
            elif _scheme == 1:  # SOLEIL yellow
                color = (251./255, 186./255, 007./255)       
            elif _scheme == 2:  # Environnement et Géoscience (green)
                color = (092./255, 171./255, 071./255)
            elif _scheme == 3:  # Matériaux avancés (magenta)
                color = (156./255, 061./255, 140./255)
            elif _scheme == 4:  # Santé et Bien-être (red)
                color = (232./255, 061./255, 041./255)
            elif _scheme == 5:  # Processus fondamentaux dans les systèmes isolés (blue/green)
                color = (000./255, 173./255, 168./255)
            elif _scheme == 6:  # Énergie durable (light blue)
                color = (232./255, 061./255, 041./255)
            elif _scheme == 7:  # Patrimoine culturel (dark green)
                color = (232./255, 061./255, 041./255)
            elif _scheme == 10:  # ESRF blue
                color = (019./255, 037./255, 119./255)
            # elif _scheme == 3:  # Light green
            #     color = (081. / 255, 160. / 255, 038. / 255)
            # elif _scheme == 4:  # magenta
            #     color = (175. / 255, 000. / 255, 124. / 255)
            # elif _scheme == 5:  # dark orange
            #     color = (237. / 255, 119. / 255, 003. / 255)
            return color
        else:
            print('Colour palette does not exist')
            return (0., 0., 0.)

    @staticmethod
    def _color_palette_2d(_scheme):
        """
        Color maps for 2D plots.

        Parameters:
            _scheme (int): Color map selection.

        Returns:
            colormap: Matplotlib colormap object.
        """
        # Colour maps reference: https://matplotlib.org/examples/color/colormaps_reference.html
        if _scheme == 0:    # white(1)/black(0) - mimics SRW
            color = cm.binary_r
        elif _scheme == 1:  # black(1)/white(0) - inverted SRW
            color = cm.binary
        elif _scheme == 2:  # Viridis (perceptually uniform colors)
            color = cm.viridis
        elif _scheme == 3:  # red(1)-yellow-green-blue(0) - Jet
            color = cm.jet
        elif _scheme == 4:  # Inverted Cube helix - adds colour to SRW
            color = cm.cubehelix_r
        elif _scheme == 5:  # Cube helix
            color = cm.cubehelix
        elif _scheme == 6:  # phase - PyNx
            color = cm.hsv
        elif _scheme == 7:  # red(1)/white/black(0) diverging
            color = cm.RdGy_r
        elif _scheme == 8:  # Diverging France
            color = cm.RdBu
        elif _scheme == 9:  # Thermal plot
            color = cm.plasma
        elif _scheme == 10:  # Inverted thermal plot
            color = cm.plasma_r
        else:              # give by hand the desired
            color = _scheme
        return color

    @staticmethod
    def _plt_settings(_k=1, _latexstyle=True, _hold=False, _silent=False, m=6.4, n=4.8):
        """
        Configure plot settings.

        Parameters:
            _k (int, optional): Scaling factor for font sizes. Defaults to 1.
            _latexstyle (bool, optional): Enable LaTeX style for text. Defaults to True.
            _hold (bool, optional): Enable hold mode for plot. Defaults to False.
            _silent (bool, optional): Suppress plotting output. Defaults to False.
            m (float, optional): Width of the figure in inches. Defaults to 6.4.
            n (float, optional): Height of the figure in inches. Defaults to 4.8.
        """
        plt.rcParams.update(rcParamsDefault)
        if _latexstyle:
            plt.rcParams.update({
                "text.usetex": False,
                # "font.family": "serif",
                # "font.serif": ["Palatino"]
                "font.family": "DeJavu Serif",
                "font.serif": ["Times New Roman"]
                # "font.family": "serif",
                # "font.serif": ["Computer Modern Roman"]
            })

        if _hold is False:
            if _silent is True:
                plt.close()
            fig = plt.figure(figsize=(m, n))

        plt.gcf().subplots_adjust(left=0.15)

        # plt.rc('font', size=18*_k)  # controls default text sizes
        plt.rc('axes', titlesize=16 * _k)   # fontsize of the axes title
        plt.rc('axes', labelsize=15 * _k)   # fontsize of the x and y labels
        plt.rc('xtick', labelsize=14 * _k)  # fontsize of the tick labels
        plt.rc('ytick', labelsize=14 * _k)  # fontsize of the tick labels
        plt.rc('legend', fontsize=13 * _k)  # legend fontsize
        # plt.rc('figure', titlesize=30*_k)  # fontsize of the figure title

    # ****************************************************************************
    # ********************** miscellaneous
    # ****************************************************************************

    def _save_and_show(self, file_name, silent, enable):
        """
        Save and optionally show the plot.

        Parameters:
            file_name (str): The file path to save the plot. If None, the plot won't be saved.
            silent (bool): If True, the plot window will be closed after saving.
            enable (bool): If True, the plot will be displayed.
        """
        if file_name is not None:
            plt.savefig(file_name, dpi=self.dpi, bbox_inches='tight')
            file_name = file_name.split('/')
            print('>>>> file %s saved to disk.' % file_name[-1])
            if silent:
                plt.close()

        if enable:
            plt.show()


if __name__ == '__main__':

    print('welcome to barc4plots!')
