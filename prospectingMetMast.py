#!/home/wiwasol/miniconda3/envs/wiwapmm/bin python
# -*- coding: utf-8 -*-
#
# Using the file system load
#
# We now assume we have a file in the same dir as this one called
# test_template.html
#
# Note: Will remove excess import statements, this was from a major refactoring from a single Python file to broken
# down into main functional tasks.
# Note: Work in Progess, still refactoring! The commented out import order below causes a cairo error and
# doesn't make PDF.


"""
# from numpy.random import random_sample, weibull, randn
from jinja2 import Environment, FileSystemLoader
import os
from decimal import *
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
# from scipy.stats import weibull_min, weibull_max
# import scipy as sp
# from weasyprint import HTML, CSS
# import pylab
from matplotlib.ticker import FuncFormatter
# import scipy.special as spe
# from windrose import WindroseAxes
# import matplotlib.cm as cm
# from matplotlib.legend_handler import HandlerLine2D
# from numpy import arange
import math
# import forecastio
# import datetime
# import numpy as np
import pandas as pd
import sys
# import json
# from sys import argv
# from io import StringIO
# from forecastio.utils import UnicodeMixin, PropertyUnavailable
# from sqlalchemy import Table, Column, Integer, String, Float, MetaData, ForeignKey
# io = StringIO()
import DarkSky.callData
"""

from numpy.random import random_sample
from jinja2 import Environment, FileSystemLoader
import os
from decimal import *
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

from numpy.random import weibull
import numpy as np
from scipy import stats
from scipy.stats import weibull_min
from scipy.stats import weibull_max
import scipy as sp
# Capture our current directory
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
#from weasyprint import HTML, CSS

import pylab
from numpy.random import randn
from matplotlib.ticker import FuncFormatter
import scipy.special as spe

#windrose
from windrose import WindroseAxes
#import windrose
import matplotlib.cm as cm
from matplotlib.legend_handler import HandlerLine2D
#import numpy as np
from numpy import arange
import math
import forecastio
import datetime
import numpy as np
import pandas as pd
import sys, json
from sys import argv
#from io import StringIO
####from forecastio.utils import UnicodeMixin, PropertyUnavailable
from sqlalchemy import Table, Column, Integer, String, Float, MetaData, ForeignKey
import csv




import DatabaseEngine.databaseEngine as dbe
import Graphs.ClimateMeansBoxPlot
import Graphs.LineGraphErrorFill
import Graphs.WindHistogramAnnual
import Graphs.WindHistogramMonths
import Graphs.WindHistogramSectors
import Graphs.WindRoseAnnual
import Graphs.WindRoseMonths
# Capture our current directory
# THIS_DIR = os.path.dirname(os.path.abspath(__file__))
# matplotlib.use('agg')

font = {'family': 'serif',
        'color':  'black',
        'weight': 'normal',
        'size': 20,
        }
fontLine = {'color':  'black',
            'weight': 'normal',
            'size': 12,
            }


def to_percent1(y, position):
    # Ignore the passed in position. This has the effect of scaling the default
    # tick locations.
    s = str(100 * y)

    # The percent symbol needs escaping in latex
    if matplotlib.rcParams['text.usetex'] is True:
        return s + r'$\%$'
    else:
        return s + '%'


def to_percent2(y, position):
    # Ignore the passed in position. This has the effect of scaling the default
    # tick locations.
    s = str(100. * y)

    # The percent symbol needs escaping in latex
    if matplotlib.rcParams['text.usetex'] is True:
        return s + r'$\%$'
    else:
        return s + '%'


def to_percent3(y, position):
    # Ignore the passed in position. This has the effect of scaling the default
    # tick locations.
    s = str(1 * y)

    # The percent symbol needs escaping in latex
    if matplotlib.rcParams['text.usetex'] is True:
        return s + r'$\%$'
    else:
        return s + '%'


def to_percent4(y, position):
    # Ignore the passed in position. This has the effect of scaling the default
    # tick locations.
    s = str((100 * round(y*100)/100))

    # The percent symbol needs escaping in latex
    if matplotlib.rcParams['text.usetex'] is True:
        return s + r'$\%$'
    else:
        return s + '%'


formatter1 = FuncFormatter(to_percent1)
formatter2 = FuncFormatter(to_percent2)
formatter3 = FuncFormatter(to_percent3)
formatter4 = FuncFormatter(to_percent4)
# Formatting histogram plots
# colourb1 = '#408754'
colourb1 = '#6DCBD5'
colourk1 = '#40423F'
colourg1 = '#09694E'


def map_bin(x, bins):
    kwargs = {}
    if x == max(bins):
        kwargs['right'] = True
    bin_name = bins[np.digitize([x], bins, **kwargs)[0]]
    bin_lower = bins[np.digitize([x], bins, **kwargs)[0]-1]
    return '[{0}-{1}]'.format(bin_lower, bin_name)


"""
def month_bin(x, bins):
    kwargs = {}
    if x == max(bins):
        kwargs['right'] = True
    bin = bins[np.digitize([x], bins, **kwargs)[0]]
    bin_lower = bins[np.digitize([x], bins, **kwargs)[0]-1]
    return '[{0}-{1}]'.format(bin_lower, bin)
"""

# First attempt at automating purchase(s) to downloading, processing data and
# creating PDF of Prospecting Met Mast(s).
# PHP script was hanging on long running Python script (waiting to download 30 years of hourly data + processing)
"""
try:
    data = json.loads(sys.argv[1])
    # data = json.load( sys.stdin )
    # result = data['testname1']
    latt2 = data["latitude"]
    long2 = data["longitude"]
    ystart = data["datayearstart"]
    yend = data["datayearend"]
    # ystart = str(data["datayearstart"]) + '-01-01'
    # yend = str(data["datayearend"]) + '-12-31'
    orderid = data["orderNum"]
    dbe.storedatabaseNEW(dbe.retrievedatabase(),"site_"+str(orderid)+"_"+str(latt2)+"_"+str(long2)+"_"+str(ystart))
except (ValueError, TypeError, IndexError, KeyError) as e:
    print(json.dumps({'error': str(e)}))
    latt2 = 0
    long2 = 0
    ystart = '2015-12-30'
    yend = '2015-12-31'
    orderid = 999
    dbe.storedatabaseNEW(dbe.retrievedatabase(),"site_"+str(orderid)+"_"+str(latt2)+"_"+str(long2)+"_"+str(ystart))
    # sys.exit(1)
"""

"""
try:
    data = json.loads(sys.argv[1])
    # data = json.load( sys.stdin )
    # result = data['testname1']
    latt2 = data["latitude"]
    long2 = data["longitude"]
    ystart = data["datayearstart"]
    yend = data["datayearend"]
    # version 1.01 is simply years, later it can have days/months
    # ystart = str(data["datayearstart"]) + '-01-01'
    # yend = str(data["datayearend"]) + '-12-31'
    ystartlong = str(data["datayearstart"]) + '-01-01'
    yendlong = str(data["datayearend"]) + '-12-31'
    orderid = data["orderNum"]
    # dbe.storedatabaseNEW(dbe.retrievedatabase(),"site_"+str(orderid)+"_"+str(latt2)+"_"+str(long2)+"_"+str(ystart))
    # vardbnew = DarkSky.callData.callday(str(latt2), str(long2), ystartlong, yendlong)
    # dbe.storedatabaseNEW(vardbnew,"site_"+str(orderid)+"_"+str(latt2)+"_"+str(long2)+"_"+str(ystart))
    # easyprintpdf(vardbnew)
except (ValueError, TypeError, IndexError, KeyError) as e:
    print(json.dumps({'error': str(e)}))
    latt2 = 0
    long2 = 0
    ystart = '2015-12-30'
    yend = '2015-12-31'
    orderid = 999
    # dbe.storedatabaseNEW(dbe.retrievedatabase(),"site_"+str(orderid)+"_"+str(latt2)+"_"+str(long2)+"_"+str(ystart))
    # sys.exit(1)
"""


def weib(x, *p):
    # Provide weibull parameters to sample within reasonable ranges
    xs_sat, lo, w, s = p
    return xs_sat*(1-np.exp(-((x-lo)/w)**s))


def easy_print_pdf(data_full, startdate='1985-01-01', enddate='2017-12-31'):
    # Datetime objects must be in consistent format
    data_full['time'] = pd.to_datetime(data_full['time'], format='%m/%d/%Y %H:%M')
    #data_full['time'] = pd.to_datetime(data_full['time'], format='%d/%m/%Y %H:%M')
    # dataFull['time'] = pd.to_datetime(dataFull['time'])
    data_full.set_index('time', inplace=True)
    # dataFull.dropna()
    data_full.dropna(subset=['windd', 'correctedwindspeed'])

    # formatter1 = FuncFormatter(to_percent1)

    fig = plt.figure(figsize=(4.34, 3.22), dpi=100, facecolor='w', edgecolor='w')
    ax1 = Graphs.WindRoseAnnual.new_axes_ra(data_full["windd"], data_full["correctedwindspeed"], font_line=fontLine)
    plt.savefig(file_path + 'WindRose_temp8888_img.png')
    annual_rose = "file://" + output_file_path + "WindRose_temp8888_img.png"
    plt.clf()
    plt.close()

    data_wind = data_full.copy()
    data_wind.dropna(axis=0, how='any', subset=['winds', 'windd', 'correctedwindspeed'], inplace=True)
    Graphs.WindHistogramAnnual.new_axesHA(data_wind, colourb1=colourb1, formatter=formatter1, file_path=file_path)
    # plt.savefig(file_path + 'WindHistogram_temp8888_img.png')
    annual_hist = "file://" + output_file_path + "WindHistogram_temp8888_img.png"
    plt.clf()
    plt.close()

    xaxis2 = ["01", "02", "03", "04", "05", "06", "07", "08", "09", "10", "11", "12"]
    xaxis_mon = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

    data_month_group = data_wind.groupby(lambda x: x.month)
    text_mon1 = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    monthly_wind_histograms = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    sector_wind_histograms = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

    for iMon, group in enumerate(data_month_group, 0):
        ax2 = Graphs.WindRoseMonths.new_axes_r(group[1].windd, group[1].correctedwindspeed, xaxis_mon[iMon], font=font)
        plt.savefig(file_path + 'weiWRose_' + xaxis2[iMon] + '_temp888_img.png')

        # add folder for each new report and image files?! Good way to keep records, and check if errors crop up

        ax, monthly_wind_histograms[iMon] = Graphs.WindHistogramMonths.new_axesH_Mon(group[1].correctedwindspeed, xaxis_mon[iMon], xaxis2[iMon],
                                                 iMon, text_mon1, formatter=formatter1, colourb1=colourb1,
                                                 file_path=file_path)
        # plt.savefig(file_path + 'weibull_' + xaxis2[iMon] + '_temp999_img.png')

    wh_columns = np.arange(0, 25, 1)
    monthly_wh_df = pd.DataFrame(data=monthly_wind_histograms, columns=wh_columns, index=xaxis_mon)
    monthly_wh_df.to_csv(file_path + "monthly_wind_histograms.csv", float_format="%2.9f")

    freq_bins2 = [0, 30, 60, 90, 120, 150, 180, 210, 240, 270, 300, 330, 360]
    # Was this changed because of a windrose error/update? From 12 to 13 array length/sectors of windrose
    # Commented out Jan 20th, added above
    # data_wind = dataFull.copy().dropna(axis=0, how='any', subset=['winds', 'windd', 'correctedwindspeed'])
    # TODO: Attempts at fixing index out of range error (seemed to work with only 2 months?), but not with 12-24 etc
    data_wind['Binned'] = data_wind['windd'].apply(map_bin, bins=freq_bins2)
    grouped = data_wind[['windd', 'correctedwindspeed', 'Binned']].groupby('Binned')

    xaxis_sec = ['30', '60', '90', '120', '150', '180', '210', '240', '270', '300', '330', '360']
    bins_h = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25]
    bins_lh = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25]
    columns_b = ['[0-30]', '[30-60]', '[60-90]', '[90-120]', '[120-150]', '[150-180]',
                 '[180-210]', '[210-240]', '[240-270]', '[270-300]', '[300-330]', '[330-360]']
    dataprint = (data_wind.assign(
                    q=pd.cut(np.clip(data_wind.correctedwindspeed, bins_h[0], bins_h[-1]), bins=bins_h, labels=bins_lh,
                             right=False)).pivot_table(index='q', columns='Binned', aggfunc='size', fill_value=0))

    text_sec4 = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    text_sec5 = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    text_sec6 = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    shape_sec = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    loc_sec = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    scale_sec = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

    for iSec, group2 in enumerate(grouped, 0):
        if iSec < 12:
            print(iSec)
            try:
                print(xaxis2[iSec])
            except IndexError:
                print("xaxis2")

            try:
                print(xaxis_sec[iSec])
            except IndexError:
                print("xaxis_sec")

            ax, sector_wind_histograms[iSec] = Graphs.WindHistogramSectors.new_axes_h_sec(group2[1].correctedwindspeed, xaxis_sec[iSec], xaxis2[iSec],
                                                       iSec, text_sec4, text_sec5, text_sec6, shape_sec, loc_sec,
                                                       scale_sec, formatter=formatter1, colourb1=colourb1,
                                                       file_path=file_path)
            # plt.savefig(file_path + xaxis2[iSec] + '_temp999_img.png')
        else:
            pass

    sector_wh_df = pd.DataFrame(data=sector_wind_histograms, columns=wh_columns, index=xaxis_sec)
    sector_wh_df.to_csv(file_path + "sector_wind_histograms.csv", float_format="%2.9f")

    count = 0
    rose_month_r = [[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]]
    hist_month_h = [[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]]
    hist_sector_h = [[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]]
    for i in range(4):
        for j in range(3):
            rose_month_r[i][j] = "file://" + output_file_path + 'weiWRose_' + xaxis2[count] + '_temp888_img.png'
            hist_month_h[i][j] = "file://" + output_file_path + 'weibull_' + xaxis2[count] + '_temp999_img.png'
            hist_sector_h[i][j] = "file://" + output_file_path + 'weibullSEC_' + xaxis2[count] + '_temp999_img.png'
            count = count + 1

    # Climate Means Fig 8 ===============================================================
    plt.clf()
    plt.close()
    fig = plt.figure(figsize=(7, 4.17), dpi=120, facecolor='w', edgecolor='w')
    Graphs.ClimateMeansBoxPlot.meanBoxplot(data_full, formatter4=formatter4, colourb1=colourb1)
    plt.savefig(file_path + 'box1_temp444_img.png')
    box1 = "file://" + output_file_path + "box1_temp444_img.png"

    # Diurnal Fig 2 ===============================================================
    diurnalm = data_full['correctedwindspeed'].groupby(lambda x: x.hour).mean().as_matrix()
    diurnals = data_full['correctedwindspeed'].groupby(lambda x: x.hour).std().as_matrix()

    xaxis2b = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23]
    xaxis_hours = ["0:00", "1:00", "2:00", "3:00", "4:00", "5:00", "6:00", "7:00", "8:00", "9:00", "10:00", "11:00",
                   "12:00", "13:00", "14:00", "15:00", "16:00", "17:00", "18:00", "19:00", "20:00", "21:00", "22:00",
                   "23:00"]
    fig = plt.figure(figsize=(7, 4.04), dpi=120, facecolor='w', edgecolor='w')
    # plt.plot(xaxis2b, diurnalm, color=colourb1, linewidth=3)
    # errorfill(xaxis2a, yline2, yerr=error2, color=colourg1, alpha_fill=0.2)
    Graphs.LineGraphErrorFill.errorfill(xaxis2b, diurnalm, yerr=diurnals, color=colourg1, alpha_fill=0.2)
    plt.xticks(xaxis2b, xaxis_hours, rotation=45)
    plt.margins(0.05)
    dmaxm = (diurnals+diurnalm).max()+1

    ax3 = fig.add_subplot(1, 1, 1)

    # major ticks every 10, minor ticks every 5
    major_ticks3 = np.arange(0, dmaxm, 2)
    minor_ticks3 = np.arange(0, dmaxm, 1)

    ax3.set_yticks(major_ticks3)
    ax3.set_yticks(minor_ticks3, minor=True)

    # and a corresponding grid
    # ax.grid(which='both')

    # or if you want different settings for the grids:
    ax3.grid(which='minor', alpha=0.4)
    ax3.grid(which='major', alpha=0.5)

    plt.ylabel('Mean Wind Speed m/s')
    plt.xlabel('Figure 2: Daily Variation in Average Hourly Wind Speeds')
    plt.tight_layout()
    plt.savefig(file_path + 'line4_temp555_img.png')
    diurnal = "file://" + output_file_path + "line4_temp555_img.png"

    # Monthly Mean, Std, StdErr Fig 9 and Table 2 ===============================================================
    mean_year_m = data_full['correctedwindspeed'].groupby(lambda x: x.year).mean()

    meanws_o = data_full['correctedwindspeed'].groupby(lambda x: x.month).mean().as_matrix()
    stdevia_o = data_full['correctedwindspeed'].groupby(lambda x: x.month).std().as_matrix()
    stderr_o = data_full['correctedwindspeed']\
        .groupby(lambda x: x.month).std().as_matrix() / math.sqrt(len(mean_year_m) - 1)
    meanws_ao = Decimal(0.00)
    stdevia_ao = Decimal(0.00)
    stderr_ao = Decimal(0.00)

    for k in range(12):
        meanws_ao += Decimal(meanws_o[k])
        stdevia_ao += Decimal(stdevia_o[k])
        stderr_ao += Decimal(stderr_o[k])

    meanws_ao = meanws_ao/12
    stdevia_ao = stdevia_ao/12
    stderr_ao = stderr_ao/12
    xaxis2a = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
    xaxis_mon = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

    # error bar values w/ different -/+ errors
    # lower_error = error2
    # upper_error = 0.9*error2
    # asymmetric_error = [lower_error, upper_error]
    fig = plt.figure(figsize=(7, 4.04), dpi=120, facecolor='w', edgecolor='w')
    Graphs.LineGraphErrorFill.errorfill(xaxis2a, meanws_o, yerr=stdevia_o, color=colourg1, alpha_fill=0.2)
    # plt.grid(True)
    dmax_o = (meanws_o+stdevia_o).max()+5

    ax2 = fig.add_subplot(1, 1, 1)

    # major ticks every 10, minor ticks every 5
    major_ticks2 = np.arange(0, dmax_o, 2)
    minor_ticks2 = np.arange(0, dmax_o, 1)

    plt.xticks(xaxis2a, xaxis_mon)
    ax2.set_yticks(major_ticks2)
    ax2.set_yticks(minor_ticks2, minor=True)

    # and a corresponding grid
    # ax.grid(which='both')

    # or if you want different settings for the grids:
    ax2.grid(which='minor', alpha=0.4)
    ax2.grid(which='major', alpha=0.5)

    plt.ylabel('Mean Wind Speed m/s')
    plt.xlabel('Figure 9: Monthly mean wind speed with 16th and 84th percentiles', fontdict=fontLine)
    plt.tight_layout()
    plt.savefig(file_path + 'line2_temp555_img.png')
    line2 = "file://" + output_file_path + "line2_temp555_img.png"

    # P50:P90 Fig 1 ===============================================================
    p50breakdown = Graphs.WindHistogramMonths.new_axesH_MonCDF(data_month_group, xaxis_mon)
    yy50_5ms = p50breakdown[0]
    yy90_10ms = p50breakdown[1]
    yy99_15ms = p50breakdown[2]

    fig = plt.figure(figsize=(7, 4.04), dpi=120, facecolor='w', edgecolor='w')

    ym50 = int(yy50_5ms.max())
    ym90 = int(yy90_10ms.max())
    ym99 = int(yy99_15ms.max())
    ymax3 = None
    if ym50 < ym90:
        if ym90 < ym99:
            ymax3 = ym99
        elif ym90 > ym99:
            ymax3 = ym90
    elif ym50 > ym90:
        if ym50 < ym99:
            ymax3 = ym99
        elif ym50 > ym99:
            ymax3 = ym50
    ynorm50 = (1-(yy50_5ms/ymax3))*100
    ynorm90 = (1-(yy90_10ms/ymax3))*100
    ynorm99 = (1-(yy99_15ms/ymax3))*100

    plt.plot(xaxis2a, ynorm50, color=colourk1, linewidth=2, label='5 m/s')
    plt.plot(xaxis2a, ynorm90, color=colourg1, linewidth=2, label='10 m/s')
    plt.plot(xaxis2a, ynorm99, color=colourb1, linewidth=2, label='15 m/s')

    plt.xticks(xaxis2a, xaxis_mon)
    plt.margins(0.05)
    plt.legend(title="Probability of Exceeding Wind Speed", loc='upper right', bbox_to_anchor=(1, 1),
               fontsize='x-small', framealpha=0.7)

    plt.ylabel('Probability of Exceedance')
    plt.xlabel('Figure 1: P50-P90 Probability of Exceedance of Wind Speed Classes')

    ax = fig.add_subplot(1, 1, 1)

    # major ticks every 10, minor ticks every 5
    major_ticks = np.arange(0, 108, 10)

    ax.set_yticks(major_ticks)
    # and a corresponding grid
    # ax.grid(which='both')

    # or if you want different settings for the grids:
    # ax.grid(which='minor', alpha=0.4)
    ax.grid(which='major', alpha=0.5)

    plt.gca().yaxis.set_major_formatter(formatter3)
    plt.tight_layout()
    plt.savefig(file_path + 'line3_temp555_img.png')
    pfifty = "file://" + output_file_path + "line3_temp555_img.png"

    # Wind Frequency Distribution TABLE===========================
    metersclass_o = []
    total_co = Decimal(000.00)
    total_ro = Decimal(000.00)
    for t in range(24):
        metersclass_o.append(str(t) + "-" + str(int(t+1)) + "m/s")
    else:
        metersclass_o.append("&ge;&nbsp;24m/s")

    meterspercent_o = np.zeros((25, 12))
    sumdata = 0
    for k in range(12):
        for i in range(25):
            # sumdata += dataprint[columns_b[k]].count()
            try:
                # print("I==============")
                # print(dataprint.iloc[i])
                sumdata += dataprint.iloc[i][columns_b[k]]
            except IndexError:
                # sumdata
                print("IndexError2")
                # print(dataprint.iloc[i][columns_b[k]])
            except KeyError:
                print("KeyError2")

    for k in range(12):
        for i in range(25):
            # meterspercent_o[i][k] = dataprint[i][columns_b[k]]/sumdata
            # meterspercent_o[i][k] = dataprint.iloc[i][columns_b[k]]/sumdata
            # meterspercent_o[i][k] = (dataprint.iloc[i][columns_b[k]]/sumdata)*100
            try:
                # print("M==============")
                # print(dataprint.iloc[i])
                # sumdata += dataprint.iloc[i][columns_b[k]]
                meterspercent_o[i][k] = (dataprint.iloc[i][columns_b[k]] / sumdata) * 100
            except IndexError:
                # sumdata
                print("IndexError3")
                meterspercent_o[i][k] = 0.0
                # print(dataprint.iloc[i][columns_b[k]])
            except KeyError:
                print("KeyError3")
                meterspercent_o[i][k] = 0.0
    # print(meterspercent_o[0][0])
    """
    meterspercent_o[1] = dataprint[columns_b[1]]/sumdata
    meterspercent_o[2] = dataprint[columns_b[2]]/sumdata
    meterspercent_o[3] = dataprint[columns_b[3]]/sumdata
    meterspercent_o[4] = dataprint[columns_b[4]]/sumdata
    meterspercent_o[5] = dataprint[columns_b[5]]/sumdata
    meterspercent_o[6] = dataprint[columns_b[6]]/sumdata
    meterspercent_o[7] = dataprint[columns_b[7]]/sumdata
    meterspercent_o[8] = dataprint[columns_b[8]]/sumdata
    meterspercent_o[9] = dataprint[columns_b[9]]/sumdata
    meterspercent_o[10] = dataprint[columns_b[10]]/sumdata
    meterspercent_o[11] = dataprint[columns_b[11]]/sumdata
    """
    sectorstotal_o = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    meterstotal_o = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

    # 10 years
    # meterspercent_o = weibull(2*np.random.random_sample()+1.2, (25, 12))
    # used to have abs() surrounding each of these, useful for random data
    getcontext().prec = 3
    for k in range(12):
        for i in range(25):
            total_ro += 100*Decimal(meterspercent_o[i][k])/100
    for i in range(25):
        for k in range(12):
            total_co += 100*Decimal(meterspercent_o[i][k])/100
    for k in range(12):
        for i in range(25):
            sectorstotal_o[k] += 100*Decimal(meterspercent_o[i][k])/100
    for i in range(25):
        for k in range(12):
            meterstotal_o[i] += 100*Decimal(meterspercent_o[i][k])/100
    getcontext().prec = 2
    # translates float to 2 place decimal
    for i in range(25):
        for k in range(12):
            meterspercent_o[i][k] = 1000*Decimal(meterspercent_o[i][k])/1000

    # for i in range(25):
    #    for k in range(12):
    #        meterspercent_o[i][k] = Decimal(meterspercent_o[i][k])
    # x1 = 10 * weibull(2*np.random.random_sample()+1, 200)
    # h1 = np.histogram(x1, bins=25, density=True)
    """
    h1 = np.histogram(x1, bins=25, density=True)
    mean = np.mean(x1)
    median = np.median(x1)
    std = np.std(x1)
    stderr = scipy.stats.stderr(x1)
    cdf5 = scipy.norm.cdf(x1, loc=5)
    cdf10 = scipy.norm.cdf(x1, loc=10)
    cdf15 = scipy.norm.cdf(x1, loc=15)
    """

    """
    meanws_o = 4.5*random_sample((12, ))+4
    stdevia_o = 2.7*random_sample((12, ))+1.2
    #meanws_o.append(0)
    #stdevia_o.append(0)
    meanws_ao = Decimal(0.00)
    stdevia_ao = Decimal(0.00)
    for k in range(12):
        #if(k <=12):
        meanws_o[k] = 10*Decimal(meanws_o[k])/10
        stdevia_o[k] = 10*Decimal(stdevia_o[k])/10
        meanws_ao += Decimal(meanws_o[k])
        stdevia_ao += Decimal(stdevia_o[k])
        #else:
        #meanws_o[k] = 10*Decimal(meanws_o[k])/10
        #stdevia_o[k] = 10*Decimal(stdevia_o[k])/10
        #meanws_ao += Decimal(meanws_o[k])
        #stdevia_ao += Decimal(stdevia_o[k])
    meanws_ao = meanws_ao/12
    stdevia_ao = stdevia_ao/12
    """

    windspeedw90 = round(stats.trim_mean(data_full["correctedwindspeed"].as_matrix(), 0.1) * 100) / 100
    fiftyyeargust = data_full["correctedwindspeed"].max().round(3)

    # Capture the current directory
    THIS_DIR = os.path.dirname(os.path.abspath(__file__))
    from weasyprint import HTML, CSS
    j2_env = Environment(loader=FileSystemLoader(THIS_DIR),
                         trim_blocks=True)

    template = j2_env.get_template('Templates/templatefioReportNOP90.html')
    template_vars = {"version": "Prospecting Met Mast v1.12",
                     "frontcover": "file://" + root_front_cover_path +
                                   "Templates/FrontOfReportWithInputsNoBorder_001.jpg",
                     "pmmreport": input_variables["PMM_REPORT"],
                     "location": input_variables["LOCATION"],
                     "monthyear": input_variables["MONTH_YEAR"],
                     "period": input_variables["PERIOD"],
                     "startyear": input_variables["START_YEAR"],
                     "endyear": input_variables["END_YEAR"],
                     "elevation": input_variables["ELEVATION"],
                     "hubheight": input_variables["HUB_HEIGHT"],
                     "surfaceroughness": input_variables["SURFACE_ROUGHNESS"],
                     "longwindspeed": round(data_full["correctedwindspeed"].mean() * 100) / 100,
                     "windspeedw90": windspeedw90,
                     "windspeedmedian": round(data_full["correctedwindspeed"].median() * 100) / 100,
                     "weibullscale": 0,
                     "weibullshape": 0,
                     "fiftyyeargust": fiftyyeargust,
                     "p50": pfifty,
                     "dailywind": diurnal,
                     "hist_month": hist_month_h,
                     "hist_sector": hist_sector_h,
                     "annualhist": annual_hist,
                     "metersclass": metersclass_o,
                     "meterspercent": meterspercent_o,
                     "sectorstotal": sectorstotal_o,
                     "meterstotal": meterstotal_o,
                     "rose_month": rose_month_r,
                     "annualrose": annual_rose,
                     "climatemeans": box1,
                     "monthdevia": line2,
                     "monthsA": ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                                 "Jul", "Aug", "Sep", "Oct", "Nov", "Dec", "Annual"],
                     "meanws": meanws_o.round(2),
                     "stdevia": stdevia_o.round(2),
                     "stderr": stderr_o.round(2),
                     "meanwsA": round(meanws_ao*1000)/1000,
                     "stdeviaA": round(stdevia_ao*1000)/1000,
                     "stderrA": round(stderr_ao*100)/100}
    html_out = template.render(template_vars)
    HTML(string=html_out).write_pdf(file_path + 'wiwasolvet' + input_variables["OUTPUT_FILE_NAME"] + '.pdf',
                                    stylesheets=[root_file_path + "Templates/reportpdf.css"])


if __name__ == '__main__':
    # print_html_doc()
    # Read .env environment variables, such as database credentials, API key, local/remote configuration details, etc.
    env_variables = {}
    env_file = ".env"
    with open(env_file) as file:
        for line in file:
            if line.startswith('#'):
                continue
            key, value = line.strip().split('=', 1)
            # env_variables.append({'name': key, 'value': value})
            env_variables[key] = value

    file_path = ""
    root_file_path = ""
    output_file_path = ""
    root_front_cover_path = ""
    if env_variables["LOCAL_FLAG"] == "True":
        file_path = env_variables["LOCAL_SAVE_ROOT"] + env_variables["LOCAL_TEMP_OUTPUT"]
        root_file_path = env_variables["LOCAL_SAVE_ROOT"]
        output_file_path = env_variables["LOCAL_WINDOWS_ROOT"] + env_variables["LOCAL_TEMP_OUTPUT"]
        root_front_cover_path = env_variables["LOCAL_WINDOWS_ROOT"]
    else:
        file_path = env_variables["REMOTE_ROOT"] + env_variables["REMOTE_TEMP_OUTPUT"]
        root_file_path = env_variables["REMOTE_ROOT"]
        output_file_path = file_path
        root_front_cover_path = root_file_path

    list_input = sys.argv[1:]
    input_file_name = list_input[0]
    # Read input.txt site parameters, such as location, site name, elevation, etc.
    input_variables = {}
    # input_file = ".env"
    with open(input_file_name) as file:
        for line in file:
            if line.startswith('#'):
                continue
            key, value = line.strip().split('=', 1)
            # env_variables.append({'name': key, 'value': value})
            input_variables[key] = value

    # ========================================================================
    api_key = env_variables["DARKSKY_API_KEY"]
    # ========================================================================

    easy_print_pdf(dbe.retrieve_database(env_variables, input_variables))

    # The callday() function costs money! Downloads new data from Darksky.net
    # Be sure that you are saving it in your database as not to have to pay twice
    # A good way to test if this works is to try a small date range at first, like 3-30 days worth
    # The first 1000 api calls in a day are free, but everything after that cost money.

    # vardb = DarkSky.callData.callday(44.6596,-63.5441,'1985-01-01', '2016-12-31', api_key=api_key)

    # Overwrites previous table based on downloaded data (raw data)
    # DatabaseEngine.databaseEngine.storedatabase(vardb, env_variables)

    # Print or eventually log data in browser when called, can update what gets printed. Large data will be...
    # compressed into ... blocks of text in long lists.
    # printdata2(vardb)
    # printdata2(dbe.retrievedatabase())

    # Creates a new database table based on downloaded data (raw data)
    # DatabaseEngine.databaseEngine.storedatabaseNEW(vardb,"site_004_44.6596_-63.5441_1985", env_variables)
