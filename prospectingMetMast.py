#!/home/wiwasol/miniconda3/envs/wiwapmm/bin python
# -*- coding: utf-8 -*-
#
# Using the file system load
#
# We now assume we have a file in the same dir as this one called
# test_template.html
#

# Note: Will remove excess import statements, this was from a major refactoring from a single Python file to broken down into main functional tasks.
# Note: Work in Progess, still refactoring!


from numpy.random import random_sample, weibull, randn
from jinja2 import Environment, FileSystemLoader
import os
from decimal import *
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

import numpy as np
from scipy import stats
from scipy.stats import weibull_min, weibull_max
import scipy as sp

# Capture our current directory
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
#from weasyprint import HTML, CSS

import pylab
from matplotlib.ticker import FuncFormatter
import scipy.special as spe

from windrose import WindroseAxes
import matplotlib.cm as cm
from matplotlib.legend_handler import HandlerLine2D
from numpy import arange
import math

import forecastio
import datetime
import numpy as np
import pandas as pd
import sys, json
from sys import argv
#from io import StringIO
#from forecastio.utils import UnicodeMixin, PropertyUnavailable
from sqlalchemy import Table, Column, Integer, String, Float, MetaData, ForeignKey
#io = StringIO()

import DarkSky.callData
import DatabaseEngine.databaseEngine as dbe

import Graphs.ClimateMeansBoxPlot
import Graphs.LineGraphErrorFill
import Graphs.WindHistogramAnnual
import Graphs.WindHistogramMonths
import Graphs.WindHistogramSectors
import Graphs.WindRoseAnnual
import Graphs.WindRoseMonths

font = {'family': 'serif',
        'color':  'black',
        'weight': 'normal',
        'size': 20,
        }
fontLine = {'color':  'black',
            'weight': 'normal',
            'size': 12,
            }


def to_percent(y, position, case):
    # Ignore the passed in position. This has the effect of scaling the default
    # tick locations.
    if case == 1:
        s = str(100 * y)

    if case == 2:
        s = str(100. * y)

    if case == 3:
        s = str(1 * y)

    if case == 4:
        s = str((100 * round(y * 100) / 100))

    # The percent symbol needs escaping in latex
    if matplotlib.rcParams['text.usetex'] is True:
        return s + r'$\%$'
    else:
        return s + '%'


formatter1 = FuncFormatter(to_percent, case=1)
formatter2 = FuncFormatter(to_percent, case=2)
formatter3 = FuncFormatter(to_percent, case=3)
formatter4 = FuncFormatter(to_percent, case=4)

# Formatting histogram plots
#colourb1 = '#408754'
colourb1 = '#6DCBD5'
colourk1 = '#40423F'
colourg1 = '#09694E'

# ========================================================================
api_key = 'ENTER YOUR DARKSKY API KEY HERE'
# ========================================================================


def map_bin(x, bins):
    # Write a short mapper that bins data
    kwargs = {}
    if x == max(bins):
        kwargs['right'] = True
    try:
        bin = bins[np.digitize([x], bins, **kwargs)[0]]
        #print("bin: " + str(bin))
    except IndexError:
        #print("binned ========")
        #print(x)
        #bin = bins[np.digitize([x], bins, **kwargs)[0]]
        # Causing IndexError?
        bin = 0
    try:
        bin_lower = bins[np.digitize([x], bins, **kwargs)[0]-1]
    except IndexError:
        bin_lower = bin
        #print("IndexError")
    # print("bin_lower: " + str(bin_lower))
    return '[{0}-{1}]'.format(bin_lower, bin)

"""
def month_bin(x, bins):
    kwargs = {}
    if x == max(bins):
        kwargs['right'] = True
    bin = bins[np.digitize([x], bins, **kwargs)[0]]
    bin_lower = bins[np.digitize([x], bins, **kwargs)[0]-1]
    return '[{0}-{1}]'.format(bin_lower, bin)
"""

# The callday() function costs money! Downloads new data from Darksky.net
# Be sure that you are saving it in your database as not to have to pay twice
# A good way to test if this works is to try a small date range at first, like 3-30 days worth
# The first 1000 api calls in a day are free, but everything after that cost money.

#vardb = DarkSky.callData.callday(44.6596,-63.5441,'1985-01-01', '2016-12-31', api_key=api_key)

# Overwrites previous table based on downloaded data (raw data)
#DatabaseEngine.databaseEngine.storedatabase(vardb)

# Print or eventually log data in browser when called, can update what gets printed. Large data will be...
# compressed into ... blocks of text in long lists.
#printdata2(vardb)
#printdata2(dbe.retrievedatabase())

# Creates a new database table based on downloaded data (raw data)
#DatabaseEngine.databaseEngine.storedatabaseNEW(vardb,"site_004_44.6596_-63.5441_1985")

# First attempt at automating purchase(s) to downloading, processing data and
# creating PDF of Prospecting Met Mast(s).
# PHP script was hanging on long running Python script (waiting to download 30 years of hourly data + processing)
"""
try:
    data = json.loads(sys.argv[1])
    #data = json.load( sys.stdin )
    #result = data['testname1']
    latt2 = data["latitude"]
    long2 = data["longitude"]
    ystart = data["datayearstart"]
    yend = data["datayearend"]
    #ystart = str(data["datayearstart"]) + '-01-01'
    #yend = str(data["datayearend"]) + '-12-31'
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
    #sys.exit(1)
"""

# ====================================================================================================================
# easyprintpdf(dbe.retrievedatabase())
# ====================================================================================================================

"""
try:
    data = json.loads(sys.argv[1])
    #data = json.load( sys.stdin )
    #result = data['testname1']
    latt2 = data["latitude"]
    long2 = data["longitude"]
    ystart = data["datayearstart"]
    yend = data["datayearend"]
    #version 1.01 is simply years, later it can have days/months
    #ystart = str(data["datayearstart"]) + '-01-01'
    #yend = str(data["datayearend"]) + '-12-31'
    ystartlong = str(data["datayearstart"]) + '-01-01'
    yendlong = str(data["datayearend"]) + '-12-31'
    orderid = data["orderNum"]
    #dbe.storedatabaseNEW(dbe.retrievedatabase(),"site_"+str(orderid)+"_"+str(latt2)+"_"+str(long2)+"_"+str(ystart))
    #vardbnew = DarkSky.callData.callday(str(latt2), str(long2), ystartlong, yendlong)
    #dbe.storedatabaseNEW(vardbnew,"site_"+str(orderid)+"_"+str(latt2)+"_"+str(long2)+"_"+str(ystart))
    #easyprintpdf(vardbnew)
except (ValueError, TypeError, IndexError, KeyError) as e:
    print(json.dumps({'error': str(e)}))
    latt2 = 0
    long2 = 0
    ystart = '2015-12-30'
    yend = '2015-12-31'
    orderid = 999
    #dbe.storedatabaseNEW(dbe.retrievedatabase(),"site_"+str(orderid)+"_"+str(latt2)+"_"+str(long2)+"_"+str(ystart))
    #sys.exit(1)
"""


def weib(x, *p):
    # Provide weibull parameters to sample within reasonable ranges
    XSsat, Lo, W, s = p
    return XSsat*(1-np.exp(-((x-Lo)/W)**s))


def easyprintpdf(dataFull, startdate='1985-01-01', enddate='2017-12-31'):
    # Datetime objects must be in consistent format
    dataFull['time'] = pd.to_datetime(dataFull['time'], format='%m/%d/%Y %H:%M')
    #dataFull['time'] = pd.to_datetime(dataFull['time'], format='%d/%m/%Y %H:%M')
    dataFull.set_index('time', inplace=True)
    #dataFull.dropna()
    dataFull.dropna(subset=['windd', 'correctedwindspeed'])

    formatter1 = FuncFormatter(to_percent, case=1)

    fig = plt.figure(figsize=(4.34, 3.22), dpi=100, facecolor='w', edgecolor='w')
    Graphs.WindRoseAnnual.new_axesRA(dataFull["windd"], dataFull["correctedwindspeed"], fontLine=fontLine)
    plt.savefig('/home/wiwasol/prospectingmm/WindRose_temp8888_img.png')
    annualRose = "file:///home/wiwasol/prospectingmm/WindRose_temp8888_img.png"
    plt.clf()
    plt.close()

    Graphs.WindHistogramAnnual.new_axesHA(dataFull)
    plt.savefig('/home/wiwasol/prospectingmm/WindHistogram_temp8888_img.png')
    annualHist = "file:///home/wiwasol/prospectingmm/WindHistogram_temp8888_img.png"
    plt.clf()
    plt.close()

    xaxis2 = ["01", "02", "03", "04", "05", "06", "07", "08", "09", "10", "11", "12"]
    xaxisMon = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

    dataMonthGroup = dataFull.groupby(lambda x: x.month)
    textMon1 = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

    for iMon, group in enumerate(dataMonthGroup, 0):
        Graphs.WindRoseMonths.new_axesR(group[1].windd, group[1].correctedwindspeed, xaxisMon[iMon], font=font)
        plt.savefig('/home/wiwasol/prospectingmm/weiWRose_' + xaxis2[iMon] + '_temp888_img.png')

        #add folder for each new report and image files?! Good way to keep records, and check if errors crop up

        Graphs.WindHistogramMonths.new_axesH_Mon(group[1].correctedwindspeed, xaxisMon[iMon], xaxis2[iMon], iMon, textMon1, formatter1=formatter1, colourb1=colourb1)

    freq_bins2 = [0, 30, 60, 90, 120, 150, 180, 210, 240, 270, 300, 330, 360]
    # Was this changed because of a windrose error/update? From 12 to 13 array length/sectors of windrose
    dataWindd = dataFull.copy().dropna(axis=0, how='any', subset=['winds', 'windd', 'correctedwindspeed'])
    # TODO: Attempts at fixing index out of range error (seemed to work with only 2 months?), but not with 12-24 etc
    dataWindd['Binned'] = dataWindd['windd'].apply(map_bin, bins=freq_bins2)
    grouped = dataWindd[['windd', 'correctedwindspeed', 'Binned']].groupby('Binned')

    xaxisSec = ['30', '60', '90', '120', '150', '180', '210', '240', '270', '300', '330', '360']
    binsH=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 50]
    binsLH=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25]
    columnsB = ['[0-30]', '[30-60]', '[60-90]', '[90-120]', '[120-150]', '[150-180]',
                '[180-210]', '[210-240]', '[240-270]', '[270-300]', '[300-330]', '[330-360]']
    dataprint = (dataWindd.assign(
        q=pd.cut(np.clip(dataWindd.correctedwindspeed, binsH[0], binsH[-1]), bins=binsH, labels=binsLH, right=False))
        .pivot_table(index='q', columns='Binned', aggfunc='size', fill_value=0)
        )

    textSec4 = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    textSec5 = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    textSec6 = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    shapeSec = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    locSec = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    scaleSec = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

    for iSec, group2 in enumerate(grouped, 0):
        Graphs.WindHistogramSectors.new_axesH_Sec(group2[1].correctedwindspeed, xaxisSec[iSec], xaxis2[iSec], iSec, textSec4, textSec5, textSec6, shapeSec, locSec, scaleSec, formatter1=formatter1, colourb1=colourb1)

    count = 0
    rose_monthR = [[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]]
    hist_monthH = [[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]]
    hist_sectorH = [[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]]
    for i in range(4):
        for j in range(3):
            rose_monthR[i][j] = 'file:///home/wiwasol/prospectingmm/weiWRose_' + xaxis2[count] + '_temp888_img.png'
            hist_monthH[i][j] = 'file:///home/wiwasol/prospectingmm/weibull_' + xaxis2[count] + '_temp999_img.png'
            hist_sectorH[i][j] = 'file:///home/wiwasol/prospectingmm/weibullSEC_' + xaxis2[count] + '_temp999_img.png'
            count = count + 1

    # Climate Means Fig 8 ===============================================================
    plt.clf()
    plt.close()
    fig = plt.figure(figsize=(7, 4.17), dpi=120, facecolor='w', edgecolor='w')
    Graphs.ClimateMeansBoxPlot.meanBoxplot(dataFull, formatter4=formatter4, colourb1=colourb1)
    plt.savefig('/home/wiwasol/prospectingmm/box1_temp444_img.png')
    box1 = "file:///home/wiwasol/prospectingmm/box1_temp444_img.png"

    # Diurnal Fig 2 ===============================================================
    diurnalm = dataFull['correctedwindspeed'].groupby(lambda x: x.hour).mean().as_matrix()
    diurnals = dataFull['correctedwindspeed'].groupby(lambda x: x.hour).std().as_matrix()

    xaxis2b = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23]
    xaxisHours = ["0:00", "1:00", "2:00", "3:00", "4:00", "5:00", "6:00", "7:00", "8:00", "9:00", "10:00", "11:00",
                  "12:00", "13:00", "14:00", "15:00", "16:00", "17:00", "18:00", "19:00", "20:00", "21:00", "22:00",
                  "23:00"]
    fig = plt.figure(figsize=(7, 4.04), dpi=120, facecolor='w', edgecolor='w')
    #plt.plot(xaxis2b, diurnalm, color=colourb1, linewidth=3)
    #errorfill(xaxis2a, yline2, yerr=error2, color=colourg1, alpha_fill=0.2)
    Graphs.LineGraphErrorFill.errorfill(xaxis2b, diurnalm, yerr=diurnals, color=colourg1, alpha_fill=0.2)
    plt.xticks(xaxis2b, xaxisHours, rotation=45)
    plt.margins(0.05)
    dmaxm = (diurnals+diurnalm).max()+1

    ax3 = fig.add_subplot(1, 1, 1)

    # major ticks every 10, minor ticks every 5
    major_ticks3 = np.arange(0, dmaxm, 2)
    minor_ticks3 = np.arange(0, dmaxm, 1)

    ax3.set_yticks(major_ticks3)
    ax3.set_yticks(minor_ticks3, minor=True)

    # and a corresponding grid
    #ax.grid(which='both')

    # or if you want different settings for the grids:
    ax3.grid(which='minor', alpha=0.4)
    ax3.grid(which='major', alpha=0.5)

    plt.ylabel('Mean Wind Speed m/s')
    plt.xlabel('Figure 2: Daily Variation in Average Hourly Wind Speeds')
    plt.tight_layout()
    plt.savefig('/home/wiwasol/prospectingmm/line4_temp555_img.png')
    diurnal = "file:///home/wiwasol/prospectingmm/line4_temp555_img.png"

    # Monthly Mean, Std, StdErr Fig 9 and Table 2 ===============================================================
    meanYearM = dataFull['correctedwindspeed'].groupby(lambda x: x.year).mean()

    meanwsO = dataFull['correctedwindspeed'].groupby(lambda x: x.month).mean().as_matrix()
    stdeviaO = dataFull['correctedwindspeed'].groupby(lambda x: x.month).std().as_matrix()
    stderrO = dataFull['correctedwindspeed'].groupby(lambda x: x.month).std().as_matrix()/math.sqrt(len(meanYearM)-1)
    meanwsAO = Decimal(0.00)
    stdeviaAO = Decimal(0.00)
    stderrAO = Decimal(0.00)

    for k in range(12):
        meanwsAO += Decimal(meanwsO[k])
        stdeviaAO += Decimal(stdeviaO[k])
        stderrAO += Decimal(stderrO[k])

    meanwsAO = meanwsAO/12
    stdeviaAO = stdeviaAO/12
    stderrAO = stderrAO/12
    xaxis2a = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
    xaxisMon = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

    # error bar values w/ different -/+ errors
    #lower_error = error2
    #upper_error = 0.9*error2
    #asymmetric_error = [lower_error, upper_error]
    fig = plt.figure(figsize=(7, 4.04), dpi=120, facecolor='w', edgecolor='w')
    Graphs.LineGraphErrorFill.errorfill(xaxis2a, meanwsO, yerr=stdeviaO, color=colourg1, alpha_fill=0.2)
    #plt.grid(True)
    dmaxO = (meanwsO+stdeviaO).max()+5

    ax2 = fig.add_subplot(1, 1, 1)

    # major ticks every 10, minor ticks every 5
    major_ticks2 = np.arange(0, dmaxO, 2)
    minor_ticks2 = np.arange(0, dmaxO, 1)

    plt.xticks(xaxis2a, xaxisMon)
    ax2.set_yticks(major_ticks2)
    ax2.set_yticks(minor_ticks2, minor=True)

    # and a corresponding grid
    #ax.grid(which='both')

    # or if you want different settings for the grids:
    ax2.grid(which='minor', alpha=0.4)
    ax2.grid(which='major', alpha=0.5)

    plt.ylabel('Mean Wind Speed m/s')
    plt.xlabel('Figure 9: Monthly mean wind speed with 16th and 84th percentiles', fontdict=fontLine)
    plt.tight_layout()
    plt.savefig('/home/wiwasol/prospectingmm/line2_temp555_img.png')
    line2 = "file:///home/wiwasol/prospectingmm/line2_temp555_img.png"

    # P50:P90 Fig 1 ===============================================================
    p50breakdown = Graphs.WindHistogramMonths.new_axesH_MonCDF(dataMonthGroup, xaxisMon)
    yy50_5ms = p50breakdown[0]
    yy90_10ms = p50breakdown[1]
    yy99_15ms = p50breakdown[2]

    fig = plt.figure(figsize=(7, 4.04), dpi=120, facecolor='w', edgecolor='w')

    ym50 = int(yy50_5ms.max())
    ym90 = int(yy90_10ms.max())
    ym99 = int(yy99_15ms.max())
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

    plt.xticks(xaxis2a, xaxisMon)
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
    #ax.grid(which='both')

    # or if you want different settings for the grids:
    #ax.grid(which='minor', alpha=0.4)
    ax.grid(which='major', alpha=0.5)

    plt.gca().yaxis.set_major_formatter(formatter3)
    plt.tight_layout()
    plt.savefig('/home/wiwasol/prospectingmm/line3_temp555_img.png')
    pfifty = "file:///home/wiwasol/prospectingmm/line3_temp555_img.png"

    # Wind Frequency Distribution TABLE===========================
    metersclassO = []
    totalCO = Decimal(000.00)
    totalRO = Decimal(000.00)
    for t in range(24):
        metersclassO.append(str(t) + "-" + str(int(t+1)) + "m/s")
    else:
        metersclassO.append("&ge;&nbsp;24m/s")

    meterspercentO = np.zeros((25, 12))
    sumdata = 0
    for k in range(12):
        for i in range(25):
            #sumdata += dataprint[columnsB[k]].count()
            try:
                #print("I==============")
                #print(dataprint.iloc[i])
                sumdata += dataprint.iloc[i][columnsB[k]]
            except IndexError:
                #sumdata
                print("IndexError2")
                #print(dataprint.iloc[i][columnsB[k]])
            except KeyError:
                print("KeyError2")

    for k in range(12):
        for i in range(25):
            #meterspercentO[i][k] = dataprint[i][columnsB[k]]/sumdata
            #meterspercentO[i][k] = dataprint.iloc[i][columnsB[k]]/sumdata
            ########meterspercentO[i][k] = (dataprint.iloc[i][columnsB[k]]/sumdata)*100
            try:
                #print("M==============")
                #print(dataprint.iloc[i])
                #sumdata += dataprint.iloc[i][columnsB[k]]
                meterspercentO[i][k] = (dataprint.iloc[i][columnsB[k]] / sumdata) * 100
            except IndexError:
                # sumdata
                print("IndexError3")
                meterspercentO[i][k] = 0.0
                # print(dataprint.iloc[i][columnsB[k]])
            except KeyError:
                print("KeyError3")
                meterspercentO[i][k] = 0.0
    #print(meterspercentO[0][0])
    """
    meterspercentO[1] = dataprint[columnsB[1]]/sumdata
    meterspercentO[2] = dataprint[columnsB[2]]/sumdata
    meterspercentO[3] = dataprint[columnsB[3]]/sumdata
    meterspercentO[4] = dataprint[columnsB[4]]/sumdata
    meterspercentO[5] = dataprint[columnsB[5]]/sumdata
    meterspercentO[6] = dataprint[columnsB[6]]/sumdata
    meterspercentO[7] = dataprint[columnsB[7]]/sumdata
    meterspercentO[8] = dataprint[columnsB[8]]/sumdata
    meterspercentO[9] = dataprint[columnsB[9]]/sumdata
    meterspercentO[10] = dataprint[columnsB[10]]/sumdata
    meterspercentO[11] = dataprint[columnsB[11]]/sumdata
    """
    sectorstotalO = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    meterstotalO = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

    # 10 years
    #meterspercentO = weibull(2*np.random.random_sample()+1.2, (25, 12))
    # used to have abs() surrounding each of these, useful for random data
    getcontext().prec = 3
    for k in range(12):
        for i in range(25):
            totalRO += 100*Decimal(meterspercentO[i][k])/100
    for i in range(25):
        for k in range(12):
            totalCO += 100*Decimal(meterspercentO[i][k])/100
    for k in range(12):
        for i in range(25):
            sectorstotalO[k] += 100*Decimal(meterspercentO[i][k])/100
    for i in range(25):
        for k in range(12):
            meterstotalO[i] += 100*Decimal(meterspercentO[i][k])/100
    getcontext().prec = 2
    # translates float to 2 place decimal
    for i in range(25):
        for k in range(12):
            meterspercentO[i][k] = 1000*Decimal(meterspercentO[i][k])/1000


    #for i in range(25):
    #    for k in range(12):
    #        meterspercentO[i][k] = Decimal(meterspercentO[i][k])
    #x1 = 10 * weibull(2*np.random.random_sample()+1, 200)
    #h1 = np.histogram(x1, bins=25, density=True)
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
    meanwsO = 4.5*random_sample((12, ))+4
    stdeviaO = 2.7*random_sample((12, ))+1.2
    #meanwsO.append(0)
    #stdeviaO.append(0)
    meanwsAO = Decimal(0.00)
    stdeviaAO = Decimal(0.00)
    for k in range(12):
        #if(k <=12):
        meanwsO[k] = 10*Decimal(meanwsO[k])/10
        stdeviaO[k] = 10*Decimal(stdeviaO[k])/10
        meanwsAO += Decimal(meanwsO[k])
        stdeviaAO += Decimal(stdeviaO[k])
        #else:
        #meanwsO[k] = 10*Decimal(meanwsO[k])/10
        #stdeviaO[k] = 10*Decimal(stdeviaO[k])/10
        #meanwsAO += Decimal(meanwsO[k])
        #stdeviaAO += Decimal(stdeviaO[k])
    meanwsAO = meanwsAO/12
    stdeviaAO = stdeviaAO/12
    """

    windspeedw90 = round(stats.trim_mean(dataFull["correctedwindspeed"].as_matrix(), 0.1)*100)/100
    fiftyyeargust = dataFull["correctedwindspeed"].max().round(3)

    # Capture the current directory
    THIS_DIR = os.path.dirname(os.path.abspath(__file__))
    from weasyprint import HTML, CSS
    j2_env = Environment(loader=FileSystemLoader(THIS_DIR),
                         trim_blocks=True)

    template = j2_env.get_template('templatefioReportNOP90.html')
    template_vars = {"title": "Prospecting Met Mast v1.11",
                     "pmmreport": "Halifax, NS",
                     "location": "46.054, -60.329",
                     "monthyear": "December 31st, 2017",
                     "period": "32",
                     "startyear": "1985",
                     "endyear": "2017",
                     "elevation": "10m",
                     "hubheight": "11m",
                     "surfaceroughness": "0.4",
                     "longwindspeed": round(dataFull["correctedwindspeed"].mean()*100)/100,
                     "windspeedw90": windspeedw90,
                     "windspeedmedian": round(dataFull["correctedwindspeed"].median()*100)/100,
                     "weibullscale": 0,
                     "weibullshape": 0,
                     "fiftyyeargust": fiftyyeargust,
                     "p50": pfifty,
                     "dailywind": diurnal,
                     "hist_month": hist_monthH,
                     "hist_sector": hist_sectorH,
                     "annualhist": annualHist,
                     "metersclass": metersclassO,
                     "meterspercent": meterspercentO,
                     "sectorstotal": sectorstotalO,
                     "meterstotal": meterstotalO,
                     "rose_month": rose_monthR,
                     "annualrose": annualRose,
                     "climatemeans": box1,
                     "monthdevia": line2,
                     "monthsA": ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec", "Annual"],
                     "meanws": meanwsO.round(2),
                     "stdevia": stdeviaO.round(2),
                     "stderr": stderrO.round(2),
                     "meanwsA": round(meanwsAO*1000)/1000,
                     "stdeviaA": round(stdeviaAO*1000)/1000,
                     "stderrA": round(stderrAO*100)/100}
    html_out = template.render(template_vars)
    HTML(string=html_out).write_pdf('/home/wiwasol/prospectingmm/wiwasolvet-p7-website8.pdf', stylesheets=["/home/wiwasol/prospectingmm/reportpdf.css"])

# ====================================================================================================================
easyprintpdf(dbe.retrievedatabase())
# ====================================================================================================================

#if __name__ == '__main__':
#    print_html_doc()
