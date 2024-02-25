#!/home/wiwasol/miniconda3/envs/wiwapmm/bin python
# -*- coding: utf-8 -*-
#
# Using the file system load
#
# We now assume we have a file in the same dir as this one called
# test_template.html
#
# Author: Jacob Thompson
# Date revised: 2023-01-05
# Date created: 2016-08-30
# Company: Wiwasolvet Total Primary Energy Solutions
# License: GPL-3.0 license
# Version: 1.12
# Data Version: DarkSky API
# TODO: Migrate to WeatherKit API
# TODO: Need to remove some hardcoding
# Use .env file to get working

import pandas
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
#formating histogram plots
#colourb1 = '#408754'
colourb1 = '#6DCBD5'
colourk1 = '#40423F'
colourg1 = '#09694E'

#========================================================================
# TODO: get new imports working sending requests on multiple cores
from joblib import Parallel, delayed
from multiprocessing import cpu_count, Pool
import time
#========================================================================
import forecastio
import datetime
import pytz
import numpy as np
import pandas as pd
import sys, json
from sys import argv
#from io import StringIO
####from forecastio.utils import UnicodeMixin, PropertyUnavailable
from sqlalchemy import Table, Column, Integer, String, Float, MetaData, ForeignKey
#io = StringIO()

#print(json.dumps(result))

api_key = ''




#latt2 = 44.6566
#long2 = -63.5963
#orderid = 123
#latt2 = data["latitude"]
#long2 = data["longitude"]
#orderid = data["orderNum"]
#datetime.datetime(2015, 1, 1, 12, 0, 0) year, month, day, hour, minute, second
#years requested / will need to set loop to make multiple api calls for each either daily or hourly request
#timerange = 1
#timeblock0 = datetime.datetime(2015, 1, 1, 0, 0, 0)

#time = datetime.datetime(2015, 1, 1, 1, 0, 0)

#rng = pd.date_range('1980-01-01', periods=365, freq='d')
#rngYears = pd.date_range('1985-01-01', '2015-12-31', freq='d')
#rngYears = pd.date_range('2015-01-01', '2015-12-31', freq='d')

#print(rng)
#time2 = "01/01/2015 00:00:00"
#def new_axesH():
def callback(forecastio=None):
    return forecastio

def callday(latt2, long2, startdate, enddate):
    #dataFull = pd.DataFrame({},columns=["time","winds","windd"])
    dataFull = pd.DataFrame({})
    #dataF24 = []
    #pre_timeA = pd.Series(0, index=np.arange(24))
    #pre_windsA = pd.Series(0, index=np.arange(24))
    #pre_winddA = pd.Series(0, index=np.arange(24))
    #dataFull = pd.concat([pre_timeA,pre_windsA,pre_winddA], axis=1)
    #rngYears = pd.date_range('2010-01-01', '2015-12-31', freq='d')
    rngYears = pd.date_range(startdate, enddate, freq='d')
    datalength = rngYears.size
    dataF24 = [i for i in range(datalength)]

    #for i, hourlyData in enumerate(byHour.data) :
    #times = []
    #data = {}
    #attributes = ["windSpeed", "windBearing"]
    #for attr in attributes:
    #    data[attr] = []
    #print(rngYears)
    #print(datalength)
    #print(dataF24)
    for v, day in enumerate(rngYears, 0):
        #time = datetime.datetime(rngYears[v])
        url_time = rngYears[v].replace(microsecond=0).isoformat()
        #print(time +" , "+url_time)
        #print(str(rngYears[v]) + ' , ' + url_time)
        #print(url_time)
        #print("<br><br>")
        #datetime.strptime(date_string, format)
        #fio [YYYY]-[MM]-[DD]T[HH]:[MM]:[SS]
        #timeblock0 = datetime.datetime.strptime(time2, '%Y-%m-%dT%H:%M:%S')
        # ====================
        # TODO: doesn't do anything useful, since request already uses local timezone based on lat/long
        #without_timezone = datetime.datetime.strptime(url_time, '%Y-%m-%dT%H:%M:%S')
        #timezone = pytz.timezone("America/Halifax")
        #with_timezone = timezone.localize(without_timezone)
        #print(with_timezone)
        #url_time3 = datetime.datetime.strftime(with_timezone, '%Y-%m-%dT%H:%M:%S%z')
        #print(url_time3)
        # ====================
        units = 'si'
        #exclude = 'flags,currently,minutely,hourly,alerts'
        exclude0 = 'minutely,currently,daily,alerts'
        #'minutely,currently,hourly,daily,alerts,flags'
        #url = 'https://api.forecast.io/forecast/%s/%s,%s,%s' \
        #              '?units=%s' % (key, lat, lng, url_time,
        #              units,)
        #url = 'https://api.forecast.io/forecast/%s/%s,%s,time=%s?units=%s&exclude=%s' % (api_key, latt, long2, url_time, units, exclude0)

        url = 'https://api.darksky.net/forecast/%s/%s,%s,%s?units=%s&exclude=%s' % (api_key, latt2, long2, url_time, units, exclude0)
        #forecast = forecastio.load_forecast(api_key,latt,long2,time=timeblock0,units,exclude=exclude0)
        #forecast = forecastio.manual(url)
        forecastCallstart = forecastio.manual(url,callback(forecastio=None))
        ####forecast = callback()
        #because of daylight savings time, won't always be 24 hours of data! Could be 23, or 24, or 25
        #responseH = forecast.http_headers()
        ####byHour = forecast.hourly()
        byHour = forecastCallstart.hourly()
        """
        time
        precipIntensity
        precipProbability
        precipType
        temperature
        humidity (convert to density later)
        wind speed
        wind bearing
        cloud cover
        pressure
        """
        #this is a bad way to do it! Caused a lot of 0 rows across the board!!!!!
        #pre_time = pd.Series(0, index=np.arange(24))
        #rngYears = pd.date_range(startdate, enddate, freq='d')
        #index = pd.date_range(startdate+datetime.timedelta(v), freq='h')
        #print(str(index)+" index_"+v)
        index = len(byHour.data)
        #print(byHour.data)
        #print(index)
        pre_time = pd.Series(0, index=np.arange(index))
        pre_winds = pd.Series(0, index=np.arange(index))
        pre_windd = pd.Series(0, index=np.arange(index))
        pre_preint = pd.Series(0, index=np.arange(index))
        pre_prepro = pd.Series(0, index=np.arange(index))
        pre_pretyp = pd.Series(0, index=np.arange(index))
        pre_temp = pd.Series(0, index=np.arange(index))
        pre_humi = pd.Series(0, index=np.arange(index))
        pre_cloud = pd.Series(0, index=np.arange(index))
        pre_pres = pd.Series(0, index=np.arange(index))
        pre_windg = pd.Series(0, index=np.arange(index))
        pre_dew = pd.Series(0, index=np.arange(index))
        pre_uv = pd.Series(0, index=np.arange(index))
        pre_ozone = pd.Series(0, index=np.arange(index))
        pre_viz = pd.Series(0, index=np.arange(index))
        pre_storm = pd.Series(0, index=np.arange(index))
        pre_apptemp = pd.Series(0, index=np.arange(index))
        pre_lat = pd.Series(latt2, index=np.arange(index))
        pre_long = pd.Series(long2, index=np.arange(index))

        for i, hourlyData in enumerate(byHour.data, 0):

            try:
                # pre_time.iloc[i] = hourlyData.time
                pre_time.iloc[i] = datetime.datetime.strftime(hourlyData.time, '%Y-%m-%dT%H:%M:%S')
            except Exception:
                pre_time.iloc[i] = None
            try:
                pre_winds.iloc[i] = hourlyData.windSpeed
            except KeyError:
                pre_winds.iloc[i] = None
            except Exception:
                pre_winds.iloc[i] = None

            try:
                pre_windd.iloc[i] = hourlyData.windBearing
            except KeyError:
                pre_windd.iloc[i] = None
            except Exception:
                pre_windd.iloc[i] = None

            try:
                pre_preint.iloc[i] = hourlyData.precipIntensity
            except KeyError:
                pre_preint.iloc[i] = None
            except Exception:
                pre_preint.iloc[i] = None

            try:
                pre_prepro.iloc[i] = hourlyData.precipProbability
            except KeyError:
                pre_prepro.iloc[i] = None
            except Exception:
                pre_prepro.iloc[i] = None

            try:
                pre_pretyp.iloc[i] = hourlyData.precipType
            except KeyError:
                pre_pretyp.iloc[i] = None
            except Exception:
                pre_pretyp.iloc[i] = None

            try:
                pre_temp.iloc[i] = hourlyData.temperature
            except KeyError:
                pre_temp.iloc[i] = None
            except Exception:
                pre_temp.iloc[i] = None

            try:
                pre_apptemp.iloc[i] = hourlyData.apparentTemperature
            except KeyError:
                pre_apptemp.iloc[i] = None
            except Exception:
                pre_apptemp.iloc[i] = None

            try:
                pre_humi.iloc[i] = hourlyData.humidity
            except KeyError:
                pre_humi.iloc[i] = None
            except Exception:
                pre_humi.iloc[i] = None

            try:
                pre_cloud.iloc[i] = hourlyData.cloudCover
            except KeyError:
                pre_cloud.iloc[i] = None
            except Exception:
                pre_cloud.iloc[i] = None

            try:
                pre_pres.iloc[i] = hourlyData.pressure
            except KeyError:
                pre_pres.iloc[i] = None
            except Exception:
                pre_pres.iloc[i] = None

            try:
                pre_windg.iloc[i] = hourlyData.windGust
            except KeyError:
                pre_windg.iloc[i] = None
            except Exception:
                pre_windg.iloc[i] = None
            try:
                pre_dew.iloc[i] = hourlyData.dewPoint
            except KeyError:
                pre_dew.iloc[i] = None
            except Exception:
                pre_dew.iloc[i] = None
            try:
                pre_uv.iloc[i] = hourlyData.uvIndex
            except KeyError:
                pre_uv.iloc[i] = None
            except Exception:
                pre_uv.iloc[i] = None
            try:
                pre_ozone.iloc[i] = hourlyData.ozone
            except KeyError:
                pre_ozone.iloc[i] = None
            except Exception:
                pre_ozone.iloc[i] = None
            try:
                pre_viz.iloc[i] = hourlyData.visibility
            except KeyError:
                pre_viz.iloc[i] = None
            except Exception:
                pre_viz.iloc[i] = None
            try:
                pre_storm.iloc[i] = hourlyData.nearestStormDistance
            except KeyError:
                pre_storm.iloc[i] = None
            except Exception:
                pre_storm.iloc[i] = None


        #####dataF24[v] = pd.concat([pre_time,pre_winds,pre_windd], axis=1)
        dataF24[v] = pd.concat([pre_time, pre_winds, pre_windd, pre_windg, pre_preint, pre_prepro, pre_pretyp, pre_temp,
                                pre_apptemp, pre_humi, pre_cloud, pre_pres, pre_dew, pre_uv, pre_ozone, pre_viz,
                                pre_storm, pre_lat, pre_long], axis=1)

        pre_time, pre_winds, pre_windd, pre_windg, pre_preint, pre_prepro, pre_pretyp, pre_temp, pre_apptemp, pre_humi, pre_cloud, pre_pres, pre_dew, pre_uv, pre_ozone, pre_viz, pre_storm
        #dataF24 = pd.DataFrame(dataF24b, columns=["time","winds","windd"])
        #dataFull = pd.DataFrame(byHour.data[0].d)
        #print(dataF24[v])
        #print("<br>========================<br>")
        #print(dataFull)
    #print(dataFull)
    dataFull = pd.concat(dataF24,axis=0)
    ####dataFull.columns = ["time","winds","windd"]
    dataFull.columns = ["time", "winds", "windd", "windg", "preint", "prepro", "pretyp", "temp",
                        "apptemp", "humid", "cloud", "press", "dew", "uv", "ozone", "viz", "nStormDist", "lat", "long"]
    #dataFull['time'] = pd.to_datetime(dataFull['time'], format='%m/%d/%Y %H:%M')
    dataFull["time"] = pd.to_datetime(dataFull["time"], format='%Y-%m-%dT%H:%M:%S')
    #print(dataFull["time"])
    dataFull.set_index("time", inplace=True)

    #dataFull.tz_localize("UTC").tz_convert("America/Halifax")
    #dataFull.tz_localize("America/Halifax").tz_convert("UTC")
    # This should be the standard method of applying datetimes
    #dataFull.index = dataFull.index.strftime('%Y-%m-%dT%H:%M:%SZ%z')
    dataFull.index = dataFull.index.strftime('%Y-%m-%dT%H:%M:%SZ')
    # Excel compatible datetime, worked for old method of manually working with CSV.
    #dataFull.index = dataFull.index.strftime('%m/%d/%Y %H:%M')

    #dataFull.tz_localize("UTC").tz_convert("US/Eastern")
    #print(dataFull.index)
    return dataFull
def storedatabase(dataFull):
    import pymysql
    from sqlalchemy import create_engine

    #engine = create_engine('mysql+mysqlconnector://[user]:[pass]@[host]:[port]/[schema]', echo=False)
    #data.to_sql(name='sample_table2', con=engine, if_exists = 'append', index=False)

    #engine = create_engine('mysql+pymysql://wiwasol_reboot:[pass]@localhost:3306/admin_wordpress2016', echo=False)
    engine = create_engine('mysql+pymysql://wiwasol_reboot:[pass]@localhost:3306/wiwasol_datapoints2017', echo=False)
    dataFull.to_sql(name='wp_trials', con=engine, if_exists = 'append', index=False)
def storedatabaseNEW(dataFull,vartablen):
    import pymysql
    from sqlalchemy import create_engine

    #engine = create_engine('mysql+mysqlconnector://[user]:[pass]@[host]:[port]/[schema]', echo=False)
    #data.to_sql(name='sample_table2', con=engine, if_exists = 'append', index=False)

    #engine = create_engine('mysql+pymysql://wiwasol_reboot:[pass]@localhost:3306/admin_wordpress2016', echo=False)
    ####engine = create_engine('mysql+pymysql://wiwasol_reboot:[pass]@localhost:3306/wiwasol_datapoints2017', echo=False)
    ####dataFull.to_sql(name='wp_trials', con=engine, if_exists = 'append', index=False)
    #####
    """
    from sqlalchemy.sql import text
    s = text(
         "SELECT users.fullname || ', ' || addresses.email_address AS title "
             "FROM users, addresses "
             "WHERE users.id = addresses.user_id "
             "AND users.name BETWEEN :x AND :y "
             "AND (addresses.email_address LIKE :e1 "
                 "OR addresses.email_address LIKE :e2)")
    conn.execute(s, x='m', y='z', e1='%@aol.com', e2='%@msn.com').fetchall()
    [(u'Wendy Williams, wendy@aol.com',)]
    """
    # VARCHAR -> String, Float, tinytext
    #from sqlalchemy import Table, Column, String, Float, MetaData, ForeignKey
    ####from sqlalchemy.dialects.mysql import TINYTEXT, VARCHAR
    ####from sqlalchemy import Table, Column, Integer, String, Float, MetaData, ForeignKey

    metadata = MetaData()
    """
    wp_testit3 = Table('wp_testit3', metadata,
         #Column('id', Integer, primary_key=True),
         Column('time', String(19), nullable=True),
         Column('winds', Float, nullable=True),
         Column('windd', Float, nullable=True),
         Column('preint', Float, nullable=True),
         Column('prepro', Float, nullable=True),
         Column('pretyp', String(256), nullable=True),
         Column('temp', Float, nullable=True),
         Column('humid', Float, nullable=True),
         Column('cloud', Float, nullable=True),
         Column('press', Float, nullable=True)
         #mysql_charset='utf8',
    )
    """
    vartable = vartablen
    #vartable = "wp_testit4"
    get_table_object(vartable,metadata)
    #from sqlalchemy import create_engine
    #e = create_engine("mysql://scott:tiger@localhost/test", echo=True)
    engine = create_engine('mysql+pymysql://wiwasol_reboot:[pass]@localhost:3306/wiwasol_datapoints2017', echo=True)
    metadata.create_all(engine)
    #engine = create_engine('mysql+pymysql://wiwasol_reboot:[pass]@localhost:3306/wiwasol_datapoints2017', echo=False)
    #dataFull.to_sql(name='wp_testit3', con=engine, if_exists = 'append', index=False)
    dataFull.to_sql(name=vartable, con=engine, if_exists = 'append', index=False)
    """

    ##separate
    from sqlalchemy.dialects.mysql import \
            BIGINT, BINARY, BIT, BLOB, BOOLEAN, CHAR, DATE, \
            DATETIME, DECIMAL, DECIMAL, DOUBLE, ENUM, FLOAT, INTEGER, \
            LONGBLOB, LONGTEXT, MEDIUMBLOB, MEDIUMINT, MEDIUMTEXT, NCHAR, \
            NUMERIC, NVARCHAR, REAL, SET, SMALLINT, TEXT, TIME, TIMESTAMP, \
            TINYBLOB, TINYINT, TINYTEXT, VARBINARY, VARCHAR, YEAR
    """
def get_table_object(vartable, metadata):
    #metadata = MetaData()
    #table_name = 'table_' + vartable
    table_name = vartable
    table_object = Table(table_name, metadata,
        #Column('id', Integer, primary_key=True),
        Column('time', String(19), nullable=True),
        Column('winds', Float, nullable=True),
        Column('windd', Float, nullable=True),
        Column('preint', Float, nullable=True),
        Column('prepro', Float, nullable=True),
        Column('pretyp', String(256), nullable=True),
        Column('temp', Float, nullable=True),
        Column('humid', Float, nullable=True),
        Column('cloud', Float, nullable=True),
        Column('press', Float, nullable=True),
        Column('correctedwindspeed', Float, nullable=True)
        #mysql_charset='utf8',
    )
    #clear_mappers()
    #mapper(ActualTableObject, table_object)
    #return ActualTableObject
    return table_object

def retrievedatabase():
    import pymysql
    from sqlalchemy import create_engine
    #dataF = pd.read_sql(session.query(wp_trial).filter(wp_trial.id == 0).statement,session.bind)
    #dataF = pd.read_sql(session.query(wp_trial).query.all(),session.bind)
    #engine = create_engine('mysql+pymysql://wiwasol_reboot:[pass]@localhost:3306/admin_wordpress2016', echo=False)
    engine = create_engine('mysql+pymysql://wiwasol_reboot:[pass]@localhost:3306/wiwasol_datapoints2017', echo=False)
    con = engine.connect()
    rs = con.execute("SELECT * FROM `site_012_44.6566_-63.5963_1990`", index_col=0)
    dataFull = pd.DataFrame(rs.fetchall())
    dataFull.columns = rs.keys()
    con.close()
    return dataFull

def retrievedatabase_csv():
    sitefile_directory_name = "./output/PEI/"

    filepaths = [sitefile_directory_name + f for f in os.listdir(sitefile_directory_name) if f.endswith('.csv')]
    wind_site_df = pd.concat(map(pd.read_csv, filepaths), ignore_index=True)

    # TODO: included Excel math with correctedwindspeed
    wind_site_df['correctedwindspeed'] = wind_site_df.loc[:, 'winds']
    #print(wind_site_df['correctedwindspeed'])

    # wind_site_df = pd.read_csv("./output/PEI/weather_1990.csv")

    return wind_site_df

def printdata(dataFull):
    #print(dataFull["winds"].to_string(index=False))
    #print("<br>========================<br>========================<br>")
    print(dataFull)
    #print(dataFull["time"])

    print("<br>")
    print(dataFull["winds"].to_string(index=False))
    print("<br>")
    #print(dataFull["time"])

    print(dataFull["winds"].mean())
    print("<br>")
    print(dataFull["winds"].std())
    print("<br>")
    print(dataFull["windd"].to_string(index=False))
    print("<br>")
    print(dataFull["windd"].mean())
    print("<br>")
    print(dataFull["windd"].std())

def printdata2(dataFull):
    #print(dataFull["winds"].to_string(index=False))
    #print("<br>========================<br>========================<br>")
    #print(dataFull)
    #print(dataFull["time"])

    #print("<br>")
    #print(dataFull["winds"].to_string(index=False))
    #print("<br>")
    #print(dataFull["time"])

    print(dataFull["winds"].mean())
    print("<br>")
    print(dataFull["winds"].std())
    print("<br>================================corrected wind speed")
    sig = dataFull["correctedwindspeed"].std()
    mean = dataFull["correctedwindspeed"].mean()
    print(mean)
    print("<br>")
    print(sig)
    print("<br>")
    #print(dataFull["windd"].to_string(index=False))
    print("<br>")
    print(dataFull["windd"].mean())
    print("<br>")
    print(dataFull["windd"].std())
    #alphaM[i][j] = 2*np.random.random_sample()+1.2
    #lambdaM[i][j] = mean[i][j]/(np.exp(spe.gammaln(1+1/alphaM[i][j])))

    fig = plt.figure(figsize=(4.34, 3.22), dpi=100, facecolor='w', edgecolor='w')
    new_axesRA(dataFull["windd"],dataFull["correctedwindspeed"])
    plt.savefig(file_path + 'WindRose_temp8888_img.png')
    #count = count + 1
    plt.clf()
    plt.close()
    new_axesHA(dataFull)
    plt.savefig(file_path + 'WindHistogram_temp8888_img.png')
    #count = count + 1
    plt.clf()
    plt.close()

def new_axesH_Mon(dataMonth, MonSec, xaxis2i, i, textMon1):
    #new_axesH_Mon(dataMonth, MonSec, xaxis2i, i, textMon1, textMon2, textMon3, shapeMon, scaleMon):
    fig = plt.figure(figsize=(4.34, 3.22), dpi=100, facecolor='w', edgecolor='w')
    binsH=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25]
    #print("i "+str(i)+"<br>")
    #print(dataMonth)
    shapeMon, locMon, scaleMon = weibull_min.fit(dataMonth)
    #shapeMon[i], locMon[i], scaleMon[i] = weibull_min.fit(dataMonth.to_numpy(), floc=0)
    #shapeMon[i], locMon[i], scaleMon[i] = dataMonth.apply(lambda x: weibull_min.fit(x, floc=0))
    #shapeMon[i], locMon[i], scaleMon[i] = dataMonth.weibull_min.fit(self, floc=0))
    textMon1[i] = plt.text(15, .18, r'$mean = %3.2f$' % dataMonth.mean(), weight=700, fontsize=15)
    #textMon2[i] = plt.text(15, .15, r'$\alpha = %3.2f$' % shapeMon[i], weight=700, fontsize=16)
    #textMon3[i] = plt.text(15, .12, r'$\lambda = %3.2f$' % scaleMon[i], weight=700, fontsize=16)
    plt.text(15, .15, r'$\alpha = %3.2f$' % shapeMon, weight=700, fontsize=15)
    plt.text(15, .12, r'$\lambda = %3.2f$' % scaleMon, weight=700, fontsize=15)
    #print(str(i)+ "i======<br>")
    #print(str(shapeMon[i])+"<br>")

    #ax = plt.hist(months[i][j],bins=binsH,facecolor=colourb1,normed=True)
    #dataFull.notnull()
    #ax = dataFull["correctedwindspeed"].diff().hist(bins=binsH,facecolor=colourb1,normed=True)
    #ax = dataFull["correctedwindspeed"].hist(bins=binsH,facecolor=colourb1,normed=True)
    # TODO: is normed right?
    #ax = dataMonth.hist(bins=binsH,facecolor=colourb1,normed=True)
    ax = dataMonth.hist(bins=binsH, facecolor=colourb1)
    #histogramMonth = np.histogram(dataMonth, bins=binsH, range=(0, 100), normed=True)
    histogramMonth = np.histogram(dataMonth, bins=binsH, range=(0, 100))
    print("#####################################################################")
    print(histogramMonth)
    #plt.plot(binsH, 1/(sig[i][j] * np.sqrt(2 * np.pi)) *
    #    np.exp( - (binsH - mean[i][j])**2 / (2 * sig[i][j]**2) ),
    #    linewidth=7, color=colourk1)
    plt.xlabel('m/s')
    #plt.xlabel("Figure 2: Long Term Annual Wind Histogram (m/s)", fontdict=fontLine)
    #plt.ylabel('Probability')
    plt.title(MonSec)
    #fontsize=28
    #plt.text(15, .18, r'$mean = %3.2f$' %(dataMonth.mean()),weight=700, fontsize=16)
    ####shapeMon, locMon, scaleMon = weibull_min.fit(dataMonth, floc=1)
   ##print("shape, loc, scale -------- new_axesH_Mon")
   ##print("%f , %f , %f" % (shape, loc, scale))

    #plt.text(18, .16, r'$\alpha = %3.2f$' %(alphaM[i][j]),weight='roman', fontsize=14)
    #plt.text(18, .14, r'$\lambda = %3.2f$' %(lambdaM[i][j]),weight='demi', fontsize=14)
    plt.axis([0, 25, 0, 0.30])
    plt.grid(True)
    plt.gca().yaxis.set_major_formatter(formatter1)
    plt.tight_layout()
    #plt.savefig('/home/wiwasol/www/python/weibull_%s_temp999_img.png',%(xaxis[count]))
    #plt.savefig('/home/wiwasol/www/python/weibull_' & xaxis[count] & '_temp999_img.png'

    plt.savefig(file_path + 'weibull_' + xaxis2i + '_temp999_img.png')
    #hist_monthH = file_path + 'weibull_' + xaxis2[i] + '_temp999_img.png'
    plt.clf()
    plt.close()
    return ax

def get_hist(ax12):
    n = 0
    binsA = 0
    n,binsA = [],[]
    for rect in ax12.patches:
        ((x0, y0), (x1, y1)) = rect.get_bbox().get_points()
        n.append(y1-y0)
        binsA.append(x0) # left edge of each bin
    binsA.append(x1) # also get right edge of last bin
    #return n
    return pd.Series(n)
#return pd.DataFrame(n)
def get_hist2(ax12):
    #n = np.zeros(26)
    n = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
    #binsA = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25]
    #n,binsA = [],[]
    for i, rect in enumerate(ax12.patches):
        ((x0, y0), (x1, y1)) = rect.get_bbox().get_points()
        n[i] = (y1-y0)
        #binsA[i] = (x0) # left edge of each bin
    #binsA[].append(x1) # also get right edge of last bin
    return n
#return pd.Series(n)
#return pd.DataFrame(n)
def new_axesH_MonCDF(dataMonthGroup,months):
    #n2 = np.zeros(26)
    #n2 = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
    #n2 = np.zeros((12,25))
    n2 = []
    binsH=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25]
    ###nAll = pd.DataFrame(np.zeros((12,25)))
    for i, group2 in enumerate(dataMonthGroup):
        #print(i)
        n2 = get_hist(group2[1].correctedwindspeed.hist(bins=binsH, cumulative=True))
        #n2[i] = get_hist(group2[1].correctedwindspeed.hist(bins=binsH, cumulative=True))
        #nAll.iloc[i] = get_hist(group2[1].correctedwindspeed.hist(bins=binsH, cumulative=True))
        #nAll.iloc[i] = get_hist(group2[1].correctedwindspeed.hist(bins=binsH, cumulative=True,normed=True))
    #nAll.index = months
    #print("<br>==============================NNNNNNNNNNNNNNNNNNNNNN1_all")
    #print(nAll.iloc[0])
    #print(n2[0])
    #print("<br>==============================NNNNNNNNNNNNNNNNNNNNNN2_all")
    #print(nAll.iloc[1])
    #print(n2[1])

    #print("<br>==============================NALL")
    #print(nAll)
    #print(n2)

    #n2 = dataMonth.hist(bins=binsH, cumulative=True,normed=True)
    #serie = dataMonth
    #hist = serie.hist(bins=binsH)
    #n, bins = get_hist(hist)
    #axMonth5 = axMonthCDF.quantile(0.5)
    #axMonth5 = axMonthCDF.
    axMonth5 = pd.DataFrame(np.zeros(12))
    axMonth10 = pd.DataFrame(np.zeros(12))
    axMonth15 = pd.DataFrame(np.zeros(12))
    nAll = pd.DataFrame(np.zeros((12,25)))
    nAll.index = months
    countiter = 0
    for i2 in range(12):
        for i3 in range(25):
            #print("countiter: "+str(countiter)+", ")
            nAll.loc[months[i2],[i3]] = n2[countiter]
            countiter = countiter +1

    ##axMonth5 = [0,0,0,0,0,0,0,0,0,0,0,0]
    ##axMonth10 = [0,0,0,0,0,0,0,0,0,0,0,0]
    ##axMonth15 = [0,0,0,0,0,0,0,0,0,0,0,0]

    #[0.030960041031379681, 0.10355760712453957, 0.19699724903249871, 0.29481978831538208, 0.39786450319391986, 0.50995477222921615, 0.59957103557607117, 0.67160908285541088, 0.74280785191402043, 0.79470322189583609, 0.84244882734181925, 0.88832937007506863, 0.91709796241898622, 0.9412505245488878, 0.95673054506457766, 0.97015899659626037, 0.97855177880356214, 0.98838998461323246, 0.99198023033524485, 0.99389191961579693, 0.99645638084580579, 0.99776192474471936, 0.99906746864363294, 0.99981349372872641, 0.99999999999999978, 0.033689024390243905, 0.10838414634146341, 0.2046239837398374, 0.3051321138211382, 0.39944105691056908, 0.50777439024390247, 0.58805894308943096, 0.66036585365853662, 0.73338414634146343, 0.78556910569105698, 0.83455284552845532, 0.88267276422764229, 0.91255081300813012, 0.93567073170731707, 0.95279471544715444, 0.96697154471544711, 0.97550813008130077, 0.98633130081300813, 0.99090447154471539, 0.99405487804878045, 0.99613821138211378, 0.99781504065040649, 0.99933943089430888, 0.99994918699186985, 0.99999999999999989, 0.031796723132463206, 0.11163565676201055, 0.2116078866981394, 0.31167268351383876, 0.41122836249190042, 0.52189206701842084, 0.60501712487272052, 0.67245209663982219, 0.74132185504026649, 0.79445524391372757, 0.84175691937424779, 0.88443025085624349, 0.91220031472739038, 0.93492548366194561, 0.95232805702119772, 0.96575025455891872, 0.97482180875682667, 0.98597611774507066, 0.99023419420531322, 0.99254836619457543, 0.99606590761825409, 0.9975932611311672, 0.99953716560214745, 0.9999074331204294, 0.99999999999999989, 0.035332785538208712, 0.11759872395959206, 0.22562714485958724, 0.32998211610034323, 0.43380540383778826, 0.54458891198221282, 0.62327807047223172, 0.69065687080090876, 0.7561022765720915, 0.80936729663106, 0.85417371550099086, 0.89936681328242063, 0.92701435545458943, 0.9453332688868481, 0.96084876021074006, 0.9727391367393301, 0.98047271496930744, 0.98859297211078356, 0.99313644932089518, 0.99574653197351248, 0.99763159166706949, 0.99850161921794189, 0.9994199816327517, 0.9999516651360626, 1.0, 0.0317371678533203, 0.11377914281725533, 0.21545907208339149, 0.31457955232909862, 0.41598026897482432, 0.52087114337568052, 0.60021406300898128, 0.66461910745032338, 0.73163013634882956, 0.78714691237377254, 0.83596258550886493, 0.88366140816231553, 0.91544511145237095, 0.93750290846479589, 0.95304574433431055, 0.96663409186095206, 0.9758946437712317, 0.9866908650937688, 0.99199590488156719, 0.99488110195914181, 0.99688212573875001, 0.99813858253059695, 0.99934850388570884, 0.99976732281632452, 0.99999999999999989, 0.024488442591890869, 0.080428192497158019, 0.15289882531261842, 0.23474801061007958, 0.32100227358848049, 0.41270367563471011, 0.48621636983705951, 0.55020841227737782, 0.61500568397120126, 0.67208222811671092, 0.7300587343690792, 0.79509283819628651, 0.83800682076544153, 0.87637362637362648, 0.90389352027283076, 0.92790829859795387, 0.94491284577491486, 0.96561197423266398, 0.97702728306176589, 0.98342175066313009, 0.98967411898446389, 0.99445812807881784, 0.99777377794619182, 0.99957370215990915, 1.0, 0.034640643788733695, 0.10956433262417908, 0.19928776246415689, 0.28258255480529093, 0.36647858662473409, 0.45948570900009256, 0.52728702247710668, 0.58574599944500971, 0.65035611876792154, 0.70622514106003142, 0.76029044491721387, 0.81463324391823133, 0.85491628896494298, 0.88840070298769758, 0.91175654426047525, 0.93159744704467651, 0.94524095828322985, 0.96448062158912196, 0.9741929516233464, 0.98089908426602512, 0.98779021367126052, 0.99153639811303274, 0.99657755989270158, 0.99902876699657728, 0.99999999999999967, 0.04363157647280079, 0.13691287789740936, 0.24208942592505522, 0.33795665052423712, 0.43170811979876811, 0.53824815459118902, 0.61173538953406359, 0.67130565611923454, 0.73256852696412622, 0.78179510085100379, 0.8274954158634632, 0.87408904979077529, 0.90319243970097318, 0.92618364756217963, 0.94198128731957298, 0.95674455780713707, 0.96699421693544585, 0.97903051389345985, 0.98523672951243579, 0.98951525694672982, 0.99327659974610916, 0.99567445578071356, 0.99830739574027916, 0.99934176501010852, 0.99999999999999989, 0.050244795015083332, 0.16265268779981207, 0.28900647841353044, 0.40012857919984168, 0.50487117353246624, 0.620147371544434, 0.6977399732950893, 0.7558973344542802, 0.81044458731022206, 0.85129321002917757, 0.88665248998565849, 0.92097324563572525, 0.93833143761436133, 0.9536125809801691, 0.9645418129667177, 0.97364126403244144, 0.98051530587013491, 0.98862568616784519, 0.99184016616388893, 0.9945106572375253, 0.99668661292715488, 0.99821967261757572, 0.99945601107759252, 0.99970327876959586, 0.99999999999999989, 0.050256751616584251, 0.16807721567135792, 0.29493153290224416, 0.41698364397109167, 0.53028718143780906, 0.65038988208444271, 0.72974515024724229, 0.78870292887029292, 0.8463769494104223, 0.88327310764549272, 0.91550969950551553, 0.94708063902624584, 0.96201027006466344, 0.97518067706352229, 0.9824077596044124, 0.98906428299733751, 0.99239254469380001, 0.9961962723469, 0.99776531000380375, 0.99862114872575125, 0.99923925446938, 0.999429440852035, 0.99985736021300875, 0.99995245340433625, 1.0, 0.03469526397515528, 0.12902756211180125, 0.2437402950310559, 0.36083074534161491, 0.46952639751552794, 0.59501164596273293, 0.68031832298136652, 0.75038819875776408, 0.81701281055900632, 0.86194681677018647, 0.90032996894409956, 0.93779114906832317, 0.95715256211180144, 0.9720011645962735, 0.98025038819875798, 0.98743206521739157, 0.99102290372670832, 0.99606948757764002, 0.99825310559006242, 0.99878687888198792, 0.99946622670807483, 0.99966032608695687, 0.99990295031055931, 0.99995147515527982, 1.0000000000000002, 0.03128615726384968, 0.11042717637802565, 0.21169065580598881, 0.31818392187716943, 0.4214837784051465, 0.54486971814689689, 0.63465543573841821, 0.70819641782755594, 0.78058036747350401, 0.83343360947840983, 0.87966862590827055, 0.92141435645855518, 0.94511038089508037, 0.96181792937473976, 0.97375850418845755, 0.98301476373397523, 0.98810570648400997, 0.99439996297496203, 0.99680659045679665, 0.99773221641134846, 0.99842643587726232, 0.99898181144999343, 0.99967603091590729, 0.99995371870227279, 1.0000000000000004]
   ##print("<br>==============================NALL___pandas")
   ##print(nAll)
   ##print("<br>==============================NALL")
   ##print(n2)

    axMonth5 = nAll.iloc[:,[4]]
    axMonth10 = nAll.iloc[:,[9]]
    axMonth15 = nAll.iloc[:,[14]]

    ####axMonth5 = n2[4]
    ####axMonth10 = n2[9]
    ####axMonth15 = n2[14]


    #axMonth5[k] = nAll.iloc[[k],[4]]
    #axMonth10[k] = nAll.iloc[[k],[9]]
    #axMonth15[k] = nAll.iloc[[k],[14]]

    """
   ##print("<br>==============================<br>")
   ##print(axMonth5)
   ##print("<br>==============================<br>")
   ##print(axMonth10)
   ##print("<br>==============================<br>")
   ##print(axMonth15)

    axMonth5 = list(set(axMonth5))
    axMonth10 = list(set(axMonth10))
    axMonth15 = list(set(axMonth15))
   ##print("<br>==============================<br>")
   ##print(axMonth5)
   ##print("<br>==============================<br>")
   ##print(axMonth10)
   ##print("<br>==============================<br>")
   ##print(axMonth15)
    """
    return (axMonth5, axMonth10, axMonth15)
def new_axesH_MonCDF2(dataMonth300, monSec):
    binsH=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25]
    #nAll = pd.DataFrame(np.zeros((12,25)))
    #ax = dataMonth.hist(bins=binsH,facecolor=colourb1,normed=True)
    #ax = dataMonth.hist(bins=binsH, cumulative=True,normed=True)
    ax = dataMonth300.hist(bins=binsH)
    bx = get_hist(ax)
    #print("<br>==========%s==============<br>" % (monSec))
    #print(bx)
    return bx
def new_axesH_Sec(dataSec, MonSec, xaxis2i, i, textSec4, textSec5, textSec6, shapeSec, locSec, scaleSec):
    fig = plt.figure(figsize=(4.34, 3.22), dpi=100, facecolor='w', edgecolor='w')
    binsH=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25]

    shapeSec[i], locSec[i], scaleSec[i] = weibull_min.fit(dataSec)
    textSec4[i] = plt.text(15, .18, r'$mean = %3.2f$' % dataSec.mean(), weight=700, fontsize=15)
    textSec5[i] = plt.text(15, .15, r'$\alpha = %3.2f$' % shapeSec[i], weight=700, fontsize=15)
    textSec6[i] = plt.text(15, .12, r'$\lambda = %3.2f$' % scaleSec[i], weight=700, fontsize=15)
    #print(str(shapeSec[i])+"<br>")

    #ax = plt.hist(months[i][j],bins=binsH,facecolor=colourb1,normed=True)
    #dataFull.notnull()
    #ax = dataFull["correctedwindspeed"].diff().hist(bins=binsH,facecolor=colourb1,normed=True)
    # TODO: is normed right?
    #ax = dataSec.hist(bins=binsH,facecolor=colourb1,normed=True)
    ax = dataSec.hist(bins=binsH, facecolor=colourb1)
    #plt.plot(binsH, 1/(sig[i][j] * np.sqrt(2 * np.pi)) *
    #    np.exp( - (binsH - mean[i][j])**2 / (2 * sig[i][j]**2) ),
    #    linewidth=7, color=colourk1)
    plt.xlabel('m/s')
    #plt.xlabel("Figure 2: Long Term Annual Wind Histogram (m/s)", fontdict=fontLine)
    #plt.ylabel('Probability')
    plt.title(MonSec)
    #fontsize=28
    ###shapeSec, locSec, scaleSec = weibull_min.fit(dataSec, floc=1)
    #plt.text(175, .18, r'$mean = %3.2f$' %(dataSec.mean()),weight=700, fontsize=16)
    #plt.text(175, .15, r'$\alpha = %3.2f$' %(shapeSec),weight=700, fontsize=16)
    #plt.text(175, .12, r'$\lambda = %3.2f$' %(scaleSec),weight=700, fontsize=16)
    #run function to find maximum frequency to set axis range for all 12
    plt.axis([0, 25, 0, 0.3])
    plt.grid(True)
    plt.gca().yaxis.set_major_formatter(formatter1)
    plt.tight_layout()
    #plt.savefig('/home/wiwasol/www/python/weibull_%s_temp999_img.png',%(xaxis[count]))
    #plt.savefig('/home/wiwasol/www/python/weibull_' & xaxis[count] & '_temp999_img.png'


    #newly added into function to see if it would keep different variable context or cascade values, or if shape/scale are global somehow?
    plt.savefig(file_path + 'weibullSEC_' + xaxis2i + '_temp999_img.png')
    #hist_sectorH = file_path + 'weibullSEC_' + xaxis2[i] + '_temp999_img.png'
    plt.clf()
    plt.close()

    return ax

def new_axesHA(dataFull):
    fig = plt.figure(figsize=(8, 6), dpi=120, facecolor='w', edgecolor='w')
    binsH=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25]
    #ax = plt.hist(dataFull,bins=binsH,facecolor=colourb1,normed=True)
    #df4.plot(kind='hist', stacked=True, bins=20)
    #dataFull.plot(kind='hist', bins=binsH, color=colourk1)
    #dataWindd2 = dataFull.dropna()
    #dataFull.dropna()
    ###dataFull.notnull()
    #dataWindd3 = dataWindd2.notnull()
    ###dataFull["correctedwindspeed"].notnull()
    # TODO: AttributeError: 'Rectangle' object has no property 'normed'
    # ax = dataFull["correctedwindspeed"].hist(bins=binsH,facecolor=colourb1,normed=True)
    ax = dataFull["correctedwindspeed"].hist(bins=binsH, facecolor=colourb1)
    #ax = dataFull["correctedwindspeed"].diff().hist(bins=binsH,facecolor=colourb1,normed=True)
    #ax = dataFull["correctedwindspeed"].diff().hist(bins=binsH,facecolor=colourb1)
    #mean = dataFull["correctedwindspeed"].mean()
    #shapeHA, locHA, scaleHA = weibull_min.fit(dataFull["correctedwindspeed"], floc=0)
    #x = np.linspace(weibull_min.ppf(0.01, dataFull["correctedwindspeed"]), weibull_min.ppf(0.99, dataFull["correctedwindspeed"]), 100)
    #print("shapeHA, locHA, scaleHA")
    #print("%f , %f , %f" % (shapeHA, locHA, scaleHA))
    #print("ax============================")
    #print(ax)
    #print("ax============================")
    #plt.plot(binsH, weibull_min.pdf(dataFull["correctedwindspeed"]), linewidth=7, color=colourk1)
    #plt.plot(binsH, 1/(sig * np.sqrt(2 * np.pi)) *
    #    np.exp( - (binsH - mean)**2 / (2 * sig**2) ),
    #    linewidth=7, color=colourk1)
    #plt.xlabel('m/s')

    #ax.legend(title="Wind Speed (m/s)", loc=(-0.48,0))
    #plt.xlabel("Figure 2: Long Term Annual Wind Histogram (m/s)", fontdict=fontLine)


    #plt.ylabel('Probability')
    #plt.title("Figure 2: Long Term Annual Wind Histogram", fontdict=font)
    #plt.title(xaxisMon[i][j])
    #fontsize=28
    #shapeHA, locHA, scaleHA = dataFull["correctedwindspeed"].apply(lambda x: weibull_min.fit(x, floc=0))
    shapeHA, locHA, scaleHA = weibull_min.fit(dataFull["correctedwindspeed"])
    #shapeHA, locHA, scaleHA = weibull_min.fit(dataFull["correctedwindspeed"].to_numpy(), floc=0)
    ###shapeHA, locHA, scaleHA = weibull_min.fit(dataFull["correctedwindspeed"].to_numpy(), floc=0)
    #shapeHA, locHA, scaleHA = dataFull["correctedwindspeed"].apply(lambda x: weibull_min.fit(x, floc=0))
    #shapeHA, locHA, scaleHA = dataFull["correctedwindspeed"].weibull_min.fit(self, floc=0))
    plt.text(15, .18, r'$mean = %3.2f$' % dataFull["correctedwindspeed"].mean(), weight=700, fontsize=28)
    plt.text(15, .16, r'$\alpha = %3.2f$' % shapeHA, weight=700, fontsize=28)
    plt.text(15, .14, r'$\lambda = %3.2f$' % scaleHA, weight=700, fontsize=28)
    #plt.text(18, .16, r'$\alpha = %3.2f$' %(shape),weight='roman', fontsize=14)
    #plt.text(18, .14, r'$\lambda = %3.2f$' %(scale),weight='demi', fontsize=14)

    plt.axis([0, 25, 0, 0.20])
    plt.grid(True)
    plt.gca().yaxis.set_major_formatter(formatter1)
    plt.tight_layout()
    #plt.savefig('/home/wiwasol/www/python/weibull_%s_temp999_img.png',%(xaxis[count]))
    #plt.savefig('/home/wiwasol/www/python/weibull_' & xaxis[count] & '_temp999_img.png'
    return ax

def new_axesR(wd,ws,MonSec):
    fig = plt.figure(figsize=(4, 4), dpi=80, facecolor='w', edgecolor='w')
    #rect = [0.1, 0.1, 0.75, 0.75] location location, size, size
    binsR=[0,2,4,6,8,10,12,14,16,18,20,22,24]
    ax = WindroseAxes.from_ax(ax=None, fig=fig, rmax = 20, rect = [0.1, 0.1, 0.75, 0.75])
    ###ax = WindroseAxes.from_ax(ax=None, fig=fig, rmax = 35)
    # TODO: is normed right?
    #ax.bar(wd, ws, nsector=12, bins=binsR, normed=True, opening=0.8, edgecolor='white')
    ax.bar(wd, ws, nsector=12, bins=binsR, opening=0.8, edgecolor='white')
    plt.title(MonSec, y=1.08, fontdict=font)
    return ax
    #4 by 3
def new_axesRA(wd,ws):
    fig = plt.figure(figsize=(10, 8), dpi=120, facecolor='w', edgecolor='w')
    binsR=[0,2,4,6,8,10,12,14,16,18,20,22,24]
    #rect = [0.25, 0.1, 0.75, 0.75]
    ax = WindroseAxes.from_ax(ax=None, fig=fig, rmax = 20, rect = [0.25, 0.1, 0.75, 0.75])
    ###ax = WindroseAxes.from_ax(ax=None, fig=fig, rmax = 35)
    # TODO: is normed right?
    #ax.bar(wd, ws, nsector=12, bins=binsR, normed=True, opening=0.8, edgecolor='white')
    ax.bar(wd, ws, nsector=12, bins=binsR, opening=0.8, edgecolor='white')
    ax.set_legend()
    ax.legend(title="Wind Speed (m/s)", loc=(-0.48,0))
    plt.xlabel("Figure 3: Long Term Annual Windrose (m/s)", fontdict=fontLine)
    return ax
def new_axesRAH(wd,ws):
    fig = plt.figure(figsize=(10, 8), dpi=120, facecolor='w', edgecolor='w')
    binsH=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25]
    #ax = WindroseAxes.from_ax(ax=None, fig=fig, rmax = 25, rect = [0.25, 0.1, 0.75, 0.75])
    ax = WindroseAxes.from_ax(ax=None, fig=fig, rmax = 25)
    # TODO: is normed right?
    #ax.histogram(wd, ws, bins=binsH, nsector=12, normed=True, opening=0.8, edgecolor='white')
    ax.histogram(wd, ws, bins=binsH, nsector=12, opening=0.8, edgecolor='white')
    #ax.set_legend()
    #ax.legend(title="Wind Speed (m/s)", loc=(-0.48,0))
    plt.xlabel("Figure 2: Long Term Annual Wind Histogram (m/s)", fontdict=fontLine)
    #plt.xlabel('m/s', fontdict=font)
    #plt.tight_layout()
    return ax
#Write a short mapper that bins data
# def map_bin(x, bins):
#     kwargs = {}
#     if x == max(bins):
#         kwargs['right'] = True
#     bin = bins[np.digitize([x], bins, **kwargs)[0]]
#     bin_lower = bins[np.digitize([x], bins, **kwargs)[0]-1]
#     return '[{0}-{1}]'.format(bin_lower, bin)


def map_bin(x, bins):
    kwargs = {}
    if x == max(bins):
        kwargs['right'] = True
    bin = bins[np.digitize([x], bins, **kwargs)[0]]
    bin_lower = bins[np.digitize([x], bins, **kwargs)[0]-1]
    return '[{0}-{1}]'.format(bin_lower, bin)


def map_bin_recent(x, bins):
    # kwargs = {}
    # if x == max(bins):
    #   kwargs['right'] = True
    try:
        #bin = bins[np.digitize([x], bins, **kwargs)[0]]
        bin = np.digitize(x, bins, right=True)
        #print("bin: " + str(bin))
    except IndexError:
        print("bin binned ========")
        print(x)
        print(np.digitize(x, bins, right=True))
        #bin = bins[np.digitize([x], bins, **kwargs)[0]]
        bin = 1
    try:
        #bin_lower = bins[np.digitize([x], bins, **kwargs)[0]-1]
        bin_lower_temp = np.digitize(x, bins, right=True)
        if bin_lower_temp > 1:
            bin_lower = bin_lower_temp-1
        else:
            bin_lower = bin_lower_temp

    except IndexError:
        #bin_lower = bin
        print("bin_lower IndexError")
        # bin_lower = bins[np.digitize(x, bins, right=True)[0]]
        print(np.digitize(x, bins, right=True))
        bin_lower = 0
    # print("bin_lower: " + str(bin_lower))
    return '[{0}-{1}]'.format(bins[bin_lower-1], bins[bin-1])

"""
def month_bin(x, bins):
    kwargs = {}
    if x == max(bins):
        kwargs['right'] = True
    bin = bins[np.digitize([x], bins, **kwargs)[0]]
    bin_lower = bins[np.digitize([x], bins, **kwargs)[0]-1]
    return '[{0}-{1}]'.format(bin_lower, bin)
"""
#hours = pd.DataFrame(predb_hr)
#winds = pd.DataFrame(predb_ws)
#directions = pd.DataFrame(predb_dir)

#dataF = pd.DataFrame(columns=[0,1,2])
#dataF = pd.DataFrame({}, columns=["time","winds","windd"])
#dataF.columns = ["time","winds","windd"]
#storedatabase(vardb)
####printdata2(vardb)
#printdata2(retrievedatabase())

# Working call function
# TODO: this makes new API calls to Darksky
#vardb = callday(44.6566, -63.5963, '1990-01-01', '2022-12-31')
#storedatabaseNEW(vardb, "site_012_44.6566_-63.5963_1990")
#print(vardb)
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
    storedatabaseNEW(retrievedatabase(),"site_"+str(orderid)+"_"+str(latt2)+"_"+str(long2)+"_"+str(ystart))
except (ValueError, TypeError, IndexError, KeyError) as e:
    print(json.dumps({'error': str(e)}))
    latt2 = 0
    long2 = 0
    ystart = '2015-12-30'
    yend = '2015-12-31'
    orderid = 999
    storedatabaseNEW(retrievedatabase(),"site_"+str(orderid)+"_"+str(latt2)+"_"+str(long2)+"_"+str(ystart))
    #sys.exit(1)
"""

#====================================================================================================================
#easyprintpdf(retrievedatabase())
#====================================================================================================================
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
    ###storedatabaseNEW(retrievedatabase(),"site_"+str(orderid)+"_"+str(latt2)+"_"+str(long2)+"_"+str(ystart))
    #vardbnew = callday(str(latt2), str(long2), ystartlong, yendlong)
    #storedatabaseNEW(vardbnew,"site_"+str(orderid)+"_"+str(latt2)+"_"+str(long2)+"_"+str(ystart))
    #easyprintpdf(vardbnew)
except (ValueError, TypeError, IndexError, KeyError) as e:
    print(json.dumps({'error': str(e)}))
    latt2 = 0
    long2 = 0
    ystart = '2015-12-30'
    yend = '2015-12-31'
    orderid = 999
    #storedatabaseNEW(retrievedatabase(),"site_"+str(orderid)+"_"+str(latt2)+"_"+str(long2)+"_"+str(ystart))
    #sys.exit(1)
"""


# Using the file system load
#
# We now assume we have a file in the same dir as this one called
# test_template.html
#
# provide weibull parameters to sample within reasonable ranges
def weib(x, *p):
    XSsat, Lo, W, s = p
    return XSsat*(1-np.exp(-((x-Lo)/W)**s))
def to_percent1(y, position):
    # Ignore the passed in position. This has the effect of scaling the default
    # tick locations.
    s = str(100 * y)

    # The percent symbol needs escaping in latex
    if matplotlib.rcParams['text.usetex'] is True:
        return s + r'$\%$'
    else:
        return s + '%'
def meanBoxplot(dataFull):
    """
    spread = random_sample(60) * 0.2
    medianbox = np.median(spread)
    yx = np.ones(60)
    fig = plt.figure(figsize=(7, 4.17), dpi=80, facecolor='w', edgecolor='w')
    bp = plt.boxplot(spread-medianbox, whis=[5,95],showcaps=None,patch_artist=True,showfliers=False)
    #bp = plt.boxplot(spread-medianbox, whis=None,showcaps=None,patch_artist=True,showfliers=False)
    #indexer=None, box_top=75,
    #                         box_bottom=25,whisker_top=99,whisker_bottom=1
    #bp = percentile_box_plot(plt,spread-medianbox)
    #pylab.setp(bp['boxes'],facecolor='#6DCBD5',alpha=0.5)
    pylab.setp(bp['boxes'], facecolor=colourb1, alpha=0.5)
    #pylab.setp(bp['boxes'], color='black', alpha=1)
    pylab.setp(bp['whiskers'], color='white', linestyle = 'solid', alpha=0)
    pylab.setp(bp['fliers'], color='black', alpha = 1, marker= '+', markersize = 32)
    pylab.setp(bp['medians'], color='black')
    pylab.setp(bp['caps'], color=colourb1, linewidth=2)
    bp2 = plt.boxplot(spread-medianbox, whis=[5,95],showcaps=None,showfliers=False)
    pylab.setp(bp2['boxes'], color='black', alpha=1)
    pylab.setp(bp2['medians'], color='black')
    pylab.setp(bp2['whiskers'], color='white', linestyle = 'solid', alpha=0)
    pylab.setp(bp2['caps'], color=colourb1, linewidth=2)
    plt.scatter(yx,spread-medianbox, alpha=1, c='black', s=320, marker='_')
    plt.tick_params(
        axis='x',          # changes apply to the x-axis
        which='both',      # both major and minor ticks are affected
        bottom='off',      # ticks along the bottom edge are off
        top='off',         # ticks along the top edge are off
        labelbottom='off') # labels along the bottom edge are off

    # adding horizontal grid lines
    plt.gca().yaxis.grid(True)
    plt.gca().yaxis.set_major_formatter(formatter1)
    plt.xlabel('Figure 8: Percentage of Variance from Mean Wind Speed with ticks for each Year', fontdict=fontLine)
    plt.tight_layout()
    plt.savefig('/home/wiwasol/www/python/box1_temp444_img.png')
    box1 = "file:///home/wiwasol/prospectingmm/box1_temp444_img.png"
    """
    #meanyears2 = dataFull['correctedwindspeed'].resample("A").to_numpy()
    #meanyears = meanyears.drop(2016)
    #meanyears = np.delete(meanyears2,
    #[2016])
    ###dataFull.dropna()
    ##dataFull.notnull()
    meanYear = dataFull['correctedwindspeed'].groupby(lambda x: x.year).mean()
    #meanYear = dataFull['correctedwindspeed'].groupby(lambda x: x.year).mean()
    #meanYear = meanYear.drop(2016)
    meanyears = meanYear.to_numpy()
    overallMean = meanYear.mean()
    overallMedian = meanYear.median()
    #meanyears = meanYearA.resample("A").to_numpy()
    #spread = random_sample(60) * 0.2
    #medianbox = np.median(spread)
    medianbox2 = dataFull['correctedwindspeed'].groupby(lambda x: x.year).median()
    #medianbox2 = medianbox2.drop(2016)
    medianbox = medianbox2.to_numpy()
    #medianbox = np.median(meanyears)
    #yxMed = np.ones(len(meanyears))*overallMedian
    #spread2 = ((meanyears-medianbox)/medianbox)
    spread2 = ((meanyears-overallMedian)/overallMean)
    #spread2 = ((meanyears-overallMean)/overallMean)
    ##print("spread2=========================")
    ##print(spread2)
    ##print("meanyears=========================")
    ##print(meanyears)
    ##print("overallMedian=========================")
    ##print(overallMedian)
    #yx = np.ones(60)
    yx = np.ones(len(meanyears))
    print("len(meanyears) "+str(len(meanyears)))
    print("yx "+str(yx))
    print("spread2 "+str(spread2))
    fig = plt.figure(figsize=(7, 4.17), dpi=120, facecolor='w', edgecolor='w')
    bp = plt.boxplot(spread2, whis=[5,95],showcaps=None,patch_artist=True,showfliers=False)
    #bp = plt.boxplot(spread-medianbox, whis=[5,95],showcaps=None,patch_artist=True,showfliers=False)
    #bp = plt.boxplot(spread-medianbox, whis=None,showcaps=None,patch_artist=True,showfliers=False)
    #indexer=None, box_top=75,
    #                         box_bottom=25,whisker_top=99,whisker_bottom=1
    #bp = percentile_box_plot(plt,spread-medianbox)
    #pylab.setp(bp['boxes'],facecolor='#6DCBD5',alpha=0.5)
    pylab.setp(bp['boxes'], facecolor=colourb1, alpha=0.5)
    #pylab.setp(bp['boxes'], color='black', alpha=1)
    pylab.setp(bp['whiskers'], color='white', linestyle = 'solid', alpha=0)
    pylab.setp(bp['fliers'], color='black', alpha = 1, marker= '+', markersize = 32)
    pylab.setp(bp['medians'], color='black')
    pylab.setp(bp['caps'], color=colourb1, linewidth=2)
    bp2 = plt.boxplot(spread2, whis=[5,95],showcaps=None,showfliers=False)
    #bp2 = plt.boxplot(spread-medianbox, whis=[5,95],showcaps=None,showfliers=False)
    pylab.setp(bp2['boxes'], color='black', alpha=1)
    pylab.setp(bp2['medians'], color='black')
    pylab.setp(bp2['whiskers'], color='white', linestyle = 'solid', alpha=0)
    pylab.setp(bp2['caps'], color=colourb1, linewidth=2)
    scatter = plt.scatter(yx,spread2, alpha=1, c='black', s=320, marker='_')
    #plt.scatter(yx,spread-medianbox, alpha=1, c='black', s=320, marker='_')
    plt.tick_params(
        axis='x',          # changes apply to the x-axis
        which='both',      # both major and minor ticks are affected
        bottom='off',      # ticks along the bottom edge are off
        top='off',         # ticks along the top edge are off
        labelbottom='off') # labels along the bottom edge are off

    # adding horizontal grid lines
    plt.gca().yaxis.grid(True)
    plt.gca().yaxis.set_major_formatter(formatter4)
    plt.xlabel('Figure 8: Percentage of Variance from Mean Wind Speed with ticks for each Year', fontdict=fontLine)
    plt.tight_layout()
    ##plt.savefig(file_path + 'box1_temp444_img.png')
    ##box1 = "file:///home/wiwasol/prospectingmm/box1_temp444_img.png"
    return (bp, bp2, scatter)


def errorfill(x, y, yerr, color=None, alpha_fill=0.3, ax=None):
    #xaxisMon = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    ax = ax if ax is not None else plt.gca()
    if color is None:
        color = ax._get_lines.color_cycle.next()

    if np.isscalar(yerr) or len(yerr) == len(y):
        ymin = y - yerr
        ymax = y + yerr
    elif len(yerr) == 2:
        ymin, ymax = yerr
    """
    ymax = y + yerr.max()
    ymin = y - yerr.min()
    """
    ax.plot(x, y, color=color, linewidth=3)
    #plt.xticks(x,xaxisMon)
    plt.margins(0.05)
    ax.fill_between(x, ymax, ymin, color=color, alpha=alpha_fill)


def easyprintpdf(dataFull, startdate='1985-01-01', enddate='2027-12-31'):
    ###dataFull['time'] = pd.to_datetime(dataFull['time'], format='%Y/%m/%d %H:%M')
    ####dataFull['time'] = pd.to_datetime(dataFull['time'], format='%m/%d/%Y %H:%M')
    dataFull['time'] = pd.to_datetime(dataFull['time'], format='%Y-%m-%dT%H:%M:%SZ')
    #dataFull['time'] = pd.to_datetime(dataFull['time'], format='%d/%m/%Y %H:%M')
    #dataFull['time'] = pd.to_datetime(dataFull['time'])
    dataFull.set_index('time', inplace=True)
    ####dataFull.dropna()
    # TODO: normally important when loaded into MySQL database with Excel created column
    dataFull.dropna(subset=['windd','correctedwindspeed'])
    #print(dataFull.head())
    #print(dataFull.info())

    formatter1 = FuncFormatter(to_percent1)
    #formating histogram plots
    #colourb1 = '#408754'
    #colourb1 = '#6DCBD5'
    #oolourk1 = '#40423F'
    #colourg1 = '#09694E'

    fig = plt.figure(figsize=(4.34, 3.22), dpi=100, facecolor='w', edgecolor='w')
    new_axesRA(dataFull["windd"],dataFull["correctedwindspeed"])
    plt.savefig(file_path + 'WindRose_temp8888_img.png')
    annualRose = "file://" + file_path + "WindRose_temp8888_img.png"
    #count = count + 1
    plt.clf()
    plt.close()

    dataWindd = dataFull.copy()
    dataWindd.dropna(axis=0, how='any', subset=['winds', 'windd', 'correctedwindspeed'], inplace=True)
    #    dataWindd.dropna(axis=0, how='any', subset=['winds', 'windd', 'correctedwindspeed'], inplace=True)

    new_axesHA(dataWindd)
    plt.savefig(file_path + 'WindHistogram_temp8888_img.png')
    annualHist = "file://" + file_path + "WindHistogram_temp8888_img.png"
    #count = count + 1
    plt.clf()
    plt.close()
    #text7 = 0
    #text8 = 0
    #text9 = 0

    xaxis2 = ["01","02","03","04","05","06","07","08","09","10","11","12"]
    xaxisMon = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

    #dataWindd = dataFull.dropna()
    #dataWindd1['Month'] = dataWindd1['time'].apply(month_bin, bins=xaxisMon)
    #dataWindd['Binned'] = dataWindd['windd'].apply(map_bin, bins=freq_bins2)
    #grouped1 = dataWindd[['windd','correctedwindspeed','Binned']].groupby('Binned')

    dataMonthGroup = dataWindd.groupby(lambda x: x.month)
    #dataMonthGroup = dataFull.groupby(lambda x: x.month)
    #p50breakdown = [0,0,0,0,0,0,0,0,0,0,0,0]
    #twelve = pd.DataFrame(np.zeros(25))
    #p50breakdown = pd.DataFrame(np.zeros((12,25)))
    ###p50breakdown = pd.DataFrame({})
    #p50breakdown.index = xaxisMon
    #nAll = pd.DataFrame(np.zeros((12,25)))
    #axMonth5 = nAll.iloc[:,[4]]
    #shapeMon = [0,0,0,0,0,0,0,0,0,0,0,0]
    #locMon = [0,0,0,0,0,0,0,0,0,0,0,0]
    #scaleMon = [0,0,0,0,0,0,0,0,0,0,0,0]
    #dataMonthGroup2 = dataWindd1.groupby(lambda x: x.month)
    #shapeMon, locMon, scaleMon = dataMonthGroup.apply(lambda x: weibull_min.fit(x, floc=0))
    textMon1 = [0,0,0,0,0,0,0,0,0,0,0,0]
    #textMon2 = [0,0,0,0,0,0,0,0,0,0,0,0]
    #textMon3 = [0,0,0,0,0,0,0,0,0,0,0,0]


    for iMon, group in enumerate(dataMonthGroup, 0):
        new_axesR(group[1].windd,group[1].correctedwindspeed, xaxisMon[iMon])
        plt.savefig(file_path + 'weiWRose_' + xaxis2[iMon] + '_temp888_img.png')
        #add folder for each new report and image files?! Good way to keep records, and check if errors crop up
        #rose_monthR = 'file:///' + file_path + 'weiWRose_' + xaxis2[i] + '_temp888_img.png'
        ##plt.clf()
        ##plt.close()
        #new_axesH_Mon(group[1].correctedwindspeed, xaxisMon[iMon], xaxis2[iMon], iMon, textMon1, textMon2, textMon3, shapeMon, scaleMon)
        new_axesH_Mon(group[1].correctedwindspeed, xaxisMon[iMon], xaxis2[iMon], iMon, textMon1)
        """
        plt.savefig('file:///' + file_path + 'weibull_' + xaxis2[i] + '_temp999_img.png')
        #hist_monthH = 'file:///' + file_path + 'weibull_' + xaxis2[i] + '_temp999_img.png'
        plt.clf()
        plt.close()
        """
        #p50breakdown = new_axesH_MonCDF2(group[1].correctedwindspeed, xaxisMon[i])
        #print(xaxisMon[i])

    #for i, group3 in enumerate(dataMonthGroup):
    #twelve = new_axesH_MonCDF2(dataMonthGroup.correctedwindspeed, xaxisMon[i])
        #for k in range(25):
    #p50breakdown = pd.DataFrame(twelve,index=xaxisMon)

    #freq_bins = [0,15,45,75,105,135,165,195,225,255,285,315,345,360]
    #print("<br>===================DirBinned=================<br>")
    #(60, 90] 74158 (90, 120] 62518 (30, 60] 30057 (120, 150] 24170 (0, 30] 18185 (300, 330] 9789 (150, 180] 8211 (270, 300] 4333 (180, 210] 3442 (240, 270] 2340 (210, 240] 2279 Name: windd, dtype: int64
    #(15, 45] 20761 (45, 75] 48187 (75, 105] 81294 (105, 135] 40790 (135, 165] 14959 (165, 195] 5353 (195, 225] 2797 (225, 255] 2080 (255, 285] 2788 (285, 315] 5433 (315, 345] 12363 (345, 375] 4966
    #(0, 15] 7693 (15, 45] 20761 (45, 75] 48187 (75, 105] 81294 (105, 135] 40790 (135, 165] 14959 (165, 195] 5353 (195, 225] 2797 (225, 255] 2080 (255, 285] 2788 (285, 315] 5433 (315, 345] 12363 (345, 360] 4966
    #freq_bins = np.arange(0, 360, 30)
    # Original bins length
    freq_bins2 = [0, 30, 60, 90, 120, 150, 180, 210, 240, 270, 300, 330, 360]  # previously had 13th item with 360
    #freq_bins2 = [0,30,60,90,120,150,180,210,240,270,300,330]
    #freq_bins2 = np.arange(0, 390, 30)
    ###dataFull.dropna()
    #dataFull.dropna(subset = ['windd','correctedwindspeed'])
    #dataFull.notnull()
    #dataFull['windd'].notnull()
    #dataFull['correctedwindspeed'].notnull()
    #dataWindd = dataFull.dropna()
    #    dataWindd = dataFull.copy()
    #    dataWindd.dropna(axis=0, how='any', subset=['winds', 'windd', 'correctedwindspeed'], inplace=True)
    print("^^^===^^^===")
    print(dataWindd.count())
    print("+++===+++===")
    print(dataFull.count())
    # TODO: Attempts at fixing index out of range error (seemed to work with only 2 months?), but not with 12-24 etc
    #np.digitize([x], bins
    #dataFull['Binned'] = np.digitize(dataFull['windd'], freq_bins2)
    #dataFull['Binned'] = dataFull['windd']
    #dataFull['Binned'] = dataFull['windd'].notnull().apply(map_bin, bins=freq_bins2)
    dataWindd['Binned'] = dataWindd['windd'].apply(map_bin, bins=freq_bins2)
    grouped = dataWindd[['windd','correctedwindspeed','Binned']].groupby('Binned')
    #print(list(enumerate(grouped)))
    # print(list(enumerate(grouped, 0)))
    print("<br>===================DirBinned=================<br>")
    print(dataWindd['Binned'].count())
    ###dataWindd = dataFull.dropna()
    #dataWindd = dataFull.dropna()
    ###dataWindd['Binned'] = dataWindd['windd'].apply(map_bin, bins=freq_bins2)
    ###grouped = dataWindd[['windd','correctedwindspeed','Binned']].groupby('Binned')
    ###dataWindd['Binned'] = dataFull['windd'].apply(map_bin, bins=freq_bins2)
    ###grouped = dataWindd[['windd','correctedwindspeed','Binned']].groupby('Binned')
    #dataWindd['Month'] = dataWindd['time'].apply(month_bin, bins=MonSec)
    """
    step = freq_bins2[1]-freq_bins2[0]
    new_index = ['[{0}-{1}]'.format(x, x+step) for x in freq_bins2]
    new_index.pop(-1) #We dont need [360-375]...
    grouped_data = grouped_data.reindex(new_index)
    #print(grouped_data)
    """
    xaxisSec = ['30','60','90','120','150','180','210','240','270','300','330','360']
    #xaxisSec = ['30°','60°','90°','120°','150°','180°','210°','240°','270°','300°','330°','360°']
    #binsH=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,50]
    #binsLH=[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,50]
    # TODO: fix index errors etc. ?
    binsH=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,50] # was previous "working" setting
    #binsH = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25]
    binsLH=[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25]
    columnsB = ['[0-30]','[30-60]','[60-90]','[90-120]','[120-150]','[150-180]','[180-210]','[210-240]','[240-270]','[270-300]','[300-330]','[330-360]']
    dataprint = (dataWindd.assign(q=pd.cut(np.clip(dataWindd.correctedwindspeed,binsH[0], binsH[-1]), bins=binsH, labels=binsLH, right=False))  # right = False was previous "working" setting
        .pivot_table(index='q', columns='Binned', aggfunc='size', fill_value=0)
    )
    ##dataprint = (dataWindd.assign(q=pd.cut(np.clip(dataWindd.correctedwindspeed,binsH[0], binsH[-1]), bins=binsH, labels=binsLH, right=False))
    ##   .pivot_table(index='q', columns='Binned', aggfunc='size', fill_value=0)
    ##)
    #####print(dataprint[columnsB[6]])
    #dataprintMon = (dataWindd.assign(q=pd.cut(np.clip(dataWindd.correctedwindspeed,binsH[0], binsH[-1]), bins=binsH, labels=binsLH, right=False))
    #   .pivot_table(index='q', columns='Binned', aggfunc='size', fill_value=0)
    #)
    #text4 = plt.text(15, .18, "some text1", weight=700, fontsize=16)
    #text5 = plt.text(15, .15, "some text2", weight=700, fontsize=16)
    #text6 = plt.text(15, .12, "some text3", weight=700, fontsize=16)
    textSec4 = [0,0,0,0,0,0,0,0,0,0,0,0]
    textSec5 = [0,0,0,0,0,0,0,0,0,0,0,0]
    textSec6 = [0,0,0,0,0,0,0,0,0,0,0,0]
    shapeSec = [0,0,0,0,0,0,0,0,0,0,0,0]
    locSec = [0,0,0,0,0,0,0,0,0,0,0,0]
    scaleSec = [0,0,0,0,0,0,0,0,0,0,0,0]

    for iSec, group2 in enumerate(grouped, 0):
        if iSec < 12:
            print(iSec)
            try:
                print(xaxis2[iSec])
            except IndexError:
                print("xaxis2")

            try:
                print(xaxisSec[iSec])
            except IndexError:
                print("xaxisSec")

            new_axesH_Sec(group2[1].correctedwindspeed, xaxisSec[iSec], xaxis2[iSec], iSec, textSec4, textSec5, textSec6, shapeSec, locSec, scaleSec)
            """
            plt.savefig(file_path + 'weibullSEC_' + xaxis2[i] + '_temp999_img.png')
            #hist_sectorH = file_path + 'weibullSEC_' + xaxis2[i] + '_temp999_img.png'
            plt.clf()
            plt.close()
            """
            #print(xaxisSec[i])
        else:
            pass
    count = 0
    rose_monthR = [[0,0,0],[0,0,0],[0,0,0],[0,0,0]]
    hist_monthH = [[0,0,0],[0,0,0],[0,0,0],[0,0,0]]
    hist_sectorH = [[0,0,0],[0,0,0],[0,0,0],[0,0,0]]
    for i in range(4):
        for j in range(3):
            rose_monthR[i][j] = 'file:///' + file_path + 'weiWRose_' + xaxis2[count] + '_temp888_img.png'
            hist_monthH[i][j] = 'file:///' + file_path + 'weibull_' + xaxis2[count] + '_temp999_img.png'
            hist_sectorH[i][j] = 'file:///' + file_path + 'weibullSEC_' + xaxis2[count] + '_temp999_img.png'
            count = count + 1
    """
    print("<br>===================GroupbyHoursMean=================<br>")
    print(dataFull['correctedwindspeed'].groupby(lambda x: x.hour).mean())
    print("<br>===================GroupbyHoursSTD_DEV=================<br>")
    print(dataFull['correctedwindspeed'].groupby(lambda x: x.hour).std())
    print("<br>================GroupbyYearsMean====================<br>")
    meanYear = dataFull['correctedwindspeed'].groupby(lambda x: x.year).mean()
    meanYear = meanYear.drop(2016)
    print(meanYear)
    #print(dataFull['correctedwindspeed'].resample('A').count())
    #count could be very useful for frequency table
    print("<br>================GroupbyMonthsMean====================<br>")
    print(dataFull['correctedwindspeed'].groupby(lambda x: x.month).mean())
    print("<br>================GroupbyMonthsSTD_DEV====================<br>")
    print(dataFull['correctedwindspeed'].groupby(lambda x: x.month).std())
    print("<br>================GroupbyMonthsSTDErr====================<br>")
    print(dataFull['correctedwindspeed'].groupby(lambda x: x.month).std()/math.sqrt(len(meanYear)))
    print("<br>====================================")
    """
    #Climate Means Fig 8 ===============================================================
    #meanYear = dataFull['correctedwindspeed'].groupby(lambda x: x.year).mean()
    #meanYear = meanYear.drop(2016)
    #print(meanYear)
    #fig = plt.figure(figsize=(7, 4.17), dpi=80, facecolor='w', edgecolor='w')
    plt.clf()
    plt.close()
    fig = plt.figure(figsize=(7, 4.17), dpi=120, facecolor='w', edgecolor='w')
    meanBoxplot(dataFull)
    plt.savefig(file_path + 'box1_temp444_img.png')
    box1 = "file://" + file_path + "box1_temp444_img.png"

    #Diurnal Fig 2 ===============================================================
    diurnalm = dataFull['correctedwindspeed'].groupby(lambda x: x.hour).mean().to_numpy()
    diurnals = dataFull['correctedwindspeed'].groupby(lambda x: x.hour).std().to_numpy()
   ##print("<br>====================================DiurnalStd")
   ##print(diurnals)
    ########pandas.Series.at_time returns values at time
    #yy2 = abs(random_sample(24)*6)+3
    xaxis2b = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23]
    xaxisHours = ["0:00","1:00","2:00","3:00","4:00","5:00","6:00","7:00","8:00","9:00","10:00","11:00","12:00","13:00","14:00","15:00","16:00","17:00","18:00","19:00","20:00","21:00","22:00","23:00"]
    #fig = plt.figure(figsize=(7, 3.94), dpi=80, facecolor='w', edgecolor='w')
    fig = plt.figure(figsize=(7, 4.04), dpi=120, facecolor='w', edgecolor='w')
    #plt.plot(xaxis2b, diurnalm, color=colourb1, linewidth=3)
    #errorfill(xaxis2a, yline2, yerr=error2, color=colourg1, alpha_fill=0.2)
    errorfill(xaxis2b, diurnalm, yerr=diurnals, color=colourg1, alpha_fill=0.2)
    plt.xticks(xaxis2b,xaxisHours, rotation=45)
    plt.margins(0.05)
    #plt.grid(True)
    dmaxm = (diurnals+diurnalm).max()+1
    #dmaxs = diurnals.max()
    #dmax = dmaxm+dmax

    ax3 = fig.add_subplot(1,1,1)

    # major ticks every 10, minor ticks every 5
    major_ticks3 = np.arange(0, dmaxm, 2)
    minor_ticks3 = np.arange(0, dmaxm, 1)

    #ax.set_xticks(major_ticks)
    #ax.set_xticks(minor_ticks, minor=True)
    ax3.set_yticks(major_ticks3)
    ax3.set_yticks(minor_ticks3, minor=True)

    # and a corresponding grid

    #ax.grid(which='both')

    # or if you want differnet settings for the grids:
    ax3.grid(which='minor', alpha=0.4)
    ax3.grid(which='major', alpha=0.5)



    plt.ylabel('Mean Wind Speed m/s')
    plt.xlabel('Figure 2: Daily Variation in Average Hourly Wind Speeds')
    plt.tight_layout()
    plt.savefig(file_path + 'line4_temp555_img.png')
    diurnal = "file://" + file_path + "line4_temp555_img.png"




    #Monthly Mean, Std, StdErr Fig 9 and Table 2 ===============================================================
    meanYearM = dataFull['correctedwindspeed'].groupby(lambda x: x.year).mean()

    #meanwsO = Decimal(0.00)
    #stdeviaO = Decimal(0.00)
    #stderrO = Decimal(0.00)
    meanwsO = dataFull['correctedwindspeed'].groupby(lambda x: x.month).mean().to_numpy()
    stdeviaO = dataFull['correctedwindspeed'].groupby(lambda x: x.month).std().to_numpy()
    stderrO = dataFull['correctedwindspeed'].groupby(lambda x: x.month).std().to_numpy()/math.sqrt(len(meanYearM)-1)
    meanwsAO = Decimal(0.00)
    stdeviaAO = Decimal(0.00)
    stderrAO = Decimal(0.00)
    for k in range(12):
        #meanwsO[k] = 10*Decimal(meanwsO[k])/10
        #stdeviaO[k] = 10*Decimal(stdeviaO[k])/10
        #stderrO[k] = 10*Decimal(stderrO[k])/10
        meanwsAO += Decimal(meanwsO[k])
        stdeviaAO += Decimal(stdeviaO[k])
        stderrAO += Decimal(stderrO[k])
    meanwsAO = meanwsAO/12
    stdeviaAO = stdeviaAO/12
    stderrAO = stderrAO/12
    xaxis2a = [1,2,3,4,5,6,7,8,9,10,11,12]
    #xaxis2b = [1,2,3,4,5,6,7,8,9,10,11,12,13]
    xaxisMon = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    # error bar values w/ different -/+ errors
    #lower_error = error2
    #upper_error = 0.9*error2
    #asymmetric_error = [lower_error, upper_error]
    fig = plt.figure(figsize=(7, 4.04), dpi=120, facecolor='w', edgecolor='w')
    errorfill(xaxis2a, meanwsO, yerr=stdeviaO, color=colourg1, alpha_fill=0.2)
    #plt.grid(True)
    dmaxO = (meanwsO+stdeviaO).max()+5

    ax2 = fig.add_subplot(1,1,1)

    # major ticks every 10, minor ticks every 5
    major_ticks2 = np.arange(0, dmaxO, 2)
    minor_ticks2 = np.arange(0, dmaxO, 1)

    #ax.set_xticks(major_ticks)
    #ax.set_xticks(minor_ticks, minor=True)
    ##plt.xticks(x,xaxisMon)
    plt.xticks(xaxis2a,xaxisMon)
    ax2.set_yticks(major_ticks2)
    ax2.set_yticks(minor_ticks2, minor=True)

    # and a corresponding grid

    #ax.grid(which='both')

    # or if you want differnet settings for the grids:
    ax2.grid(which='minor', alpha=0.4)
    ax2.grid(which='major', alpha=0.5)



    plt.ylabel('Mean Wind Speed m/s')
    plt.xlabel('Figure 9: Monthly mean wind speed with 16th and 84th percentiles',fontdict=fontLine)
    plt.tight_layout()
    plt.savefig(file_path + 'line2_temp555_img.png')
    line2 = "file://" + file_path + "line2_temp555_img.png"


    #P50:P90 Fig 1 ===============================================================
    #yy50_5ms = abs(random_sample(12)*6)+3
    #yy90_10ms = abs(yy50_5ms - 5)
    #yy99_15ms = threeline(yy50_5ms - 8.75)
    #yy99_15ms = yy50_5ms - 8.75
    #xaxisMon = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    #dataMonthGroup = dataFull.groupby(lambda x: x.month)
    #p50breakdown = [0,0,0]
    #for i, group in enumerate(dataMonthGroup):
    #p50breakdown = new_axesH_MonCDF(dataMonthGroup.correctedwindspeed)
    #p50breakdown = new_axesH_MonCDF(dataMonthGroup)
    #(yy50_5ms, yy90_10ms, yy99_15ms) = new_axesH_MonCDF(dataMonthGroup)
    #p50breakdown = new_axesH_MonCDF(dataFull["correctedwindspeed"])

    p50breakdown = new_axesH_MonCDF(dataMonthGroup,xaxisMon)
    yy50_5ms = p50breakdown[0]
    yy90_10ms = p50breakdown[1]
    yy99_15ms = p50breakdown[2]

    #p50breakdown = new_axesH_MonCDF(dataMonthGroup)
    #yy50_5ms = p50breakdown.iloc[:,[4]]
    #yy90_10ms = p50breakdown.iloc[:,[9]]
    #yy99_15ms = p50breakdown.iloc[:,[14]]
   ##print("<br>============5==============<br>")
   ##print(yy50_5ms)
   ##print("<br>============10==============<br>")
   ##print(yy90_10ms)
   ##print("<br>============15==============<br>")
   ##print(yy99_15ms)
    #print("<br>======p50breakdown=====<br>")
    #print(p50breakdown)
    #xaxis2b = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23]
    #xaxisHours = ["0:00","1:00","2:00","3:00","4:00","5:00","6:00","7:00","8:00","9:00","10:00","11:00","12:00","13:00","14:00","15:00","16:00","17:00","18:00","19:00","20:00","21:00","22:00","23:00"]
    fig = plt.figure(figsize=(7, 4.04), dpi=120, facecolor='w', edgecolor='w')
    #plt.errorbar(xaxis2a, yline2, yerr=error2, fmt='-o')
    #errorfill(xaxis2a, yline2, yerr=error2, color=colourg1, alpha_fill=0.2)
    #test["x"][5:10].plot()
    ##xaxis300a = np.arange(0,len(yy50_5ms),1)
    #plt.plot(xaxis2a, yy50_5ms, color=colourk1, linewidth=2)
    ym50 = int(yy50_5ms.max())
    ym90 = int(yy90_10ms.max())
    ym99 = int(yy99_15ms.max())
    if ym50 < ym90:
        #ymax3 = ym90
        if ym90 < ym99:
            ymax3 = ym99
        elif ym90 > ym99:
            ymax3 = ym90
    elif ym50 > ym90:
        #ymax3 = ym50
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
    #yy50_5ms[0:11].plot(color=colourk1, linewidth=2)
    #yy90_10ms[0:11].plot(color=colourg1, linewidth=2)
    #yy99_15ms[0:11].plot(color=colourb1, linewidth=2)
    plt.xticks(xaxis2a,xaxisMon)
    plt.margins(0.05)
    #leg = plt.legend(title="Probability of Exceeding Wind Speed", fancybox=True, loc='upper right', bbox_to_anchor=(1,1))
    #leg.get_frame().set_alpha(0.7)
    plt.legend(title="Probability of Exceeding Wind Speed", loc='upper right', bbox_to_anchor=(1,1), fontsize='x-small', framealpha=0.7)
    #loc=(0.85,0.85))
    #, loc=(-0.48,0)
    #marker='o',


    ####plt.grid(True)
    #plt.xlabel(xaxisMon)
    #plt.ylabel('Mean Wind Speed m/s', fontdict=fontLine)
    plt.ylabel('Probability of Exceedance')
    plt.xlabel('Figure 1: P50-P90 Probability of Exceedance of Wind Speed Classes')
    #plt.tight_layout()
    #plt.grid(True)
    #plt.axis([0, 12, 0, 1])

    ax = fig.add_subplot(1,1,1)

    # major ticks every 10, minor ticks every 5
    major_ticks = np.arange(0, 108, 10)
    #major_ticks = np.arange(0, 101., 20)
    #minor_ticks = np.arange(0, 101., 10)

    #ax.set_xticks(major_ticks)
    #ax.set_xticks(minor_ticks, minor=True)
    ax.set_yticks(major_ticks)
    ###ax.set_yticks(minor_ticks, minor=True)

    # and a corresponding grid

    #ax.grid(which='both')

    # or if you want differnet settings for the grids:
    ###ax.grid(which='minor', alpha=0.4)
    ax.grid(which='major', alpha=0.5)
    #plt.show()

    #plt.axis([0, 1])
    ##plt.gca().yaxis.grid(True)
    plt.gca().yaxis.set_major_formatter(formatter3)
    plt.tight_layout()
    #locs, labels = xticks()
    #xticks = plt.gca().xaxis.get_major_ticks()
    #xticks(arange(12),('Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'))
    #xticks[0].label1.set_visible(False)
    #xticks[-1].label1.set_visible(True)
    #sizeSTE = fig.get_size_inches()*fig.dpi
    #plt.title('Figure 9: Monthly mean wind speed with 15th and 85th percentiles',fontdict=fontLine)
    plt.savefig(file_path + 'line3_temp555_img.png')
    pfifty = "file://" + file_path + "line3_temp555_img.png"
    #===============================================================================================

    #Wind Frequency Distribution TABLE===========================
    metersclassO = []
    totalCO = Decimal(000.00)
    totalRO = Decimal(000.00)
    for t in range(24):
        metersclassO.append(str(t) + "-" + str(int(t+1)) + "m/s")
    else :
        metersclassO.append("&ge;&nbsp;24m/s")

    #meterspercentO = 0
    meterspercentO = np.zeros((25,12))
    #meterspercentO = random_sample((25,12))
    ###meterspercentO = weibull(2*np.random.random_sample()+1.2,(25,12))
    #dataprint[columnsB[6]]
   ##print(dataprint[columnsB])
    #1 347 2 1032 3 1089 4 1131 5 978 6 1044 7 562 8 381 9 362 10 243 11 179 12 214 13 109 14 92 15 67 16 42 17 21 18 25 19 8 20 3 21 7 22 6 23 4 24 1 25 15
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

   ##print("<br>=========================================<br>")
   ##print(sumdata)

    #meterspercentO[0] = dataprint[columnsB[0]]/sumdata
    #print(meterspercentO[0])
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
    sectorstotalO = [0,0,0,0,0,0,0,0,0,0,0,0]
    meterstotalO = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]

    #10 years
    #meterspercentO = weibull(2*np.random.random_sample()+1.2,(25,12))
    #used to have abs() surrounding each of these, useful for random data
    getcontext().prec = 3
    #getcontext().prec = 5
    for k in range(12):
        for i in range(25):
            totalRO += 100*Decimal(meterspercentO[i][k])/100
    for i in range(25):
        for k in range(12):
            totalCO += 100*Decimal(meterspercentO[i][k])/100
    #getcontext().prec = 4
    for k in range(12):
        for i in range(25):
            sectorstotalO[k] += 100*Decimal(meterspercentO[i][k])/100
    #getcontext().prec = 3
    for i in range(25):
        for k in range(12):
            meterstotalO[i] += 100*Decimal(meterspercentO[i][k])/100
    getcontext().prec = 2
    #tranlates float to 2 place decimal
    for i in range(25):
        for k in range(12):
            meterspercentO[i][k] = 1000*Decimal(meterspercentO[i][k])/1000


    #for i in range(25) :
    #    for k in range(12) :
    #        meterspercentO[i][k] = Decimal(meterspercentO[i][k])
    #x1 = 10 * weibull(2*np.random.random_sample()+1,200)
    #h1 = np.histogram(x1,bins=25,density=True)
    """
    h1 = np.histogram(x1,bins=25,density=True)
    mean = np.mean(x1)
    median = np.median(x1)
    std = np.std(x1)
    stderr = scipy.stats.stderr(x1)
    cdf5 = scipy.norm.cdf(x1,loc=5)
    cdf10 = scipy.norm.cdf(x1,loc=10)
    cdf15 = scipy.norm.cdf(x1,loc=15)
    """


    """
    meanwsO = 4.5*random_sample((12,))+4
    stdeviaO = 2.7*random_sample((12,))+1.2
    #meanwsO.append(0)
    #stdeviaO.append(0)
    meanwsAO = Decimal(0.00)
    stdeviaAO = Decimal(0.00)
    for k in range(12) :
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
    #q1 = dataFull["correctedwindspeed"].quantile(0.05)
    #q3 = dataFull["correctedwindspeed"].quantile(0.95)
    #iqr = q3 - q1
    #filtered = dataFull["correctedwindspeed"].query('(@q1 - 1.644854 * @iqr) <= nb <= (@q3 + 1.644854 * @iqr)')
    #def getIQR(series):
    #    q1 = series.quantile(0.05)
    #    q3 = series.quantile(0.95)
    #    iqr = (series > q1) & (series < q3)
    #    return series[iqr]
    #filtered = dataFull["correctedwindspeed"].apply(getIQR)
    #dataFull.notnull()
    #dataFull["correctedwindspeed"].dropna()
    #dataFull["correctedwindspeed"].notnull()
    #filtered = dataFull["correctedwindspeed"].apply(lambda x : x[ (x > x.quantile(0.05)) & (x < x.quantile(0.95)) ])

    #windspeedw90= Decimal(0.00)
    windspeedw90 = round(stats.trim_mean(dataFull["correctedwindspeed"].to_numpy(),0.1)*100)/100
    longwindspeed= Decimal(0.00)
    windspeedmedian = Decimal(0.00)
    weibullscale = Decimal(0.00)
    weibullshape = Decimal(0.00)
    #fiftyyeargust = Decimal(0.00)
    fiftyyeargust = dataFull["correctedwindspeed"].max().round(3)
    #html_out = template.render(template_vars)
    #HTML(string=html_out).write_pdf(args.outfile.name,stylesheets=["style.css"])
    #HTML(file_path + 'poorplotpy12.php').write_pdf('/home/wiwasol/www/python/wiwasolvet-p7-website4.pdf', stylesheets=["reportpdf.css"])
    #print_html_doc(dataFull)
    # Capture our current directory
    THIS_DIR = os.path.dirname(os.path.abspath(__file__))
    from weasyprint import HTML, CSS
    j2_env = Environment(loader=FileSystemLoader(THIS_DIR),
                             trim_blocks=True)

    template = j2_env.get_template(file_path + 'templatefioReportNOP90.html')
    template_vars = {"title" : "Prospecting Met Mast v1.12",
        #"longwindspeed" : 100*Decimal(np.mean(x1))/100,
        "pmmreport" : "3 Street Address",
        "location" : "~44.6566, -63.5963",
        "monthyear" : "December 31st, 2022",
        "period" : "32",
        "startyear" : "1990",
        "endyear" : "2022",
        "elevation" : "195m",
        "hubheight" : "64m",
        "surfaceroughness" : "0.2",
        "longwindspeed" : round(dataFull["correctedwindspeed"].mean()*100)/100,
        #"cheap 90th percentile, actually work this out!"
        #"windspeedw90" : 100*Decimal(dataFull["correctedwindspeed"].mean()*0.9)/100,
        #"windspeedw90" : round((dataFull["correctedwindspeed"].quantile(0.5)-dataFull["correctedwindspeed"].quantile(0.9))*100)/100,
        #"windspeedw90" : round(dataFull["correctedwindspeed"].var()*100)/100,
        "windspeedw90" : windspeedw90,
        "windspeedmedian" : round(dataFull["correctedwindspeed"].median()*100)/100,
        #"weibullscale" : 100*Decimal(2.1+sp.stats.sem(dataFull["correctedwindspeed"])*0.87)/100,
        #"weibullshape" : 100*Decimal(1.5+sp.stats.sem(dataFull["correctedwindspeed"]))/100,
        "weibullscale" : 0,
        "weibullshape" : 0,
        "fiftyyeargust" : fiftyyeargust,
        #fiftyyearsteady" : Decimal(0,
        #windclass" : 0,
        #meanwindspeedredu" : 0,
        "p50" : pfifty,
        "dailywind" : diurnal,
        "hist_month" : hist_monthH,
        "hist_sector" : hist_sectorH,
        "annualhist" : annualHist,
        "metersclass" : metersclassO,
        "meterspercent" : meterspercentO,
        "sectorstotal" : sectorstotalO,
        "meterstotal" : meterstotalO,
        #"totalr" : totalRO,
        #totalc : totalCO,
        #totalr : totalRO,
        #"totalc" : totalCO,
        "rose_month" : rose_monthR,
        "annualrose" : annualRose,
        "climatemeans" : box1,
        "monthdevia" : line2,
        "monthsA" : ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec", "Annual"],
        "meanws" : meanwsO.round(2),
        "stdevia" : stdeviaO.round(2),
        "stderr" : stderrO.round(2),
        "meanwsA" : round(meanwsAO*1000)/1000,
        "stdeviaA" : round(stdeviaAO*1000)/1000,
        "stderrA" : round(stderrAO*100)/100 }
    html_out = template.render(template_vars)
    #HTML(string=html_out).write_pdf(file_path + 'wiwasolvet-p7-website8.pdf', stylesheets=["" + file_path + "reportpdf.css"])
    HTML(string=html_out).write_pdf(file_path + 'wiwasolvet-p7-website8.pdf', stylesheets=[file_path + "reportpdf.css"])
    #base_url=request.build_absolute_uri()
    #base_url='https://www.wiwasolvet.ca/dev/python/'



#====================================================================================================================
# TODO: this is to create the PDF
# easyprintpdf(retrievedatabase())

# ====================================================================================================================

if __name__ == '__main__':

    # print_html_doc()

    multisite = False
    createpdf = True

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
        cpu_parallel_capable = env_variables["Parallel_CPU"]
        year_based_parallel = env_variables["YEAR_BASED_PARALLEL"]
        num_cores = cpu_count() - 2
    else:
        file_path = env_variables["REMOTE_ROOT"] + env_variables["REMOTE_TEMP_OUTPUT"]
        root_file_path = env_variables["REMOTE_ROOT"]
        output_file_path = file_path
        root_front_cover_path = root_file_path
        cpu_parallel_capable = env_variables["Parallel_CPU"]
        num_cores = cpu_count() - 2

    if createpdf:
        # TODO: this is to create the PDF from CSVs
        # new way of making PDF from yearly CSV files
        easyprintpdf(retrievedatabase_csv())

    # don't need this on full run?

    #list_input = sys.argv[1:]
    #input_file_name = list_input[0]
    # Read input.txt site parameters, such as location, site name, elevation, etc.
    #input_variables = {}
    # input_file = ".env"
    #with open(input_file_name) as file:
    #    for line in file:
    #        if line.startswith('#'):
    #            continue
    #        key, value = line.strip().split('=', 1)
    #        # env_variables.append({'name': key, 'value': value})
    #        input_variables[key] = value

    if multisite:
        # ========================================================================
        api_key = env_variables["DARKSKY_API_KEY"]
        # ========================================================================
        #vardb = callday(44.6566, -63.5963, '1990-01-01', '2022-12-31')

        Yearbins = pd.date_range('1990-01-01', '2022-12-31', freq='AS')
        Yearbinsend = pd.date_range('1990-01-01', '2022-12-31', freq='A')

        listannual = []
        listannualend = []
        for i in Yearbins:
            listannual.append(i.strftime('%Y-%m-%d'))
        for k in Yearbinsend:
            listannualend.append(k.strftime('%Y-%m-%d'))

        ##latlonglist = [[44.7451, -63.1604], [44.7475, -63.1578]]

        #coordinates_filename = "./cwa_230_region_5precision_NS_latlong.csv"
        coordinates_filename = "./cwa_41_region_5precision_PEI_latlong.csv"
        latlong_df = pd.read_csv(coordinates_filename)

        #latlong_df["lat"].to_list()
        #latlong_df["long"].to_list()
        # working on multiple lat/longs
        latlonglist = [[x, y] for x, y in zip(latlong_df["lat"].to_list(), latlong_df["long"].to_list())]
        #latlonglist = [[x, y] for x, y in zip(latlong_df["lat"][4:].to_list(), latlong_df["long"][4:].to_list())]
        # print(latlonglist)
        yearlistwrap = [listannual, listannualend]
        total_sites = len(latlonglist)
        sys.stdout.write("There are {0:d} sites!\n".format(total_sites))
        sys.stdout.flush()
        #print("There are {0:d} sites!".format(total_sites))

        yearbegin = 0
        yearend = 0
        for yearbegin, yearend in zip(yearlistwrap[0], yearlistwrap[1]):
            #print(yearbegin + ", " + yearend)
            sys.stdout.write(yearbegin + ", " + yearend + "\n")
            sys.stdout.flush()
            yearbegin = yearbegin
            yearend = yearend

        t4 = time.time()
        site_count = 1

        if year_based_parallel == 'True':
            for latlong in latlonglist:

                t3 = time.time()

                if cpu_parallel_capable == 'True':
                    results = Parallel(n_jobs=num_cores)(
                        delayed(callday)(latlong[0], latlong[1], yearbegin, yearend) for yearbegin, yearend in zip(yearlistwrap[0], yearlistwrap[1]))
                else:
                    with Pool(num_cores) as p:
                        results = p.starmap(callday, [(latlong[0], latlong[1], yearbegin, yearend) for yearbegin, yearend in zip(yearlistwrap[0], yearlistwrap[1])])

                length_results = len(results)
                timetotal3 = time.time() - t3
                #print(
                #    "Darksky connected: Total time for location {0:d}: {1:.5f}, {2:.5f} roughly 8760 hr * {3:d} years records = {4:.2f} seconds, or {5:.2f} minutes.".format(
                #       site_count, latlong[0], latlong[1], length_results, timetotal3, timetotal3/60))
                sys.stdout.write(
                    "Darksky connected: Total time for location {0:d}: {1:.5f}, {2:.5f} roughly 8760 hr * {3:d} years records = {4:.2f} seconds, or {5:.2f} minutes.\n".format(
                        site_count, latlong[0], latlong[1], length_results, timetotal3, timetotal3 / 60))
                sys.stdout.flush()

                for year_csv in results:
                    try:
                        filename = "weather_" + str(year_csv.index[0])[:4] + ".csv"
                    except IndexError:
                        filename = "weather_indexerror.csv"
                    except Exception:
                        filename = "weather_exception.csv"

                    count_str = "{:03d}".format(site_count)
                    outdir = "./output/" + count_str + "_" + str(latlong[0]) + "_" + str(latlong[1]) + "/"

                    if not os.path.exists(outdir):
                        os.mkdir(outdir)

                    fullpathname = os.path.join(outdir, filename)
                    year_csv.to_csv(fullpathname)

                site_count = site_count + 1
        else:
            #for latlong in latlonglist:

            t3 = time.time()
            #2022-01-01, 2022-12-31
            if cpu_parallel_capable == 'True':
                results = Parallel(n_jobs=num_cores)(
                    delayed(callday)(latlong1, latlong2, "1990-01-01", "2022-12-31") for latlong1, latlong2 in latlonglist)
            else:
                with Pool(num_cores) as p:
                    results = p.starmap(callday, [(latlong1, latlong2, "1990-01-01", "2022-12-31") for latlong1, latlong2
                                                  in latlonglist])
            #if cpu_parallel_capable == 'True':
            #    results = Parallel(n_jobs=num_cores)(
            #        delayed(callday)(latlong1, latlong2, yearlistwrap[0], yearlistwrap[1]) for latlong1, latlong2 in
            #        zip(latlonglist[0], latlonglist[1]))
            #else:
            #    with Pool(num_cores) as p:
            #        results = p.starmap(callday, [(latlong[0], latlong[1], yearbegin, yearend) for yearbegin, yearend in
            #                                      zip(yearlistwrap[0], yearlistwrap[1])])

            length_results = len(results)
            timetotal3 = time.time() - t3
            # print(
            #    "Darksky connected: Total time for location {0:d}: {1:.5f}, {2:.5f} roughly 8760 hr * {3:d} years records = {4:.2f} seconds, or {5:.2f} minutes.".format(
            #       site_count, latlong[0], latlong[1], length_results, timetotal3, timetotal3/60))
            sys.stdout.write(
                "Darksky connected: Total time for year {0:d}: {1}, {2} roughly 8760 hr * {3:d} years records = {4:.2f} seconds, or {5:.2f} minutes.\n".format(
                    site_count, yearlistwrap[0], yearlistwrap[1], length_results, timetotal3, timetotal3 / 60))
            sys.stdout.flush()

            for year_csv in results:
                #sys.stdout.write("{0}".format(year_csv))
                try:
                    filename = "weather_" + str(year_csv.index[0])[:4] + ".csv"
                except IndexError:
                    filename = "weather_indexerror.csv"
                except Exception:
                    filename = "weather_exception.csv"

                count_str = "{:03d}".format(site_count)
                sys.stdout.write("{0:.5f}, {1:.5f} \n".format(year_csv["lat"][0], year_csv["long"][0]))
                sys.stdout.flush()
                outdir = "./output/{0}_{1:.5f}_{2:.5f}/".format(count_str, year_csv["lat"][0], year_csv["long"][0])

                if not os.path.exists(outdir):
                    os.mkdir(outdir)

                fullpathname = os.path.join(outdir, filename)
                year_csv.to_csv(fullpathname)

                site_count = site_count + 1

        timetotal = time.time() - t4
        #print("Completed run! For gathering {0:d} sites and CSV files = {1:.2f} seconds, or {2:.2f} minutes, or {3:.2f} hours.".format(total_sites, timetotal, timetotal/60, timetotal/3600))
        sys.stdout.write(
            "Completed run! For gathering {0:d} sites and CSV files = {1:.2f} seconds, or {2:.2f} minutes, or {3:.2f} hours.\n".format(
                total_sites, timetotal, timetotal / 60, timetotal / 3600))
        sys.stdout.flush()

