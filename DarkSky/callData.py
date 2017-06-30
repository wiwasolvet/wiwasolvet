# Initial separation of functions into files
# Note: DarkSky API is accessed via forecastio, I will post a forked repository with the gzip headers speed-up patch.

def callback(forecastio=None):
    """
    Callback function for DarkSky.net API
    """
    return forecastio


def callday(latt2, long2, startdate, enddate, api_key=None):
    import forecastio
    import numpy as np
    import pandas as pd

    dataFull = pd.DataFrame({})
    rngYears = pd.date_range(startdate, enddate, freq='d')
    datalength = rngYears.size
    dataF24 = [i for i in range(datalength)]

    for v, day in enumerate(rngYears, 0):
        url_time = rngYears[v].replace(microsecond=0).isoformat()
        units = 'si'
        exclude0 = 'minutely,currently,daily,alerts'
        # url = 'https://api.forecast.io/forecast/%s/%s,%s,time=%s?units=%s&exclude=%s' %
        # (api_key, latt, long2, url_time, units, exclude0)

        url = 'https://api.darksky.net/forecast/%s/%s,%s,%s?units=%s&exclude=%s' % (api_key, latt2, long2, url_time,
                                                                                    units, exclude0)

        forecastCallstart = forecastio.manual(url, callback(forecastio=None))
        # because of daylight savings time, won't always be 24 hours of data! Could be 23, or 24, or 25

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

        index = len(byHour.data)

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

        for i, hourlyData in enumerate(byHour.data, 0):
            try:
                pre_time.iloc[i] = hourlyData.time.strftime('%Y/%m/%d %H:%M')
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
        dataF24[v] = pd.concat([pre_time, pre_winds, pre_windd, pre_preint, pre_prepro, pre_pretyp, pre_temp, pre_humi,
                                pre_cloud, pre_pres], axis=1)

    dataFull = pd.concat(dataF24, axis=0)
    dataFull.columns = ["time", "winds", "windd", "preint", "prepro", "pretyp", "temp", "humid", "cloud", "press"]
    return dataFull
