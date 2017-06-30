def new_axesH_Mon(dataMonth, MonSec, xaxis2i, i, textMon1, formatter1=None, colourb1=None):
    """
    Wind Histograms Months
    """

    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.stats import weibull_min  # , weibull_max

    fig = plt.figure(figsize=(4.34, 3.22), dpi=100, facecolor='w', edgecolor='w')
    binsH=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25]

    shapeMon, locMon, scaleMon = weibull_min.fit(dataMonth, floc=2)
    textMon1[i] = plt.text(15, .18, r'$mean = %3.2f$' % dataMonth.mean(), weight=700, fontsize=15)
    plt.text(15, .15, r'$\alpha = %3.2f$' % shapeMon, weight=700, fontsize=15)
    plt.text(15, .12, r'$\lambda = %3.2f$' % scaleMon, weight=700, fontsize=15)

    ax = dataMonth.hist(bins=binsH,facecolor=colourb1,normed=True)
    histogramMonth = np.histogram(dataMonth, bins=binsH, range=(0, 100), normed=True)
    print("#####################################################################")
    print(histogramMonth)
    #plt.plot(binsH, 1/(sig[i][j] * np.sqrt(2 * np.pi)) *
    #    np.exp( - (binsH - mean[i][j])**2 / (2 * sig[i][j]**2) ),
    #    linewidth=7, color=colourk1)
    plt.xlabel('m/s')
    plt.title(MonSec)

    plt.axis([0, 25, 0, 0.30])
    plt.grid(True)
    plt.gca().yaxis.set_major_formatter(formatter1)
    plt.tight_layout()

    plt.savefig('/home/wiwasol/prospectingmm/weibull_' + xaxis2i + '_temp999_img.png')
    # hist_monthH = '/home/wiwasol/prospectingmm/weibull_' + xaxis2[i] + '_temp999_img.png'
    plt.clf()
    plt.close()
    return ax


def get_hist(ax12):
    import pandas as pd
    n = 0
    binsA = 0
    n, binsA = [], []
    for rect in ax12.patches:
        ((x0, y0), (x1, y1)) = rect.get_bbox().get_points()
        n.append(y1-y0)
        binsA.append(x0) # left edge of each bin
    binsA.append(x1) # also get right edge of last bin
    return pd.Series(n)


def get_hist2(ax12):
    n = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

    for i, rect in enumerate(ax12.patches):
        ((x0, y0), (x1, y1)) = rect.get_bbox().get_points()
        n[i] = (y1-y0)

    return n


def new_axesH_MonCDF(dataMonthGroup,months):
    import pandas as pd
    import numpy as np

    n2 = []
    binsH = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25]
    for i, group2 in enumerate(dataMonthGroup):
        n2 = get_hist(group2[1].correctedwindspeed.hist(bins=binsH, cumulative=True))

    axMonth5 = pd.DataFrame(np.zeros(12))
    axMonth10 = pd.DataFrame(np.zeros(12))
    axMonth15 = pd.DataFrame(np.zeros(12))
    nAll = pd.DataFrame(np.zeros((12, 25)))
    nAll.index = months
    countiter = 0
    for i2 in range(12):
        for i3 in range(25):
            nAll.loc[months[i2], [i3]] = n2[countiter]
            countiter = countiter +1


    axMonth5 = nAll.iloc[:,[4]]
    axMonth10 = nAll.iloc[:,[9]]
    axMonth15 = nAll.iloc[:,[14]]

    return axMonth5, axMonth10, axMonth15


def new_axesH_MonCDF2(dataMonth300, monSec):
    binsH = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25]
    ax = dataMonth300.hist(bins=binsH)
    bx = get_hist(ax)

    return bx
