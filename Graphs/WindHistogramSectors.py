def new_axesH_Sec(dataSec, MonSec, xaxis2i, i, textSec4, textSec5, textSec6, shapeSec, locSec, scaleSec, formatter1=None, colourb1=None):
    """
    Wind Histograms by degree sectors (directions on a compass)
    """

    import matplotlib.pyplot as plt
    from scipy.stats import weibull_min  # , weibull_max
    fig = plt.figure(figsize=(4.34, 3.22), dpi=100, facecolor='w', edgecolor='w')
    binsH=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25]

    shapeSec[i], locSec[i], scaleSec[i] = weibull_min.fit(dataSec, floc=2)
    textSec4[i] = plt.text(15, .18, r'$mean = %3.2f$' % dataSec.mean(), weight=700, fontsize=15)
    textSec5[i] = plt.text(15, .15, r'$\alpha = %3.2f$' % shapeSec[i], weight=700, fontsize=15)
    textSec6[i] = plt.text(15, .12, r'$\lambda = %3.2f$' % scaleSec[i], weight=700, fontsize=15)

    ax = dataSec.hist(bins=binsH,facecolor=colourb1,normed=True)
    #plt.plot(binsH, 1/(sig[i][j] * np.sqrt(2 * np.pi)) *
    #    np.exp( - (binsH - mean[i][j])**2 / (2 * sig[i][j]**2) ),
    #    linewidth=7, color=colourk1)
    plt.xlabel('m/s')
    plt.title(MonSec)
    # Note: run new function to find maximum frequency to set axis range for all 12?
    plt.axis([0, 25, 0, 0.3])
    plt.grid(True)
    plt.gca().yaxis.set_major_formatter(formatter1)
    plt.tight_layout()

    plt.savefig('/home/wiwasol/prospectingmm/weibullSEC_' + xaxis2i + '_temp999_img.png')
    # hist_sectorH = '/home/wiwasol/prospectingmm/weibullSEC_' + xaxis2[i] + '_temp999_img.png'
    plt.clf()
    plt.close()

    return ax
