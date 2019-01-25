import matplotlib.pyplot as plt
from scipy.stats import weibull_min  # , weibull_max


def new_axes_h_sec(data_sec, mon_sec, xaxis2i, i, text_sec4, text_sec5, text_sec6, shape_sec, loc_sec, scale_sec,
                   formatter=None, colourb1=None, file_path=None):
    """
    Wind Histograms by degree sectors (directions on a compass)
    """

    # import matplotlib.pyplot as plt
    # from scipy.stats import weibull_min  # , weibull_max
    fig = plt.figure(figsize=(4.34, 3.22), dpi=100, facecolor='w', edgecolor='w')
    bins_h = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25]

    shape_sec[i], loc_sec[i], scale_sec[i] = weibull_min.fit(data_sec)
    text_sec4[i] = plt.text(15, .18, r'$mean = %3.2f$' % data_sec.mean(), weight=700, fontsize=15)
    text_sec5[i] = plt.text(15, .15, r'$\alpha = %3.2f$' % shape_sec[i], weight=700, fontsize=15)
    text_sec6[i] = plt.text(15, .12, r'$\lambda = %3.2f$' % scale_sec[i], weight=700, fontsize=15)

    ax = data_sec.hist(bins=bins_h, facecolor=colourb1, normed=True)
    # plt.plot(bins_h, 1/(sig[i][j] * np.sqrt(2 * np.pi)) *
    #    np.exp( - (bins_h - mean[i][j])**2 / (2 * sig[i][j]**2) ),
    #    linewidth=7, color=colourk1)
    plt.xlabel('m/s')
    plt.title(mon_sec)
    # Note: run new function to find maximum frequency to set axis range for all 12?
    plt.axis([0, 25, 0, 0.3])
    plt.grid(True)
    plt.gca().yaxis.set_major_formatter(formatter)
    plt.tight_layout()

    plt.savefig(file_path + 'weibullSEC_' + xaxis2i + '_temp999_img.png')
    # hist_sectorH = '/home/wiwasol/prospectingmm/weibullSEC_' + xaxis2[i] + '_temp999_img.png'
    plt.clf()
    plt.close()

    return ax
