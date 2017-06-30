def new_axesRAH(wd, ws):
    import matplotlib.pyplot as plt
    from windrose import WindroseAxes

    fig = plt.figure(figsize=(10, 8), dpi=120, facecolor='w', edgecolor='w')
    binsH=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25]
    ax = WindroseAxes.from_ax(ax=None, fig=fig, rmax = 25)
    ax.histogram(wd, ws, bins=binsH, nsector=12, normed=True, opening=0.8, edgecolor='white')
    plt.xlabel("Figure 2: Long Term Annual Wind Histogram (m/s)", fontdict=fontLine)
    return ax


def new_axesHA(dataFull):
    """
    Annual Wind Histogram
    """
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(8, 6), dpi=120, facecolor='w', edgecolor='w')
    binsH=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25]
    ax = dataFull["correctedwindspeed"].hist(bins=binsH, facecolor=colourb1, normed=True)
    shapeHA, locHA, scaleHA = weibull_min.fit(dataFull["correctedwindspeed"], floc=2)
    plt.text(15, .18, r'$mean = %3.2f$' % dataFull["correctedwindspeed"].mean(), weight=700, fontsize=28)
    plt.text(15, .16, r'$\alpha = %3.2f$' % shapeHA, weight=700, fontsize=28)
    plt.text(15, .14, r'$\lambda = %3.2f$' % scaleHA, weight=700, fontsize=28)
    plt.axis([0, 25, 0, 0.20])
    plt.grid(True)
    plt.gca().yaxis.set_major_formatter(formatter1)
    plt.tight_layout()
    # Referencing which image name this function creates
    # plt.savefig('/home/wiwasol/prospectingmm/WindHistogram_temp8888_img.png')
    # annualHist = "file:///home/wiwasol/prospectingmm/WindHistogram_temp8888_img.png"
    return ax
