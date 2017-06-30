def meanBoxplot(dataFull,formatter4=None, colourb1=None, fontLine=None):
    """
    Box Plot with each year represented as a tick, including 30-60 years historical
    and 10 forecast climate years
    """

    import numpy as np
    import pylab
    import matplotlib
    matplotlib.use('agg')
    import matplotlib.pyplot as plt

    meanYear = dataFull['correctedwindspeed'].groupby(lambda x: x.year).mean()
    meanyears = meanYear.as_matrix()
    overallMean = meanYear.mean()
    overallMedian = meanYear.median()
    # medianbox2 = dataFull['correctedwindspeed'].groupby(lambda x: x.year).median()
    # medianbox = medianbox2.as_matrix()
    spread2 = ((meanyears-overallMedian)/overallMean)
    yx = np.ones(len(meanyears))
    print("len(meanyears) "+str(len(meanyears)))
    print("yx "+str(yx))
    print("spread2 "+str(spread2))
    fig = plt.figure(figsize=(7, 4.17), dpi=120, facecolor='w', edgecolor='w')
    bp = plt.boxplot(spread2, whis=[5,95],showcaps=None,patch_artist=True,showfliers=False)
    pylab.setp(bp['boxes'], facecolor=colourb1, alpha=0.5)
    pylab.setp(bp['whiskers'], color='white', linestyle = 'solid', alpha=0)
    pylab.setp(bp['fliers'], color='black', alpha = 1, marker= '+', markersize = 32)
    pylab.setp(bp['medians'], color='black')
    pylab.setp(bp['caps'], color=colourb1, linewidth=2)
    bp2 = plt.boxplot(spread2, whis=[5,95],showcaps=None,showfliers=False)
    pylab.setp(bp2['boxes'], color='black', alpha=1)
    pylab.setp(bp2['medians'], color='black')
    pylab.setp(bp2['whiskers'], color='white', linestyle = 'solid', alpha=0)
    pylab.setp(bp2['caps'], color=colourb1, linewidth=2)
    scatter = plt.scatter(yx, spread2, alpha=1, c='black', s=320, marker='_')
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
    # Referencing which image name this function creates
    # plt.savefig('/home/wiwasol/prospectingmm/box1_temp444_img.png')
    # box1 = "file:///home/wiwasol/prospectingmm/box1_temp444_img.png"
    return bp, bp2, scatter
