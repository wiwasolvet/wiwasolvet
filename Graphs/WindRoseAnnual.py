def new_axesRA(wd, ws, fontLine=None):
    """
    Annual Wind Rose
    """

    import matplotlib.pyplot as plt
    from windrose import WindroseAxes

    fig = plt.figure(figsize=(10, 8), dpi=120, facecolor='w', edgecolor='w')
    binsR = [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24]
    ax = WindroseAxes.from_ax(ax=None, fig=fig, rmax=35, rect=[0.25, 0.1, 0.75, 0.75])
    ax.bar(wd, ws, nsector=12, bins=binsR, normed=True, opening=0.8, edgecolor='white')
    ax.set_legend()
    ax.legend(title="Wind Speed (m/s)", loc=(-0.48,0))
    plt.xlabel("Figure 3: Long Term Annual Windrose (m/s)", fontdict=fontLine)
    return ax
