def new_axesR(wd, ws, MonSec, font=None):
    """
    Wind Rose for Months
    """

    import matplotlib.pyplot as plt
    from windrose import WindroseAxes

    fig = plt.figure(figsize=(4, 4), dpi=80, facecolor='w', edgecolor='w')
    binsR = [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24]
    ax = WindroseAxes.from_ax(ax=None, fig=fig, rmax=35, rect=[0.1, 0.1, 0.75, 0.75])
    ax.bar(wd, ws, nsector=12, bins=binsR, normed=True, opening=0.8, edgecolor='white')
    plt.title(MonSec, y=1.08, fontdict=font)
    # 4 by 3
    return ax
