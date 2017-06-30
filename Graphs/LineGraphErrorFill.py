def errorfill(x, y, yerr, color=None, alpha_fill=0.3, ax=None):
    """
    Line Graphs with Error Fills (shaded areas between max and min data from main data line)
    """

    import matplotlib.pyplot as plt
    import numpy as np

    # xaxisMon = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
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
    # plt.xticks(x, xaxisMon)
    plt.margins(0.05)
    ax.fill_between(x, ymax, ymin, color=color, alpha=alpha_fill)
