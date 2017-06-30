# Used to print data in browser, or eventually log data to file


def printdata(dataFull):
    print(dataFull)

    print("<br>")
    print(dataFull["winds"].to_string(index=False))
    print("<br>")

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
    import Graphs.WindRoseMonths
    import matplotlib.pyplot as plt
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
    print("<br>")
    print(dataFull["windd"].mean())
    print("<br>")
    print(dataFull["windd"].std())

    fig = plt.figure(figsize=(4.34, 3.22), dpi=100, facecolor='w', edgecolor='w')
    Graphs.WindRoseMonths.new_axesR(dataFull["windd"],dataFull["correctedwindspeed"])
    plt.savefig('/home/wiwasol/prospectingmm/WindRose_temp8888_img.png')
    plt.clf()
    plt.close()
    import Graphs.WindHistogramAnnual
    Graphs.WindHistogramAnnual.new_axesHA(dataFull)
    plt.savefig('/home/wiwasol/prospectingmm/WindHistogram_temp8888_img.png')
    plt.clf()
    plt.close()
