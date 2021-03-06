import matplotlib.pyplot as plt


def plot(x_data, y_data, y_data_2, metric):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    fig.subplots_adjust(top=0.85)
    ax.set_title("Comparación {} nuestro vs scikit".format(metric))
    ax.set_ylim([0, 120])
    ax.plot(x_data, y_data, '.r-')
    ax.plot(x_data, y_data_2, '.b-')
    ax.legend(['Nuestro', 'Scikit'])
    ax.grid()
    ax.set_xscale('log', basex=2)
    ax.plot([1, x_data[-1]], [100, 100], 'k')
    plt.show()
