import matplotlib.pyplot as plt
import matplotlib.lines as mlines


def contourf(z, RR, ZZ, title: str = ""):
    plt.figure()
    plt.contourf(RR, ZZ, z, 20)
    plt.axis("equal")
    plt.colorbar()
    plt.title(title)
    plt.show()
    return


def contour(z, RR, ZZ):
    plt.figure()
    plt.contour(RR, ZZ, z, 20)
    plt.axis("equal")
    plt.colorbar()
    plt.show()
    return


def contour_diff(z_ref, z, RR, ZZ):
    l1 = mlines.Line2D([], [], label="DNN")
    l2 = mlines.Line2D([], [], color="black", label="FRIDA")

    plt.figure()
    plt.contour(RR, ZZ, z, 10)
    plt.colorbar()
    plt.contour(RR, ZZ, z_ref, 10, colors="black", linestyles="dashed")
    plt.legend(handles=[l1, l2])
    plt.axis("equal")
    plt.show()
    return
