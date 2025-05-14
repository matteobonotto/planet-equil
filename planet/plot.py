import matplotlib.pyplot as plt
import matplotlib.lines as mlines

from .types import _TypeNpFloat

def contourf(z:_TypeNpFloat, RR:_TypeNpFloat, ZZ:_TypeNpFloat, title: str = ""):
    plt.figure()
    plt.contourf(RR, ZZ, z, 20)
    plt.axis("equal")
    plt.colorbar()
    plt.title(title)
    plt.show()


def contour(z:_TypeNpFloat, RR:_TypeNpFloat, ZZ:_TypeNpFloat):
    plt.figure()
    plt.contour(RR, ZZ, z, 20)
    plt.axis("equal")
    plt.colorbar()
    plt.show()


def contour_diff(z_ref:_TypeNpFloat, z:_TypeNpFloat, RR:_TypeNpFloat, ZZ:_TypeNpFloat) -> None:
    l1 = mlines.Line2D([], [], label="DNN")
    l2 = mlines.Line2D([], [], color="black", label="FRIDA")

    plt.figure()
    plt.contour(RR, ZZ, z, 10)
    plt.colorbar()
    plt.contour(RR, ZZ, z_ref, 10, colors="black", linestyles="dashed")
    plt.legend(handles=[l1, l2])
    plt.axis("equal")
    plt.show()
