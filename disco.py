"""This Class Disco represents the CGM of a galaxy from a disc model"""
import numpy as np
#logN = np.arange(12, 16, 0.1)
#logN_PDF = logN**(-0.4)
#logN_dist = RanDist(logN, logN_PDF)

class Disco:
    """Represents the CGM of a idealized disc projected in the sky"""

    def __init__(self,h, incl, Rcore=0.1):
        """
        :param h: float, height of disc in kpc
        :param incl: float, inclination angle of disc in degrees
        :param Rcore: float, radius of disk core in kpc, where the probability is maximum
        """

        self.Rcore = Rcore
        self.h = h
        self.incl = incl
        self.incl_rad = np.radians(self.incl)
