import numpy as np

class Angle:
    R2D = 180.0 / 3.141592653589793238462643383279
    D2R = 3.141592653589793238462643383279 / 180.0

    @staticmethod
    def rad2deg(rad):
        return rad * Angle.R2D

    @staticmethod
    def deg2rad(deg):
        return deg * Angle.D2R