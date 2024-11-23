from Units import *
from Vectors import *

class Cartesian:
    def __init__(self, r, v):
        self.r = r
        self.v = v

    @property
    def rmag(self):
        return self.r.magnitude
    
    @property
    def vmag(self):
        return self.v.magnitude

class Cartesian2D(Cartesian):
    def __init__(self, r, v):
        self.r = r
        self.v = v
        

class Cartesian3D(Cartesian):
    def __init__(self, r, v):
        self.r = r
        self.v = v

class ClassicalOrbitalElements:
    def __init__(self, sma:Distance, ecc:float, inc:Angle , raan:Angle, aop:Angle, ta:Angle) -> None:
        self.SMA = sma
        self.ECC = ecc
        self.INC = inc
        self.RAAN = raan
        self.AOP = aop
        self.TA = ta