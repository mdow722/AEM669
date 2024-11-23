class UnitValue:
    def __init__(self, value, unit):
        self.value = value
        self.unit = unit

class Distance(UnitValue):
    # using units of m, km, ft, mi, AU
     
    def Convert(value, initial_unit, target_unit):
        if initial_unit == target_unit:
            return value
        if initial_unit == "m":
            if target_unit == "km":
                return value / 1000
            elif target_unit == "ft":
                return value * 3.281
            elif target_unit == "mi":
                return value / 1609
            elif target_unit == "AU":
                return value / 1.496e11
        elif initial_unit == "km":
            if target_unit == "m":
                return value * 1000
            elif target_unit == "ft":
                return value * 3281
            elif target_unit == "mi":
                return value / 1.609
            elif target_unit == "AU":
                return value / 1.496e8
        elif initial_unit == "ft":
            if target_unit == "m":
                return value /3.281
            elif target_unit == "km":
                return value / 3281
            elif target_unit == "mi":
                return value / 5280
            elif target_unit == "AU":
                return value / 4.908e11
        elif initial_unit == "mi":
            if target_unit == "m":
                return value * 1609
            elif target_unit == "km":
                return value * 1.609
            elif target_unit == "ft":
                return value * 5280
            elif target_unit == "AU":
                return value / 9.296e7
        elif initial_unit == "AU":
            if target_unit == "m":
                return value * 1.496e11
            elif target_unit == "km":
                return value * 1.496e8
            elif target_unit == "ft":
                return value * 4.908e11
            elif target_unit == "mi":
                return value * 9.296e7
        else:
            return print("Initial Unit not recognized.")

        return print("Target Unit not recognized.")
    
    @property
    def meters(self):
        return Distance.Convert(self.value, self.unit, "m")
    
    @property
    def kilometers(self):
        return Distance.Convert(self.value, self.unit, "km")
    
    @property
    def feet(self):
        return Distance.Convert(self.value, self.unit, "ft")
    
    @property
    def miles(self):
        return Distance.Convert(self.value, self.unit, "mi")
    
    @property
    def AU(self):
        return Distance.Convert(self.value, self.unit, "AU")
    
class Duration(UnitValue):
    # using units of s, min, hr, d, yr
    
    def Convert(value, initial_unit, target_unit):
        if initial_unit == target_unit:
            return value
        if initial_unit == "s":
            if target_unit == "min":
                return value / 60
            elif target_unit == "hr":
                return value / 3600
            elif target_unit == "d":
                return value / 86400
            elif target_unit == "yr":
                return value / 31553280
        elif initial_unit == "min":
            if target_unit == "s":
                return value * 60
            elif target_unit == "hr":
                return value / 60
            elif target_unit == "d":
                return value / 1440
            elif target_unit == "yr":
                return value / 525888
        elif initial_unit == "hr":
            if target_unit == "s":
                return value * 3600
            elif target_unit == "min":
                return value * 60
            elif target_unit == "d":
                return value / 24
            elif target_unit == "yr":
                return value / 8764.8
        elif initial_unit == "d":
            if target_unit == "s":
                return value * 86400
            elif target_unit == "min":
                return value * 1440
            elif target_unit == "hr":
                return value * 24
            elif target_unit == "yr":
                return value / 365.2
        elif initial_unit == "yr":
            if target_unit == "s":
                return value * 31553280
            elif target_unit == "min":
                return value * 525888
            elif target_unit == "hr":
                return value * 8764.8
            elif target_unit == "d":
                return value * 365.2
        else:
            return print("Initial Unit not recognized.")

        return print("Target Unit not recognized.")
    
    @property
    def seconds(self):
        return Duration.Convert(self.value, self.unit, "s")
    
    @property
    def minutes(self):
        return Duration.Convert(self.value, self.unit, "min")
    
    @property
    def hours(self):
        return Duration.Convert(self.value, self.unit, "hr")
    
    @property
    def days(self):
        return Duration.Convert(self.value, self.unit, "d")
    
    @property
    def years(self):
        return Duration.Convert(self.value, self.unit, "yr")

class Angle(UnitValue):
    # using units of deg, rad
    def Convert(value, initial_unit, target_unit):
        if initial_unit == target_unit:
            return value
        if initial_unit == "deg":
            if target_unit == "rad":
                return value * (3.141592654 / 180)
        elif initial_unit == "rad":
            if target_unit == "deg":
                return value / (3.141592654 / 180)
        else:
            return print("Initial Unit not recognized.")

        return print("Target Unit not recognized.")
    
    @property
    def degrees(self):
        return Angle.Convert(self.value, self.unit, "deg")
    
    @property
    def radians(self):
        return Angle.Convert(self.value, self.unit, "rad")