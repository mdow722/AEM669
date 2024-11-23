import math

class Vector2:
    def __init__(self, x: float, y: float, unit=None):
        self.x = x
        self.y = y
        self.unit = unit
    
    def __repr__(self) -> str:
        if self.unit == None:
            return f"<{self.x},{self.y}> with no units"
        else:
            return f"<{self.x},{self.y}> with units of {self.unit}"

    @property    
    def magnitude(self):
        return math.sqrt(self.x**2 + self.y**2)
        
class Vector3:
    def __init__(self, x: float, y: float, z: float, unit=None):
        self.x = x
        self.y = y
        self.z = z
        self.unit = unit

    def __repr__(self) -> str:
        if self.unit == None:
            return f"<{self.x},{self.y},{self.z}> with no units"
        else:
            return f"<{self.x},{self.y},{self.z}> with units of {self.unit}"
        
    @property    
    def magnitude(self):
        return math.sqrt(self.x**2 + self.y**2 + self.z**2)