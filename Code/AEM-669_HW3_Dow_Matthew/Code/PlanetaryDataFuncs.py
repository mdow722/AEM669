import csv
import re
import numpy as np

G_dim = 6.67430e-11 # N * m^2 / kg^2

def PrintAllPlanetData():
    """
        Parses all data in the CelestialBodyData.csv file and prints it to the terminal.
    """
    with open('CelestialBodyData.csv', newline='') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=' ', quotechar='|')
        for row in spamreader:
            print(', '.join(row))
          
def GetPlanetProperty(planet_name:str, property_name:str, print_result=False):
    """
        Finds the desired value from the CelestialBodyData.csv file that correlates to the input planet name {planet_name}
        and property name {property_name}
    """  
    property_col_index = -1
    with open('CelestialBodyData.csv', newline='') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=',', quotechar='|')
        row_index = 0
        for row in spamreader:
            if row_index == 0:
                property_col_index = row.index(property_name)
            else:
                if row[0].upper() == planet_name.upper():
                    if print_result:
                        print(f"The '{property_name}' of body '{planet_name.capitalize()}' is '{row[property_col_index]}'.")
                    
                    if property_name != 'mass':
                        return row[property_col_index]
                    else:
                        return ParseFloatFromScientificString(row[property_col_index])

            row_index += 1
    return print(f"Body '{planet_name}' not found")

def ParseFloatFromScientificString(string_value):
    pattern = re.compile('-?\ *[0-9]+\.?[0-9]*(?:[Ee]\ *-?\ *[0-9]+)?')
    return [float(x) for x in re.findall(pattern, string_value)][0]

def Get3BodyCharacteristics(primary_name, secondary_name):
    Lstar = float(GetPlanetProperty(secondary_name,"semi_major_axis"))

    primary_mass = GetPlanetProperty(primary_name,"mass")
    secondary_mass = GetPlanetProperty(secondary_name,"mass")
    Mstar = primary_mass + secondary_mass

    Tstar = np.sqrt((Lstar**3)/(G_dim * Mstar))
    mu = secondary_mass/Mstar


    return Lstar,Mstar,Tstar,mu



class CelestialBody:
    """
        This class can be used to store instances of related celestial body data with one instance containing
        data for that one body. Each instance has the following fields (in order):\n
        name {str}\n
        axial_rotational_rate {float}\n
        equatorial_radius {float}\n
        gravitational_parameter {float}\n
        semi_major_axis {float}\n
        orbital_period {float}\n
        eccentricity {float}\n
        inclination {float}\n
        central_body {str}\n
        This class also contains a static method 'ImportFromPlanetObject(planet_name)' which returns a 
        CelestialBody instance built from data from the CelestialBodyData.csv file.
    """  
    def __init__(self,name,axial_rotational_rate,equatorial_radius,gravitational_parameter,semi_major_axis,orbital_period,eccentricity,inclination,central_body) -> None:
        self.name=name
        self.axial_rotational_rate=axial_rotational_rate
        self.equatorial_radius=equatorial_radius
        self.gravitational_parameter=gravitational_parameter
        self.semi_major_axis=semi_major_axis
        self.orbital_period=orbital_period
        self.eccentricity=eccentricity
        self.inclination=inclination
        self.central_body=central_body


    @staticmethod
    def ImportFromPlanetObject(planet_name:str):
        """
        Returns a CelestialBody instance built from data from the CelestialBodyData.csv file related to the
        input string {planet_name}.
        """
        with open('CelestialBodyData.csv', newline='') as csvfile:
            spamreader = csv.reader(csvfile, delimiter=',', quotechar='|')
            for row in spamreader:
                if row[0] == planet_name:
                    return CelestialBody(*row)
        return print(f"Body '{planet_name}' not found")