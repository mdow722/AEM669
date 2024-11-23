import pandas as pd
import csv
from States import *

class Planet:
    def __init__(self, name, mass, radius, surface_gravity, GM, J2, orbital_period, semi_major_axis, eccentricity, inclination, RAAN, AOP):
        self.name = name
        self.mass = mass
        self.radius = Distance(radius,"km")
        self.surface_gravity = surface_gravity
        self.GM = GM
        self.J2 = J2
        self.orbital_period = Duration(orbital_period,"d")
        self.state_J2000 = ClassicalOrbitalElements(Distance(semi_major_axis, "km"), eccentricity, Angle(inclination, "deg"), Angle(RAAN, "deg"), Angle(AOP, "deg"), Angle(0, "deg"))

    @staticmethod
    def ImportAllPlanetData():
        return pd.read_csv("PlanetaryData.csv")

    @staticmethod
    def GetPlanetProperty(planet_name, property_name):
        all_data = pd.read_csv("PlanetaryData.csv")
        return all_data.loc[all_data["name"] == planet_name][property_name].values[0]

    @staticmethod
    def ImportPlanetObject(planet_name):
        all_data = pd.read_csv("PlanetaryData.csv")
        planet_row = all_data.loc[all_data['name'] == planet_name]
        return Planet(planet_name, planet_row.mass.values[0], planet_row.radius.values[0], planet_row.surface_gravity.values[0], planet_row.GM.values[0], planet_row.J2.values[0], planet_row.orbital_period.values[0], planet_row.semimajor_axis_KM.values[0], planet_row.orbital_eccentricity.values[0], planet_row.orbital_inclination.values[0], planet_row.orbital_RAAN.values[0], planet_row.orbital_AOP.values[0])
    
class CelesitalBody:
    def __init__(self,name,axial_rotational_rate,equatorial_radius,gravitational_parameter,semi_major_axis,orbital_period,eccentricity,inclination,central_body) -> None:
        self.name=name
        self.axial_rotational_rate=axial_rotational_rate
        self.equatorial_radius=Distance(equatorial_radius,"km")
        self.gravitational_parameter=gravitational_parameter
        self.semi_major_axis=Distance(semi_major_axis,"km")
        self.orbital_period=Duration(orbital_period,"d")
        self.eccentricity=eccentricity
        self.inclination=Angle(inclination,"deg")
        self.central_body=central_body

    @staticmethod
    def ImportAllPlanetData():
        with open('ProvidedData.csv', newline='') as csvfile:
            spamreader = csv.reader(csvfile, delimiter=' ', quotechar='|')
            for row in spamreader:
                print(', '.join(row))
        # return pd.read_csv("ProvidedData.csv")

    @staticmethod
    def GetPlanetProperty(planet_name, property_name):
        all_data = pd.read_csv("ProvidedData.csv")
        return all_data.loc[all_data["name"] == planet_name][property_name].values[0]

    @staticmethod
    def ImportPlanetObject(planet_name):
        all_data = pd.read_csv("ProvidedData.csv")
        planet_row = all_data.loc[all_data['name'] == planet_name]
        return CelesitalBody(planet_name, planet_row.axial_rotational_rate.values[0], planet_row.equatorial_radius.values[0], planet_row.gravitational_parameter.values[0], planet_row.semi_major_axis.values[0], planet_row.orbital_period.values[0], planet_row.eccentricity.values[0], planet_row.inclination_to_elciptic.values[0], planet_row.central_body.values[0])