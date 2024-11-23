# **************************************************************************************
# PROBLEM 1 - CALLING DATA FROM CSV

from PlanetaryDataFuncs import *
# Ex A - printing all data in csv
print("Example 1A:")
PrintAllPlanetData()
print()


# Ex B - Obtaining 1 property value of a specific planet
# Note: Third argument is optional, defaults to False, if True prints a pretty string with the result
print("Example 1B:")
Venus_GM = GetPlanetProperty("veNus", "gravitational_parameter", True)
print()


# Ex C - Create and save an instance for a planet
print("Example 1C:")
Titan = CelestialBody.ImportFromPlanetObject("Titan")
print("Titan's Orbital Period: ",Titan.orbital_period)
print()
# Note: CelestialBody class instances can also be created manually with custom values if desired with CelestialBody(args)

# **************************************************************************************
# PROBLEM 2 - OBTAINING EPHEMERIS DATA FROM JPL DATABASE

from PlanetaryDataRequests import *
# Ex A - Obtaining NAIF id of a body
print("Example 2A:")
body_id = FindNaifId("ADrASTEA")
print("Adrastea's Naif Id is ",body_id)
print()

# Ex B - Obtaining ephemeris and other data from JPL Horizons API
PrintPlanetaryDataFromHorizonsAPI("KALYKE")