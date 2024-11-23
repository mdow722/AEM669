from PlanetaryDataApi import *
from Units import *
from Vectors import *
from States import *
from PlanetaryData import *

# body_name = "MARS"
# result_json = GetPlanetaryDataFromHorizonsAPI(body_name)
# # print(dir(result_json))
# print(result_json["result"])

# dis = Distance(12, "mi")
# print(dis.value)
# print(dis.unit)
# print(dis.meters)

# dur = Duration(13, "hr")
# print(dur.value)
# print(dur.unit)
# print(dur.years)

# ang = Angle(13, "deg")
# print(ang.value)
# print(ang.unit)
# print(ang.radians)

# r2 = Vector2(3, 4)
# v2 = Vector2(6, 7)

# print(r2.magnitude)

# cart = Cartesian2D(r2, v2)
# print(type(cart))
# print(cart.rmag)
# print(cart.vmag)

# r3 = Vector3(3, 4, 7)
# v3 = Vector3(1, 5, 2)
# print(type(r3))
# cart = Cartesian(r3, v3)
# print(type(cart))

# df = Planet.ImportAllPlanetData()
# print(df.loc[df['name'] == 'Mercury'].GM.values[0])

df = CelesitalBody.ImportAllPlanetData()
print(df.loc[df['name'] == 'Mercury'].gravitational_parameter.values[0])

# print(Planet.GetPlanetProperty("Mercury", "GM"))
# print(CelesitalBody.GetPlanetProperty("Mercury", "gravitational_parameter"))

# merc = Planet.ImportPlanetObject("Mercury")
# print(merc.state_J2000.INC.degrees)
# merc = CelesitalBody.ImportPlanetObject("Mercury")
# print(merc.inclination.degrees)