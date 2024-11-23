import requests
import pandas as pd
import json
# Uses an API GET call to produce the properties of the planet with the name 'planet_name'
# API "documentation", very limited
# Response is organized as a JSON string as the following examples show:
# [{"name": "Jupiter", "mass": 1.0, "radius": 1.0, "period": 4331.0, "semi_major_axis": 5.204, "temperature": 165.0, "distance_light_year": 8.8e-05, "host_star_mass": 1.0, "host_star_temperature": 6000.0}]
# [{"name": "Neptune", "mass": 0.0537, "radius": 0.346, "period": 59800.0, "semi_major_axis": 30.07, "temperature": 72.0, "distance_light_year": 0.000478, "host_star_mass": 1.0, "host_star_temperature": 6000.0}]
# Where mass is normalized by Jupiter's mass (1.898e27 kg),
# Radius is normalized by Jupiter's radius (69911 km),
# Period is given in days
# Semi_major_axis is given in AU
# Temperatures are in Kelvin
# Star mass is normalized by our sun's mass (1.989e30 kg)
def GetPlanetaryDataFromNinjaAPI(planet_name):
    api_url = 'https://api.api-ninjas.com/v1/planets?name={}'.format(planet_name)
    response = requests.get(api_url, headers={'X-Api-Key': '+gc+uhBWnrmr6k5cF6J9Lw==kACTZw0XrRBwgSt4'})
    if response.status_code == requests.codes.ok:
        print(response.text)
    else:
        print("Error:", response.status_code, response.text)

def GetPlanetaryDataFromHorizonsAPI(body_name, start_date="2024-01-20", end_date="2024-01-21", step_size="1 mo", center_name="SUN"):
    body_naif_id = FindNaifId(body_name)
    center_naif_id = FindNaifId(center_name)
    api_url = f"https://ssd.jpl.nasa.gov/api/horizons.api?format=json&COMMAND='{body_naif_id}'&OBJ_DATA='YES'&MAKE_EPHEM='YES'&EPHEM_TYPE='ELEMENTS'&CENTER='{center_naif_id}'&START_TIME='{start_date}'&STOP_TIME='{end_date}'&STEP_SIZE='{step_size}'"
    response = requests.post(api_url, headers={})
    if response.status_code == requests.codes.ok:
        return response.json()
    else:
        print("Error:", response.status_code, response.text)

def FindNaifId(body_name:str):
    all_data = pd.read_csv("NAIF_IDs.csv")
    body_row = all_data.loc[all_data['name'] == body_name.upper()]
    return body_row.naif_id.values[0]