import requests
import csv

def PrintPlanetaryDataFromHorizonsAPI(body_name:str, start_date="2024-01-20", end_date="2024-01-21", step_size="1 mo", center_name="SUN"):
    """
    Utilizes the NASA Horizons API to get info about an input body.
    API documentation here: https://ssd-api.jpl.nasa.gov/doc/horizons.html
    """
    body_naif_id = FindNaifId(body_name)
    center_naif_id = FindNaifId(center_name)
    api_url = f"https://ssd.jpl.nasa.gov/api/horizons.api?format=text&COMMAND='{body_naif_id}'&OBJ_DATA='YES'&MAKE_EPHEM='YES'&EPHEM_TYPE='ELEMENTS'&CENTER='{center_naif_id}'&START_TIME='{start_date}'&STOP_TIME='{end_date}'&STEP_SIZE='{step_size}'"
    response = requests.post(api_url, headers={})
    if response.status_code == requests.codes.ok:
        print(response.text)
    else:
        print("Error:", response.status_code, response.text)

def FindNaifId(body_name:str):
    """
    Obtains the official NAIF integer ID code for an input body.
    NAIF IDs were obtained from here: https://naif.jpl.nasa.gov/pub/naif/toolkit_docs/C/req/naif_ids.html
    and organized into a csv file for local use.
    """
    with open('NAIF_IDs.csv', newline='') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=',', quotechar='|')
        for row in spamreader:
            if row[0] == body_name.upper():
                return row[1]
        return print(f"Body '{body_name}' not found")