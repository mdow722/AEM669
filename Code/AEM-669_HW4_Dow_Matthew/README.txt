AEM-669 Homework 1
Due Date: Jan 28, 2024

Authored By: Matthew Dow

INCLUDED FILES:
- CelestialBodyData.csv
    - Contains the celestial body data provided on the course home page.

- PlanetaryDataFuncs.py
    - Contains functions and a class definition related to reading and returning
      data from the CelestialBodyData.csv file in different useful formats

- PlanetaryDataRequests.py
    - Contains functions related to obtaining information for almost any desired space object
      using the Horizons API
    - API usage requires importing the 'requests' package, which is not installed by default for some reason

- NAIF_IDs.csv
    - Contains NAIF IDs for most space objects with official NAIF ID codes

- ExampleUseCases.py
    - Imports functions from PlanetaryDataFuncs.py and runs a few examples to get common body data
    - Imports functions from PlanetaryDataRequests.py and runs a couple examples to get info on less common bodies

INSTRUCIONS:
- Relocate to the "AEM-669_HW1_Dow_Matthew" directory
- For the solution for problem 2 to run, the 'requests' package must be installed
  - If it is not already installed, you can just use 'pip install requests' to do so
  - If you don't want to do this, the example file will still run the examles for problem 1
    and then output an error once trying to import the functions for problem 2
- Run the "ExampleUseCases.py" file
  - Problem 1 is shown with 3 examples:
    - 1A: Printing all data in csv
    - 1B: Obtaining 1 property value of a specific planet
    - 1C: Create and save an instance for a planet
  - Problem 2 is shown with 2 examples (if 'requests' is installed):
    - 2A: Obtaining NAIF id of a body
    - 2B: Obtaining ephemeris and other data from JPL Horizons API