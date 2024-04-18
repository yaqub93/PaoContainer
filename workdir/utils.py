import pyomo.environ as pe
from sllib.datatypes.vessel import VesselState
from sllib.conversions.geo_conversions import m2nm, mps2knots, knots2mps
from typing import Union, Tuple, List, Dict

def degrees(radians):
    return radians * (180 / 3.14159)

def radians(degrees):
    return degrees * (3.14159/180.0)

def lat_lon_to_north_east(lat: float, lon: float, lat0: float, lon0: float) -> (float, float):
    """ Converts a point in (latitude, longitude) into (North, East) w.r.t. a reference point (lat_0, lon_0)

    Args:
        lat: Latitude component of the point
        lon: Longitude component of the point
        lat0: Latitude of the reference point
        lon0: Longitude of the reference point

    Returns:
        north, east: North and East of the point
    """
    # Earth radius in meters
    R = 6371000.0

    # Convert degrees to radians
    lat = radians(lat)
    lon = radians(lon)
    lat0 = radians(lat0)
    lon0 = radians(lon0)

    # Calculate differences
    d_lat = lat - lat0
    d_lon = lon - lon0

    # Calculate North and East
    north = R * d_lat
    east = R * pe.cos(lat0) * d_lon

    return north, east

def convert_two_vessels_to_relative(vessel_A: VesselState, vessel_B: VesselState, lat0: float, lon0: float, 
                                               sog_unit = "knots", output_unit = "nm") -> (float, float, float, float, float):
    lat_A = vessel_A.lat
    lon_A = vessel_A.lon
    sog_A = vessel_A.sog
    cog_A = vessel_A.cog

    lat_B = vessel_B.lat
    lon_B = vessel_B.lon
    sog_B = vessel_B.sog
    cog_B = vessel_B.cog

    return convert_lat_lon_sog_cog_to_relative(lat_A, lon_A, sog_A, cog_A,
                                               lat_B, lon_B, sog_B, cog_B,
                                               lat0, lon0, sog_unit, output_unit)
    
def convert_lat_lon_sog_cog_to_relative(lat_A: float, lon_A: float, sog_A: float, cog_A: float, 
                                        lat_B: float, lon_B: float, sog_B: float, cog_B: float, 
                                        lat0: float, lon0: float, sog_unit = "knots", output_unit = "nm") -> (float, float, float, float, float):
    """
    Converts latitude, longitude, speed over ground (SOG), and course over ground (COG) data for two ships
    into relative position and velocity information.

    Args:
        lat_A (float): Latitude of ship A. (Degree)
        lon_A (float): Longitude of ship A. (Degree)
        sog_A (float): Speed over ground of ship A. (Knots or m/s)
        cog_A (float): Course over ground of ship A. (Degree)
        lat_B (float): Latitude of ship B. (Degree)
        lon_B (float): Longitude of ship B. (Degree)
        sog_B (float): Speed over ground of ship B. (Knots or m/s)
        cog_B (float): Course over ground of ship B. (Degree)
        lat0 (float): Reference latitude. (Degree)
        lon0 (float): Reference longitude. (Degree)
        sog_unit (str, optional): Unit for SOG, can be "knots" (default) or "mps".
        output_unit (str, optional): Unit for the output, can be "nm" (default) or "m".

    Returns:
        Tuple of relative position and velocity information:
        - R_x (float): Relative position in the north-south direction. (Nautical Miles or meters)
        - R_y (float): Relative position in the east-west direction. (Nautical Miles)
        - theta (float): Relative heading of ship B with respect to ship A (0-360 degrees).
        - V_x_rel (float): Relative velocity in the east-west direction. (Knots or m/s)
        - V_y_rel (float): Relative velocity in the north-south direction. (Knots or m/s)
    """

    if sog_unit not in ["knots", "mps"]:
        raise ValueError("Invalid SOG unit (sog_unit) parameter. Should be knots or mps")

    if output_unit not in ["nm", "m"]:
        raise ValueError("Invalid output unit (output_unit) parameter. Should be nm or m")

    x_A, y_A = lat_lon_to_north_east(lat_A, lon_A, lat0, lon0) # in meters
    x_B, y_B = lat_lon_to_north_east(lat_B, lon_B, lat0, lon0) # in meters

    # Calculate the relative position vector
    R_x = x_B - x_A # meters
    R_y = y_B - y_A # meters
    if output_unit == "nm":
        R_x = m2nm(R_x) # NM
        R_y = m2nm(R_y) # NM

    # Calculate the heading of ship B relative to ship A
    theta = cog_B - cog_A

    # Ensure the heading is between 0 and 360 degrees
    #theta = (theta + 360) % 360
    #theta = (theta + 360) - 360 * ((theta + 360) // 360)
    
    if sog_unit == "mps" and output_unit == "nm":
        sog_A = mps2knots(sog_A)
        sog_B = mps2knots(sog_B)
    elif sog_unit == "knots" and output_unit == "m":
        sog_A = knots2mps(sog_A)
        sog_B = knots2mps(sog_B)

    V_x_A = sog_A*pe.cos(radians(cog_A))
    V_y_A = sog_A*pe.sin(radians(cog_A))

    V_x_B = sog_B*pe.cos(radians(cog_B))
    V_y_B = sog_B*pe.sin(radians(cog_B))

    return R_x, R_y, theta, V_x_B-V_x_A, V_y_B-V_y_A


