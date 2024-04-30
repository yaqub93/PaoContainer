import multiprocessing
from tqdm import tqdm
import numpy as np

#import pyomo.environ as pe
from pao.pyomo import *
from pyomo.environ import *
import pao 
from utils import * 
#from pyomo.contrib.pynumero.sparse import BlockVector
#from pyomo.core import *
from sllib.calculations.seamanship_score import DV
from sllib.datatypes import VesselState
import pandas as pd 
import math
import matplotlib.pyplot as plt 
from concurrent.futures import ThreadPoolExecutor

def rotation_matrix(angle):
    """
    Returns a 2D rotation matrix for the given angle in radians.
    
    Parameters:
        angle (float): The angle in radians.
        
    Returns:
        numpy.ndarray: The 2D rotation matrix.
    """
    cos_theta = np.cos(angle)
    sin_theta = np.sin(angle)
    
    return np.array([[cos_theta, -sin_theta, 0.0],
                     [sin_theta, cos_theta, 0.0],
                     [0.0, 0.0, 1.0]])

def xdot(yaw, u):
    return rotation_matrix(yaw)*u 

def f(x, u, delta_t):
    return x+xdot(x[2],u)*delta_t 

def J(u, theta1, theta2, delta_t, df, i):
    return theta1*u*u + theta2*DDV(u,delta_t, df, i)

def linear_interpolation(x, x1, y1, x2, y2):
    return y1 + (y2 - y1) * ((x - x1) / (x2 - x1))

def J2(u, theta1, theta2, ddv_fun, i):
    u1 = u-1
    u2 = u
    ddv1 = ddv_fun[(i,u1)]
    ddv2 = ddv_fun[(i,u2)]
    ddv = linear_interpolation(u,u1,u2,ddv1,ddv2)
    return theta1*u*u + theta2*ddv

def l2_distance(list1, list2):
    if len(list1) != len(list2):
        raise ValueError("Lists must have the same length")
    
    squared_diff_sum = sum((x - y) ** 2 for x, y in zip(list1, list2))
    distance = math.sqrt(squared_diff_sum)
    return distance

def X_L2(x_expert, x_estimated):
    return np.mean(l2_distance(x_expert, x_estimated))

def U_L2(u_expert, u_estimated):
    return np.mean(l2_distance(u_expert, u_estimated))

#delta_t = 120
sampling_time = 10.0

def DDV(u, delta_t, df, i):
    # DV(alpha: float, domain_geometry: Union[float, Tuple[float, float, float, float]], X: float, Y: float, 
    # V_X: float, V_Y: float, TDV_unit: str = "hours")
    length = df["length"].iloc[0]
    # get relative state
    
    cog_A = df["cog"].iloc[i]+u*delta_t/sampling_time
    sog_A = df["sog"].iloc[i]
    
    # Convert SOG from knots to nautical miles per second
    sog_nautical_miles_per_sec = sog_A / 3600
    
    # Calculate distance traveled
    distance = sog_nautical_miles_per_sec * delta_t
    lat_chg = (distance / 60) * pe.cos(radians(cog_A))
    lon_chg = (distance / 60) * pe.sin(radians(cog_A)) / pe.cos(radians(df["lat"].iloc[i]))
    
    lat_A = df["lat"].iloc[i]+lat_chg
    lon_A = df["lon"].iloc[i]+lon_chg
    vessel_A = VesselState(timestamp=0, lat=lat_A, lon=lon_A, 
                            sog=sog_A, cog=cog_A)
    
    cog_B = df["cog_ts"].iloc[i]
    sog_B = df["sog_ts"].iloc[i]
    
    # Convert SOG from knots to nautical miles per second
    sog_nautical_miles_per_sec = sog_B / 3600
    
    # Calculate distance traveled
    distance = sog_nautical_miles_per_sec * delta_t
    lat_chg = (distance / 60) * pe.cos(radians(cog_B))
    lon_chg = (distance / 60) * pe.sin(radians(cog_B)) / pe.cos(radians(df["lat_ts"].iloc[i]))
    
    lat_B = df["lat_ts"].iloc[i]+lat_chg
    lon_B = df["lon_ts"].iloc[i]+lon_chg
    
    vessel_B = VesselState(timestamp=0, lat=lat_B, lon=lon_B, 
                            sog=sog_B, cog=cog_B)
    lat0, lon0 = vessel_A.lat, vessel_A.lon
    sog_unit = "mps"
    output_unit = "nm"
    R_x, R_y, theta, V_x, V_y = convert_two_vessels_to_relative(vessel_A, vessel_B, lat0, lon0, sog_unit, output_unit) # theta is deg
    length = m2nm(length)
    length_domain = (10*length, 5*length, 2.5*length, 1.25*length)
    #theta = (theta + 180) % 360 - 180
    f_min, DDV_, TDV = DV(radians(theta),length_domain, R_x, R_y, V_x, V_y)
    if np.isnan(DDV_):
        return 0.0 #-df["DDV"].iloc[i]
    else:
        return DDV_ #-df["DDV"].iloc[i]

def get_next_state():
    pass

def J3(u, ddv, theta1, theta2):
    return theta1*u*u + theta2*ddv*ddv
            
df = pd.read_csv("filtered_samso_complete2.csv") # from data
df = df[(df["mmsi"] == 356234000) & (df["scenario"] == "466fbc3092")].reset_index()

max_data = 5
df_final = df.iloc[:max_data].reset_index()

i1s = []
i2s = []
i=0
delta_i = max_data
while i < max_data:
    i1s.append(i)
    i += delta_i
    i2s.append(i)

def upper_obj(theta1, theta2, delta_t, len_u, df):
    ddv_fun = {}
    for i in range(len(df)):
        ddv_fun[i] = []
        for u in input_values:
            ddv = DDV(u, delta_t, df, i)
            ddv_fun[i].append(ddv)
            
    M_o_expr = 0
    for i in range(len_u):
        L = ConcreteModel()
        output_values = ddv_fun[i]

        L.u_est = Var(bounds=(-2,2))
        L.ddv = Var(bounds=(0,1))
        L.con = Piecewise(L.ddv, L.u_est, 
                          pw_pts=input_values, pw_constr_type='EQ', 
                          f_rule=output_values, pw_repn='INC', 
                          warn_domain_coverage= False,
                          warning_tol=0.0)

        L.o = Objective(expr=J3(L.u_est, L.ddv, theta1, theta2),
                        sense=minimize)
        
        SolverFactory('mindtpy').solve(L, mip_solver='glpk', nlp_solver='ipopt') 

        u_est = L.u_est.value
        #print("(", u_est, df["cog_dot"].iloc[i], ")", end=", ")
        #print(theta1.value, theta2.value, u_est)
        M_o_expr += (df["cog_dot"].iloc[i]-u_est)**2
    #print()
    #print("F:",M_o_expr)
    return M_o_expr
    
theta1s = []
theta2s = []
delta_t_opts = []
idx = 0
output_pickle = []
num_cpus = 16
for i1,i2 in zip(i1s, i2s):
    df = df_final.iloc[i1:i2]

    input_values = list(np.round(np.arange(-2,2.1,0.1),1))

    len_u = len(df)

    theta1 = np.arange(-0.5,1.51,0.05)
    theta2 = 1.0-theta1
    delta_ts = [30,60,90,120,150,180]

    def calculate_upper_level(task):
        #print("============")
        #print(t1, t2, ts)
        t1, t2, ts = task
        return (t1, t2, ts), upper_obj(t1, t2, ts, len_u, df)

    dict_out = {}
    with multiprocessing.Pool(processes=num_cpus) as pool:
        tasks = [(t1, t2, ts) for t1, t2 in zip(theta1, theta2) for ts in delta_ts]
        
        # Use tqdm to create a progress bar
        with tqdm(total=len(tasks)) as pbar:
            # Use pool.imap_unordered to get results asynchronously
            results = []
            for result in pool.imap_unordered(calculate_upper_level, tasks):
                results.append(result)
                # Update the progress bar for each completed task
                pbar.update(1)
    print("results")
    print(results)
    for key, value in results:
        dict_out[key] = value
    """
    for t1, t2 in zip(theta1, theta2):
        for ts in delta_ts:
            #print("============")
            #print(t1,t2,ts)
            dict_out[(t1, t2, ts)] = upper_obj(t1, t2, ts, len_u, df)
    """

    min_key = min(dict_out, key=dict_out.get)
    min_value = dict_out[min_key]

    print(min_key, min_value)

    theta1, theta2, delta_t_opt = min_key
    theta1s.append(theta1)
    theta2s.append(theta2)
    delta_t_opts.append(delta_t_opt)
    
    lats, lons = [], []
    lats_c, lons_c = [], []
    lat_A = df["lat"].iloc[0]
    lon_A = df["lon"].iloc[0]
    lat_A_c = df["lat"].iloc[0]
    lon_A_c = df["lon"].iloc[0]
    cog_A = df["cog"].iloc[0]
    cog_A_c = df["cog"].iloc[0]
    print(lat_A, lon_A)
    
    ddv_fun = {}
    for i in range(len(df)):
        ddv_fun[i] = []
        for u in input_values:
            ddv = DDV(u, delta_t_opt, df, i)
            ddv_fun[i].append(ddv)
    
    for i in range(len_u):
        L = ConcreteModel()
        output_values = ddv_fun[i]

        L.u_est = Var(bounds=(-2,2))
        L.ddv = Var(bounds=(0,1))
        
        L.con = Piecewise(L.ddv, L.u_est, 
                          pw_pts=input_values, pw_constr_type='EQ', 
                          f_rule=output_values, pw_repn='INC', 
                          warn_domain_coverage= False,
                          warning_tol=0.0)
        
        L.o = Objective(expr=J3(L.u_est, L.ddv, theta1, theta2),
                        sense=minimize)
        
        SolverFactory('mindtpy').solve(L, mip_solver='glpk', nlp_solver='ipopt') 

        u_est = L.u_est.value
        
        cog_A += u_est*sampling_time/sampling_time
        sog_A = df["sog"].iloc[i]
        
        # Convert SOG from knots to nautical miles per second
        sog_nautical_miles_per_sec = sog_A / 3600
        
        # Calculate distance traveled
        distance = sog_nautical_miles_per_sec * sampling_time
        lat_chg = (distance / 60) * np.cos(np.radians(cog_A))
        lon_chg = (distance / 60) * np.sin(np.radians(cog_A)) / np.cos(np.radians(lat_A))
        
        lats.append(lat_A)
        lons.append(lon_A)
        
        lats_c.append(lat_A_c)
        lons_c.append(lon_A_c)
        
        lat_A += lat_chg
        lon_A += lon_chg
        
        cog_A_c += df["cog_dot"].iloc[i]*sampling_time/sampling_time
        sog_A = df["sog"].iloc[i]
        
        # Convert SOG from knots to nautical miles per second
        sog_nautical_miles_per_sec = sog_A / 3600
        
        # Calculate distance traveled
        distance = sog_nautical_miles_per_sec * sampling_time
        lat_chg_c = (distance / 60) * np.cos(np.radians(cog_A_c))
        lon_chg_c = (distance / 60) * np.sin(np.radians(cog_A_c)) / np.cos(np.radians(lat_A_c))
        
        lat_A_c += lat_chg_c
        lon_A_c += lon_chg_c
    
    output_pickle.append((dict_out, lats, lons, df, theta1, theta2, delta_t_opt))
    
    plt.plot(lats, lons,label="est"+str(idx))
    idx += 1

output_pickle.append(df_final)
# Save the dict_out dictionary to a pickle file
import pickle
with open('output_pickle.pkl', 'wb') as f:
    pickle.dump(output_pickle, f)

plt.plot(df_final["lat"], df_final["lon"],label="gt")
#plt.plot(df["lat_ts"], df["lon_ts"])
#plt.plot(lats_c, lons_c, label="check")
plt.grid()
plt.legend()
plt.savefig("traj.png")

plt.clf()
#for i in range(len(ddv_fun)):
#    plt.plot(input_values,ddv_fun[i], label = i)
#plt.grid()
#plt.legend()
#plt.savefig("u vs ddv.png")

plt.plot(theta1s,label="theta1")
plt.plot(theta2s,label="theta2")
plt.grid()
plt.legend()
plt.savefig("thetas.png")

plt.clf()

plt.plot(delta_t_opts)
plt.grid()
plt.legend()
plt.savefig("delta_t_opts.png")

