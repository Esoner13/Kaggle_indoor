import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import glob
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
import plotly.graph_objs as go
import json

from dataclasses import dataclass
import scipy.signal as signal

from PIL import Image, ImageOps
from skimage import io
from skimage.color import rgba2rgb, rgb2xyz
from tqdm import tqdm
from dataclasses import dataclass
from math import floor, ceil
import cv2

import warnings
warnings.filterwarnings('ignore')
f = open("../input/indoor-location-navigation/test/00ff0c9a71cc37a2ebdd0f05.txt", "r")
print("Dataframe Info:\n")
for line in range(10):
    print(f.readline())

print("Dataframe head in txt form:\n")
for line in range(5):
    print(f.readline())

ss = pd.read_csv("../input/indoor-location-navigation/sample_submission.csv")
ss.head()
spt = ss.site_path_timestamp.values
spliter = lambda id_spt: id_spt.split('_')
spt = np.array([spliter(id_spt) for id_spt in spt])
spt = pd.DataFrame(spt, columns=["Site", "Path", "Timestamp"])
spt.head()
test_sites = spt["Site"].unique()
len_sites = len(test_sites)
print(f"There are {len_sites} sites in the test(submission) file:") 
print(test_sites)
print()

test_paths = spt["Path"].unique()
len_paths = len(test_paths)
print(f"There are {len_paths} paths in the test(submission) file") 
spt['Datetime'] = pd.to_datetime(spt['Timestamp'].astype('int64'), unit='ms')
spt['Datetime'] = spt['Datetime'].apply( lambda d : d.time() ) 
spt.head(10)

FLOOR_MAP = {"B2": -2, 
             "B1": -1, 
             "F1": 0, 
             "F2": 1, 
             "F3": 2, 
             "F4": 3, 
             "F5": 4, 
             "F6": 5, 
             "F7": 6, 
             "F8": 7, 
             "F9": 8,
             "1F": 0, "2F": 1, "3F": 2, "4F": 3, "5F": 4, "6F": 5, "7F": 6, "8F": 7, "9F": 8} 
from indoor_location_competition_20.io_f import read_data_file 
from indoor_location_competition_20.compute_f import split_ts_seq
from indoor_location_competition_20.compute_f import correct_trajectory
from indoor_location_competition_20.compute_f import correct_positions
from indoor_location_competition_20.compute_f import init_parameters_filter
from indoor_location_competition_20.compute_f import get_rotation_matrix_from_vector
from indoor_location_competition_20.compute_f import get_orientation
from indoor_location_competition_20.compute_f import compute_steps
from indoor_location_competition_20.compute_f import compute_stride_length
from indoor_location_competition_20.compute_f import compute_headings
from indoor_location_competition_20.compute_f import compute_step_heading
from indoor_location_competition_20.compute_f import compute_rel_positions
from indoor_location_competition_20.compute_f import compute_step_positions





from main import calibrate_magnetic_wifi_ibeacon_to_position
from main import extract_magnetic_strength
from main import extract_wifi_rssi
from main import extract_ibeacon_rssi
from main import extract_wifi_count

from visualize_f import visualize_trajectory
from visualize_f import visualize_heatmap


ex_building = "5a0546857ecc773753327266"
ex_floor = "F4"
ex_path = "5d11dc28ffe23f0008604f67"

ex_file_path = f"../input/indoor-location-navigation/train/{ex_building}/{ex_floor}/{ex_path}.txt"

ex_file = open(ex_file_path, "r")
col_names = list()
for i in range(10):
    ex_file.readline()
    col_names.append(f"col_{i}")
ex_site = pd.read_csv(ex_file, names=col_names, delimiter='\t')
ex_site.head()

ex_db = read_data_file(ex_file_path) #create a sample database

print("Structure and Shape: ")
print("acce: {}".format(ex_db.acce.shape), "\n" +
      "acacce_uncalice: {}".format(ex_db.acce_uncali.shape), "\n" +
      "ahrs: {}".format(ex_db.ahrs.shape), "\n" +
      "gyro: {}".format(ex_db.gyro.shape), "\n" +
      "gyro_uncali: {}".format(ex_db.gyro_uncali.shape), "\n" +
      "ibeacon: {}".format(ex_db.ibeacon.shape), "\n" +
      "magn: {}".format(ex_db.magn.shape), "\n" +
      "magn_uncali: {}".format(ex_db.magn_uncali.shape), "\n" +
      "waypoint: {}".format(ex_db.waypoint.shape), "\n" +
      "wifi: {}".format(ex_db.wifi.shape))
  
  
  
ex_acce = pd.DataFrame(ex_db.acce, columns=['time','x','y','z'])
ex_gyro = pd.DataFrame(ex_db.gyro, columns=['time','x','y','z'])
ex_magn = pd.DataFrame(ex_db.magn, columns=['time','x','y','z'])
ex_ahrs = pd.DataFrame(ex_db.ahrs, columns=['time','x','y','z'])

ex_waypoints = pd.DataFrame(ex_db.waypoint, columns=['time','X','Y'])

ex_acce.head()

def plot_sensor_info(df, name):    
    cols = ["x", "y", "z"]
    plt.subplots(3, 3, sharex='col', sharey='row', figsize=(16,10))
    plt.suptitle(name, fontsize=22)
    for i in range(3):
        col=cols[i]
        plt.subplot(3, 3, i+1)
        sns.distplot(df[col], axlabel=col+"_axis")
        
        plt.subplot(3, 3, i+4)
        sns.boxplot(df[col], color="#96bcfa")
        
    plt.subplot(3, 1, 3)
    plt.plot(df['z'], color='#69cf83', label='z_axis')
    plt.plot(df['y'], color='#d6b258', label='y_axis')
    plt.plot(df['x'], color='#96bcfa', label='x_axis')
    plt.xlabel('Time')
    plt.ylabel('Sensor Value')
    plt.legend()
    plt.show()


plot_sensor_info(ex_acce, "ACCE")


plot_sensor_info(ex_gyro, "GYRO")


plot_sensor_info(ex_magn, "MAGN")

plot_sensor_info(ex_ahrs, "AHRS")


mwi = calibrate_magnetic_wifi_ibeacon_to_position([ex_file_path])
mwi_df = pd.DataFrame(mwi).T
mwi_df.head()

magnetic_strength = extract_magnetic_strength(mwi)

magn_df = pd.DataFrame(magnetic_strength, index=[0]).T
magn_df.columns = ["magn"]
magn_df.head()


wifi_rssi = extract_wifi_rssi(mwi)
wifi_bssid = list(wifi_rssi.keys())
wifi_rssi_df = pd.DataFrame(dict(wifi_rssi[wifi_bssid[0]])).T
wifi_rssi_df.columns=["RSSI", "old_count"]
wifi_rssi_df.head()


wifi_counts = extract_wifi_count(mwi)


wifi_counts_df = pd.DataFrame(wifi_counts, index=[0]).T
wifi_counts_df.columns = ["count"]
wifi_counts_df.head()

ibeacon_rssi = extract_ibeacon_rssi(mwi)

iBeacon_ummid = list(ibeacon_rssi.keys())
iBeacon_rssi_df = pd.DataFrame(dict(ibeacon_rssi[iBeacon_ummid[0]])).T
iBeacon_rssi_df.columns=["RSSI", "old_count"]
iBeacon_rssi_df.head()

def show_site_png(site):
    '''This functions outputs the visualization of the .png images available
    in the metadata.
    sites: the code coresponding to 1 site (or building)'''
    
    base = '../input/indoor-location-navigation'
    site_path = f"{base}/metadata/{site}/*/floor_image.png"
    floor_paths = glob.glob(site_path)
    n = len(floor_paths)

    # Create the custom number of rows & columns
    ncols = [ceil(n / 3) if n > 4 else 4][0]
    nrows = [ceil(n / ncols) if n > 4 else 1][0]

    plt.figure(figsize=(20, 10))
    plt.suptitle(f"Site no. '{site}'", fontsize=18)

    # Plot image for each floor
    for k, floor in enumerate(floor_paths):
        plt.subplot(nrows, ncols, k+1)

        image = Image.open(floor)

        plt.imshow(image)
        plt.axis("off")
        title = floor.split("/")[5]
        plt.title(title, fontsize=15)

show_site_png(site='5cd56b64e2acfd2d33b592b3')


trajectory = ex_db.waypoint
trajectory = trajectory[:, 1:3]

ex_png_path = f"../input/indoor-location-navigation/metadata/{ex_building}/{ex_floor}/floor_image.png"
ex_json_path = f"../input/indoor-location-navigation/metadata/{ex_building}/{ex_floor}/floor_info.json"

with open(ex_json_path) as json_file:
    json_data = json.load(json_file)
    
width_meter = json_data["map_info"]["width"]
height_meter = json_data["map_info"]["height"]

visualize_trajectory(trajectory = trajectory,
                     floor_plan_filename = ex_png_path,
                     width_meter=width_meter,
                     height_meter=height_meter,
                     title="Waypoint Path",
                     g_size=750,
                     point_color='#76C1A0',
                     start_color='#007B51',
                     end_color='#9B0000')
heat_positions = np.array(list(magnetic_strength.keys()))
heat_values = np.array(list(magnetic_strength.values()))

visualize_heatmap(heat_positions, 
                  heat_values, 
                  ex_png_path,
                  width_meter, 
                  height_meter, 
                  colorbar_title='strength', 
                  title='Magnetic Strength',
                  g_size=750,
                  colorscale='temps')

heat_positions = np.array(list(wifi_counts.keys()))
heat_values = np.array(list(wifi_counts.values()))
# filter out positions that no wifi detected
mask = heat_values != 0
heat_positions = heat_positions[mask]
heat_values = heat_values[mask]

# The heatmap
visualize_heatmap(heat_positions, 
                  heat_values, 
                  ex_png_path, 
                  width_meter, 
                  height_meter, 
                  colorbar_title=' WiFi Counts', 
                  title=f'WiFi Count',
                  g_size=755,
                  colorscale='temps')


print(f'This floor has {len(wifi_rssi.keys())} wifis.')

wifi_bssid = list(wifi_rssi.keys())
target_wifi = wifi_bssid[0]
heat_positions = np.array(list(wifi_rssi[target_wifi].keys()))
heat_values = np.array(list(wifi_rssi[target_wifi].values()))[:, 0]

# The heatmap
visualize_heatmap(heat_positions, 
                  heat_values, 
                  ex_png_path, 
                  width_meter, 
                  height_meter, 
                  colorbar_title='dBm', 
                  title=f'WiFi RSSI ({target_wifi})',
                  g_size=755,
                  colorscale='temps')


print(f'This floor has {len(ibeacon_rssi.keys())} ibeacons.')

ibeacon_ummids = list(ibeacon_rssi.keys())
target_ibeacon = ibeacon_ummids[0]
heat_positions = np.array(list(ibeacon_rssi[target_ibeacon].keys()))
heat_values = np.array(list(ibeacon_rssi[target_ibeacon].values()))[:, 0]

# The heatmap
visualize_heatmap(heat_positions, 
                  heat_values, 
                  ex_png_path, 
                  width_meter, 
                  height_meter, 
                  colorbar_title='dBm', 
                  title=f'iBeacon RSSI ({target_ibeacon})',
                  g_size=755,
                  colorscale='temps')

sensor_df = pd.DataFrame()


step_timestamps, step_indexs, step_acce_max_mins = compute_steps(ex_db.acce)

sensor_df = pd.DataFrame(step_acce_max_mins, index=step_indexs)
sensor_df.columns = ["timestamp", "acce_max", "acce_min", "acce_std"]
sensor_df.head()

stride_lengths = compute_stride_length(step_acce_max_mins)

sensor_df["stride_length"] = stride_lengths[:, 1]
sensor_df.head()

headings = compute_headings(ex_db.ahrs)

headings.shape

step_headings = compute_step_heading(step_timestamps, headings)

sensor_df["step_heading"] = step_headings[:, 1]
sensor_df.head()

rel_positions = compute_rel_positions(stride_lengths, step_headings)

sensor_df["rel_pos_x"] = rel_positions[:, 1]
sensor_df["rel_pos_y"] = rel_positions[:, 2]
sensor_df.head()


step_positions = compute_step_positions(ex_db.acce, ex_db.ahrs, ex_db.waypoint)

sensor_df["step_pos_x"] = step_positions[:, 1]
sensor_df["step_pos_y"] = step_positions[:, 2]
sensor_df.head()

#Fix timestamp
def time_float_to_str(time):
    return str(int(time))

sensor_df["timestamp"] = sensor_df["timestamp"].apply(time_float_to_str)
sensor_df


def plot_sensor_info(df, name='Computed Sensor Info'):
    plt.subplots(6, 2, figsize=(18,24))
    plt.suptitle(name, fontsize=22)
    
    plt.subplot(6, 1, 1)
    plt.plot(df['acce_max'], color='#db5046', label='acce_max')
    plt.plot(df['acce_min'], color='#96bcfa', label='acce_min')
    plt.xlabel('Time')
    plt.ylabel('Value')
    plt.title('ACCE Max & Min Over Time')
    plt.legend(loc='upper left')
    
    plt.subplot(6, 2, 3)
    plt.title('ACCE Max Boxplot')
    sns.boxplot(df['acce_min'], color="#db5046").set(xlabel=None)
    
    plt.subplot(6, 2, 4)
    plt.title('ACCE Min Boxplot')
    sns.boxplot(df['acce_min'], color="#96bcfa").set(xlabel=None)
    
    plt.subplot(6, 1, 3)
    plt.plot(df['stride_length'], color='#69cf83')
    plt.xlabel('Time')
    plt.ylabel('Length')
    plt.title('Stride Length Over Time')
    
    plt.subplot(6, 2 , 7)
    sns.distplot(df['acce_std'], color='#eba834', axlabel='acce_std')
    plt.title('acce_std')
    
    plt.subplot(6, 2, 8)
    sns.distplot(df['step_heading'], color='#34c8ed', axlabel='step_heading')
    plt.title('step_heading')
    
    plt.subplot(6, 1, 5)
    plt.plot(df['rel_pos_x'], color='#96bcfa', label='rel_pos_x')
    plt.plot(df['rel_pos_y'], color='#db5046', label='rel_pos_y')
    plt.legend(loc='upper left')
    plt.title('rel_pos')
    
    plt.subplot(6, 1, 6)
    plt.plot(df['step_pos_x'], color='#96bcfa', label='step_pos_x')
    plt.plot(df['step_pos_y'], color='#db5046', label='step_pos_y')
    plt.legend(loc='upper left')
    plt.title('step_pos')
    
plt.show()


plot_sensor_info(sensor_df)


sensor_df.describe()
