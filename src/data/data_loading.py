import pandas as pd
import os

BASE_DIR = BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(BASE_DIR, 'raw_data', 'CMaps')

index_names = ['Engine', 'Cycle']
setting_names = ['Setting 1', 'Setting 2', 'Setting 3']
sensor_names = ['Fan Inlet Temperature (◦R)',
               'LPC Outlet Temperature (◦R)',
               'HPC Outlet Temperature (◦R)',
               'LPT Outlet Temperature (◦R)',
               'Fan Inlet Pressure (psia)',
               'Bypass-Duct Pressure (psia)',
               'HPC Outlet Pressure (psia)',
               'Physical Fan Speed (rpm)',
               'Physical Core Speed (rpm)',
               'Engine Pressure Ratio (P50/P2)',
               'HPC Outlet Static Pressure (psia)',
               'Ratio of Fuel Flow to Ps30 (pps/psia)',
               'Corrected Fan Speed (rpm)',
               'Corrected Core Speed (rpm)',
               'Bypass Ratio',
               'Burner Fuel-Air Ratio',
               'Bleed Enthalpy',
               'Required Fan Speed',
               'Required Fan Conversion Speed',
               'High-Pressure Turbines Cool Air Flow',
               'Low-Pressure Turbines Cool Air Flow',
               'Sensor 26',
               'Sensor 27']

col_names = index_names + setting_names + sensor_names

def load_data(file_name, col_names=None):
    file_path = os.path.join(DATA_DIR, file_name)
    data = pd.read_csv(file_path, sep='\s+', header=None, names=col_names)
    print(f'Data loaded successfully ✅')
    return data

def load_train_data():
    return load_data('train_FD001.txt', col_names=col_names)

def load_test_data():
    return load_data('test_FD001.txt', col_names=col_names)

def load_test_RUL():
    return load_data('RUL_FD0001.txt', col_names=col_names)
