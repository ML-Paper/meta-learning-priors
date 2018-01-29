import os

def get_data_path():
    # The path of the directory in which raw data is saved:
    data_path = os.path.join(os.path.expanduser("~"), 'ML_data_sets')
    return data_path

