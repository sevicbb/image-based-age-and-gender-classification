import glob
import pandas as pd

def frontal_data_loader():
    data_path = 'csv_data/*.csv'

    mapped_data = map(pd.read_csv, glob.glob(data_path))
    data = pd.concat(mapped_data).to_dict('records')

    return data