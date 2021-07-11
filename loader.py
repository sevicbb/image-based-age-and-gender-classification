import glob
import pandas as pd

def data_loader():
    data_path = 'csv_data/*.csv'

    mapped_df = map(pd.read_csv, glob.glob(data_path))
    df = pd.concat(mapped_df).to_dict('records')

    return df