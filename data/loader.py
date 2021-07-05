import pandas as pd

def data_loader():
    data = pd.read_csv('data.csv')
    print(data)

def frontal_data_loader():
    frontal_data = pd.read_csv('frontal_data.csv')
    print(frontal_data)