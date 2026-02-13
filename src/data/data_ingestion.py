import pandas as pd
from sklearn.model_selection import train_test_split
import yaml
import os 
import kagglehub
from kagglehub import KaggleDatasetAdapter
# pip install kagglehub[pandas-datasets]


def load_data(dataset_path:str)->pd.DataFrame:
    print('loading data')
    df = kagglehub.load_dataset(KaggleDatasetAdapter.PANDAS,
                                dataset_path,
                                "placementdata.csv" )

    print("First 5 records:", df.head())
    return df

def load_yml(url)->float:
    print('loading yaml')
    content=yaml.safe_load(open(url,'r'))
    test_size=content['data_ingestion']['test_size']
    return test_size


def split_data(df:pd.DataFrame,test_size:float)->tuple:
   print('splitted data')
   train_ds,test_ds= train_test_split(df,test_size=test_size,random_state=0,shuffle=True)
   return train_ds,test_ds


def save_data(train_ds:pd.DataFrame,test_ds:pd.DataFrame,path:str)->None:
    print('save data')
    os.makedirs(path,exist_ok=True)
    train_ds.to_csv(os.path.join(path,'raw_train_ds.csv'))
    test_ds.to_csv(os.path.join(path,'raw_test_ds.csv'))


def main()->None:
    load_path="ruchikakumbhar/placement-prediction-dataset"
    save_path=os.path.join('data','raw')
    df=load_data(load_path)
    test_size=load_yml('params.yaml')
    datas=split_data(df,test_size)
    save_data(datas[0],datas[1],save_path)


main()


