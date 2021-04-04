import typing as t
import typing_extensions as te

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


class DatasetReader(te.Protocol):
    def __call__(self) -> pd.DataFrame:
        ...


SplitName = te.Literal["train", "test"]

def splitter(X,y):
    train_indices = X['yr'] == 0
    X_train, y_train = X[train_indices] , y[train_indices]
    X_test, y_test = X[~train_indices], y[~train_indices]
    return X_train, X_test, y_train, y_test

def get_dataset(reader: DatasetReader, splits: t.Iterable[SplitName]):
    df = reader()
    df = clean_dataset(df)
    y = df["cnt"]
    X = df
    X_train, X_test, y_train, y_test = train_test_split(X, y)
    split_mapping = {"train": (X_train, y_train), "test": (X_test, y_test)}
    return {k: split_mapping[k] for k in splits}


def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
    cleaning_fn = _chain(
        [
            _fix_missing_dates,
            _fix_dteday,
            _fix_yr,
            _fix_mnth,
            _fix_hr,
            _fix_weekday,
            _fix_holiday,
            _fix_workingday,
            _fix_season,
            _fix_weathersit,
            _fix_temp,
            _fix_atemp,
            _fix_hum,
            _fix_windspeed,
            _fix_casual,
            _fix_registered,
            _fix_cnt,
            _fix_columns
        ]
    )
    df = cleaning_fn(df)
    return df


def _chain(functions: t.List[t.Callable[[pd.DataFrame], pd.DataFrame]]):
    def helper(df):
        for fn in functions:
            df = fn(df)
        return df

    return helper


def _fix_missing_dates(df):
    fecha_inicio=df["dteday"][0]

    if df["hr"][0]<10:
        fecha_inicio+=" 0"+str(df["hr"][0])+":00:00"
    else:
        fecha_inicio+=" "+str(df["hr"][0])+":00:00"
        
    print(fecha_inicio)

    fecha_fin = df["dteday"][len(df)-1]
    print(fecha_fin)

    if df["hr"][len(df)-1] < 10:
        fecha_fin += " 0"+str(df["hr"][len(df)-1])+":00:00"
    else:
        fecha_fin += " "+str(df["hr"][len(df)-1])+":00:00"
        

    dteday=df['dteday']


    df['dteday'] = pd.to_datetime(df['dteday'])
    df.index = df['dteday'] + df['hr'].apply(pd.Timedelta,unit='hour')
    df['dteday']=np.array(dteday)
    
    ### Create an index with all datetimes bewtween the specific period
    
    rng = pd.date_range(fecha_inicio, fecha_fin, freq='1H')
    
    ### Re index the df to obtain null values
    
    df = df.reindex(rng)
    
    ### get nan index
    
    df["index_control"]=list(df.reset_index()["index"].astype(str))
    df = df.reset_index()
    df = df.drop(["index","instant"],axis=1)
    df = df.reset_index()
    df = df.rename({"index":"instant"},axis=1) 

    return df





def _fix_dteday(df):

    index_na = df[pd.isna(df["yr"])].index
    for i in index_na:    
        df.at[i,"dteday"] = df["index_control"][i][0:10]
    return df
    

def _fix_yr(df):
    index_na = df.loc[pd.isna(df["yr"]), :].index
    for i in index_na:
        if df["index_control"][i][3]=="1":
            df.at[i,"yr"] = 0
        else:
            df.at[i,"yr"] = 1
    return df

def _fix_mnth(df):
    index_na = df.loc[pd.isna(df["mnth"]), :].index
    for i in index_na:
        df.at[i,"mnth"] = int(df["index_control"][i][5:7])
    return df


def _fix_hr(df):
    index_na = df.loc[pd.isna(df["hr"]), :].index
    for i in index_na:
        df.at[i,"hr"] = int(df["index_control"][i][11:13])
    return df

def _fix_weekday(df):
    index_na = df.loc[pd.isna(df["weekday"]), :].index
    for i in index_na:
        if df.loc[i-1]["dteday"][-2:]==df.loc[i]["dteday"][-2:]:
            df.at[i,"weekday"] = int(df.loc[i-1]["weekday"])
        else:
            if df.loc[i-1]["weekday"]+1==7:
                df.at[i,"weekday"] = 0
            else:
                df.at[i,"weekday"] = df.loc[i-1]["weekday"]+1
    return df


def _fix_holiday(df):
    index_na = df.loc[pd.isna(df["holiday"]), :].index
    for i in index_na:
        if df.loc[i-1]["dteday"][-2:]==df.loc[i]["dteday"][-2:]:
            df.at[i,"holiday"] = int(df.loc[i-1]["holiday"])
        else:
            df.at[i,"holiday"] = 0
    return df


def _fix_workingday(df):
    index_na = df.loc[pd.isna(df["workingday"]), :].index
    for i in index_na:
        if (df.loc[i]["weekday"]==6 or df.loc[i]["weekday"]==0):
            df.at[i,"workingday"] = 0
        else:
            df.at[i,"workingday"] = 1
    return df

def _fix_season(df):
    index_na = df.loc[pd.isna(df["season"]), :].index
    for i in index_na:
        df.at[i,"season"] = df.loc[i-1]["season"]
    return df


def _fix_weathersit(df):
    index_na = df.loc[pd.isna(df["weathersit"]), :].index
    for i in index_na:
        df.at[i,"weathersit"] = df.loc[i-1]["weathersit"]
    return df

def _fix_temp(df):
    index_na = df.loc[pd.isna(df["temp"]), :].index
    for i in index_na:
        df.at[i,"temp"] = df["temp"][i-1]
    return df

def _fix_atemp(df):
    index_na = df.loc[pd.isna(df["atemp"]), :].index
    for i in index_na:
        df.at[i,"atemp"] = df["atemp"][i-1]
    return df

def _fix_hum(df):
    index_na = df.loc[pd.isna(df["hum"]), :].index
    for i in index_na:
        df.at[i,"hum"] = df["hum"][i-1]
    return df

def _fix_windspeed(df):
    index_na = df.loc[pd.isna(df["windspeed"]), :].index
    for i in index_na:
        df.at[i,"windspeed"] = df["windspeed"][i-1]
    return df

def _fix_casual(df):
    index_na = df.loc[pd.isna(df["casual"]), :].index
    for i in index_na:
        df.at[i,"casual"] = df["casual"][i-1]
    return df

def _fix_registered(df):
    index_na = df.loc[pd.isna(df["registered"]), :].index
    for i in index_na:
        df.at[i,"registered"] = df["registered"][i-1]
    return df

def _fix_cnt(df):
    index_na = df.loc[pd.isna(df["cnt"]), :].index
    for i in index_na:   
        df.at[i,"cnt"] = df["cnt"][i-1]
    return df  


def _fix_columns(df):
    df=df.drop(["index_control"],axis=1)
    return df














