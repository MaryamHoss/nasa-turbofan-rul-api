import pandas as pd
from sklearn.preprocessing import StandardScaler

# Define column names
# In the dataset description:
#1) unit number
#2) time, in cycles
#3) operational setting 1
#4) operational setting 2
#5) operational setting 3
#6) sensor measurement 1
#7) sensor measurement 2 .....
#26) sensor measurement 21

def load_data_p(path):
    cols=(['units', 'cycles'] +
          [f"op_setting_{i}" for i in range(1,4)] +
          [f"sensor_{i}" for i in range(1,22)])
    df = pd.read_csv(
        path,
        sep=r"\s+",
        header=None,
        names=cols
    )
    return df

def extract_healthy(df):
    healthy_list = []
    for unit_id, groups in df.groupby("units"):
        T = len(groups)
        cutoff = int(0.3 * T)
        healthy_list.append(groups.iloc[:cutoff])
    healthy_df = pd.concat(healthy_list).copy()
    healthy_df.columns = healthy_df.columns.str.strip()
    return healthy_df

def scale_data(healthy_df, df, columns):
    scaler = StandardScaler().set_output(transform='pandas')
    healthy_df[columns] = scaler.fit_transform(healthy_df[columns])
    df[columns] = scaler.transform(df[columns])
    return healthy_df, df, scaler



def load_and_preprocess(path):
    original_df = load_data_p(path)

    df_new = original_df.drop(columns=[f"op_setting_{i}" for i in range(1,4)]).copy()
    df_new.columns = df_new.columns.str.strip()

    healthy_df = extract_healthy(df_new)

    sensor_cols = [f"sensor_{i}" for i in range(1, 22)]
    healthy_df, df_new, scaler = scale_data(healthy_df, df_new, sensor_cols)


    '''
    Check 1 — Number of unique engines
    df_new['units'].nunique() #should match the following
    df['units'].nunique()
    Check 2 — Value counts per engine
    df_new['units'].value_counts().sort_index()
    Check 3 — Data type check
    df_new[['units', 'cycles']].dtypes #both should be integers
    
    --
    (df_new[['units', 'cycles']] 
    == df[['units', 'cycles']]).all()'''
    #save the scaler
    #joblib.dump(scaler, 'scaler.pkl')
    #load with: scaler = joblib.load("scaler.pkl")
    return df_new, original_df, healthy_df, scaler



