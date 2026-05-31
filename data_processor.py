import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from autogluon.tabular import TabularDataset
from autogluon.features.generators import AutoMLPipelineFeatureGenerator
import pandas as pd
import pickle
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def chicago_storm(c):
  A = 14.004
  t = 120
  b = 11.305
  n = 0.557
  r = 0.35
  rain = []
  for i in range(41, -1, -1):
    x = A * ((1 - n) * i / r + b)
    y = (i / r + b) ** (n + 1)
    rain.append(x/y)

  for i in range(1,79):
    x = A * ((1 - n) * i / (1 - r) + b) / ((i / (1 - r) + b) ** (n + 1))
    rain.append(x)

  return np.array(rain) * c / sum(rain)


def rainfall_generator(data, chicago = True):
  if chicago:
    a = []
    basic = chicago_storm(1)
    for _, row in data.iterrows():
      rain = (row['Precipitation_Depth_mm'] * basic).reshape((120,1))
      # convert precipitation after rain generation
      row['Precipitation_Depth_mm'] = 5 * np.arctan(row['Precipitation_Depth_mm']/20)
      other = np.repeat([row.values], 120, axis = 0)
      ts = np.concatenate((rain, other), axis=1)
      a.append(ts)
    return torch.tensor(np.array(a))
  else:
    return torch.tensor([np.repeat([np.insert(row.values, 0, row['Precipitation_Depth_mm']/120)], 120, axis=0) for _, row in data.iterrows()], dtype=torch.float)
  


# Convert dataframe to dataloader of torch
def data_loader(data, batch_size):
  label = torch.tensor(data[['TSS_mgL', 'Runoff_mm']].values, dtype=torch.float).to(torch.float32)
  x = rainfall_generator(data.drop('TSS_mgL', axis = 1), chicago = True).to(torch.float32)
  dataset = TensorDataset(x, label)
  data_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=False, drop_last = True)
  return data_loader


def raw_data(path):
  data = TabularDataset(path)
  # Extract the rows with valid TSS data
  data = data.loc[data['TSS (mg/L)'].notna()].copy()

  # Replace the expression of range with its ceiling and floor values
  # Note: 8 h = 1/3 d
  data['TSS (mg/L)'] = data['TSS (mg/L)'].str.replace('<','')
  data['Days since last rain'] = data['Days since last rain'].str.replace('>8hrs','0.34')

  # Convert the type of columns
  data['TSS (mg/L)'] = data['TSS (mg/L)'].astype(float)
  data['Days since last rain'] = data['Days since last rain'].astype(float)
  data['EPA_Rain_Zone'] = data['EPA_Rain_Zone'].astype(str)

  # Convert the units to SI
  inch2mm = 25.4
  acre2ha = 0.404685642
  data['Runoff_mm'] = data['Runoff_(in)'] * inch2mm
  data['Precipitation_Depth_mm'] = data['Precipitation_Depth_(in)'] * inch2mm
  data['Drainage_Area_ha'] = data['Drainage_Area_Acres'] * acre2ha

  # Screen the unused columns
  unused_col = ['ID_V4.02', 'Station_Name', 'Latitude', 'Longitude', 'Start_Date',
                'Start_Time', 'End_Date', 'End_Time','Runoff_(in)',
                'Precipitation_Depth_(in)','Drainage_Area_Acres' ]

  data.drop(columns = unused_col, inplace = True)
  
# Fill nan with mean of column
  data.fillna({'Percent_Impervious': data['Percent_Impervious'].mean(),
              'Days since last rain': data['Days since last rain'].mean(),
              'Precipitation_Depth_mm': data['Precipitation_Depth_mm'].mean(),
              'Runoff_mm': data['Runoff_mm'].mean(),
              'Drainage_Area_ha': data['Drainage_Area_ha'].mean()},
              inplace=True)

  data.rename(columns={'Principal Landuses': 'Principal_Landuses', 'Days since last rain': 'ADD',
                      'TSS (mg/L)': 'TSS_mgL' }, inplace = True)
  
  data = data[data['TSS_mgL'] < 1000]
  data = data[data['TSS_mgL'] > 1]
  for col in data.columns:
    if col.startswith('Percent'):
      data[col] = data[col]/100

  return data


def data_transform(x):
    df = x.copy()
  
    df['TSS_mgL'] = np.log1p(df['TSS_mgL'])
    df['Runoff_mm'] = np.log1p(df['Runoff_mm'])
    df['Drainage_Area_ha'] = np.log1p(df['Drainage_Area_ha'])
    df['ADD'] = np.log1p(df['ADD'])

    with open('data/my_transformer','rb') as f:
        auto_ml_pipeline_feature_generator = pickle.load(f)

    with open('data/obj_dict','rb') as f:
        obj_dict = pickle.load(f)

    converted_data = auto_ml_pipeline_feature_generator.transform(df)

    print(auto_ml_pipeline_feature_generator.feature_metadata_in)
    return converted_data, obj_dict


def data_retransform(x, obj_dict):

  df = x.copy()

  df['TSS_mgL'] = np.expm1(df['TSS_mgL'])
  df['Runoff_mm'] = np.expm1(df['Runoff_mm'])
  df['Drainage_Area_ha'] = np.expm1(df['Drainage_Area_ha'])
  df['ADD'] = np.expm1(df['ADD'])

  # When replacing category codes with labels some mappings may contain null (np.nan).
  # Replacing directly on a categorical dtype can raise:
  #   ValueError: Categorical categories cannot be null
  # Convert categorical columns to object before applying replace.
  for col in obj_dict:
    if col in df.columns:
      if isinstance(df[col].dtype, pd.CategoricalDtype) or getattr(df[col].dtype, 'name', None) == 'category':
        df[col] = df[col].astype(object)
      df[col] = df[col].replace(obj_dict[col])

  return df
  


