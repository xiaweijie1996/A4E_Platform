import pandas as pd
import numpy as np

# read meter data
meter_path = r'data\smart_meter.xlsx'
meter_data = pd.read_excel(meter_path, sheet_name='Sheet1')
meter_data = meter_data.iloc[:,:-2]

# read condition data
condition_path = r'data\house_conditions.xlsx'
con_data = pd.read_excel(condition_path, sheet_name='Sheet1')

#%%
import datetime as dt
# process the smart meter data to have daily comsunption profile
meter_data['Time'] = pd.to_datetime(meter_data['Time'])
meter_data.set_index('Time', inplace=True)
meter_data['Date'] = meter_data.index.date
meter_data['Exact_time'] = meter_data.index.time

#%%
# rearrage the data based on day, time and klant_id
all_user_data = pd.DataFrame()
for c in range(0, meter_data.shape[1]-2):
    _user_data = meter_data.iloc[:,[c,-2,-1]]
    # pivot the data based on day, time
    _user_data = _user_data.pivot_table(index='Date', columns='Exact_time', values=_user_data.columns[0])
    _user_data['klant_id'] = meter_data.columns[c]
    # combine data into one dataframe
    all_user_data = pd.concat([all_user_data, _user_data])

# %%
# merge the condition data with the smart meter data based on klant_id
all_user_data = all_user_data.reset_index()
con_data = con_data.rename(columns={'House_number':'klant_id'})
all_user_data = pd.merge(all_user_data, con_data, on='klant_id', how='left')

# save the data
all_user_data.to_csv(r'data\all_user_data.csv', index=False)

# %%
# cancel 'klant_id'
all_user_data = all_user_data.drop(columns='klant_id')

# use one hot encoding to encode house type, house_age and family_type
all_user_data = pd.get_dummies(all_user_data, columns=['House_type', 'House_age', 'Family_type'])

# change false to 0 and true to 1 in dummy columns
all_user_data = all_user_data.replace({False:0, True:1})

# save
all_user_data.to_csv(r'data\all_user_data_dummy.csv', index=False)

