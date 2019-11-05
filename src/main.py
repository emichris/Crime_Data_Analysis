import pandas as pd
import datetime as dt

print("Christian Emiyah's Project - TheDataIncubator")
data = pd.read_csv("Dataset/Arrest_Data_from_2010_to_Present.csv")
pd.set_option('display.max_columns', None)
data.head(5);
data['Arrest Date'] = data['Arrest Date'].astype('datetime64[ns]')
``
# Delete data past Jan 1, 2019
dataset = data[data['Arrest Date'].dt.year != 2019]

#Crimes in 2018
B2018 = dataset[dataset['Arrest Date'].dt.year == 2018]
# to check we only have dates in 2018 use:  B2018['Arrest Date'].groupby(B2018['Arrest Date'].dt.year).count()
crimes2018 = B2018['Arrest Date'].count() 
print("Number of crimes in 2018: %d" % crimes2018);

# 2018 Crimes by areas
mst_crime_by_area_2018 = max(B2018['Area Name'].groupby(B2018['Area Name']).count())
print("Most crimes in one area in 2018: %d" % mst_crime_by_area_2018);

Q = B2018[(B2018['Charge Group Description'] == "Burglary") |       \
          (B2018['Charge Group Description'] == "Robbery") |        \
          (B2018['Charge Group Description'] == "Vehicle Theft") |  \
          (B2018['Charge Group Description'] == "Receive Stolen Property")]
# to check we only have the requested groups only us: 
# Q['Charge Group Description'].groupby(Q['Charge Group Description']).count()

print("The 95% quantile of the age in the selected charge groups: ", Q['Age'].quantile(.95))

# Z-Score of the age groups
Q = B2018[(B2018['Charge Group Description'] != "Pre-Delinquency") | (B2018['Charge Group Description'] != "Non-Criminal Detention")];
group_means = Q['Age'].groupby(Q['Charge Group Description']).mean();
Z = (group_means - group_means.mean())/group_means.std();
print("The largest absolute value of Z-score: ", max(abs(Z)))

# Forecast using trendline
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

num_crimes = dataset['Arrest Date'].groupby(dataset['Arrest Date'].dt.year).count()
num_crimes.plot(); plt.show()

years = num_crimes.keys(); X = np.array(years).reshape(-1, 1)
count = num_crimes.values
reg = LinearRegression().fit(X.reshape(-1, 1), count)
predicted_crimes = reg.predict([[2019]])
print("Number of crimes predicted for 2019 using Linear Estimation: %f" % predicted_crimes[0])


# Location work - within 2km
import re
from math import sqrt, pi, cos

def get_dist_from_Building(x):
    R = 6371; c = (34.050536, -118.247861); #c is location of Bradbury Building 
    dphi = (c[0] - x[0])*pi/180.0;
    d_lamda = (c[1] - x[1])*pi/180.0;
    phi_m = (c[0] - x[0])*pi/(180.0* 2.0)
    return R * sqrt((dphi*dphi) + (cos(phi_m)*d_lamda)**2)

L = dataset[(dataset['Location'] != '(0.0, 0.0)') & (dataset['Arrest Date'].dt.year == 2018)]
lat_lon_re = re.compile(r'\((.*),\ (.*)\)')
L['Lat_Lng'] = [tuple(map(float,lat_lon_re.findall(p)[0])) for p in L['Location']]
L['dist'] = list(get_dist_from_Building(x) for x in L['Lat_Lng'])
count_crimes = L[(L['dist'] <= 2.0)].count()['Arrest Date'];
print("Number of arrest incidents that occurred within 2 km from the Bradbury Building in 2018: %.0f" % count_crimes)

# Arrests made on Pico Boulevard
Pico = B2018[(B2018['Address'].str.contains('PICO')) & (B2018['Location'] != '(0.0, 0.0)')]
Pico.head(5)
Pico['Lat_Lng'] = [tuple(map(float,lat_lon_re.findall(p)[0])) for p in Pico['Location']]
Pico['Lat'] = [x[0] for x in Pico['Lat_Lng']]; Pico['Lng'] = [x[1] for x in Pico['Lat_Lng']];
Pico['Lat'].head(10)

mean_lat = Pico.Lat.mean(); mean_lng = Pico.Lng.mean(); std_lat = Pico.Lat.std(); std_lng = Pico.Lng.std();

Pico = Pico[((Pico['Lat'] - mean_lat) < 2*std_lat) | ((Pico['Lng'] - mean_lng) < 2*std_lng)]
min_loc = Pico.loc[Pico['Lat'].idxmin()][['Lat', 'Lng']]
max_loc = Pico.loc[Pico['Lng'].idxmax()][['Lat', 'Lng']]

def get_dist_two_points(x, c):
    R = 6371; # Earth's radius
    dphi = (c[0] - x[0])*pi/180.0;
    d_lamda = (c[1] - x[1])*pi/180.0;
    phi_m = (c[0] - x[0])*pi/(180.0* 2.0)
    return R * sqrt((dphi*dphi) + (cos(phi_m)*d_lamda)**2)

length_pico = get_dist_two_points(min_loc, max_loc);
num_crimes = Pico.Lat.count()
print("Number of arrest incidents per kilometer on Pico Boulevard in 2018: %.10f" %(num_crimes/length_pico))

# COnditional Probability Analysis
cond_prob_set = dataset[(dataset['Charge Group Code'] != 99)]
cond_prob_set = cond_prob_set.dropna(subset=['Charge Group Code'])

table = pd.crosstab(cond_prob_set['Area ID'], cond_prob_set['Charge Group Code'])
charge_groups = table.columns.values
cond_prob = table/table[charge_groups].sum()
uncond_prob = table[charge_groups].sum()/table[charge_groups].sum().sum()
ratios = cond_prob/uncond_prob
ratio_list = ratios.values.flatten()
ratio_list.sort()
avg_top_5 = ratio_list[-6:-1].mean()
print("The average of the top 5 of the calculated ratios: %.10f"%avg_top_5)
