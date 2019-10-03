#!/usr/bin/env python
# coding: utf-8

# # IBM Data Science Final Project - Capstone

# ## Install all the Libraries

# In[3]:


get_ipython().system('conda install -c conda-forge beautifulsoup4 --yes')

get_ipython().system('conda install -c conda-forge geopy --yes')

get_ipython().system('conda install -c conda-forge folium=0.5.0 --yes')

print('Libraries installed!')


# ## Import all the Libraries

# In[4]:


import numpy as np
import pandas as pd
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

import requests
from pandas.io.json import json_normalize
import json

import requests

from bs4 import BeautifulSoup

from geopy.geocoders import Nominatim

import folium
import matplotlib.cm as cm
import matplotlib.colors as colors

from sklearn.cluster import KMeans

print('Libraries imported!')


# ## Part 1: Data Extraction and Cleaning 

# In[5]:


#scrapping neighborhoods in Canada
url  = "https://en.wikipedia.org/wiki/List_of_postal_codes_of_Canada:_M"
page = requests.get(url)
if page.status_code == 200:
    print('Page download successful')
else:
    print('Page download error. Error code: {}'.format(page.status_code))


# In[6]:


# open Wiki page with Beautiful Soup
data = requests.get('https://en.wikipedia.org/wiki/List_of_postal_codes_of_Canada:_M').text
soup = BeautifulSoup(data, 'html.parser')


# ### Get the data using the Beautiful Soup library and put it in three columns. Post Code, borough List and neighbouhood list
# 
# * Create the empty lists for Post Code, Borough and Neighbourhood
# * Loop through the html file and append in the empty lists

# In[7]:



# Create empty lists

Postcode = []
Borough = []
Neighbourhood = []

# Loop through the data and append the data into the empty lists

for row in soup.find('table').find_all('tr'):
    cells = row.find_all('td')
    if(len(cells) > 0):
        Postcode.append(cells[0].text)
        Borough.append(cells[1].text)
        Neighbourhood.append(cells[2].text.rstrip('\n')) # remove the new line char from neighborhood cell


# * All the lists generated in the above steps will be converted into a dictionary
# * Then, this dictionary is converted into Pandas data frame for better data manipulation 

# In[8]:


toronto_neighorhood = [('PostalCode', Postcode),
                      ('Borough', Borough),
                      ('Neighborhood', Neighbourhood)]

## Conver the Dictionary to data frame
toronto_df = pd.DataFrame.from_dict(dict(toronto_neighorhood))
toronto_df.head()


# From the above table, there are few Borough's and Neighbourhoods that are 'Not assigned' and thus, we are going to remove these with the help of below code:

# In[9]:


toronto_df_dropna = toronto_df[toronto_df.Borough != 'Not assigned'].reset_index(drop=True)
toronto_df_dropna.head()


# **Group neighborhoods by postal and borough**
# * There are some neighborhoods that belongs to the same postal code and borough and thus, we will concatenate them in the same row which will be separated by a colon as learnt in this module
# 

# In[10]:


toronto_df_grouped = toronto_df_dropna.groupby(['PostalCode','Borough'], as_index=False).agg(lambda x: ','.join(x))
toronto_df_grouped.head()


# Deal with Not assigned Neighborhood
# For M7A Queen's Park, there is no neighborhood assigned.
# We will replace the 'Not assigned' with the value of the corresponding Borough

# In[11]:


na_neigh_rows = toronto_df_grouped.Neighborhood == 'Not assigned'
toronto_df_grouped.loc[na_neigh_rows, 'Neighborhood'] = toronto_df_grouped.loc[na_neigh_rows, 'Borough']
toronto_df_grouped[na_neigh_rows]


# In[12]:


toronto_df_cleaned = toronto_df_grouped
toronto_df_cleaned.shape


# ## Add Co-ordinates to to the cleaned Data frame

# In[13]:


get_ipython().system('wget -q -O "toronto_coordinates.csv" http://cocl.us/Geospatial_data')
print('Coordinates downloaded!')
coors = pd.read_csv('toronto_coordinates.csv')


# In[14]:


print(coors.shape)
coors.head()


# We will merge the two dataframes.
# To get the correct result independant on the order of data, we need to set indexes of two dataframes to its Postal Code columns

# In[15]:


toronto_df_temp = toronto_df_cleaned.set_index('PostalCode')
coors_temp = coors.set_index('Postal Code')
toronto_df_coordinates = pd.concat([toronto_df_temp, coors_temp], axis=1, join='inner')


# Reset index and we will get the toronto dataframe with coordinates

# In[16]:


toronto_df_coordinates.index.name = 'PostalCode'
toronto_df_coordinates.reset_index(inplace=True)


# Check the toronto dataset with the coordinates

# In[17]:


print(toronto_df_coordinates.shape)
toronto_df_coordinates.head()


# In[18]:


type(toronto_df_coordinates)


# In[19]:


print('The dataframe has {} boroughs and {} neighborhoods.'.format(
        len(toronto_df_coordinates['Borough'].unique()),
        toronto_df_coordinates.shape[0]
    )
)


# In[21]:


df = toronto_df_coordinates


# In[22]:


# Check the number of neighbourhoods in each borough

df.groupby('Borough').count()['Neighborhood']


# In[23]:


## Get the data only for Toronto

df_toronto = df[df['Borough'].str.contains('Toronto')]
df_toronto.reset_index(inplace=True)
df_toronto.drop('index', axis=1, inplace=True)
df_toronto.head()


# In[24]:


# check again
df_toronto.groupby('Borough').count()['Neighborhood']


# * Now we have a dataframe called df_toronto which consists on data only for Toronto

# ## Get the Venues Data using FourSquare for Indian Restaurants

# In[25]:


df_toronto.head()


# In[26]:


CLIENT_ID = 'IUP3ZOPDAJHQCY0MPLSCT0BTR34CT3VTG4R52EWJIJEF1DUD' # your Foursquare ID
CLIENT_SECRET = 'ZYGCFP4W2YBPMPYGRTTZ4JZDSFOV1CAZHL1ZIB4VUTRU1VG3' # your Foursquare Secret
VERSION = '20180604' # Foursquare API version
LIMIT = 100 # limit of number of venues returned by Foursquare API
radius = 500 # define radius


# In[35]:


# Create a function learnt during the module for getting venues

def getNearbyVenues(names, latitudes, longitudes, radius=500):
    
    venues_list=[]
    for name, lat, lng in zip(names, latitudes, longitudes):
        print(name)
            
        # create the API request URL
        url = 'https://api.foursquare.com/v2/venues/explore?&client_id={}&client_secret={}&v={}&ll={},{}&radius={}&limit={}'.format(
            CLIENT_ID, 
            CLIENT_SECRET, 
            VERSION, 
            lat, 
            lng, 
            radius, 
            LIMIT)
            
        # make the GET request
        results = requests.get(url).json()["response"]['groups'][0]['items']
        
        # return only relevant information for each nearby venue
        venues_list.append([(
            name, 
            lat, 
            lng, 
            v['venue']['name'], 
            v['venue']['location']['lat'], 
            v['venue']['location']['lng'],  
            v['venue']['categories'][0]['name']) for v in results])

    nearby_venues = pd.DataFrame([item for venue_list in venues_list for item in venue_list])
    nearby_venues.columns = ['Neighborhood', 
                  'Neighborhood Latitude', 
                  'Neighborhood Longitude', 
                  'Venue', 
                  'Venue Latitude', 
                  'Venue Longitude', 
                  'Venue Category']
    
    return(nearby_venues)


# In[36]:


#Get venues for all neighborhoods in our dataset
toronto_venues = getNearbyVenues(names=df_toronto['Neighborhood'],
                                latitudes=df_toronto['Latitude'],
                                longitudes=df_toronto['Longitude'])


# In[37]:


# Check the number of Unique venue categories

print('There are {} uniques categories.'.format(len(toronto_venues['Venue Category'].unique())))


# In[38]:


toronto_venues['Venue Category'].unique()[:100]


# Check Indian Restaurants 

# In[39]:


#
toronto_venues.head()


# In[40]:


# Venues per neighbourhood 
toronto_venues.groupby('Neighborhood').count()


# In[41]:


# One hot encoding to normalize

to_onehot = pd.get_dummies(toronto_venues[['Venue Category']], prefix="", prefix_sep="")

# add neighborhood column back to dataframe
to_onehot['Neighborhoods'] = toronto_venues['Neighborhood'] 

# move neighborhood column to the first column
fixed_columns = [to_onehot.columns[-1]] + list(to_onehot.columns[:-1])
to_onehot = to_onehot[fixed_columns]

print(to_onehot.shape)
to_onehot.head()


# Next, let's group rows by neighborhood and by taking the mean of the frequency of occurrence of each category

# In[42]:


to_grouped = to_onehot.groupby(["Neighborhoods"]).mean().reset_index()

print(to_grouped.shape)
to_grouped


# In[43]:


# Count Indain Restaurants
len(to_grouped[to_grouped["Indian Restaurant"] > 0])


# Create separate Data Frame for Indian Restaurants

# In[45]:


to_indian = to_grouped[["Neighborhoods","Indian Restaurant"]]


# In[46]:


to_indian.head()


# ## Clustering
# 
# Run K-means algorithm with 3 clusters

# In[48]:


# set number of clusters
toclusters = 3

to_clustering = to_indian.drop(["Neighborhoods"], 1)

# run k-means clustering
kmeans = KMeans(n_clusters=toclusters, random_state=0).fit(to_clustering)

# check cluster labels generated for each row in the dataframe
kmeans.labels_[0:10]


# In[49]:


# create a new dataframe that includes the cluster as well as the top 10 venues for each neighborhood.
to_merged = to_indian.copy()

# add clustering labels
to_merged["Cluster Labels"] = kmeans.labels_


# In[53]:


# Join both the tables
to_merged.rename(columns={"Neighborhoods": "Neighborhood"}, inplace=True)
to_merged.head()


# In[ ]:


# Join the table with the Toronto data to get the latitude and longitude (key: neighbourhood)
to_merged = to_merged.join(toronto_venues.set_index("Neighborhood"), on="Neighborhood")

print(to_merged.shape)
to_merged.head()


# In[54]:


to_merged.head()


# In[55]:


# sort the results by Cluster Labels
print(to_merged.shape)
to_merged.sort_values(["Cluster Labels"], inplace=True)
to_merged


# In[56]:


# Matplotlib and associated plotting modules
import matplotlib.cm as cm
import matplotlib.colors as colors


# In[58]:


#Obtain the coordinates from the dataset itself, just averaging Latitude/Longitude of the current dataset 
lat_toronto = df_toronto['Latitude'].mean()
lon_toronto = df_toronto['Longitude'].mean()
print('The geographical coordinates of Toronto are {}, {}'.format(lat_toronto, lon_toronto))


# create map
map_clusters = folium.Map(location=[lat_toronto, lon_toronto], zoom_start=11)

# set color scheme for the clusters
x = np.arange(toclusters)
ys = [i+x+(i*x)**2 for i in range(toclusters)]
colors_array = cm.rainbow(np.linspace(0, 1, len(ys)))
rainbow = [colors.rgb2hex(i) for i in colors_array]

# add markers to the map
markers_colors = []
for lat, lon, poi, cluster in zip(to_merged['Neighborhood Latitude'], to_merged['Neighborhood Longitude'], to_merged['Neighborhood'], to_merged['Cluster Labels']):
    label = folium.Popup(str(poi) + ' - Cluster ' + str(cluster))
    folium.CircleMarker(
        [lat, lon],
        radius=5,
        popup=label,
        color=rainbow[cluster-1],
        fill_color=rainbow[cluster-1],
        fill_opacity=0.7).add_to(map_clusters)
       
map_clusters


# In[59]:


# save the map as HTML file
map_clusters.save('map_clusters.html')


# ## Examine Clusters

# In[60]:


#Cluster 0
to_merged.loc[to_merged['Cluster Labels'] == 0]


# In[62]:


#Cluster 1
to_merged.loc[to_merged['Cluster Labels'] == 1]


# In[63]:


#Cluster 2
to_merged.loc[to_merged['Cluster Labels'] == 2]


# ## Observation and Conclusion

# There are lot of Indian restaurants in Toronto but after the cluster analysis, we see that there are no Indian restaurants in cluster 0 while there are many of them in cluster 1 and cluster 2. Cluster 0 consists of areas like Dominion Centre, St. James street, North Toronto, Davisville and thus these areas lack Indian restaurants. 
# There are many Asian restaurants in the area which serves India/Chinese/Thai and Malaysian food as well, but lacks authentic Indian restaurant. Thus, there is a scope to have a restaurant in these areas. 

# ## Future Work

# We can have further analysis about the demographics of that region which will help us in making this decision quicker. Understanding the average age in the area, income and the ethnicity can add value to our analysis but the data for demographics is not publicly available. 

# In[ ]:




