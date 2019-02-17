#!/usr/bin/env python
# coding: utf-8

# # Data Wrangling Project #

# ## Introduction ##
# 
# We will analyze the tweet archive of Twitter user @dog_rates, also known as WeRateDogs. WeRateDogs is a Twitter account with over 4 million followers that rates and comments people's dogs. 

# In[1]:


# IMPORT statements
import pandas as pd
import requests
import os
import numpy as np
import tweepy
import json
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.dates as mdates
import datetime as dt
from scipy import stats
get_ipython().run_line_magic('matplotlib', 'inline')

pd.set_option('display.float_format', lambda x: '%.2f' % x)


# ## Gather ##
# In this section we will download the following files:
# - The WeRateDogs Twitter archive
# - The tweet image predictions
# - Each tweet's information (for instance, favourite and retweet count)
# 
# Each file will be described and discussed in more detail when gathered.  
# 
# ### WeRateDogs Twitter archive ###
# 
# WeRateDogs Twitter archive contains basic tweet data (tweet ID, timestamp, text, etc.) for all 5000+ of their tweets as they stood on August 1, 2017. Additional tweet information will be gathered in the next steps.  
# This file was provided already, I have therefore downloaded it and uploaded it to the Jupyter Notebook Workspace.

# In[2]:


#create dataframe from archive
archive = pd.read_csv('twitter-archive-enhanced.csv')

#display archive
archive.head()


# ### Tweet Image Predictions ###
# 
# Every image in the WeRateDogs Twitter archive was run through a neural network that can classify breeds of dogs. It consists in a table with up to three image predictions alongside each tweet ID and an indication of the image number that correspond to each prediction (a tweet can have more than one image). 
# 
# The file containint this information is in a flat file structure or tabular form (with extension *.tsv*).

# In[3]:


url = 'https://d17h27t6h515a5.cloudfront.net/topher/2017/August/599fd2ad_image-predictions/image-predictions.tsv'

#request and write file
response = requests.get(url)
with open (url.split('/')[-1], mode='wb') as file:
    file.write(response.content)
    
#check file got created
os.listdir()


# In[4]:


#create dataframe out of .tsv file we just saved
predictions = pd.read_csv('image-predictions.tsv', sep='\t')

predictions.head()


# ### Tweets Information ###

# In[5]:


#create empty list
twitter_api = []

#create list tweet ids
tweet_list = archive.tweet_id.tolist()

#access twitter api
consumer_key = ''
consumer_secret = ''
access_token = ''
access_secret = ''

auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_secret)

api = tweepy.API(auth, wait_on_rate_limit=True, wait_on_rate_limit_notify=True)

#retrieve tweet info
with open('tweet_json.txt', 'w', encoding='utf-8') as json_file:
    for t_id in tweet_list:
        
        try:
            #counter
            ranking = tweet_list.index(t_id)+1
            print(ranking)

            #tweet info
            tweet = api.get_status(t_id, tweet_mode='extended')
 
            retweet_count = tweet.retweet_count
            favorite_count = tweet.favorite_count

            #append to list
            twitter_api = {'tweet_id':int(t_id), 'retweet_count':int(retweet_count), 'favorite_count':int(favorite_count)}
            
            json_file.write(json.dumps(twitter_api))
            json_file.write('\n')
    
        except Exception as e:
            print(e)
            
json_file.close()    


# In[6]:


df_list= []

# read json file
with open('tweet_json.txt') as file:  
    for line in file:
        df_list.append(json.loads(line))
        
# create dataframe
tweet_info = pd.DataFrame(df_list, columns = ['tweet_id', 'retweet_count', 'favorite_count'])


# In[7]:


tweet_info.head()


# In[8]:


tweet_info.shape


# **Note** that we could not retrieve information for some tweet IDs as some tweets were deleted or became invalid. 

# ## Defining scope ##
# 
# We would like to look only at tweets that are original (not retweeted) and that have images. We can filter from now the information to what we will need to analyse to decrease waiting time and computational power. 
# 
# **Note**: This is where I take out retweets and tweets with no images as per the project requirements. 
# 
# We will start by looking at the list of retweet tweet ids. 

# In[9]:


retweet_list = archive[archive['retweeted_status_id'].notnull()].retweeted_status_id.unique().astype(int).tolist()


# In[10]:


retweet_list


# In[11]:


#create new column in archive indicating whether retweet or not

archive['is_retweet'] = 'N' #default value

m1 = archive['tweet_id'].isin(retweet_list)

archive['is_retweet'] = archive['is_retweet'].mask(m1, 'Y')


# In[12]:


# list of tweet IDs having images
images_list = predictions['tweet_id'].unique().tolist()


# In[13]:


#create new column in archive indicating whether tweet has images

archive['has_images'] = 'N' #default value

m2 = archive['tweet_id'].isin(images_list)

archive['has_images'] = archive['has_images'].mask(m2, 'Y')


# In[14]:


archive_scope = archive[(archive['is_retweet']=='N')& (archive['has_images']=='Y')]


# In[15]:


#check 1
archive_scope['is_retweet'].value_counts()


# In[16]:


#check 2
archive_scope['has_images'].value_counts()


# In[17]:


#do the same for tweet_info table

#create new columns to indicate whether tweet is retweet
tweet_info['is_retweet'] = 'N' #default value

m1_t = tweet_info['tweet_id'].isin(retweet_list)

tweet_info['is_retweet'] = tweet_info['is_retweet'].mask(m1_t, 'Y')

#create new column to indicate whether tweet has images
tweet_info['has_images'] = 'N' #default value

m2_t = tweet_info['tweet_id'].isin(images_list)

tweet_info['has_images'] = tweet_info['has_images'].mask(m2_t, 'Y')


# In[18]:


#create new dataframe with only info for tweets in scope
tweet_scope = tweet_info[(tweet_info['is_retweet']=='N')& (tweet_info['has_images']=='Y')]


# In[19]:


#do the same with predictions to check if retweet

#create new columns to indicate whether tweet is retweet
predictions['is_retweet'] = 'N' #default value

m1_p = predictions['tweet_id'].isin(retweet_list)

predictions['is_retweet'] = predictions['is_retweet'].mask(m1_p, 'Y')


# In[20]:


#create new dataframe with only info for tweets in scope
predictions_scope = predictions[predictions['is_retweet']=='N']


# ## Assess ##
# 
# This section of the project will identify any quality and tidiness issues that need to be addressed before proceeding with the analysis.

# In[21]:


archive_scope.info()


# We can already notice a quality issue regarding data format. Timestamp should have datetime format, as well as retweeted_status_timestamp. Also, status ids and user ids should have string format but since we won't use them for our analysis we can ignore for now. Tweet id should be string and rating_numerator and rating_denominator should be float.

# In[22]:


pd.set_option('display.max_colwidth', -1)
archive_scope.head()


# In[23]:


archive_scope['source'].value_counts()


# In[24]:


archive_scope['puppo'].value_counts()


# We could tidy up by having one column indicating the dog stage rather than 4 different columns for this.
# 
# We will now check for consistency in the retweets columns, i.e. if we look at non null *retweeted_status_id*, we expect also the other *retweeted* columns to be non null. If instead we have null *retweeted_status_id* we would then expect all other *retweeted* columns to be null. 

# In[25]:


#check for consistency (not null retweets in all retweets relevant columns)
archive_scope[archive_scope['retweeted_status_id'].notnull()]


# In[26]:


archive_scope[archive_scope['retweeted_status_id'].isnull()]


# In[27]:


#same check for in_reply_to_status_id
archive_scope[archive_scope['in_reply_to_status_id'].notnull()]


# In[28]:


#same check for in_reply_to_status_id
archive_scope[archive_scope['in_reply_to_status_id'].isnull()]


# In[29]:


#check for name typos
archive_scope['name'].value_counts()


# Some names are obviously wrong, like "a", "the", "quite", "an". As it can be seen below, we can generalise and say that all names with a lower case are not real dog names.

# In[30]:


archive_scope[archive_scope['name'].astype(str).str.islower()]['name'].value_counts()


# In[31]:


#check date makes sense
min(archive_scope['timestamp'])
max(archive_scope['timestamp'])


# In[32]:


# check numerators 
archive_scope['rating_numerator'].value_counts()


# In[33]:


#check tweets with high numerators manually
archive_scope[archive_scope['rating_numerator']==1776]


# In[34]:


archive_scope[archive_scope['rating_numerator']==420]


# Even though they are high numerators, checking manually the tweets confirms that the values are correct.

# In[35]:


# check denominators
archive_scope['rating_denominator'].value_counts()


# In[36]:


#check the high denominators
archive_scope[archive_scope['rating_denominator']==170]


# By checking this tweet manually, we can see that the numerator and the denominator match with what we have in the table. 

# In[37]:


archive_scope.isnull().sum()


# In[38]:


archive_scope.duplicated().sum()


# In[39]:


archive_scope[archive_scope['expanded_urls'].str.startswith('https')==False]


# In[40]:


tweet_scope.info()


# In[41]:


tweet_scope.head()


# In[42]:


tweet_scope.isnull().sum()


# In[43]:


tweet_scope.describe()


# In[44]:


#check min e max for retweet
tweet_scope[tweet_scope['retweet_count'] == tweet_scope['retweet_count'].max()]


# There doesn't seem to be an issue with the minimum and maximum values for retweets. 

# In[45]:


#something wrong with favorite count
tweet_scope[tweet_scope['favorite_count']==0]


# I have spent quite some time trying to understand how to fix the issue above, i.e. incorrectly displaying favourite count of 0 for many tweets. I have found out that this issue has occurred in the past https://twittercommunity.com/t/retweet-count-and-favorite-count-always-zero/11092/31
# 
# I will look at the issue in more detail and keep for now the code as it is, noting that the favorite count won't be useful as an analysis variable later on. 

# In[46]:


predictions_scope.shape


# In[47]:


predictions_scope.info()


# In[48]:


predictions_scope.head()


# In[49]:


predictions_scope.describe()


# In[50]:


predictions_scope.isnull().sum()


# In[51]:


predictions_scope[predictions_scope['jpg_url'].str.endswith('.jpg')==False]


# *.png* is still an image format so that does not create an issue. 

# ### Quality ###
# #### archive_scope ####
# - Erroneous time datatypes (Timestamp, retweet timestamp) 
# - Erroneous int datatypes (int should be float)
# - Erroneous int datatypes (int should be string)
# - Some dog names are wrong ("the", "quite", etc)
# - Text column contains other components too
# - The source column is not clear
# - 2 expanded urls messy
# 
# 
# #### predictions_scope ####
# - Some dog species are lower case and some upper case 

# ### Tidiness ###
# 
# #### tweet_scope####
# - Unnecessary to have it separate from the archive dataframe (same for predictions)
# 
# #### archive_scope ####
# - Dog stage variable occupies 4 columns 
# 
# 

# ## Clean ## 
# 
# Here, we will start actioning the issues previously identified to then go back to the **Assess** stage . For each item, we will define the required action, perform it and test it. Please note that we will reassess the data at a later stage after the tidiness steps are made. These led to two additional quality issues that have to be addressed. 
# 
# First of all, we will create copies of the dataframes.
# 

# In[89]:


archive_clean = archive_scope.copy()
tweet_clean = tweet_scope.copy()
predictions_clean = predictions_scope.copy()


# ### Quality ###
# 
# #### 1) Erroneous Time Datatypes ####
# 
# ##### Define #####
# 
# Change datatypes in *archive_scope* table for timestamp and retweeted_status_timestamp.
# 
# ##### Code #####

# In[90]:


#timestamp
archive_clean.timestamp = pd.to_datetime(archive_clean.timestamp)


# In[91]:


#retweeted_status_timestamp
archive_clean['retweeted_status_timestamp'] = pd.to_datetime(archive_clean.retweeted_status_timestamp)


# #### Test ####

# In[92]:


archive_clean.info()


# #### 2) Erroneous Int Datatypes - a ####
# 
# ##### Define #####
# 
# Change datatypes in *archive_scope* table for rating_numerator and rating_denominator from int to float
# 
# ##### Code #####

# In[93]:


archive_clean['rating_numerator'] = archive_clean['rating_numerator'].astype(float)
archive_clean['rating_denominator'] = archive_clean['rating_denominator'].astype(float)


# ##### Test #####

# In[94]:


archive_clean.info()


# #### 3) Erroneous Int Datatypes - b ####
# 
# ##### Define #####
# 
# Change datatypes in *archive_scope* table for tweet_id from int to string
# 
# ##### Code #####

# In[95]:


archive_clean['tweet_id'] = archive_clean['tweet_id'].astype(str)


# ##### Test #####

# In[96]:


archive_clean.info()


# #### 4) Wrong dog names ####
# 
# ##### Define #####
# 
# Set wrong dog names to N/A
# 
# ##### Code #####

# In[97]:


m3 = archive_clean['name'].str.islower()

archive_clean['name'] = archive_clean['name'].mask(m3, 'N/A')


# #### Test ####

# In[98]:


archive_clean['name'].str.islower().value_counts()


# #### 5) Dog Species different cases ####
# 
# ##### Define #####
# 
# Change all dog species to lower case
# 
# ##### Code #####

# In[99]:


predictions_clean['p1'] = predictions_clean['p1'].str.lower()
predictions_clean['p2'] = predictions_clean['p2'].str.lower()
predictions_clean['p3'] = predictions_clean['p3'].str.lower()


# #### Test ####

# In[100]:


predictions_clean.p1.str.isupper().sum()


# In[101]:


predictions_clean.p2.str.isupper().sum()


# In[102]:


predictions_clean.p3.str.isupper().sum()


# Taking the opportunity to delete the *is_retweet* column.

# In[103]:


predictions_clean = predictions_clean.drop(['is_retweet'], axis=1)


# In[104]:


predictions_clean.head()


# #### 6) Text column contains other components too ####
# 
# #### Define ####
# Keep only true tweet text.
# 
# ##### Code #####

# In[105]:


archive_clean['text'] = archive_clean.text.astype(str)

def slicer(s):
    return s[:s.find(' https')]

archive_clean['text'] = archive_clean['text'].apply(lambda s: slicer(s))   


# #### Test ####

# In[106]:


archive_clean.head()


# #### 7) The source column is not clear ####
# 
# #### Define ####
# Tidy up the *source* column
# 
# 
# ##### Code #####

# In[107]:


archive_clean.source.value_counts()


# In[108]:


archive_clean['source'] = archive_clean.source.astype(str)

def source_slicer (s):
    s = s[s.find('>')+1:]
    return s[:s.find('<')]

archive_clean['source'] = archive_clean['source'].apply(lambda s: source_slicer(s))   


# #### Test ####

# In[109]:


archive_clean.head()


# #### 8) 2 expanded urls are messy ####
# 
# #### Define ####
# Fix the 2 messy expanded urls in order to keep only the relevant one. 
# 
# ##### Code #####

# In[110]:


archive_clean.head()


# In[111]:


t_list = archive_clean[archive_clean['expanded_urls'].str.startswith('https')==False]['tweet_id'].tolist()


# In[112]:


archive_clean['expanded_urls'] = archive_clean.expanded_urls.astype(str)

def url_slicer(s):
    return s[s.find('https'):]

for t in t_list:
    archive_clean['expanded_urls'] = archive_clean['expanded_urls'].apply(lambda s: url_slicer(s))   


# #### Test ####

# In[113]:


archive_clean[archive_clean['expanded_urls'].str.startswith('https')==False]


# ### Tidiness ###
# 
# #### 1) Separate dataframes ####
# 
# ##### Define #####
# 
# Merge *tweet_scope* and *predictions* into *archive_scope*
# 
# ##### Code #####

# In[114]:


# delete columns is_retweet and has_images before merging as they are not needed
archive_clean = archive_clean.drop(['is_retweet', 'has_images'], axis = 1)
tweet_clean = tweet_clean.drop(['is_retweet', 'has_images'], axis = 1)


# In[115]:


tweet_clean['tweet_id'] = tweet_clean['tweet_id'].astype(str)


# In[116]:


new_archive = archive_clean.merge(tweet_clean, on='tweet_id', how='left')


# In[120]:


predictions_clean['tweet_id'] = predictions_clean['tweet_id'].astype(str)


# In[121]:


new_archive = new_archive.merge(predictions_clean, on='tweet_id', how='left')


# #### Test ####

# In[122]:


new_archive.head()


# In[123]:


new_archive.info()


# In[124]:


new_archive.isnull().sum()


# New quality issues arise after this step, i.e. the retweet_count and favorite_count is null for 7 tweets and their format is float rather than integer. 

# #### 2) 1 variable in 4 columns ####
# 
# #### Define####
# Combine the floofer, puppo, pupper and doggo columns in archive_scope into one. 
# 
# ##### Code #####

# In[125]:


def new_column(x):
    name = [i for i in x if i!="None"]
    if name:
        return str(name)
    else:
        return None

new_archive["type"] = new_archive[['doggo','pupper', 'puppo', 'floofer']].apply(lambda x: new_column(x),axis =1 )   


# In[129]:


new_archive.type = new_archive.type.astype(str)


# #### Test ####

# In[132]:


new_archive.head()


# In[133]:


new_archive[(new_archive['pupper']=='None')&(new_archive['doggo']=='None')]


# In[134]:


new_archive.shape


# In[135]:


new_archive['type'].value_counts()


# We notice here that we have many "None" values for the type of dog. 

# In[136]:


new_archive.isnull().sum()


# In[137]:


new_archive = new_archive.drop(['doggo', 'floofer', 'puppo', 'pupper'], axis=1)


# ## Reassess ##
# 
# ### Quality ###
# 
# - 7 null retweet_count and favorite_count values
# - Wrong format of retweet_count and favorite_count

# #### 9) Seven null retweet_count and favorite_count values ####
# 
# #### Define ####
# Check manually each of these tweets
# 
# ##### Code #####

# In[138]:


list_null = new_archive[new_archive['retweet_count'].isnull()]['tweet_id'].index.tolist()


# In[139]:


list_null


# I was not able to open any of these tweets therefore I assume these were deleted. We can delete them from the dataframe. 

# In[140]:


for t in list_null:
    new_archive = new_archive.drop(t)


# #### Test ####

# In[141]:


new_archive.isnull().sum()


# #### 10) Wrong format retweet_count and favorite_count ####
# 
# #### Define ####
# Fix format of the two columns
# 
# ##### Code #####

# In[142]:


new_archive['retweet_count'] = new_archive['retweet_count'].astype(int)
new_archive['favorite_count'] = new_archive['favorite_count'].astype(int)


# #### Test ####

# In[143]:


new_archive.info()


# ## Storing data ##

# In[144]:


new_archive = new_archive.reset_index(drop=True)


# In[146]:


new_archive.to_csv('twitter_archive_master.csv', index=False)


# ## Analysis ## 
# 
# Let's start by looking at a historical trend of tweet's retweet and favorite counts. 

# In[147]:


x = [pd.to_datetime(i) for i in  new_archive.timestamp.values]

sns.set_style("darkgrid")
fig=plt.figure(figsize=(12, 10))
plt.plot(x,new_archive.retweet_count.values, linewidth=2)
plt.plot(x,new_archive.favorite_count.values, linewidth=2, alpha = 0.4)
plt.ylabel('Count', fontsize=20)
plt.title('Historical trend of retweet and favorite count', fontsize=18)
plt.legend(["Retweet Count", "Favorite Count"], fontsize=14)
plt.gcf().autofmt_xdate()


# As expected, the peaks in retweet count correspond to the peaks in favorite count. Note also that the biggest peak is in summer 2016. We can see an increasing, yet very volatile trend. 
# 
# Below, a scatter plot that shows a roughly positive relation between retweet count and favorite count. 

# In[148]:


x = new_archive.retweet_count.values
y = new_archive.favorite_count.values

slope, intercept, r_value, p_value, std_err = stats.linregress(x,y)
line = slope*x+intercept
sns.set_style("darkgrid")
fig=plt.figure(figsize=(10, 10))

plt.plot(x,y,'o',x,line)
plt.xlabel('Number of retweets', fontsize=18)
plt.ylabel('Number of favorites', fontsize=18)
plt.title('Relation between number of retweets and number of favorites', fontsize=20)

plt.show();


# **Insight 1:** It looks like retweet and favorite count have an increasing yet volatile trend up to August 2017 and they are positively correlated. On average, tweets with high retweet count would have a high favorite count and viceversa. 

# We now want to look at a potential relationship between retweet/favorite count and the type of dog, i.e. doggo, pupper, puppo, floofer. 

# In[149]:


new_archive.groupby(['type'])['retweet_count'].mean().sort_values()


# In[150]:


df_1 = pd.DataFrame(new_archive.groupby('type')['retweet_count'].mean().sort_values()).reset_index()
df_2 = pd.DataFrame(new_archive.groupby('type')['favorite_count'].mean().sort_values()).reset_index()


# In[151]:


df = df_1.merge(df_2, on='type', how='left')
df


# In[155]:


x1 = df.favorite_count.values
x2 = df.retweet_count.values

groups = df.type.tolist()
axis = []

for l in groups: 
    l = str(l)
    l = l.replace("[\'","").replace("\']","").replace("\'","")
    axis.append(l)


# In[165]:


fig, ax = plt.subplots(figsize=(13, 10))

index = np.arange(8)
bar_width = 0.35

opacity = 0.5

rects1 = ax.bar(index, x1, bar_width,
                alpha=opacity, color='b',
                label='Likes')

rects2 = ax.bar(index + bar_width, x2, bar_width,
                alpha=opacity, color='r',
                label='Retweets')

ax.set_xlabel('Dog Type',fontsize=16)
ax.set_ylabel('Average Count', fontsize=16)
ax.set_title('Relation between dog type and average number of retweets and favorites', fontsize=20)
ax.set_xticks(index + bar_width / 2)
ax.set_xticklabels(axis, fontsize=12)
ax.legend(fontsize=14)
plt.rcParams.update({'font.size': 10})

fig.tight_layout()
plt.show();


# **Insight 2:** It looks like dogs considered as both doggo and puppo have a better chance to get a high retweet count and favorite count. However, as a reminder we have quite little data on the dog type (most of the data appears with "None" as dog type). Hence, we can't draw any conclusions with high confidence. 

# We want to look now at whether a higher rating leads to a higher retweet/favorite count or a lower one.

# In[166]:


new_archive['fraction'] = new_archive.rating_numerator/new_archive.rating_denominator


# In[167]:


new_archive.describe()


# In[168]:


#function for bucketing retweet count

def retweet_bucketing(x):
    if x>0 and x<201:
        return "0-200"
    elif  x>200 and x<401:
        return "201-400"
    elif  x>400 and x<601:
        return "401-600"
    elif  x>600 and x<801:
        return "601-800"
    elif  x>800 and x<1001:
        return "801-1000"
    elif  x>1000 and x<1501:
        return "1001-1500"
    elif  x>1500 and x<2001:
        return "1501-2000"
    elif  x>2000 and x<3001:
        return "2001-3000"
    elif  x>3000 and x<4001:
        return "3001-4000"
    elif  x>4000 and x<6001:
        return "4001-6000"
    elif  x>6000 and x<8001:
        return "6001-8000"
    else:
        return "8000+"
    
new_archive['retweet_bucket'] = new_archive['retweet_count'].apply(lambda t:retweet_bucketing(t))


# In[169]:


#function for bucketing favorite count
def favorite_bucketing(x):
    if x>0 and x<501:
        return "0-500"
    elif  x>500 and x<1001:
        return "501-1000"
    elif  x>1000 and x<1501:
        return "1001-1500"
    elif  x>1500 and x<2001:
        return "1501-2000"
    elif  x>2000 and x<3001:
        return "2001-3000"
    elif  x>3000 and x<4001:
        return "3001-4000"
    elif  x>4000 and x<5001:
        return "4001-5000"
    elif  x>5000 and x<6001:
        return "5001-6000"
    elif  x>6000 and x<7001:
        return "6001-7000"
    elif  x>7000 and x<8001:
        return "7001-8000"
    elif  x>8000 and x<10001:
        return "8001-10000"
    else:
        return "10000+"
    

new_archive['favorite_bucket'] = new_archive['favorite_count'].apply(lambda z:favorite_bucketing(z))


# In[171]:


#remove outliers
#as I noticed that it would not give interesting results in the heatmap 
new_archive = new_archive.drop(new_archive[new_archive['fraction']==42].index)
new_archive = new_archive.drop(new_archive[new_archive['fraction']==177.6].index)
new_archive = new_archive.reset_index()

#check
new_archive[new_archive['fraction']==177.60]


# In[172]:


new_archive.describe()


# In[173]:


#data for heatmap
data_heat = pd.DataFrame(new_archive.groupby(["retweet_bucket", "favorite_bucket"])['fraction'].mean().reset_index())

data = data_heat.pivot("retweet_bucket", "favorite_bucket", "fraction")


# In[174]:


fig, ax = plt.subplots(figsize=(13, 10))

x = sns.heatmap(data, cmap="viridis")

ax.set_xlabel('Number of Favorites',fontsize=14)
ax.set_ylabel('Number of Retweets', fontsize=14)
ax.set_title('Relation between rating and number of likes/retweets', fontsize=20)
ax.legend(fontsize=10)
plt.rcParams.update({'font.size': 10})


# **Insight 3:** Many observations can be drawn from the heatmap above 
# - all the posts with 8000+ retweets have 10000+ likes
# - looking at the 10000+ favorites, they all have a quite high rating (with rating here considered as quotient between numerator and denominator)
# - also the RHS of the heatmap, with likes ranging from 6000 to 10000 have all quite high ratings
# - the highest ratings can be found in posts having 7001-8000 likes and 4001-6000 retweets
# - the lowest ratings can be found in posts having 3001-4000 likes and 201-400 retweets
# 
