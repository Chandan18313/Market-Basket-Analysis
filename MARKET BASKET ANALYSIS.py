#!/usr/bin/env python
# coding: utf-8

# # Reading and Cleaning Data

# In[163]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

import seaborn as sns


# In[164]:


df = pd.read_csv("C:/Users/Windows/Documents/KARE/SEM 5/DA/bread basket .csv")


# In[165]:


df.head()


# In[166]:


df.describe()


# In[167]:


df.info()


# In[168]:


# Converting the 'date_time' column into the right format
df['date_time'] = pd.to_datetime(df['date_time'])


# In[169]:


df.head(10)


# In[170]:


# Count of unique customers
df['Transaction'].nunique()


# In[171]:


# Extracting date
df['date'] = df['date_time'].dt.date

#Extracting time
df['time'] = df['date_time'].dt.time

# Extracting month and replacing it with text
df['month'] = df['date_time'].dt.month
df['month'] = df['month'].replace((1,2,3,4,5,6,7,8,9,10,11,12), 
                                          ('January','February','March','April','May','June','July','August',
                                          'September','October','November','December'))

# Extracting hour
df['hour'] = df['date_time'].dt.hour
# Replacing hours with text
hour_in_num = (1,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23)
hour_in_obj = ('1-2','7-8','8-9','9-10','10-11','11-12','12-13','13-14','14-15',
               '15-16','16-17','17-18','18-19','19-20','20-21','21-22','22-23','23-24')
df['hour'] = df['hour'].replace(hour_in_num, hour_in_obj)

# Extracting weekday and replacing it with text
df['weekday'] = df['date_time'].dt.weekday
df['weekday'] = df['weekday'].replace((0,1,2,3,4,5,6), 
                                          ('Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday'))

# dropping date_time column
df.drop('date_time', axis = 1, inplace = True)


# In[172]:


df.head()


# In[173]:


# cleaning the item column
df['Item'] = df['Item'].str.strip()
df['Item'] = df['Item'].str.lower()


# # Data Visualization

# In[174]:


plt.figure(figsize=(15,5))
sns.barplot(x = df.Item.value_counts().head(20).index, y = df.Item.value_counts().head(20).values, palette = 'gnuplot')
plt.xlabel('Items', size = 15)
plt.xticks(rotation=45)
plt.ylabel('Count of Items', size = 15)
plt.title('Top 20 Items purchased by customers', color = 'green', size = 20)
plt.show()


# In[175]:


#Coffee has the highest transactions.
#Coke is the 20th most buyed product.


# In[176]:


monthTran = df.groupby('month')['Transaction'].count().reset_index()
monthTran.loc[:,"monthorder"] = [4,8,12,2,1,7,6,3,5,11,10,9]
monthTran.sort_values("monthorder",inplace=True)

plt.figure(figsize=(12,5))
sns.barplot(data = monthTran, x = "month", y = "Transaction")
plt.xlabel('Months', size = 15)
plt.ylabel('Orders per month', size = 15)
plt.title('Number of orders received each month', color = 'green', size = 20)
plt.show()


plt.show()


# In[177]:


#Most transactions were in March, January, February, November, December


# In[178]:


weekTran = df.groupby('weekday')['Transaction'].count().reset_index()
weekTran.loc[:,"weekorder"] = [4,0,5,6,3,1,2]
weekTran.sort_values("weekorder",inplace=True)

plt.figure(figsize=(12,5))
sns.barplot(data = weekTran, x = "weekday", y = "Transaction")
plt.xlabel('Week Day', size = 15)
plt.ylabel('Orders per day', size = 15)
plt.title('Number of orders received each day', color = 'green', size = 20)
plt.show()


plt.show()


# In[179]:


#People order more on weekends.


# In[180]:


hourTran = df.groupby('hour')['Transaction'].count().reset_index()
hourTran.loc[:,"hourorder"] = [1,10,11,12,13,14,15,16,17,18,19,20,21,22,23,7,8,9]
hourTran.sort_values("hourorder",inplace=True)

plt.figure(figsize=(12,5))
sns.barplot(data = hourTran, x = "Transaction", y = "hour")
plt.ylabel('Hours', size = 15)
plt.xlabel('Orders each hour', size = 15)
plt.title('Count of orders received each hour', color = 'green', size = 20)
plt.show()


# In[181]:


#People order more during the afternoon, since there are a lot of maximum order percentage between 12-5.


# In[182]:


dayTran = df.groupby('period_day')['Transaction'].count().reset_index()
# dayTran.loc[:,"hourorder"] = [1,10,11,12,13,14,15,16,17,18,19,20,21,22,23,7,8,9]
# dayTran.sort_values("hourorder",inplace=True)

plt.figure(figsize=(12,5))
sns.barplot(data = dayTran, x = "Transaction", y = "period_day")
plt.ylabel('Period', size = 15)
plt.xlabel('Orders each period of a day', size = 15)
plt.title('Count of orders received each period of a day', color = 'green', size = 20)
plt.show()


# In[183]:


#People prefer to order in the morning and afternoon.


# In[184]:


dates = df.groupby('date')['Transaction'].count().reset_index()
dates = dates[dates['Transaction']>=200].sort_values('date').reset_index(drop = True)

dates = pd.merge(dates, df[['date','weekday']], on = 'date', how = 'inner')
dates.drop_duplicates(inplace =True)
dates


# In[185]:


#Mostly transactions are on weekends, as we saw earlier in our graph.


# In[186]:


data = df.groupby(['period_day','Item'])['Transaction'].count().reset_index().sort_values(['period_day','Transaction'],ascending=False)
day = ['morning','afternoon','evening','night']

plt.figure(figsize=(15,8))
for i,j in enumerate(day):
    plt.subplot(2,2,i+1)
    df1 = data[data.period_day==j].head(10)
    sns.barplot(data=df1, y=df1.Item, x=df1.Transaction, color='pink')
    plt.xlabel('')
    plt.ylabel('')
    plt.title('Top 10 items people like to order in "{}"'.format(j), size=13)

plt.show()


# # Apriori Algorithm

# In[187]:


from mlxtend.frequent_patterns import association_rules, apriori


# In[188]:


transactions_str = df.groupby(['Transaction', 'Item'])['Item'].count().reset_index(name ='Count')
transactions_str


# In[189]:


# making a mxn matrice where m=transaction and n=items and each row represents whether the item was in the transaction or not
my_basket = transactions_str.pivot_table(index='Transaction', columns='Item', values='Count', aggfunc='sum').fillna(0)


my_basket.head()


# In[190]:


# making a function which returns 0 or 1
# 0 means item was not in that transaction, 1 means item present in that transaction

def encode(x):
    if x<=0:
        return 0
    if x>=1:
        return 1

# applying the function to the dataset

my_basket_sets = my_basket.applymap(encode)
my_basket_sets.head()


# In[191]:


frequent_items = apriori(my_basket_sets, min_support = 0.01,use_colnames = True)
frequent_items


# In[192]:


# now making the rules from frequent itemset generated above

rules = association_rules(frequent_items, metric = "lift", min_threshold = 1)
rules.sort_values('confidence', ascending = False, inplace = True)
rules


# In[193]:


# arranging the data from highest to lowest with respect to 'confidence'

rules.sort_values('confidence', ascending=False)


# # FREQUENT ITEMS SOLD TOGETHER

# In[194]:


# Assuming you've generated frequent_items DataFrame

frequent_groceries = frequent_items['itemsets']
print(frequent_groceries)


# In[195]:


# Generate frequent itemsets
frequent_itemsets = apriori(my_basket_sets, min_support=0.01, use_colnames=True)

# Filter for itemsets of size 2, 3, and 4
frequent_itemsets_2 = frequent_itemsets[frequent_itemsets['itemsets'].apply(lambda x: len(x) == 2)]
frequent_itemsets_3 = frequent_itemsets[frequent_itemsets['itemsets'].apply(lambda x: len(x) == 3)]


# Display the frequent itemsets
print("Frequent Itemsets of Size 2:")
print(frequent_itemsets_2)

print("\nFrequent Itemsets of Size 3:")
print(frequent_itemsets_3)


# In[196]:


import matplotlib.pyplot as plt

def plot_frequent_itemsets(frequent_itemsets, size):
    plt.figure(figsize=(10, 6))
    plt.barh(range(len(frequent_itemsets)), frequent_itemsets['support'], align='center')
    plt.yticks(range(len(frequent_itemsets)), [', '.join(itemset) for itemset in frequent_itemsets['itemsets']])
    plt.xlabel('Support')
    plt.title(f'Frequent Itemsets of Size {size}')
    plt.show()

# Plot frequent itemsets of size 2, 3, and 4
plot_frequent_itemsets(frequent_itemsets_2, 2)
plot_frequent_itemsets(frequent_itemsets_3, 3)


# In[197]:


import pandas as pd
import matplotlib.pyplot as plt

# Assuming you have a DataFrame 'frequent_items' with columns 'itemsets' and 'support'

# Flatten the itemsets column
frequent_items['itemsets'] = frequent_items['itemsets'].apply(lambda x: ', '.join(x))

# Group by item and calculate total support
item_support = frequent_items.groupby('itemsets')['support'].sum().reset_index()

# Sort items by support in descending order
item_support = item_support.sort_values(by='support', ascending=False)

# Create a pie chart
plt.figure(figsize=(10, 6))
plt.pie(item_support['support'], labels=item_support['itemsets'], autopct='%1.1f%%', startangle=140)
plt.axis('equal')  # Equal aspect ratio ensures the pie chart is circular.
plt.title('Frequent Items ')
plt.show()


# In[198]:


item_counts = df['Item'].value_counts()
top_n = 10 
# Adjust this number based on the number of items you want to display
top_items = item_counts.head(top_n)
import matplotlib.pyplot as plt

# Create a bar chart
plt.figure(figsize=(12, 6))
plt.bar(top_items.index, top_items.values, color='skyblue')
plt.xlabel('Items')
plt.ylabel('Frequency')
plt.title(f'Top {top_n} Sold Items')
plt.xticks(rotation=45, ha='right')
plt.show()


# # Association rule diagrams and Support-Confidence plots

# In[199]:


import seaborn as sns
import matplotlib.pyplot as plt

# Assuming you have the 'rules' DataFrame

# Create a scatter plot of support vs confidence
plt.figure(figsize=(10, 6))
sns.scatterplot(x="support", y="confidence", data=rules, alpha=0.5)
plt.title("Support-Confidence Plot")
plt.show()


# In[200]:


# List of minimum support values
min_support_values = [0.07, 0.05, 0.03, 0.02]

# Confidence levels to evaluate
confidence_levels = list(np.arange(0.05, 1.05, 0.05))

# Empty lists to store results
num_rules_lists = []

# Calculate and store the number of rules for each combination of minimum support and confidence level
for min_support in min_support_values:
    frequent_itemsets = apriori(onehot_encoded, min_support=min_support, use_colnames=True)
    rules_list = []
    for confidence_level in confidence_levels:
        rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=confidence_level)
        num_rules = len(rules)
        rules_list.append(num_rules)
    num_rules_lists.append(rules_list)

# Plot the results
plt.figure(figsize=(10, 6))

colors = ["#4e79a7", "#f28e2b", "#e15759", "#76b7b2"]

for i, min_support in enumerate(min_support_values):
    plt.plot(confidence_levels, num_rules_lists[i], marker="o", color=colors[i], label=f"Min Support: {min_support}")

plt.xlabel("Confidence Level")
plt.ylabel("Number of Rules")
plt.title("Number of Rules vs. Confidence Level for Different Minimum Support")

# Set the desired x-axis labels
plt.xticks([0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 1])

# Add grid lines for better readability
plt.grid(True, linestyle="--", alpha=0.7)

# Add legend
plt.legend()

plt.show()


# In[ ]:




