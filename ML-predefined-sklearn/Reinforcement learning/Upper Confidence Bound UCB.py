# -*- coding: utf-8 -*-
"""
Created on Sat Jun 19 13:41:04 2021

@author: alpha
"""


import math
# import pandas for importing csv files
import pandas as pd
#data visualization
import matplotlib.pyplot as plt    

# reading csv files
df = pd.read_csv('../data/Ads_CTR_Optimisation.csv',sep=',')


N = 10000
d = 10
ads_selected = []
numbers_of_selections = [0] * d
sums_of_rewards = [0] * d
total_reward = 0

for n in range(0, N):
    ad = 0
    max_upper_bound = 0
    for i in range(0, d):
        if numbers_of_selections[i] > 0:
            # 3 lines below is the algorithm shown above
            average_reward = sums_of_rewards[i] / numbers_of_selections[i]
            delta_i = math.sqrt(3/2 * math.log(n + 1) / numbers_of_selections[i])
            upper_bound = average_reward + delta_i
        else:
            upper_bound = 1e400 #makes this large so that the first round gives every category a chance 
        if upper_bound > max_upper_bound:
            ad = i
            max_upper_bound = upper_bound
    ads_selected.append(ad)
    numbers_of_selections[ad] = numbers_of_selections[ad] + 1
    reward = df.values[n, ad]
    sums_of_rewards[ad] = sums_of_rewards[ad] + reward
    total_reward = total_reward + reward

#Visualizing the result
plt.hist(ads_selected) #histogram of number of times each ad is clicked
plt.title('Histogram of ads_selected')
plt.xlabel('Ad No')
plt.ylabel('Number of times each add is selected')
plt.show()

#to view the reward
print(total_reward)