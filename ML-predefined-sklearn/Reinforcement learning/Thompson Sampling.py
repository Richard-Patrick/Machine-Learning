# -*- coding: utf-8 -*-
"""
Created on Sat Jun 19 13:41:04 2021

@author: alpha
"""

import random
# import pandas for importing csv files
import pandas as pd
#data visualization
import matplotlib.pyplot as plt    



# reading csv files
df = pd.read_csv('../data/Ads_CTR_Optimisation.csv',sep=',')

N = 10000
d = 10  
ads_selected = []
numbers_of_rewards_1 = [0] * d
numbers_of_rewards_0 = [0] * d #number of 0 rewards for each ad
total_reward = 0

for n in range(0, N):
    ad = 0
    max_random = 0 #maximum random draw
    for i in range(0, d):
        random_beta = random.betavariate(numbers_of_rewards_1[i] + 1, numbers_of_rewards_0[i] + 1)
        if random_beta > max_random:
            ad = i
            max_random = random_beta
    ads_selected.append(ad)
    reward = df.values[n, ad]
    if reward == 1:
        numbers_of_rewards_1[ad] = numbers_of_rewards_1[ad] + 1
    else:
        numbers_of_rewards_0[ad] = numbers_of_rewards_0[ad] + 1
    total_reward = total_reward + reward

plt.hist(ads_selected) #histogram of number of times each ad is clicked
plt.title('Histogram of ads_selected')
plt.xlabel('Ad No')
plt.ylabel('Number of times each add is selected')
plt.show()

#to view the reward
print(total_reward)