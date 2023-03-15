import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

with open('./to_plot/chertala.pkl', 'rb') as f:
	our_file = pickle.load(f)


melodia_file = []
with open('./to_plot/melodia_results.txt', 'r') as f:
	melodia_file = np.genfromtxt(f, delimiter=',')

durrieu_file = []
with open('./to_plot/durrieu_results.txt', 'r') as f:
	durrieu_file = np.genfromtxt(f, delimiter=',')

melodia_list = []
for i in melodia_file:
	melodia_list.append(i[:-1])

durrieu_list = []
for i in durrieu_file:
	durrieu_list.append(i[:-1])

'''
results_df = pd.DataFrame(our_results_OA, columns=['VR', 'VFA', 'RPA', 'RCA', 'OA'])
plt.figure(figsize=[6, 7.5])
plt.ylabel('Score (%)')
plt.xlabel('Melody metrics') 
results_df.boxplot(grid=True, showfliers=False, showmeans=True, meanprops=
	{"marker":"x",
     "markerfacecolor":"white", 
     "markeredgecolor":"black",
     "markersize":"7.5"}
    )
plt.show()
'''

print(melodia_list[1])


durrieu_proc = []
for i in durrieu_list:
	durrieu_proc.append([-10, -10, i[2], i[3], i[4]])


plt.figure(figsize=[10, 7])
our_df = pd.DataFrame(our_file, columns=['VR', 'VFA', 'RPA', 'RCA', 'OA']).assign(Model='Our model')
melodia_df = pd.DataFrame(melodia_list, columns=['VR', 'VFA', 'RPA', 'RCA', 'OA']).assign(Model='Melodia')
durrieu_df = pd.DataFrame(durrieu_proc, columns=['VR', 'VFA', 'RPA', 'RCA', 'OA']).assign(Model='Durrieu')

cdf = pd.concat([our_df, melodia_df, durrieu_df])    
mdf = pd.melt(cdf, id_vars=['Model'], var_name=['Score'])
print(mdf.head())

#    Location Letter     value
# 0         1      A  0.223565
# 1         1      A  0.515797
# 2         1      A  0.377588
# 3         1      A  0.687614
# 4         1      A  0.094116

ax = sns.boxplot(x="Score", y="value", hue="Model", data=mdf, showfliers=False, showmeans=True, 
	meanprops= {"marker":"o",
     "markerfacecolor":"white", 
     "markeredgecolor":"black",
     "markersize":"7"})

lines = ax.get_lines()
categories = ax.get_xticks()

plt.grid()  #just add this
plt.xlabel('Melody metrics') 
plt.ylabel('Score (%)') 
plt.ylim([-5, 105])
plt.yticks([0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
plt.show()