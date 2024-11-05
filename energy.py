#%%
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
#%%
labels = ['month','day','ek','ape','eb','temp','salinity','0.1']
da_tide = pd.read_csv('./ocn.log_tide',skiprows=4,sep='\s+',
                      names=labels,
                      ) 
da_tide = da_tide.drop(columns=['0.1'])
rows_to_delete1 = [960 * i + (i - 1) for i in range(1, (len(da_tide) // 960) + 1)]
da_tide = da_tide.drop(rows_to_delete1)
da_tide['time'] = pd.date_range(start='2016-01-01', periods=len(da_tide), freq='90s')
da_tide[labels[2:-1]] = da_tide[labels[2:-1]].apply(lambda x: x.str.replace('D', 'E').astype(float))  

da_notide = pd.read_csv('./ocn.log_notide.log',skiprows=4,sep='\s+',
                        names=labels,
                        )
da_notide = da_notide.drop(columns=['0.1'])
rows_to_delete2 = [960 * i + (i - 1) for i in range(1, (len(da_notide) // 960) + 1)]
da_notide = da_notide.drop(rows_to_delete2)
da_notide['time'] = pd.date_range(start='2016-01-01', periods=len(da_notide), freq='90s')
da_notide[labels[2:-1]] = da_notide[labels[2:-1]].apply(lambda x: x.str.replace('D', 'E').astype(float))  

# %%
sns.relplot(data=da_notide,x='time',y="ek", kind="line")
sns.relplot(data=da_tide,x='time',y="ek", kind="line")

# %%
sns.set_style("ticks")
sns.set_context("poster")
sns.set_palette("husl")

fig, axs = plt.subplots(1, 1, figsize=(20, 8))

sns.lineplot(ax=axs,data=da_notide,x='time',y="ek",label='no tide')
sns.lineplot(ax=axs,data=da_tide,x='time',y="ek",label='tide')
axs.set_xlabel("Time")
axs.set_ylabel("EK(W)")

# ax2 = axs.twinx()
# sns.lineplot(ax=ax2, data=da_notide, x='time', y='another_variable', label='another variable', color='r')
# ax2.set_ylabel("Another Variable")
sns.despine(fig,offset=10, trim=True)
# axs[0].set_xlabel("Lead days")
# axs[0].set_title("SST drifter RMSE")
# axs[1].set_xlabel("Lead days")
# axs[1].set_title("SST drifter bias")

# axs[0].set_ylabel("RMSE(°C)",labelpad=10)
# axs[1].set_ylabel("Bias(°C)",labelpad=10)

# # axs[0].set_xlim(-1,6)
# # axs[1].set_xlim(-1,6)
# # patches = [Patch(facecolor=palettes[i], label=model_names[i]) for i in range(len(model_names))]

# axs[1].legend(loc='lower center', frameon=False,
#           bbox_to_anchor=(-0.1, -0.38),ncol=8,
#           mode="none") 
# plt.subplots_adjust(wspace=0.3)
# fig.savefig("/home/work/tzw/code/pics/sst_ivtt.png",bbox_inches='tight') 
# %%
fig, axs = plt.subplots(1, 1, figsize=(20, 8))

sns.lineplot(ax=axs,data=da_notide,x='time',y="ape",label='no tide')
sns.lineplot(ax=axs,data=da_tide,x='time',y="ape",label='tide')
axs.set_xlabel("Time")
axs.set_ylabel("APE(W)")

# ax2 = axs.twinx()
# sns.lineplot(ax=ax2, data=da_notide, x='time', y='another_variable', label='another variable', color='r')
# ax2.set_ylabel("Another Variable")
sns.despine(fig,offset=10, trim=True)
# %%
fig, axs = plt.subplots(1, 1, figsize=(20, 8))

sns.lineplot(ax=axs,data=da_notide.loc[10000:,:],x='time',y="eb",label='no tide')
sns.lineplot(ax=axs,data=da_tide.loc[10000:,:],x='time',y="eb",label='tide')
axs.set_xlabel("Time")
axs.set_ylabel("temp(°C)")

# ax2 = axs.twinx()
# sns.lineplot(ax=ax2, data=da_notide, x='time', y='another_variable', label='another variable', color='r')
# ax2.set_ylabel("Another Variable")
sns.despine(fig,offset=10, trim=True)
# %%
