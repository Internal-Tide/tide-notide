#%%
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
#%%
def skip_separator_rows(x):
    # x是行号，从0开始，每482行中的第481行（索引为481的行）是分隔行
    return x % 482 == 481

labels = ['month','day','KE','APE','SST','Temp','Salinity','0.1']
da_tide = pd.read_csv('./ocn.log_tidep',sep='\s+',
                      names=labels,skiprows=skip_separator_rows
                      ) 
da_tide = da_tide.drop(columns=['0.1'])
da_tide['time'] = pd.date_range(start='2016-06-01', periods=len(da_tide), freq='180s')
da_tide[labels[2:-1]] = da_tide[labels[2:-1]].apply(lambda x: x.str.replace('D', 'E').astype(float))  
da_tide["KE+APE"] = da_tide["KE"] + da_tide["APE"]
da_tide.set_index('time', inplace=True)
da_tide['KE_smooth'] = da_tide['KE'].rolling(window=14400, center=True).mean()
da_tide['APE_smooth'] = da_tide['APE'].rolling(window=14400, center=True).mean()
da_tide['KE+APE_smooth'] = da_tide['KE+APE'].rolling(window=14400, center=True).mean()

da_notide = pd.read_csv('./ocn.log_notide',sep='\s+',
                        names=labels,skiprows=skip_separator_rows
                        )
da_notide = da_notide.drop(columns=['0.1'])
da_notide['time'] = pd.date_range(start='2016-06-01', periods=len(da_notide), freq='180s')
da_notide[labels[2:-1]] = da_notide[labels[2:-1]].apply(lambda x: x.str.replace('D', 'E').astype(float))  
da_notide["KE+APE"] = da_notide["KE"] + da_notide["APE"]
# %%
sns.set_style("ticks")
sns.set_context("poster")
sns.set_palette("husl")
#%%
def plot_timeseries(da_tide,da_notide,variable):
    fig, axs = plt.subplots(1, 1, figsize=(40, 8))
    if variable == 'Salinity':
        axs.text(-0.04, 1.02, '(+35)', fontsize=24,transform=axs.transAxes)
    if variable == 'KE':
        sns.lineplot(ax=axs,data=da_tide,x='time',y='KE_smooth',label='tide_smooth',lw=2,color='r')
    if variable == 'APE':
        sns.lineplot(ax=axs,data=da_tide,x='time',y='APE_smooth',label='tide_smooth',lw=2,color='r')
    if variable == 'KE+APE':
        sns.lineplot(ax=axs,data=da_tide,x='time',y='KE+APE_smooth',label='tide_smooth',lw=2,color='r')
    sns.lineplot(ax=axs,data=da_notide,x='time',y=variable,label='no tide',lw=2,color='b')
    sns.lineplot(ax=axs,data=da_tide,x='time',y=variable,label='tide',lw=1,color='y')
    axs.legend(loc='upper right')
    axs.set_xlabel("Time")
    axs.set_ylabel(variable)
    sns.despine(fig,offset=0, trim=True)
#%%
plot_timeseries(da_tide,da_notide,'KE')
plot_timeseries(da_tide,da_notide,'APE')
plot_timeseries(da_tide,da_notide,'SST')
plot_timeseries(da_tide,da_notide,'Temp')
plot_timeseries(da_tide,da_notide,'Salinity')
plot_timeseries(da_tide,da_notide,'KE+APE')
