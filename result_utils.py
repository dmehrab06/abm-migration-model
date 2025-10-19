import pandas as pd
from file_paths_and_consts import *
import os
from matplotlib import rcParams
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import json

def get_config_for_sim(sim):
    filename = f'LHC_configs/config_LHC_sim_{sim}.json'
    # Opening JSON file
    with open(filename) as json_file:
        default_param = json.load(json_file)
    return default_param

def optimize_scale(sim_data,true_data,sim_col='total_refugee',true_col='daily',sim_time_col='date',true_time_col='time',upto='2022-05-06'):
    combined_data = sim_data.merge(true_data,left_on=sim_time_col,right_on=true_time_col,how='inner')
    combined_data = combined_data.rename(columns={sim_col:'y',true_col:'x'})
    combined_data = combined_data[combined_data.date<=pd.to_datetime(upto)]
    return (combined_data['x']*combined_data['y']).sum()/(combined_data['y']*combined_data['y']).sum()

def get_daily_total_count_for_sim(all_dfs,interested_column='total_refugee',roll=7):
    
    assert interested_column in all_dfs[0].columns.tolist(), f'{interested_column} not a valid column'
    
    daily_total_refugee_df = (pd.concat(all_dfs)).groupby('date')[interested_column].sum().reset_index()
    daily_total_refugee_df[interested_column] = daily_total_refugee_df[interested_column].rolling(roll).mean()
    daily_total_refugee_df = daily_total_refugee_df.dropna(subset=[interested_column])
    daily_total_refugee_df['date'] = pd.to_datetime(daily_total_refugee_df['date'])
    
    return daily_total_refugee_df

def get_summary_counts(sim,interested_column='total_refugee'):
    all_dfs = get_result_for_sim(sim)
    daily_total_refugee_df = get_daily_total_count_for_sim(all_dfs,interested_column=interested_column)
    
    study_time_migration =  daily_total_refugee_df[(daily_total_refugee_df.date>=pd.to_datetime('2022-02-24')) 
                                             & (daily_total_refugee_df.date<=pd.to_datetime('2022-05-15'))][interested_column].sum()
    march_migration = daily_total_refugee_df[(daily_total_refugee_df.date>=pd.to_datetime('2022-03-01')) 
                                             & (daily_total_refugee_df.date<=pd.to_datetime('2022-03-31'))][interested_column].sum()
    april_migration =  daily_total_refugee_df[(daily_total_refugee_df.date>=pd.to_datetime('2022-04-01')) 
                                             & (daily_total_refugee_df.date<=pd.to_datetime('2022-04-30'))][interested_column].sum()
    may_migration = daily_total_refugee_df[(daily_total_refugee_df.date>=pd.to_datetime('2022-05-01')) 
                                             & (daily_total_refugee_df.date<=pd.to_datetime('2022-05-31'))][interested_column].sum()
    max_migration = daily_total_refugee_df[interested_column].max()
    
    # ob_tot_migration = compare_df['refugee'].sum()
    # ob_march_migration = compare_df[(compare_df.time>=pd.to_datetime('2022-03-01')) & (compare_df.time<=pd.to_datetime('2022-03-31'))]['refugee'].sum()
    # ob_april_migration = compare_df[(compare_df.time>=pd.to_datetime('2022-04-01')) & (compare_df.time<=pd.to_datetime('2022-04-30'))]['refugee'].sum()
    # ob_may_migration = compare_df[(compare_df.time>=pd.to_datetime('2022-05-01')) & (compare_df.time<=pd.to_datetime('2022-05-31'))]['refugee'].sum()
    # ob_max_migration = compare_df[str(pp)].max()
    
    output_dict ={'hypercube':sim,'output_tot':study_time_migration,'output_MAR':march_migration,
                  'output_APR':april_migration,'output_MAY':may_migration,'output_MAX':max_migration}
    return output_dict
    

def get_daily_uncertainty_count(selected_simulations,interested_column='total_refugee',q1=0.25,q3=0.75,summary_result_id=None,cache_if_compute=True):
    
    if summary_result_id is not None and os.path.isfile(f'{CACHE_DIR}forward_result_{interested_column}_uncertainty_{summary_result_id}_{q1}_{q3}.pq'):
        print('loading result from cache directory')
        df = pd.read_parquet(f'{CACHE_DIR}forward_result_{interested_column}_uncertainty_{summary_result_id}_{q1}_{q3}.pq')
        return df
    
    daily_dfs = []

    for simidx, sim in enumerate(selected_simulations):
        print(f'gathering simulation {sim} results')
        all_dfs = get_result_for_sim(sim)
        #print(len(all_dfs))
        if len(all_dfs)!=121:
            print(sim,'has not completed')
            continue
        daily_total_refugee_df = get_daily_total_count_for_sim(all_dfs,interested_column=interested_column)
        daily_dfs.append(daily_total_refugee_df)
    
    daily_df_across_sim = pd.concat(daily_dfs)
    
    daily_df_median = ((daily_df_across_sim.groupby('date')[interested_column].quantile(q=0.5)).reset_index()).rename(columns={interested_column:'median'})
    daily_df_q1 = ((daily_df_across_sim.groupby('date')[interested_column].quantile(q=q1)).reset_index()).rename(columns={interested_column:'q1'})
    daily_df_q3 = ((daily_df_across_sim.groupby('date')[interested_column].quantile(q=q3)).reset_index()).rename(columns={interested_column:'q3'})
    
    uncertain_df = (daily_df_median.merge(daily_df_q1,on='date',how='inner')).merge(daily_df_q3,on='date',how='inner')
    
    if summary_result_id is not None and cache_if_compute:
        print('caching results')
        uncertain_df.to_parquet(f'{CACHE_DIR}forward_result_{interested_column}_uncertainty_{summary_result_id}_{q1}_{q3}.pq',index=False)
    return uncertain_df

def get_selective_daily_uncertainty_count(selected_simulations,ground_truth_data,interested_column='total_refugee',q1=0.25,q3=0.75,
                                          summary_result_id=None,nrmse_threshold = 0.12, pcc_threshold = 0.9, cache_if_compute=True):
    
    file_name = f'forward_result_selective_{interested_column}_nrmse_below_{nrmse_threshold}_pcc_above_{pcc_threshold}_{summary_result_id}_{q1}_{q3}.pq'
    
    if summary_result_id is not None and os.path.isfile(f'{CACHE_DIR}{file_name}'):
        print('loading result from cache directory')
        df = pd.read_parquet(f'{CACHE_DIR}{file_name}')
        return df
    
    daily_dfs = []

    for simidx, sim in enumerate(selected_simulations):
        print(f'gathering simulation {sim} results')
        all_dfs = get_result_for_sim(sim)
        #print(len(all_dfs))
        if len(all_dfs)!=121:
            print(sim,'has not completed')
            continue
        daily_total_refugee_df = get_daily_total_count_for_sim(all_dfs,interested_column=interested_column)
        nrmse,_,corr,_,_ = get_metrics_of_daily_sim(daily_total_refugee_df,ground_truth_data)
        if nrmse<nrmse_threshold and corr>pcc_threshold:
            print(sim,'has been selected with nrmse',nrmse,'and pcc',pcc)
            daily_dfs.append(daily_total_refugee_df)
        else:
            print(sim,'has not been selected for violating some threshod with nrmse',nrmse,'and pcc',pcc)
    daily_df_across_sim = pd.concat(daily_dfs)
    
    daily_df_median = ((daily_df_across_sim.groupby('date')[interested_column].quantile(q=0.5)).reset_index()).rename(columns={interested_column:'median'})
    daily_df_q1 = ((daily_df_across_sim.groupby('date')[interested_column].quantile(q=q1)).reset_index()).rename(columns={interested_column:'q1'})
    daily_df_q3 = ((daily_df_across_sim.groupby('date')[interested_column].quantile(q=q3)).reset_index()).rename(columns={interested_column:'q3'})
    
    uncertain_df = (daily_df_median.merge(daily_df_q1,on='date',how='inner')).merge(daily_df_q3,on='date',how='inner')
    
    if summary_result_id is not None and cache_if_compute:
        print('caching results')
        uncertain_df.to_parquet(f'{CACHE_DIR}{file_name}',index=False)
    return uncertain_df

def get_daily_total_refugee_for_sim(all_dfs,upto='2022-09-15'):
    daily_total_refugee_df = (pd.concat(all_dfs)).groupby('date')['total_refugee'].sum().reset_index()
    daily_total_refugee_df['total_refugee'] = daily_total_refugee_df['total_refugee'].rolling(7).mean()
    daily_total_refugee_df = daily_total_refugee_df.dropna(subset=['total_refugee'])
    daily_total_refugee_df['date'] = pd.to_datetime(daily_total_refugee_df['date'])
    daily_total_refugee_df = daily_total_refugee_df[daily_total_refugee_df.date<=pd.to_datetime(upto)]
    return daily_total_refugee_df

def get_ofatconfig_for_sim(sim,param):
    filename = f'OFAT_configs/config_OFAT_{param}_sim_{sim}.json'
    # Opening JSON file
    with open(filename) as json_file:
        default_param = json.load(json_file)
    return default_param

def get_result_for_sim(simidx):
    simidx = simidx if simidx!=84 else 199
    RESULT_DIR = f'{OUTPUT_DIR}forward_Migration/Agg-Result-Sim-{str(simidx).zfill(9)}/'
    all_dfs = []
    dates = pd.date_range(start='2022-02-24', end='2022-09-01')
    for f in os.listdir(RESULT_DIR):
        if f.endswith('daily_aggregated_migrant.csv'):
            df = pd.read_csv(f'{RESULT_DIR}{f}')
            #print(df.shape[0])
            df['date'] = pd.to_datetime(df['date'])
            df['region'] = f.split('_')[0]
            all_dfs.append(df)
    return all_dfs

def get_timing_result_for_sim(simidx):
    RESULT_DIR = f'{OUTPUT_DIR}forward_Migration/Other-Log-Sim-{str(simidx).zfill(9)}/'
    all_dfs = []
    for f in os.listdir(RESULT_DIR):
        if f.endswith('timing_information.csv'):
            df = pd.read_csv(f'{RESULT_DIR}{f}')
            #print(df.shape[0])
            #df['date'] = dates
            df['region'] = f.split('_')[0]
            all_dfs.append(df)
    return all_dfs


def get_metrics_of_daily_sim(daily_total_refugee_df,pnas_refugee_data,scale=1.0,sim_column='total_refugee',true_column='daily',upto='2022-05-06'):
    
    assert sim_column in daily_total_refugee_df, f'{sim_column} not a valid simulation result column'
    assert true_column in pnas_refugee_data, f'{true_column} not a valid ground truth result column'
    
    assert 'date' in daily_total_refugee_df, f'time series column is not called "date" in simulation result'
    assert 'time' in pnas_refugee_data, f'time series column is not called "time" in ground truth data'
    
    result_compare = daily_total_refugee_df.merge(pnas_refugee_data,left_on='date',right_on='time',how='inner')
    result_compare = result_compare[result_compare.date<=pd.to_datetime(upto)]
    rmse = ((((result_compare[sim_column]*scale-result_compare[true_column])**2).mean())**0.5)
    nrmse = rmse/(result_compare[true_column].max()-result_compare[true_column].min())
    corr = result_compare[sim_column].corr(result_compare[true_column])
    rmspe = ((((result_compare[sim_column] - result_compare[true_column])/result_compare[true_column])**2).mean())**0.5
    mape = (abs((result_compare[sim_column]-result_compare[true_column])/result_compare[true_column])).mean()
    return nrmse,rmse,corr,rmspe,mape