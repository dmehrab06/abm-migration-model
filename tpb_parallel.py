import numpy as np
import time
import math
import s2sphere
import multiprocessing as mp
import pandas as pd
#import fireducks.pandas as pd
import logging
from InputClass import  AgentConflictInputs, AttitudePBCInputs, NetworkInputs, TimingCheckPoints, InputSizeInfo, FinalActionInputs
from utils import *

'''multi core version of the peer influence part'''
def peer_effect_parallel_old(args):
    temp_household,neighbor_chunk,tau_lo,tau_hi,lambda_1,lambda_2 = args
    
    network_creation_start = time.time()
    #h_copy = temp_household.sort_values(by='hid')
    #h_copy = temp_household.copy()
    staying_hh_list = temp_household.hid.tolist()
    
    N_stay_set = neighbor_chunk[neighbor_chunk.hid_x.isin(staying_hh_list)]
    N_stay_set = N_stay_set[N_stay_set.hid_y.isin(staying_hh_list)] #dataframe with two neighbors as columns
    x_move_df = N_stay_set.merge(temp_household[['hid','moves','P(move)']],left_on='hid_x',right_on='hid',how='inner')
    x_move_df = x_move_df.rename(columns={'moves':'moves_x','P(move)':'own_move_prob'})
    x_move_df = x_move_df.drop(columns=['hid'])   #hidx-movex-hidy
    h_g = x_move_df.merge(temp_household[['hid','moves']],left_on='hid_y',right_on='hid',how='inner')
    h_g = h_g.rename(columns={'moves':'moves_y'})
    h_g = h_g.drop(columns=['hid']) #hidx--moves_x--hidy--moves_y
    #print('network creation through node merging by s2 takes',time.time()-network_creation_start,'seconds')
    
    neighbor_count_start = time.time()
    h_g_move = (h_g.groupby('hid_x')['moves_y'].sum().reset_index()).merge(h_g[['hid_x','moves_x','own_move_prob']].drop_duplicates(),on='hid_x',how='inner')
    h_g_move['moves_y'] = h_g_move['moves_y'] - h_g_move['moves_x'] #discard oneself while looking at neighbors
    
    h_g_in_peer_cnt = h_g.groupby('hid_x')['moves_y'].count().reset_index()
    h_g_in_peer_cnt = h_g_in_peer_cnt.merge(temp_household[['hid','N_size']],left_on='hid_x',right_on='hid',how='inner')
    h_g_in_peer_cnt['N_gone'] =  h_g_in_peer_cnt['N_size'] - h_g_in_peer_cnt['moves_y']
    h_g_in_peer_cnt = h_g_in_peer_cnt.drop(columns=['hid','moves_y']) #hid_x,N_size,'N_gone -- how many have already migrated'
    
    h_state = h_g_move[['hid_x','moves_y','moves_x','own_move_prob']].merge(h_g_in_peer_cnt,on='hid_x',how='inner')
    #print(h_network_peer_move_count.shape,h_network_peer_rem_count.shape,h_state.shape)
    h_state['psi_1'] = h_state['moves_y']/h_state['N_size']
    h_state['psi_2'] = h_state['N_gone']/h_state['N_size']
    
    threshold_func_start = time.time()
    h_state['m_state'] = h_state.apply(lambda x: apply_voter_model(x['moves_x'],x['psi_1'],x['psi_2'],lambda_1,lambda_2,tau_lo,tau_hi),axis=1)
    h_state['m_state_2'] = h_state.apply(lambda x: apply_voter_model(x['moves_x'],x['own_move_prob'],x['psi_2'],lambda_1,lambda_2,tau_lo,tau_hi),axis=1)
    #print('applying threshold function takes',time.time()-threshold_func_start,'seconds')
    #h_state = h_state.sort_values(by='hid_x')
    '''there should be assertion that these two have the same households in order
    h_state.hid_x  == h_copy.hid'''
    # h_copy['moves'] = h_state['m_state']
    # h_copy['moves_2'] = h_state['m_state_2']
    temp_household = temp_household.merge(h_state[['hid_x','m_state','m_state_2']],left_on='hid',right_on='hid_x',how='inner')
    temp_household['moves'] = temp_household['m_state']
    temp_household['moves_2'] = temp_household['m_state_2']
    temp_household = temp_household.drop(columns=['hid_x'])
    return temp_household

'''multi core version of the peer influence part'''
def peer_effect_parallel_old2(args):
    temp_household,neighbor_chunk,tau_lo,tau_hi,lambda_1,lambda_2 = args
    
    h_copy = temp_household[['hid','N_size','P(move)','moves']].copy()
    staying_hh = h_copy.hid.tolist()
    self_staying_chunk = neighbor_chunk[neighbor_chunk.hid_x.isin(staying_hh)]
    self_and_neighbor_staying_chunk = self_staying_chunk[self_staying_chunk.hid_y.isin(staying_hh)]
    hh_remaining_neighbor = self_and_neighbor_staying_chunk.groupby('hid_x')['hid_y'].count().reset_index()
    hh_remaining_neighbor = hh_remaining_neighbor.rename(columns={'hid_y':'N_remain'})
    
    peer_count_df = hh_remaining_neighbor.merge(h_copy,left_on='hid_x',right_on='hid',how='inner')
    peer_count_df = peer_count_df.drop(columns=['hid'])
    peer_count_df = peer_count_df.rename(columns={'hid_x':'hid'})
    peer_count_df['N_left'] = peer_count_df['N_size']-peer_count_df['N_remain'] # hid - N_left - N_size - P(move) - moves
    peer_count_df['psi'] = peer_count_df['N_left']/peer_count_df['N_size']
    
    peer_count_df['moves_2'] = peer_count_df.apply(lambda x: apply_voter_model(x['moves'],x['P(move)'],x['psi'],lambda_1,lambda_2,tau_lo,tau_hi),axis=1)
    temp_household = temp_household.merge(peer_count_df[['hid','moves_2']],on='hid',how='inner')
    
    return temp_household

'''even more optimized hopefully'''
def peer_effect_parallel(args):
    temp_household,neighbor_chunk,tau_lo,tau_hi,lambda_1,lambda_2 = args
    
    h_copy = temp_household[['hid', 'N_size', 'P(move)', 'moves']]

    staying_hh = h_copy['hid'].tolist()  # Set for fast lookup

    # Only keep neighbor relationships where both are staying
    self_staying_chunk = neighbor_chunk[ neighbor_chunk['hid_x'].isin(staying_hh) & neighbor_chunk['hid_y'].isin(staying_hh)]
    # Count remaining neighbors for each agent
    hh_remaining_neighbor = self_staying_chunk.groupby('hid_x').size().reset_index(name='N_remain')

    # Merge neighbor counts with household info
    peer_count_df = h_copy.merge(hh_remaining_neighbor, left_on='hid', right_on='hid_x', how='left')
    peer_count_df = peer_count_df.drop(columns=['hid_x'])
    peer_count_df['N_remain'] = peer_count_df['N_remain'].fillna(0).astype(int)
    peer_count_df['N_left'] = peer_count_df['N_size'] - peer_count_df['N_remain']
    peer_count_df['psi'] = peer_count_df['N_left'] / peer_count_df['N_size']

    # ---- Key optimization: Vectorize apply_voter_model ----
    p_fraction = peer_count_df['P(move)']
    m_fraction = peer_count_df['psi']
    cur_decision = peer_count_df['moves']
    tot_fraction = (p_fraction * lambda_1 + m_fraction * lambda_2) / 2.0
    peer_count_df['moves_2'] = np.select([tot_fraction >= tau_hi, tot_fraction < tau_lo],[1, 0],default=cur_decision)
    # Merge back
    temp_household = temp_household.merge(peer_count_df[['hid', 'moves_2']], on='hid', how='left')
    
    return temp_household


'''multi-core version for the attitude part'''
def calc_attitude_parallel(args):
    agent_data,conflict_data,t,event_scale,eps,delta,q,v,theta = args
    
    if conflict_data.shape[0]==0:
        agent_fear_data = agent_data.groupby(['hid'])['prob_conflict'].sum().reset_index()
        #agent_fear_data['prob_conflict'] = agent_fear_data['prob_conflict'].apply(lambda x: memory_decay(x,theta))
        agent_fear_data['prob_conflict'] = agent_fear_data['prob_conflict'] * (theta / 100.0) # for fast computation possibly
        #agent_data = agent_data.drop(columns='prob_conflict')
        #agent_fear_data['P(violence)'] = agent_fear_data['prob_conflict'].apply(lambda x: aggregated_prob_conflict(x,q,v))
        agent_fear_data['P(violence)'] = 1.0 / (1 + q * np.exp(-v * agent_fear_data['prob_conflict']))
        #agent_fear_data = agent_fear_data.merge(agent_data,on='hid',how='inner')
        return agent_fear_data.merge(agent_data.drop(columns='prob_conflict'), on='hid', how='inner') #for less shuffling
        #return agent_fear_data
    
    conflict_data = conflict_data.rename(columns={'latitude':'impact_lat','longitude':'impact_lng'})
    conflict_data['cur_time'] = t
    conflict_data['time_diff_to_event'] = (conflict_data['cur_time'] - conflict_data['time']) / np.timedelta64(1,'D')
    conflict_data['w_c'] = conflict_data['event_weight']*conflict_data['event_intensity']*event_scale
    conflict_data['w_c'].replace(to_replace = 0, value = eps, inplace=True)
    
    axc_data = conflict_data.merge(agent_data,on='matching_place_id',how='inner')
    axc_data['dis_conflict_home'] = haversine(axc_data['h_lng'],axc_data['h_lat'],axc_data['impact_lng'],axc_data['impact_lat'])
    
    #axc_data['prob_conflict'] = axc_data['prob_conflict'].apply(lambda x: memory_decay(x,theta))
    axc_data['prob_conflict'] = axc_data['prob_conflict']*(theta/100.0)  #hopefully this is faster
    
    #axc_data['g'] = axc_data.apply(lambda x: prob_conflict(x['w_c'],x['dis_conflict_home'],x['time_diff_to_event'],delta),axis=1)
    axc_data['g'] = (axc_data['w_c'] / (axc_data['dis_conflict_home'] ** delta)) * (1.0 / (1 + axc_data['time_diff_to_event'])) #hopefully this is faster
    
    #print(impact_in_homes.shape[0],flush=True)
    axc_data['prob_conflict'] = axc_data['prob_conflict'] + axc_data['g']
    agent_data = agent_data.drop(columns='prob_conflict')
    agent_fear_data = axc_data.groupby(['hid'])['prob_conflict'].sum().reset_index()
    
    #agent_fear_data['P(violence)'] = agent_fear_data['prob_conflict'].apply(lambda x: aggregated_prob_conflict(x,q,v))
    agent_fear_data['P(violence)'] = 1.0 / (1 + q * np.exp(-v * agent_fear_data['prob_conflict'])) #hopefully this is faster
    
    agent_fear_data = agent_fear_data.merge(agent_data,on='hid',how='inner')
    return agent_fear_data

'''helper function for calc_attitude_parallel, creates chunks of agents and each chunk is assigned to a separate core
conflict data is copied across every chunk. might trigger memory issue'''
def multiproc_attitude(house_data, conflict_data, t_begin, t_end, t_min, event_scale, eps, delta, q, v, theta, num_cores):
    cpus = num_cores
    house_splits = np.array_split(house_data, cpus) #--this a list with multiple dataframe.. each dataframe is used by one core
    current_conflict_data = conflict_data[(conflict_data.time>=t_begin) & (conflict_data.time<=t_end)]
    pool_args = [(h,current_conflict_data,t_min,event_scale,eps,delta,q, v, theta) for h_idx,h in enumerate(house_splits)]
    #print('total time taken to split',time.time()-st_time)
    pool = mp.Pool(processes = cpus)
    attitude_results = pool.map(calc_attitude_parallel, pool_args)
    pool.close()
    pool.join()
    return pd.concat(attitude_results)

'''helper function for peer_effect_parallel, creates chunks of households and each chunk is assigned to a separate core
neighbor data for every chunk is also passed. So, one house can be passed as neighbors across multiple chunks'''
def multiproc_peer_effect(house_data, neighbor_chunks, tau_lo, tau_hi, lambda_1, lambda_2, num_cores):
    house_core_assignment = house_data.groupby('core_id')
    house_chunks = [house_core_assignment.get_group(x) for x in house_core_assignment.groups]
    core_lists = [x for x in house_core_assignment.groups]
    
    house_chunk_sizes = [h_chunk.shape[0] for h_chunk in house_chunks]
    #print('hh chunk sizes before sending to peer effect workers',flush=True)
    #print(chunk_sizes,flush=True)
    
    cpus = min(num_cores,len(core_lists))
    pool_args = [(h_chunk,neighbor_chunks[core_lists[h_idx]],tau_lo,tau_hi, lambda_1,lambda_2) for h_idx,h_chunk in enumerate(house_chunks)]
    pool = mp.Pool(processes = cpus)
    subjective_norm_results = pool.map(peer_effect_parallel, pool_args)
    pool.close()
    pool.join()
    return pd.concat(subjective_norm_results)
    
def step_parallel(main_params : AgentConflictInputs, risk_params: AttitudePBCInputs, net_params: NetworkInputs, act_params: FinalActionInputs):
    logger = logging.getLogger(__name__)
    
    '''theory of planned behavior: attitude'''
    attitude_start = time.time()
    C = main_params.cc_data
    H = main_params.hh_data
    CUR_T = main_params.cur_t
    CT_S = main_params.ct_start
    CT_E = main_params.ct_end
    CPU_CORE = main_params.use_core
    home_conflict_df = multiproc_attitude(H,C,CT_S, CT_E, CUR_T,risk_params.es, risk_params.e, risk_params.delta, risk_params.Q, risk_params.V, risk_params.theta, CPU_CORE)            
    attitude_end = time.time()
    logger.info(f'{attitude_end-attitude_start} second to process attitude')

    '''theory of planned behavior: pbc -- maybe this part can be a little bit cleaned up further, especially the nested if-else s'''
    pcb_start = time.time()
    if main_params.flag_p:
        if risk_params.pbc_agent:
            home_conflict_df['scaled_prob_conflict'] = home_conflict_df['prob_conflict']*home_conflict_df['P(move|violence)']
            #hcdf['P(move)'] = hcdf['scaled_prob_conflict'].apply(lambda x: aggregated_prob_conflict(x,risk_params.Q, risk_params.V))
            home_conflict_df['P(move)'] = 1.0 / (1 + risk_params.Q * np.exp(-risk_params.V * home_conflict_df['scaled_prob_conflict']))
            home_conflict_df = home_conflict_df.drop(columns=['scaled_prob_conflict'])
        else:
            home_conflict_df['P(move)'] = home_conflict_df['P(violence)']*home_conflict_df['P(move|violence)']
    else:
        home_conflict_df['P(move)'] = home_conflict_df['P(violence)']
                
    home_conflict_df['random'] = np.random.random(home_conflict_df.shape[0])
    #home_conflict_df['moves'] = home_conflict_df.apply(lambda x: bernoulli(x['random'],x['P(move)']),axis=1)
    home_conflict_df['moves'] = (home_conflict_df['random'] <= home_conflict_df['P(move)']).astype(int) #hopefully faster
    
    cols_to_delete = home_conflict_df.columns.intersection(main_params.del_col1)
    logger.debug(f'After attitude {cols_to_delete} columns will be deleted')
    home_conflict_df = home_conflict_df.drop(columns=cols_to_delete)
    logger.debug(f'Agent x Conflict data has the columns: {home_conflict_df.columns.tolist()} after attitude')
    
    pcb_end = time.time()
    logger.info(f'{pcb_end-pcb_start} second to process PBC')
    
    '''theory of planned behavior: subjective norm'''
    subjective_norm_start = time.time()
    temp_households = home_conflict_df
    nodes = temp_households.shape[0]
    phase = (0 if (main_params.simtime < net_params.phase_shift_day) else 1)
    
    logger.debug(f'{nodes} houses in temp household file before calling peer effect function')
    if main_params.flag_s:
        neighbor_household_data = main_params.neighbor_data
        for peer_it in range(0,net_params.thresh_steps):
            temp_households = multiproc_peer_effect(temp_households,neighbor_household_data,net_params.tau_lo,net_params.tau_hi,net_params.lambda_1,net_params.lambda_2,CPU_CORE)
            logger.debug(f'{temp_households.shape[0]} houses in temp household file after calling peer effect function {peer_it+1} times')
            
        if not net_params.knows_neighbor:
            temp_households['moves'] = temp_households['moves_2']
            
    cols_to_delete = temp_households.columns.intersection(main_params.del_col2)
    logger.debug(f'After subjective norm {cols_to_delete} columns will be deleted')
    temp_households = temp_households.drop(columns=cols_to_delete)
    
    #return temp_households
    leaving_agent = temp_households[temp_households.moves==1]['hh_size'].sum()
    leaving_hh = temp_households[temp_households.moves==1].shape[0]
    logger.debug(f'After peer_effect {leaving_hh} houses are ready to migrate having {leaving_agent} agents')
    
    '''decide on final action'''
    temp_households['move_type_random'] = np.random.random(temp_households.shape[0])
    # temp_households['move_type'] = temp_households.apply(lambda x: bernoulli_border(x['move_type_random'],x['moves'],phase,act_params.refugee_ratio,act_params.lo,act_params.hi),axis=1)
    temp_households['move_type'] = np.select([temp_households['moves']==0,temp_households['move_type_random']<=act_params.refugee_ratio],
                                                [0,2],default=1) ## hopefully faster

    temp_households = temp_households.drop(columns=['move_type_random'])
    
    idp_agent = temp_households[temp_households.move_type==1]['hh_size'].sum()
    idp_hh = temp_households[temp_households.move_type==1].shape[0]
    refugee_agent = temp_households[temp_households.move_type==2]['hh_size'].sum()
    refugee_hh = temp_households[temp_households.move_type==2].shape[0]
    
    logger.debug(f'After coin toss {idp_agent} agents assigned idp and {refugee_agent} agents assigned refugee, totalling {idp_agent+refugee_agent} migrant agents')
    logger.debug(f'After coin toss {idp_hh} houses assigned idp and {refugee_hh} houses assigned refugee, totalling {idp_hh+refugee_hh} migrant houses')
    
    subjective_norm_end = time.time()
    logger.info(f'{subjective_norm_end-subjective_norm_start} second to process subjective norm')
    
    t_ckpt = TimingCheckPoints(att_start=attitude_start,att_end=attitude_end,pcb_start=pcb_start,pcb_end=pcb_end,sn_start=subjective_norm_start,sn_end=subjective_norm_end)
    
    chunk_sizes = [cs.shape[0] for cs in neighbor_household_data]
    input_info = InputSizeInfo(num_agents = H['hh_size'].sum(), num_households = H.shape[0], num_events = C.shape[0],
                              num_edges_network = sum(chunk_sizes))
    
    return temp_households,t_ckpt,input_info