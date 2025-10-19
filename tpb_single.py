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

def refine_through_peer_effect(temp_household,neighbor_data,lambda_1,lambda_2,tau_lo,tau_hi,current_phase=0):
    ## current_phase is not used in this function???
    '''temp_household is a subset of households with their move status
    we need hid,moves,N_size, maybe some optimization can be done if we only pass these 3 columns'''
    network_creation_start = time.time()
    h_copy = temp_household.copy() ## additionally has P(move)
    '''only take neighbors who have not migrated yet'''
    N_stay_set = neighbor_data[neighbor_data.hid_x.isin(h_copy.hid.tolist())]
    N_stay_set = N_stay_set[N_stay_set.hid_y.isin(h_copy.hid.tolist())] #dataframe with two neighbors as columns
    x_move_df = N_stay_set.merge(h_copy[['hid','moves','P(move)']],left_on='hid_x',right_on='hid',how='inner')
    x_move_df = x_move_df.rename(columns={'moves':'moves_x','P(move)':'own_move_prob'})
    x_move_df = x_move_df.drop(columns=['hid'])   #hidx-movex-hidy
    h_g = x_move_df.merge(h_copy[['hid','moves']],left_on='hid_y',right_on='hid',how='inner')
    h_g = h_g.rename(columns={'moves':'moves_y'})
    h_g = h_g.drop(columns=['hid']) #hidx--moves_x--hidy--moves_y--own_move_prob
    #print('network creation through node merging by s2 takes',time.time()-network_creation_start,'seconds')
    
    neighbor_count_start = time.time()
    h_g_move = (h_g.groupby('hid_x')['moves_y'].sum().reset_index()).merge(h_g[['hid_x','moves_x','own_move_prob']].drop_duplicates(),on='hid_x',how='inner')
    h_g_move['moves_y'] = h_g_move['moves_y'] - h_g_move['moves_x'] #discard oneself while looking at neighbors
    ##h_g_move contains for each hid, how many of their neighbors are moving? --hid_x, moves_y, moves_x, own_move_prob
    
    h_g_in_peer_cnt = h_g.groupby('hid_x')['moves_y'].count().reset_index()
    h_g_in_peer_cnt = h_g_in_peer_cnt.merge(h_copy[['hid','N_size']],left_on='hid_x',right_on='hid',how='inner')
    h_g_in_peer_cnt['N_gone'] =  h_g_in_peer_cnt['N_size'] - h_g_in_peer_cnt['moves_y']
    h_g_in_peer_cnt = h_g_in_peer_cnt.drop(columns=['hid','moves_y']) #hid_x,N_size,'N_gone -- how many have already migrated'
    
    h_state = h_g_move[['hid_x','moves_y','moves_x', 'own_move_prob']].merge(h_g_in_peer_cnt,on='hid_x',how='inner')
    #print(h_network_peer_move_count.shape,h_network_peer_rem_count.shape,h_state.shape)
    h_state['psi_1'] = h_state['moves_y']/h_state['N_size']
    h_state['psi_2'] = h_state['N_gone']/h_state['N_size']
    ## h_state -- #hid_x,'moves_y','moves_x','own_move_prob','N_size','N_gone','psi_1','psi_2','m_state_1','m_state_2'
    
    threshold_func_start = time.time()
    h_state['m_state'] = h_state.apply(lambda x: apply_voter_model(x['moves_x'],x['psi_1'],x['psi_2'],lambda_1,lambda_2,tau_lo,tau_hi),axis=1)
    h_state['m_state_2'] = h_state.apply(lambda x: apply_voter_model(x['moves_x'],x['own_move_prob'],x['psi_2'],lambda_1,lambda_2,tau_lo,tau_hi),axis=1)
    #print('applying threshold function takes',time.time()-threshold_func_start,'seconds')
    #h_state = h_state.sort_values(by='hid_x')
    '''there should be assertion that these two have the same households in order
    h_state.hid_x  == h_copy.hid'''
    # h_copy['moves'] = h_state['m_state']
    # h_copy['moves_2'] = h_state['m_state_2']
    h_copy = h_copy.merge(h_state[['hid_x','m_state','m_state_2']],left_on='hid',right_on='hid_x',how='inner')
    h_copy['moves'] = h_copy['m_state']
    h_copy['moves_2'] = h_copy['m_state_2']
    h_copy = h_copy.drop(columns=['hid_x'])
    return h_copy


'''Calculate each agents fear at the current time, takes agent data and a set of observed events as input
uses a gravity model and a discounting utility model to calculate the final output'''
def calc_attitude(conflict_data,agent_data,eps,event_scale,theta,delta,q,v,t):
    
    if conflict_data.shape[0]==0:
        agent_fear_data = agent_data.groupby(['hid'])['prob_conflict'].sum().reset_index()
        agent_fear_data['prob_conflict'] = agent_fear_data['prob_conflict'].apply(lambda x: memory_decay(x,theta))
        agent_data = agent_data.drop(columns='prob_conflict')
        agent_fear_data['P(violence)'] = agent_fear_data['prob_conflict'].apply(lambda x: aggregated_prob_conflict(x,q,v))
        agent_fear_data = agent_fear_data.merge(agent_data,on='hid',how='inner')
        return agent_fear_data
    
    conflict_data = conflict_data.rename(columns={'latitude':'impact_lat','longitude':'impact_lng'})
    conflict_data['cur_time'] = t
    conflict_data['time_diff_to_event'] = (conflict_data['cur_time'] - conflict_data['time']) / np.timedelta64(1,'D')
    conflict_data['w_c'] = conflict_data['event_weight']*conflict_data['event_intensity']*event_scale
    conflict_data['w_c'].replace(to_replace = 0, value = eps, inplace=True)
    
    axc_data = conflict_data.merge(agent_data,on='matching_place_id',how='inner')
    axc_data['dis_conflict_home'] = haversine(axc_data['h_lng'],axc_data['h_lat'],axc_data['impact_lng'],axc_data['impact_lat'])
    axc_data['prob_conflict'] = axc_data['prob_conflict'].apply(lambda x: memory_decay(x,theta)) #-----do we need a separate function----, we can just multiply... (however, there may be other forms)
    axc_data['g'] = axc_data.apply(lambda x: prob_conflict(x['w_c'],x['dis_conflict_home'],x['time_diff_to_event'],delta),axis=1)
    #print(impact_in_homes.shape[0],flush=True)
    axc_data['prob_conflict'] = axc_data['prob_conflict'] + axc_data['g']
    agent_data = agent_data.drop(columns='prob_conflict')
    agent_fear_data = axc_data.groupby(['hid'])['prob_conflict'].sum().reset_index()
    agent_fear_data['P(violence)'] = agent_fear_data['prob_conflict'].apply(lambda x: aggregated_prob_conflict(x,q,v))
    agent_fear_data = agent_fear_data.merge(agent_data,on='hid',how='inner')
    return agent_fear_data

def calc_attitude_fast(conflict_data, agent_data, eps, event_scale, theta, delta, q, v, t):
    if conflict_data.empty:
        # Only memory decay needed
        agent_data['prob_conflict'] *= (theta / 100.0)
        agent_fear_data = agent_data.groupby('hid', as_index=False)['prob_conflict'].sum()
        agent_fear_data['P(violence)'] = 1.0 / (1 + q * np.exp(-v * agent_fear_data['prob_conflict']))
        return agent_fear_data.merge(agent_data.drop(columns='prob_conflict'), on='hid', how='inner')

    # Prepare conflict data
    conflict_data = conflict_data.rename(columns={'latitude': 'impact_lat', 'longitude': 'impact_lng'})
    conflict_data['cur_time'] = t
    conflict_data['time_diff_to_event'] = (conflict_data['cur_time'] - conflict_data['time']) / np.timedelta64(1, 'D')
    conflict_data['w_c'] = conflict_data['event_weight'] * conflict_data['event_intensity'] * event_scale
    conflict_data['w_c'] = conflict_data['w_c'].replace(0, eps)

    # Merge agents with conflict events
    axc_data = conflict_data.merge(agent_data, on='matching_place_id', how='inner')

    # Vectorized haversine
    axc_data['dis_conflict_home'] = haversine_vec(axc_data['h_lng'], axc_data['h_lat'],
                                                  axc_data['impact_lng'], axc_data['impact_lat'])

    # Memory decay
    axc_data['prob_conflict'] *= (theta / 100.0)

    # Gravity model vectorized
    axc_data['g'] = (axc_data['w_c'] / (axc_data['dis_conflict_home'] ** delta)) * (1.0 / (1 + axc_data['time_diff_to_event']))

    # Update prob_conflict
    axc_data['prob_conflict'] += axc_data['g']

    # Aggregate at household level
    agent_fear_data = axc_data.groupby('hid', as_index=False)['prob_conflict'].sum()

    # Apply sigmoid function
    agent_fear_data['P(violence)'] = 1.0 / (1 + q * np.exp(-v * agent_fear_data['prob_conflict']))

    # Merge back other agent fields (except outdated prob_conflict)
    agent_fear_data = agent_fear_data.merge(agent_data.drop(columns='prob_conflict'), on='hid', how='inner')

    return agent_fear_data

def step_single(main_params : AgentConflictInputs, risk_params: AttitudePBCInputs, net_params: NetworkInputs, act_params: FinalActionInputs):
    logger = logging.getLogger(__name__)
    
    '''theory of planned behavior: attitude'''
    attitude_start = time.time()
    cur_impact_data = main_params.cc_data
    cur_household_data = main_params.hh_data
    T_CURRENT = main_params.cur_t
    home_conflict_df = calc_attitude(cur_impact_data,cur_household_data,risk_params.e,risk_params.es, risk_params.theta, risk_params.delta, risk_params.Q, risk_params.V, T_CURRENT)            
    attitude_end = time.time()
    logger.info(f'{attitude_end-attitude_start} second to process attitude')

    '''theory of planned behavior: pbc -- maybe this part can be a little bit cleaned up further, especially the nested if-else s'''
    pcb_start = time.time()
    if main_params.flag_p:
        if risk_params.pbc_agent:
            home_conflict_df['scaled_prob_conflict'] = home_conflict_df['prob_conflict']*home_conflict_df['P(move|violence)']
            home_conflict_df['P(move)'] = home_conflict_df['scaled_prob_conflict'].apply(lambda x: aggregated_prob_conflict(x,risk_params.Q, risk_params.V))
            home_conflict_df = home_conflict_df.drop(columns=['scaled_prob_conflict'])
        else:
            home_conflict_df['P(move)'] = home_conflict_df['P(violence)']*home_conflict_df['P(move|violence)']
    else:
        home_conflict_df['P(move)'] = home_conflict_df['P(violence)']
                
    home_conflict_df['random'] = np.random.random(home_conflict_df.shape[0])
    home_conflict_df['moves'] = home_conflict_df.apply(lambda x: bernoulli(x['random'],x['P(move)']),axis=1)
    
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

    if main_params.flag_s:
        neighbor_household_data = main_params.neighbor_data
        for peer_it in range(0,net_params.thresh_steps):
            temp_households = refine_through_peer_effect(temp_households,neighbor_household_data,net_params.lambda_1,net_params.lambda_2,net_params.tau_lo,net_params.tau_hi,phase)
                
        if not net_params.knows_neighbor:
            temp_households['moves'] = temp_households['moves_2']
            #temp_households = temp_households.drop(columns=main_params.del_col2)
    
    cols_to_delete = temp_households.columns.intersection(main_params.del_col2)
    logger.debug(f'After subjective norm {cols_to_delete} columns will be deleted')
    temp_households = temp_households.drop(columns=cols_to_delete)
    
    '''decide on final action'''
    temp_households['move_type_random'] = np.random.random(temp_households.shape[0])
    temp_households['move_type'] = temp_households.apply(lambda x: bernoulli_border(x['move_type_random'],x['moves'],phase,act_params.refugee_ratio,act_params.lo,act_params.hi),axis=1)
    temp_households = temp_households.drop(columns=['move_type_random'])
    subjective_norm_end = time.time()
    logger.info(f'{subjective_norm_end-subjective_norm_start} second to process subjective norm')
    
    t_ckpt = TimingCheckPoints(att_start=attitude_start,att_end=attitude_end,pcb_start=pcb_start,pcb_end=pcb_end,sn_start=subjective_norm_start,sn_end=subjective_norm_end)
    
    input_info = InputSizeInfo(num_agents = cur_household_data['hh_size'].sum(), num_households = cur_household_data.shape[0], num_events = cur_impact_data.shape[0],
                              num_edges_network = neighbor_household_data.shape[0])
    
    return temp_households,t_ckpt,input_info
    
