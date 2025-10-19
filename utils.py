import numpy as np
import time
import math
import s2sphere
import multiprocessing as mp
import pandas as pd
import logging
from InputClass import  AgentConflictInputs, AttitudePBCInputs, NetworkInputs, TimingCheckPoints, InputSizeInfo, FinalActionInputs

'''finds the great earth distance between two locations
returns the distance in kilometers
ln() should be applied to get a natural distance'''
def haversine(lon1, lat1, lon2, lat2):
    KM = 6372.8 #Radius of earth in km instead of miles
    lat1, lon1, lat2, lon2 = map(np.deg2rad, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    total_km = KM * c
    return total_km



'''Get S2 geometry of a household given level
by default L13 cell is considered'''
def getl13(lat,lng,req_level=13):
    p = s2sphere.LatLng.from_degrees(lat, lng) 
    cell = s2sphere.Cell.from_lat_lng(p)
    cellid = cell.id()
    for i in range(1,30):
        #print(cellid)
        if cellid.level()==req_level:
            return cellid
        cellid = cellid.parent()
        
'''hash-like function for assigning a cell 
to a CPU, this can be modified'''
def get_core_id(s2cellid,num_cores):
    return (int(s2cellid.to_token(),16)//16)%num_cores

'''Risk function, gravity model'''
def prob_conflict(impact,dis,t_diff=0,d=2.212):
    return ((impact)/(dis**d))*(1.0/(1+t_diff))

'''given a fear, conversion to intention. Sigmoid function used'''
def aggregated_prob_conflict(x,q=55,v=0.8):  
    return 1.0 / (1 + q*math.exp(-v*x))

'''Discount factor applied to previous fear'''
def memory_decay(x,theta=98.67):
    return x*(theta/100.00)

'''get a unified migration probability of a house given the demographic composition
this is pre-calculated due to the fact that every location of every agent is the same
as the household. Otherwise, this would not be possible'''
def get_move_prob(member_info,family_movement,non_family_movement): ##this function can be played with
    tot_size = sum(member_info)
    move_prob_family = sum([x*y for x,y in zip(member_info,family_movement)])/tot_size
    move_prob_single = sum([x*y for x,y in zip(member_info,non_family_movement)])/tot_size
    return (move_prob_family if tot_size>1 else move_prob_single)

'''coin toss'''
def bernoulli(val,p):
    return (1 if val<=p else 0)

'''Change weight of event based on type'''
def get_event_weight(event_type,sub_event_type,ablation_type='None'):
    if sub_event_type==ablation_type or event_type==ablation_type:
        return 0
    event_weight_dict = {'Battles':3,'Civilian':8,'Explosions':5,'Violence':3}
    for e in event_weight_dict:
        if event_type.startswith(e):
            return event_weight_dict[e]
    return 0

'''given a person migration state (0 or 1) return the migration status
0 means the person stays
1 means the person migrates as IDP
2 means the person migrates as refugee'''
def bernoulli_border(val,moves,current_phase,refugee_ratio=0.33,multiply_phase_1=0.8,multiply_phase_2=1.5):
    multiply = multiply_phase_1 if current_phase==0 else multiply_phase_2
    return (0 if moves==0 else (2 if val<=refugee_ratio*multiply else 1))

'''peer influnce by threshold function
current decision to migrate (1) can go to stay (0), if less than thresh_lo neighbors are migrating
current decision to stay (0) can go to migrate (1), if more than thresh_hi neighbors are migrating'''
def apply_threshold_function(cur_decision,cur_neighbor_migrating,thresh_lo=0,thresh_hi=10):
    return (1 if cur_neighbor_migrating>=thresh_hi else( 0 if cur_neighbor_migrating<thresh_lo else cur_decision))

'''peer influence is divided in two parts
outside -- m fraction, how many have already migrated
inside -- p fraction, how many wants to migrate'''
def apply_voter_model(cur_decision,p_fraction,m_fraction,lambda_1=0.99,lambda_2=0.5,thresh_lo=0.01,thresh_hi=0.4):
    try:
        tot_fraction = (p_fraction*lambda_1 + m_fraction*lambda_2)/2.0
        assert tot_fraction <= 1, "Combination of in/out peer-threshold cannot be larger than 1"
        return (1 if tot_fraction>=thresh_hi else(0 if tot_fraction<thresh_lo else cur_decision))
    except AssertionError as msg:
        logger = logging.getLogger(__name__)
        logger.error(msg)

'''takes a set of households along with their current intention, their neighbors intention
and their neighbors migration status, based on a threshold function, it produces the next state which
represents their final migration decision'''
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

'''multi core version of the previous function'''
def peer_effect_parallel(args):
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

'''multi-core version for the previous function'''
def calc_attitude_parallel(args):
    agent_data,conflict_data,t,event_scale,eps,delta,q,v,theta = args
    
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


def trim_neighborhood(hh_series,num_cores,graph):
    if num_cores==1:
        #graph is just a single dataframe
        graph = graph[graph.hid_x.isin(hh_series.tolist())]
    else:
        for i in range(len(graph)):
            graph[i] = graph[i][graph[i].hid_x.isin(hh_series.tolist())]


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
    temp_households['move_type'] = temp_households.apply(lambda x: bernoulli_border(x['move_type_random'],x['moves'],phase,act_params.refugee_ratio,act_params.lo,act_params.hi),axis=1)
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