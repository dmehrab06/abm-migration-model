import pandas as pd
#import fireducks.pandas as pd
import sys
import resource
import datetime
import gc
import os
import random
import json
from file_paths_and_consts import *
from utils import *
from tpb_single import *
from tpb_parallel import *
import warnings
import argparse
from parameter_parse import parse_args
import logging
from InputClass import  AgentConflictInputs, AttitudePBCInputs, NetworkInputs, TimingCheckPoints, InputSizeInfo, FinalActionInputs
warnings.filterwarnings('ignore')
##################################


if __name__ == "__main__":
    mp.set_start_method('forkserver')
    warnings.filterwarnings('ignore')

    #random.seed(time.time())
    #random.seed(time.time())
    random.seed(42)
    np.random.seed(42)
    start_time = time.time()
    
    args = parse_args()
    
    '''initialize social theory flags'''
    ATT_FLAG = args.attitude ##only used twice
    PBC_FLAG = args.perceived_behavior ##only used twice
    SN_FLAG = args.subjective_norm ##only used twice
    
    '''initialize variables unrelated with models'''
    verbose = args.verbose
    
    '''initialize non-parameteric variables of models'''
    EPS = args.eps
    
    CONFLICT_DATA_PREFIX = args.conflict_data_prefix
    USE_NEIGHBOR = args.conflict_neighbor_buffer
    conflict_file_name = f'{IMPACT_DIR}{CONFLICT_DATA_PREFIX}_buffer_{USE_NEIGHBOR}_km.csv'
    
    HOUSEHOLD_DATA_PREFIX = args.household_data_prefix
    household_file_name = f'{HOUSEHOLD_DIR}{HOUSEHOLD_DATA_PREFIX}.pq'
    
    '''load conflict_data, agent_data'''
    total_impact_data = pd.read_csv(conflict_file_name)
    total_impact_data['time'] = pd.to_datetime(total_impact_data['time'])
    total_household_data = pd.read_parquet(household_file_name)
    total_household_data = total_household_data.rename(columns={'latitude':'h_lat','longitude':'h_lng'})
    
    NEIGHBOR_DATA_PREFIX = args.graph_data_prefix
    NETWORK_TYPE = args.graph_params
    START_DATE = args.simulation_start
    END_DATE = args.simulation_end

    #############################2
    '''initialize parameteric variables of models'''
    
    MOVE_PROB = [args.elderly_beta[0],args.child_beta[0],args.adult_male_beta[0],args.adult_female_beta[0]]
    FAMILY_PROB = [args.elderly_beta[1],args.child_beta[1],args.adult_male_beta[1],args.adult_female_beta[1]]
    
    ### somewhat hacky variable parameters to make life easier ###
    PBC_SCALE_BEFORE_INTENTION = args.scale_before_sigmoid
    T_WINDOW_LEFT = args.time_lag
    T_WINDOW_RIGHT = T_WINDOW_LEFT
    PROB_SCALAR = args.move_scale
    EVENT_WEIGHT_SCALAR = args.event_scale ##only used twice
    PARAM_THRESH_LO = args.param_threshold_lo ##only used twice
    REFUGEE_RATIO = args.refugee_among_displaced ##only used twice
    PHASE_SHIFT = args.phase_shift ##only used twice
    LO_SCALE = args.scale_before_phase_shift ##only used twice
    MID_SCALE = args.scale_after_phase_shift ##only used twice
    MAX_PEER_IT = args.peer_influence_steps_per_sim ##only used twice
    NEIGHBOR_KNOWLEDGE = args.observe_neighbor_intention ##only used twice
    
    ########--highly vairable parameters and arguments passed externally--########
    PARENT_REGION = args.region_name
    SIMULATION_INDEX = str(args.simulation_index).zfill(9)
    PARAM_DELTA = args.param_distance_decay ##only used twice
    PARAM_Q = args.param_migration_bias ## only used twice
    PARAM_v = args.param_risk_growth_rate ## only used 
    PARAM_theta = args.param_discount_rate
    PARAM_THRESH_HI = args.param_threshold_hi
    PARAM_LAMBDA_1 = args.param_lambda1
    PARAM_LAMBDA_2 = args.param_lambda2
    
    ## processing spec parameters ##
    STRUCT = args.S2_LEVEL
    USE_CORE = args.cpu_core_usage
    RAION_GROUP_FILE = args.group_region_file
    
    TRAIN_MODE = args.mode if 'mode' in args else 'no-rerun-infer'
    
    logging.basicConfig(
        filename = f'{args.log_file}/sim_{SIMULATION_INDEX}_{PARENT_REGION}.log',
        level = logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s: %(asctime)s [%(filename)s:%(lineno)s - %(funcName)s() ] >>> %(message)s",
        datefmt="%Y-%m-%d--%H:%M:%SZ",
    )
    logger = logging.getLogger(__name__)
    
    logger.info('Log Starts here')
    logger.info("Agent data and conflict Data loaded")
    logger.info('Parsed all parameters and arguments')
    logger.debug(f'Dump of all parameters {args}')
    ########--parameters and arguments passed externally ends--########
    
    with open(RAION_GROUP_FILE) as f_in:
        REGION_BIN_DICT = json.load(f_in)    
    CURRENT_BIN = [PARENT_REGION]+REGION_BIN_DICT[PARENT_REGION]
    
    logger.debug(f'List of regions to process {CURRENT_BIN}')
    
    for i in range(len(MOVE_PROB)):
        MOVE_PROB[i] = MOVE_PROB[i]*PROB_SCALAR
        FAMILY_PROB[i] = FAMILY_PROB[i]*PROB_SCALAR
    
    resource_log_history = []
    
    resource_log_file = f'resource_usage_information.csv'
    output_dir_agg = f'{OUTPUT_DIR}/forward_Migration/Agg-Result-Sim-{SIMULATION_INDEX}' ## should be created when config file is generated
    output_dir_detail = f'{OUTPUT_DIR}/forward_Migration/Detail-Result-Sim-{SIMULATION_INDEX}' ## should be created when config file is generated
    log_dir_detail = f'{OUTPUT_DIR}/forward_Migration/Other-Log-Sim-{SIMULATION_INDEX}'
    temp_output_dir = f'{TEMPORARY_DIR}/forward_Migration/Simulation-{SIMULATION_INDEX}'
    ## DONE UPTO THIS PART ##
    
    '''perform region specific action from here'''
    
    for REGION_NAME in CURRENT_BIN:
        logger.info(f'processing {REGION_NAME} region')
        
        daily_agg_refugee_file = f'{REGION_NAME}_daily_aggregated_migrant.csv'
        daily_detail_refugee_file = f'{REGION_NAME}_daily_detailed_migrant_info.pq'
        temp_household_file = f'{REGION_NAME}_last_saved_household_state.pq' ## cannot do parquet because of s2Cell info here
        timing_info_file = f'{REGION_NAME}_timing_information.csv'
        
        if TRAIN_MODE=='no-rerun-infer' and os.path.isfile(f'{output_dir_agg}/{daily_agg_refugee_file}'):
            logger.debug(f'{REGION_NAME} already simulated for this configuration, skipping duplicate computation..')
            continue
        
        impact_data = total_impact_data[total_impact_data.matching_place_id==REGION_NAME]
        logger.info(f'this region experienced {impact_data.shape[0]} conflicts in total')
        
        if impact_data.shape[0]==0:
            logger.debug(f'{REGION_NAME} did not experience any events to trigger migration, skipping computation..')
            continue
        
        impact_data['event_weight'] = impact_data.apply(lambda x: get_event_weight(x['event_type'],x['sub_event_type']),axis=1)
        
        graph_file_name = f'{HOUSEHOLD_DIR}{NEIGHBOR_DATA_PREFIX}_{REGION_NAME}_{NETWORK_TYPE}.pq'
        logger.debug(f'trying to load {graph_file_name} as network data')
        s2_graph_file_name = f'{HOUSEHOLD_DIR}ukraine_neighbor_{REGION_NAME}_{STRUCT}_s2.csv'
        if os.path.isfile(graph_file_name):
            logger.debug(f'KSW network loaded as parquet file')
            neighbor_household_data = pd.read_parquet(graph_file_name)
        else:
            logger.debug(f'KSW network not found')
            neighbor_household_data = pd.read_csv(s2_graph_file_name,usecols=['hid_x','hid_y'])

        logger.info(f'network data loaded for {REGION_NAME}')

        '''preprocess data'''
        neighbor_cnts = neighbor_household_data['hid_x'].value_counts().reset_index().rename(columns={'hid_x':'hid','count':'N_size'})

        cur_household_data = total_household_data[total_household_data.matching_place_id==REGION_NAME]
        cur_household_data['s2_cell'] = cur_household_data.apply(lambda x: getl13(x['h_lat'],x['h_lng'],STRUCT),axis=1)
        cur_household_data = cur_household_data.merge(neighbor_cnts,on='hid',how='inner')     

        '''additional preprocessing for multicore execution'''
        if USE_CORE>1:
            neighbor_household_data['core_id'] = neighbor_household_data['s2_id']%USE_CORE
            neighbor_household_data = neighbor_household_data.drop(columns=['s2_id'])
            gb = neighbor_household_data.groupby('core_id')
            neighbor_chunks = [gb.get_group(x) for x in gb.groups]
            cur_household_data['core_id'] = cur_household_data['s2_cell'].apply(lambda x: get_core_id(x,USE_CORE))
            del neighbor_household_data
            gc.collect()
            logger.info("Nodes have been partitioned for multi-core processing")
        
        cur_household_data = cur_household_data.drop(columns=['s2_cell']) ## to enable parquet file load/save
        #print('data loaded until garbage collector',flush=True)

        '''initialize simulation attributes'''
        cur_household_data['hh_size'] = cur_household_data[DEMO_TYPES].sum(axis = 1, skipna = True)
        cur_household_data['P(move|violence)'] = cur_household_data.apply(lambda x: get_move_prob([x['OLD_PERSON'],x['CHILD'],x['ADULT_MALE'],x['ADULT_FEMALE']],FAMILY_PROB,MOVE_PROB),axis=1)
        cur_household_data['prob_conflict'] = 0
        cur_household_data['moves'] = 0
        cur_household_data['move_type'] = 0 # 0 means did not move, 1 means IDP, 2 means outside
        f = 0
        region_start_time = time.time()
        cur_checkpoint = 1000
        prev_temp_checkpoint = 0
        last_saved_checkpoint = -1

        '''initialize simulation tables'''
        DEL_COLUMNS_1 = ['P(violence)','random']
        DEL_COLUMNS_2 = ['P(move)', 'moves_2','m_state','m_state_2']
        T_CURRENT = pd.to_datetime(START_DATE)
        T_FINAL = pd.to_datetime(END_DATE)

        migrant_history = []
        timing_history = []

        hid_displacement_df = []
        logger.info('Simulation_starting')


        #########################################5
        for simtime in range(0,300):

            logger.info(f'Simulation for {T_CURRENT}:')
            preprocess_start = time.time()
            prev_temp_checkpoint = prev_temp_checkpoint + 1

            C_TIME_BEGIN = T_CURRENT - pd.DateOffset(days=T_WINDOW_LEFT)
            C_TIME_END = T_CURRENT - pd.DateOffset(days=T_WINDOW_RIGHT)

            '''check if simulation limit is reached'''
            if(f==1 and T_CURRENT > T_FINAL):
                break

            '''load agents who did not migrate from last simulation'''
            if(f!=0):
                logger.debug(f'trying to read intermediate state as parquet file')
                cur_household_data = pd.read_parquet(f'{temp_output_dir}/{temp_household_file}')
                cur_household_data = cur_household_data[cur_household_data.moves==0]
                if USE_CORE==1:
                    trim_neighborhood(cur_household_data['hid'],USE_CORE,neighbor_household_data)
                else:
                    trim_neighborhood(cur_household_data['hid'],USE_CORE,neighbor_chunks)

            '''check boundary conditions
            no household remaining'''
            if(cur_household_data.shape[0]<2):
                logger.debug(f'only {cur_household_data.shape[0]} agents remaining in {REGION_NAME}, possibly skipping further computation..')
                refugee_info = {'total':0,'old_people':0,'child':0,'male':0,'female':0}
                idp_info = {'total':0,'old_people':0,'child':0,'male':0,'female':0}
                agg_info_ref = {x+'_refugee':refugee_info[x] for x in refugee_info}
                agg_info_idp = {x+'_idp':idp_info[x] for x in idp_info}
                agg_info_ref.update(agg_info_idp)
                T_CURRENT = T_CURRENT + pd.DateOffset(days=1)
                migrant_history.append(agg_info_ref)
                continue

            cur_impact_data = impact_data[(impact_data.time>=C_TIME_BEGIN) & (impact_data.time<=C_TIME_END)]
            logger.info(f'{REGION_NAME} faced {cur_impact_data.shape[0]} events during current timeframe')

            preprocess_end = time.time()
            logger.info(f'{preprocess_end-preprocess_start} second to preprocess')

            step_start = time.time()
            cur_net = neighbor_household_data if USE_CORE==1 else neighbor_chunks
            mps = AgentConflictInputs(cc_data = cur_impact_data, hh_data=cur_household_data, neighbor_data=cur_net, cur_t=T_CURRENT, 
                                              simtime=simtime, flag_a=ATT_FLAG,flag_p=PBC_FLAG, flag_s=SN_FLAG, del_col1 = DEL_COLUMNS_1, 
                                              del_col2 = DEL_COLUMNS_2, use_core = USE_CORE, ct_start = C_TIME_BEGIN, ct_end = C_TIME_END)
            rps = AttitudePBCInputs(e = EPS, es = EVENT_WEIGHT_SCALAR, theta = PARAM_theta, delta = PARAM_DELTA, 
                                            Q = PARAM_Q, V = PARAM_v, pbc_agent = PBC_SCALE_BEFORE_INTENTION)
            nps = NetworkInputs(lambda_1 = PARAM_LAMBDA_1, lambda_2 = PARAM_LAMBDA_2, tau_lo = PARAM_THRESH_LO, tau_hi = PARAM_THRESH_HI, 
                                       thresh_steps = MAX_PEER_IT, knows_neighbor = NEIGHBOR_KNOWLEDGE, phase_shift_day = PHASE_SHIFT)
            aps = FinalActionInputs(refugee_ratio = REFUGEE_RATIO, lo = LO_SCALE, hi = MID_SCALE)

            ## main magic in this call
            logger.info('Calling Theory of Planned Behavior Modules')
            temp_households,t_ckpt,input_info = step_parallel(mps,rps,nps,aps) if USE_CORE>1 else step_single(mps,rps,nps,aps)

            step_end = time.time()
            logger.info(f'{step_end-step_start} second to finish episode')

            '''Sanity Check that failed previously at first episode'''
            leaving = temp_households[temp_households.moves==1]['hh_size'].sum()
            idp = temp_households[temp_households.move_type==1]['hh_size'].sum()
            refugee = temp_households[temp_households.move_type==2]['hh_size'].sum()
            logger.debug(f'In this step, a) total leaving: {leaving}, b) refugee: {refugee}, c) idp: {idp}')
            status = 'OK' if (idp+refugee==leaving) else 'something wrong'
            logger.debug(f'Status {status}')

            '''generate temps results from this simulation'''
            refugee_df = temp_households[temp_households.move_type==2]
            idp_df = temp_households[temp_households.move_type==1]
            refugee_info = {'total':refugee_df['hh_size'].sum(),'old_people':refugee_df['OLD_PERSON'].sum(),'child':refugee_df['CHILD'].sum(),
                           'male':refugee_df['ADULT_MALE'].sum(),'female':refugee_df['ADULT_FEMALE'].sum()}
            idp_info = {'total':idp_df['hh_size'].sum(),'old_people':idp_df['OLD_PERSON'].sum(),'child':idp_df['CHILD'].sum(),
                           'male':idp_df['ADULT_MALE'].sum(),'female':idp_df['ADULT_FEMALE'].sum()}
            agg_info_ref = {x+'_refugee':refugee_info[x] for x in refugee_info}
            agg_info_idp = {x+'_idp':idp_info[x] for x in idp_info}
            agg_info_ref.update(agg_info_idp)
            agg_info_ref['date'] = str(T_CURRENT)

            migrant_history.append(agg_info_ref)

            temp_households['move_date'] = str(T_CURRENT)
            hid_displacement_df.append(temp_households[temp_households.move_type!=0])
            temp_households = temp_households.drop(columns=['move_date'])

            logger.debug(f'Before saving last household state it has columns {temp_households.columns.tolist()}')
            logger.debug(f'trying to save intermediate state as parquet file')
            temp_households.to_parquet(f'{temp_output_dir}/{temp_household_file}',index=False)
            ## CHecked upto here, need to return other things from episode function
            last_saved_checkpoint = prev_temp_checkpoint
            T_CURRENT = T_CURRENT + pd.DateOffset(days=1)
            f = 1

            timing_row = {'step':simtime,'h_size':input_info.num_households,'a_size':input_info.num_agents,'c_size':input_info.num_events,
                          'G_edges':input_info.num_edges_network, 'ATT_time':t_ckpt.att_end-t_ckpt.att_start,
                          'PBC_time':t_ckpt.pcb_end-t_ckpt.pcb_start,'SN_time':t_ckpt.sn_end-t_ckpt.sn_start}
            timing_history.append(timing_row)

        ##################################6
        hid_all_displacement_df = pd.concat(hid_displacement_df)
        logger.debug('Attempting to save detailed migrant state output as parquet file')
        hid_all_displacement_df.to_parquet(f'{output_dir_detail}/{daily_detail_refugee_file}',index=False) ##cannot be done because of CellID column, need to change to csv or remove celliD column

        simulated_migrant_df = pd.DataFrame.from_dict(migrant_history)
        simulated_migrant_df.to_csv(f'{output_dir_agg}/{daily_agg_refugee_file}',index=False)

        region_runtime = (time.time()-region_start_time)/60.0

        timing_log_df = pd.DataFrame.from_dict(timing_history)
        timing_log_df.to_csv(f'{log_dir_detail}/{timing_info_file}',index=False)

        logger.info(f'simulation for {REGION_NAME} ended in {region_runtime} minutes')
        log_data = {'raion': REGION_NAME,'T':region_runtime,'Sindex':SIMULATION_INDEX,'M':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}
        resource_log_history.append(log_data)
        
    logger.info(f'Simulation finished in {(time.time()-start_time)/60.0} minutes')
    resource_log_df = pd.DataFrame.from_dict(resource_log_history)
    resource_log_df.to_csv(f'{log_dir_detail}/{resource_log_file}',index=False)
        