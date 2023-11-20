############################1
import pandas as pd
import numpy as np
import sys
import random
import time
import warnings
from file_paths_and_consts import *
import math
import s2sphere

warnings.filterwarnings('ignore')

random.seed(time.time())

start_time = time.time()

############################# parameters and other hyperparameters
APPLY_PEER = 1 ## SHOULD NOT CHANGE THIS
EPS = 0.0001 ## SHOULD NOT CHANGE THIS

PLACE_NAME = sys.argv[1] ## RAION_NAME
hyper_comb = int(sys.argv[2]) ## SIMULATION NO, should not be more than 5 digits

DIS_EXPONENT = float(sys.argv[3]) #\delta parameter
A = float(sys.argv[4]) #Q parameter
T = float(sys.argv[5]) #v parameter
S = float(sys.argv[6]) #98.67 #\theta parameter
lookbefore_days_left = int(sys.argv[7]) #time window 1
lookbefore_days_right = int(sys.argv[8]) #time window 2

PROB_SCALAR = float(sys.argv[9]) #bias scaling parameter
EVENT_WEIGHT_SCALAR = float(sys.argv[10]) #event weight scaling parameter
USE_PEER_EFFECT = int(sys.argv[11]) ## from generator, keep default
USE_CIVIL_DATA = 0
THRESH_HI = int(sys.argv[12]) #threshold parameter
USE_NEIGHBOR = int(sys.argv[13]) #from generator, keep default
BORDER_CROSS_PROB = float(sys.argv[14]) #from generator, keep default

ablation_conflict_type = 'None'
print(ablation_conflict_type)

#MOVE_PROB = [0.15,0.5,0.02,0.2]
#FAMILY_PROB = [0.2,0.7,0.05,0.7]

MOVE_PROB = [0.25,0.7,0.02,0.7]
FAMILY_PROB = [0.25,0.85,0.1,0.85]

for i in range(len(MOVE_PROB)):
    MOVE_PROB[i] = MOVE_PROB[i]*PROB_SCALAR
    FAMILY_PROB[i] = FAMILY_PROB[i]*PROB_SCALAR

################################## helper functions #####################################
def haversine(lon1, lat1, lon2, lat2):
    KM = 6372.8 #Radius of earth in km instead of miles
    lat1, lon1, lat2, lon2 = map(np.deg2rad, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    total_km = KM * c
    return total_km

## rule about how each agent is affected by each impact
def prob_conflict(impact,dis,t_diff=0):
    #print(fatality)
    #print(dis)
    return ((impact)/(dis**DIS_EXPONENT))*(1.0/(1+t_diff))

## rule about how an agent is affected overall by all impact   (P(violence))
def aggregated_prob_conflict(x):  
    return 1 / (1 + A*math.exp(-T*x))

## theta for previous fear values
def memory_decay(x):
    return x*(S/100.00)

## rule about how different types of demographic group decides to move
def get_move_prob(DEMO_NUMS): ##this function can be played with
    tot_size = 0
    for v in DEMO_NUMS:
        tot_size = tot_size+v
    if tot_size>1:
        move_prob = 0.0
        for i in range(0,len(DEMO_NUMS)):
            move_prob = move_prob + DEMO_NUMS[i]*FAMILY_PROB[i]
        return move_prob/tot_size
    else:
        move_prob = 0.0
        for i in range(0,len(DEMO_NUMS)):
            move_prob = move_prob + DEMO_NUMS[i]*MOVE_PROB[i]
        return move_prob/tot_size

def bernoulli(val,p):
    if (val<=p):
        return 1
    else:
        return 0
        
def bernoulli_border(val,moves):
    if moves==0:
        return 0
    else:
        if val<=BORDER_CROSS_PROB:
            return 2
        else:
            return 1
    
def getl13(lat,lng,req_level=13):
    p = s2sphere.LatLng.from_degrees(lat, lng) 
    cell = s2sphere.Cell.from_lat_lng(p)
    cellid = cell.id()
    for i in range(1,30):
        #print(cellid)
        if cellid.level()==req_level:
            return cellid
        cellid = cellid.parent()

        
## thresh_type==0
## if my current decision is to migrate (1), if less than thresh_lo of my neighbors are migrating, i will change my decision to (0)

## thresh_type==1
## if my current decision is not to migrate(0), if at least thresh_hi of my neigbhors are migrating, i will change my decision to (1)

def apply_threshold_function(cur_decision,cur_neighbor_migrating,thresh_lo=1,thresh_hi=THRESH_HI):
        if cur_decision==1:
            if cur_neighbor_migrating<thresh_lo:
                return 0
        if cur_decision==0:
            if cur_neighbor_migrating>=thresh_hi:
                return 1
        return cur_decision
    
def refine_through_peer_effect(temp_household):
    new_temp_households = temp_household.sort_values(by='hid')
    new_temp_households['s2_cell'] = new_temp_households.apply(lambda x: getl13(x['h_lat'],x['h_lng'],13),axis=1)
    h_decision = new_temp_households[['hid','s2_cell','moves']]

    h_network = h_decision.merge(h_decision,on='s2_cell',how='inner')
    #h_network = h_network[h_network.hid_x!=h_network.hid_y]
    h_network_peer_move_count = (h_network.groupby('hid_x')['moves_y'].sum().reset_index()).merge(h_network[['hid_x','moves_x']].drop_duplicates(),on='hid_x',how='inner')
    h_network_peer_move_count['moves_y'] = h_network_peer_move_count['moves_y'] - h_network_peer_move_count['moves_x']
    h_network_peer_move_count['peer_affected_move'] = h_network_peer_move_count.apply(lambda x: apply_threshold_function(x['moves_x'],x['moves_y']),axis=1)
    h_network_peer_move_count = h_network_peer_move_count.sort_values(by='hid_x')

    #print(h_network_peer_move_count[h_network_peer_move_count.moves_x==1].shape,h_network_peer_move_count[h_network_peer_move_count.peer_affected_move==1].shape)
    
    new_temp_households['moves'] = h_network_peer_move_count['peer_affected_move']
    return new_temp_households

def get_event_weight(event_type,sub_event_type):
    if sub_event_type==ablation_conflict_type:
        return 0
    if event_type=="Battles":
        return 3
    if event_type.startswith('Civilian'):
        return 8
    if event_type.startswith('Explosions'):
        return 5
    if event_type.startswith('Violence'):
        return 3
    if event_type.startswith('Protests') or event_type.startswith('Riots'):
        return 0
    return 0

#####################################4
neighbor_data = pd.read_csv('neighbor_raions.csv')
neighbor_list = neighbor_data[neighbor_data.ADM2_EN_x==PLACE_NAME]['ADM2_EN_y'].unique().tolist()

if USE_NEIGHBOR==0:
    impact_data = pd.read_csv(IMPACT_DIR+'ukraine_conflict_data_ADM2_HDX.csv')
else:
    impact_data = pd.read_csv(IMPACT_DIR+'ukraine_conflict_data_ADM2_HDX_buffer_'+str(USE_NEIGHBOR)+'_km.csv')

impact_data['time'] = pd.to_datetime(impact_data['time'])

if USE_CIVIL_DATA==1:
    civilian_data = pd.read_csv(IMPACT_DIR+'ukraine_civilian_conflict_data_ADM2_HDX.csv')
    civilian_data['time'] = pd.to_datetime(civilian_data['time'])

    impact_data = pd.concat([impact_data,civilian_data])

if USE_NEIGHBOR==1:
    impact_data = impact_data[impact_data.matching_place_id.isin(neighbor_list)]
    impact_data = impact_data.drop(columns=['matching_place_id'])
    
cur_household_data = pd.read_csv(HOUSEHOLD_DIR+'ukraine_household_data_ADM2_HDX.csv')
refugee_data = pd.read_csv('ukraine_refugee_data_2.csv')
refugee_data['time'] = pd.to_datetime(refugee_data['time'])


impact_data['event_weight'] = impact_data.apply(lambda x: get_event_weight(x['event_type'],x['sub_event_type']),axis=1)
print('impact events reassigned')

print('total size',cur_household_data.shape)
cur_household_data = cur_household_data[cur_household_data.matching_place_id==PLACE_NAME]
cur_household_data['hh_size'] = cur_household_data[DEMO_TYPES].sum(axis = 1, skipna = True)
cur_household_data['P(move|violence)'] = cur_household_data.apply(lambda x: get_move_prob([x['OLD_PERSON'],x['CHILD'],x['ADULT_MALE'],x['ADULT_FEMALE']]),axis=1)
cur_household_data['prob_conflict'] = 0
cur_household_data['moves'] = 0

if 'h_lat' not in cur_household_data.columns.tolist():
    cur_household_data = cur_household_data.rename(columns={'latitude':'h_lat','longitude':'h_lng'})

print('household properties updated')
print('this oblast size',cur_household_data.shape)



min_date = pd.to_datetime('2022-03-01')
end_date = pd.to_datetime('2022-05-15')
simulated_refugee_df = pd.DataFrame(columns=['id','time','refugee','old_people','child','male','female'])

temp_prefix = ''
if ablation_conflict_type!='None':
    OUTPUT_DIR = ABLATION_DIR
    temp_prefix = 'ablation_'
    
print(OUTPUT_DIR)
print(temp_prefix)

f = 0

start = time.time()
cur_checkpoint = 1000
print('combination_no',hyper_comb)

prev_temp_checkpoint = 0
last_saved_checkpoint = -1

peer_used = 0

DEL_COLUMNS = ['P(violence)','P(move)','random']

who_went_where = []
hid_displacement_df = []

#########################################5
for i in range(0,300):
    prev_temp_checkpoint = prev_temp_checkpoint + 1
    
    max_date = min_date + pd.DateOffset(days=1)
    
    lookahead_date_1 = min_date - pd.DateOffset(days=lookbefore_days_left)
    lookahead_date_2 = min_date - pd.DateOffset(days=lookbefore_days_right)
    
    if(f==0 and (min_date not in refugee_data['time'].tolist())):
        min_date = max_date
        continue
    
    if(f==1 and min_date > end_date):
        break
    ##################
    print(min_date)
    if(f!=0):
        cur_household_data = pd.read_csv(TEMPORARY_DIR+'last_saved_household_data_'+str(temp_prefix)+str(PLACE_NAME)+'_'+str(hyper_comb)+'.csv')
        #print(cur_household_data.info())
        #break
        cur_household_data = cur_household_data[cur_household_data.moves==0]
        
    ##################
    
    #print('number of households',cur_household_data.shape)
    
    #print('persons remaining',person_df.shape)
    cur_impact_data = impact_data[(impact_data.time>=lookahead_date_1) & (impact_data.time<=lookahead_date_2)]
    #cur_impact_data = cur_conflict_data[cur_conflict_columns]
    
    cur_impact_data = cur_impact_data.rename(columns={'latitude':'impact_lat','longitude':'impact_lng'})
    cur_impact_data['cur_time'] = min_date
    cur_impact_data['time_diff_to_event'] = (cur_impact_data['cur_time'] - cur_impact_data['time']) / np.timedelta64(1,'D')
    cur_impact_data['impact_intensity'] = cur_impact_data['event_weight']*cur_impact_data['event_intensity']*EVENT_WEIGHT_SCALAR
    cur_impact_data['impact_intensity'].replace(to_replace = 0, value = EPS, inplace=True)
    
    if(cur_impact_data.shape[0]==0):
        new_row = {'id':PLACE_NAME,'time':min_date,'refugee':0,'old_people':0,'child':0,'male':0,'female':0}
        min_date = max_date
        simulated_refugee_df = simulated_refugee_df.append(new_row,ignore_index=True)
        continue
    
    if USE_NEIGHBOR==1:
        cur_household_data['key'] = 1
        cur_impact_data['key'] = 1
        impact_in_homes = cur_impact_data.merge(cur_household_data,on='key',how='inner')
    else:
        impact_in_homes = cur_impact_data.merge(cur_household_data,on='matching_place_id',how='inner')
    
    impacted_household_list = impact_in_homes.hid.unique().tolist()
    
    household_not_impacted = cur_household_data[~cur_household_data.hid.isin(impacted_household_list)]
    
    #print('number of impacted households',len(impact_in_homes.hid.unique().tolist()))
    #print('number of non impacted households',household_not_impacted.shape)
    
    if impact_in_homes.shape[0]==0:
        new_row = {'id':PLACE_NAME,'time':min_date,'refugee':0,'old_people':0,'child':0,'male':0,'female':0}
        min_date = max_date
        simulated_refugee_df = simulated_refugee_df.append(new_row,ignore_index=True)
        continue
    #print(ukraine_conflict_in_homes['fatalities'].describe())
    impact_in_homes['dis_conflict_home'] = haversine(impact_in_homes['h_lng'],impact_in_homes['h_lat'],impact_in_homes['impact_lng'],impact_in_homes['impact_lat'])
    impact_in_homes['prob_conflict'] = impact_in_homes['prob_conflict'].apply(lambda x: memory_decay(x))
    impact_in_homes['prob_conflict'] = impact_in_homes['prob_conflict'] + impact_in_homes.apply(lambda x: prob_conflict(x['impact_intensity'],x['dis_conflict_home'],x['time_diff_to_event']),axis=1)
    
    cur_household_data = cur_household_data.drop(columns='prob_conflict')
    home_conflict_df = impact_in_homes.groupby(['hid'])['prob_conflict'].sum().reset_index()
    home_conflict_df['P(violence)'] = home_conflict_df.prob_conflict.apply(aggregated_prob_conflict)
    home_conflict_df = home_conflict_df.merge(cur_household_data,on='hid',how='inner')
    #print('number of households with p(violence)',len(home_conflict_df.hid.unique().tolist()),home_conflict_df.shape)
    
    home_conflict_df['P(move)'] = home_conflict_df['P(violence)']*home_conflict_df['P(move|violence)']
    home_conflict_df['random'] = np.random.random(home_conflict_df.shape[0])
    home_conflict_df['moves'] = home_conflict_df.apply(lambda x: bernoulli(x['random'],x['P(move)']),axis=1)
    
    curtime = time.time()
    if((curtime-start)>=cur_checkpoint):
        print('checkpoint for ',str(PLACE_NAME))
        simulated_refugee_df.to_csv(OUTPUT_DIR+'mim_result_'+str(PLACE_NAME)+'_'+str(hyper_comb).zfill(5)+'.csv',index=False)
        start = curtime
    
    
    home_conflict_df = home_conflict_df.drop(columns=DEL_COLUMNS)
    
    #print('cur_save_household_size',home_conflict_df.shape)
    #print('saving columns',home_conflict_df.columns.tolist())
    print('###########################')
    
    temp_households = pd.concat([home_conflict_df,household_not_impacted])
    
    if APPLY_PEER==1 and peer_used<USE_PEER_EFFECT:
        temp_households = refine_through_peer_effect(temp_households)
        peer_used = peer_used + 1
    
    temp_households['move_type_random'] = np.random.random(temp_households.shape[0])
    temp_households['move_type'] = temp_households.apply(lambda x: bernoulli_border(x['move_type_random'],x['moves']),axis=1)
    temp_households = temp_households.drop(columns=['move_type_random'])
    
    new_row = {'id':PLACE_NAME,'time':min_date,'refugee':temp_households[temp_households.move_type==2]['hh_size'].sum(),
               'old_people':temp_households[temp_households.move_type==2]['OLD_PERSON'].sum(),'child':temp_households[temp_households.move_type==2]['CHILD'].sum(),
                'male':temp_households[temp_households.move_type==2]['ADULT_MALE'].sum(),'female':temp_households[temp_households.move_type==2]['ADULT_FEMALE'].sum()}
    simulated_refugee_df = simulated_refugee_df.append(new_row,ignore_index=True)
    
    temp_households['move_date'] = str(min_date)
    hid_displacement_df.append(temp_households[temp_households.move_type!=0])
    temp_households = temp_households.drop(columns=['move_date'])
    
    temp_households.to_csv(TEMPORARY_DIR+'last_saved_household_data_'+str(temp_prefix)+str(PLACE_NAME)+'_'+str(hyper_comb)+'.csv',index=False)
    
    print('intention module done')
    
    ##apply destination module from here
    ## this is for testing purpose, this module is not complete and not part of the submitted paper and it does not affect the result any way
    
    moving_households_df = temp_households[temp_households.moves==1].reset_index()
    #print(moving_households_df.shape, 'people moved',end=' ')
    dest_prob_distribution_df = pd.read_csv(OUTPUT_DIR+'destination_probability_distribution.csv')
    
    if(str(min_date) not in dest_prob_distribution_df.columns.tolist()):
        #print(str(min_date))
        cur_dest_prob_df = dest_prob_distribution_df[['ISO','2022-03-01 00:00:00']]
        weight_col = '2022-03-01 00:00:00'
        #print(cur_dest_prob_df.shape)
    else:
        #print(str(min_date))
        cur_dest_prob_df = dest_prob_distribution_df[['ISO',str(min_date)]]
        weight_col = str(min_date)
    #print('projected')
    
    dest_sampled = cur_dest_prob_df.sample(moving_households_df.shape[0],replace=True,weights=cur_dest_prob_df[weight_col]).reset_index()
    #print(dest_sampled)
    #print(dest_sampled.shape,'sampled')
    #print(dest_sampled.ISO.value_counts())
    
    #break
    moving_households_df['destination'] = dest_sampled['ISO']
    moving_households_df['moved_date'] = str(min_date)
    
    who_went_where.append(moving_households_df)
    
    #print(moving_households_df['destination'].value_counts())
    
    ##destination  module done
    
    print('destination module done')
    ### irrelevnat part ends here------ ####
    last_saved_checkpoint = prev_temp_checkpoint
    min_date = max_date
    f = 1

##################################saving files########
simulated_refugee_with_dest_df = pd.concat(who_went_where)
hid_all_displacement_df = pd.concat(hid_displacement_df)

simulated_refugee_with_dest_df.to_csv(OUTPUT_DIR+'mdm_result_completed_'+str(PLACE_NAME)+"_"+str(hyper_comb).zfill(5)+'.csv',index=False)
simulated_refugee_df.to_csv(OUTPUT_DIR+'mim_result_completed_'+str(PLACE_NAME)+'_'+str(hyper_comb).zfill(5)+'.csv',index=False)
hid_all_displacement_df.to_csv(OUTPUT_DIR+'mim_hid_completed_'+str(PLACE_NAME)+'_'+str(hyper_comb).zfill(5)+'.csv',index=False)

end = time.time()
data = {'raion': [PLACE_NAME],'runtime':[end-start_time],'hyper_comb':[hyper_comb]}
run_df = pd.DataFrame(data)
 
# append data frame to CSV file
run_df.to_csv('runtime_log/runtime_raion.csv', mode='a', index=False, header=False)
