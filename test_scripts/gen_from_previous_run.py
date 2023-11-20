all_ids = ['Kalmiuskyi', 'Synelnykivskyi', 'Fastivskyi', 'Khustskyi', 'Berezivskyi', 'Beryslavskyi', 'Dubenskyi', 'Novomoskovskyi', 'Shepetivskyi', 'Chernihivskyi', 'Chortkivskyi', 'Cherkaskyi', 'Melitopolskyi', 'Poltavskyi', 'Zaporizkyi', 'Volnovaskyi', 'Nizhynskyi', 'Kremenchutskyi', 'Prylutskyi', 'Volodymyrskyi', 'Henicheskyi', 'Starobilskyi', 'Kovelskyi', 'Sievierodonetskyi', 'Kropyvnytskyi', 'Kamianskyi', 'Khersonskyi', 'Zviahelskyi', 'Chernivetskyi', 'Khmelnytskyi', 'Bilotserkivskyi', 'Yavorivskyi', 'Bolhradskyi', 'Polohivskyi', 'Odeskyi', 'Sumskyi', 'Berdianskyi', 'Sarnenskyi', 'Varaskyi', 'Vyzhnytskyi', 'Rakhivskyi', 'Zhmerynskyi', 'Dovzhanskyi', 'Kamin-Kashyrskyi', 'Podilskyi', 'Haisynskyi', 'Skadovskyi', 'Rivnenskyi', 'Sambirskyi', 'Kupianskyi', 'Mariupolskyi', 'Kryvorizkyi', 'Bashtanskyi', 'Myrhorodskyi', 'Bakhchysaraiskyi', 'Voznesenskyi', 'Konotopskyi', 'Kamianets-Podilskyi', 'Chervonohradskyi', 'Lubenskyi', 'Izmailskyi', 'Korostenskyi', 'Rozdilnianskyi', 'Tulchynskyi', 'Vyshhorodskyi', 'Lozivskyi', 'Kramatorskyi', 'Lutskyi', 'Ternopilskyi', 'Koriukivskyi', 'Mykolaivskyi', 'Romenskyi', 'Svativskyi', 'Novoukrainskyi', 'Bilohirskyi', 'Khmilnytskyi', 'Umanskyi', 'Alchevskyi', 'Dniprovskyi', 'Zhytomyrskyi', 'Krasnoperekopskyi', 'Kolomyiskyi', 'Krasnohradskyi', 'Kakhovskyi', 'Berdychivskyi', 'Pervomaiskyi', 'Shchastynskyi', 'Bohodukhivskyi', 'Zvenyhorodskyi', 'Kharkivskyi', 'Vinnytskyi', 'Oleksandriiskyi', 'Uzhhorodskyi', 'Obukhivskyi', 'Zolochivskyi', 'Nikopolskyi', 'Horlivskyi', 'Lvivskyi', 'Donetskyi', 'Ivano-Frankivskyi', 'Kaluskyi', 'Iziumskyi', 'Krasnohvardiiskyi', 'Bilhorod-Dnistrovskyi', 'Pokrovskyi', 'Kremenetskyi', 'Brovarskyi', 'Yevpatoriiskyi', 'Buchanskyi', 'Stryiskyi', 'Novhorod-Siverskyi', 'Zolotoniskyi', 'Pavlohradskyi', 'Tiachivskyi', 'Boryspilskyi', 'Holovanivskyi', 'Drohobytskyi', 'Okhtyrskyi', 'Verkhovynskyi', 'Rovenkivskyi', 'Vasylivskyi', 'Yaltynskyi', 'Shostkynskyi', 'Dnistrovskyi', 'Mukachivskyi', 'Simferopolskyi', 'Mohyliv-Podilskyi', 'Chuhuivskyi', 'Feodosiiskyi', 'Dzhankoiskyi', 'Nadvirnianskyi', 'Bakhmutskyi', 'Berehivskyi', 'Kyiv', 'Luhanskyi', 'Kerchynskyi', 'Kosivskyi', 'Chornobyl Exclusion Zone', 'Sevastopol']

sub_event_type = ['None','Shelling_or_artillery_or_missile_attack', 'Air_or_drone_strike', 'Peaceful_protest', 'Remote_explosive_or_landmine_or_IED', 'Armed_clash', 'Grenade', 'Abduction_or_forced_disappearance', 'Attack', 'Sexual_violence', 'Excessive_force_against_protesters', 'Looting_or_property_destruction', 'Protest_with_intervention', 'Non-state_actor_overtakes_territory', 'Change_to_group_or_activity', 'Disrupted_weapons_use', 'Agreement', 'Government_regains_territory', 'Non-violent_transfer_of_territory', 'Violent_demonstration', 'Mob_violence', 'Arrests', 'Headquarters_or_base_established']

import random
import time
import sys
import pandas as pd

hyper_comb = int(sys.argv[1])

random.seed(time.time())

prev_run_df = pd.read_csv('../runtime_log/parameter_comb.csv')

prev_run_df['hyper_comb'] = prev_run_df['hyper_comb'].astype(int)

prev_run_df = prev_run_df[prev_run_df.hyper_comb==hyper_comb]

D = prev_run_df.iloc[0]['DISTANCE_DECAY']
A = prev_run_df.iloc[0]['SIGMOID_SCALAR']
T = prev_run_df.iloc[0]['SIGMOID_EXPONENT']
S = prev_run_df.iloc[0]['MEMORY_DECAY']
t_l = int(prev_run_df.iloc[0]['TIME_LEFT'])
t_r = int(prev_run_df.iloc[0]['TIME_RIGHT'])
ps = prev_run_df.iloc[0]['BIAS_SCALE']
ews = prev_run_df.iloc[0]['EVENT_WEIGHT_SCALE']
pactive = int(prev_run_df.iloc[0]['PEER_AFFECT_ACTIVE_DAY'])
peer_thresh_hi = int(prev_run_df.iloc[0]['THRESH_HI'])
use_neighbor = int(prev_run_df.iloc[0]['USE_NEIGHBOR'])
border_cross_prob = prev_run_df.iloc[0]['BORDER_CROSS_PROB']

for name in all_ids:
    if name.startswith('Chornobyl'):
        print('sbatch ukr_mim_mdm_sample_no_idp.sbatch '+'"'+name+'"',end=' ')
        print(hyper_comb,D,A,T,S,t_l,t_r,ps,ews,pactive,peer_thresh_hi,use_neighbor,border_cross_prob)
    #print('sbatch ukr_mim_mdm_sample.sbatch',name,hyper_comb+900,D,A,T,S,t_l,t_r,ps,ews,pactive,use_civil_data,'1')
                
