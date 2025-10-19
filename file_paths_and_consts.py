#BASE_DIR = '/gpfs/gpfs0/project/XMode/evaluation/datasets_from_home/migration_data/'
BASE_DIR = '/project/biocomplexity/UKR_forecast/migration_data/'
IMPACT_DIR = BASE_DIR+'conflict_data/'
GROUND_TRUTH_DIR = BASE_DIR+'ground_truth_data/'
HOUSEHOLD_DIR = BASE_DIR+'household_data/'
POPULATION_DIR = BASE_DIR+'population_data/'
TEMPORARY_DIR = BASE_DIR+'temporary_data_2025/'
UNCLEANED_DATA_DIR = BASE_DIR+'raw_data/'
OUTPUT_DIR = BASE_DIR+'output_data_2025/'
ABLATION_DIR = BASE_DIR+'ablation_data_2025/'
CACHE_DIR = BASE_DIR+'cached_data_2025/forward/'


IMPACT_FIELDS = ['event_id','time','latitude','longitude','event_type','sub_event_type','event_weight','event_intensity','matching_place_name','matching_place_id']
DEMO_TYPES = ['OLD_PERSON','CHILD','ADULT_MALE','ADULT_FEMALE']
HOUSEHOLD_FIELDS = ['hid','h_lat','h_lng','matching_place_name','matching_place_id']
GT1_FIELDS = ['time','refugee']
OPP_FEATURES = ['pop_d','gdppc_d','stockOinD','eu_d']