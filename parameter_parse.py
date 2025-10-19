import argparse
import json
import sys

def parse_args():
    # First parse just the --config argument
    config_parser = argparse.ArgumentParser(description="Config loader", add_help=False)
    config_parser.add_argument("--config", type=str, help="Path to config.json", default="config_default.json")
    args, remaining_argv = config_parser.parse_known_args()

    # Default config dict
    config = {}
    if args.config:
        with open(args.config, 'r') as f:
            config = json.load(f)

    # Now define full parser and set defaults from config
    parser = argparse.ArgumentParser(parents=[config_parser])
    parser.set_defaults(**config)  # apply json as defaults

    # miscallenous variables, not main parameters, usually don't change
    parser.add_argument("--attitude", help="enable observation of events")
    parser.add_argument("--perceived_behavior", help="enable agent-specific perception")
    parser.add_argument("--subjective_norm", help="Enable peer influence")
    parser.add_argument("--verbose", action="store_true", help="Enable detailed outputs")
    parser.add_argument("--eps", type=float, help="Default weight of events")
    parser.add_argument("--conflict_data_prefix", type=str, help="filename prefix for conflict data")
    parser.add_argument("--conflict_neighbor_buffer", type=int, choices=[5,10], help="buffer km used for regions during data processing")
    parser.add_argument("--household_data_prefix", type=str, help="filename prefix for agent data")
    parser.add_argument("--graph_data_prefix", type=str, help="graph model for agent network data, currently only KSW model")
    parser.add_argument("--graph_params", type=str, help="graph parameter used for generating agent network, currently only KSW model")
    parser.add_argument("--simulation_start", type=str, help="YYYY-MM-DD format, when should simulation begin")
    parser.add_argument("--simulation_end", type=str, help="YYYY-MM-DD format, when should simulation begin")
    parser.add_argument('--adult_male_beta', nargs=2, type=float, help='male agent perception values (non-family, family)')
    parser.add_argument('--adult_female_beta', nargs=2, type=float, help='female agent perception values (non-family, family)')
    parser.add_argument('--elderly_beta', nargs=2, type=float, help='elderly perception values (non-family, family)')
    parser.add_argument('--child_beta', nargs=2, type=float, help='children perception values (non-family, family)')
    
    # somewhat hacky parameters, might need to be changed to make life easier, but not theoretically justified strongly
    parser.add_argument('--scale_before_sigmoid', help='whether to apply scale before or after sigmoid during pbc calculation')
    parser.add_argument('--time_lag', type=int, help='Lag from latest observable events for current simulation')
    parser.add_argument('--move_scale', type=float, help='scalar to be used with beta values, set to 1.0 usually for no effect')
    parser.add_argument('--event_scale', type=float, help='scalar to be used with event weights, set to 1.0 usually for no effect')
    parser.add_argument('--param_threshold_lo', type=float, help='If effective peer influnce below this, no migration, set to 0 for no effect')
    parser.add_argument('--refugee_among_displaced', type=float, help='Ratio of refugee migrant among total displaced, ~0.3 from historical data')
    parser.add_argument('--phase_shift', type=float, help='Days after which refugee_among_displaced may shift, set to 1000 for no effect')
    parser.add_argument('--scale_before_phase_shift', type=float, help='Scaling amount before phase shift, set to 1 for no effect')
    parser.add_argument('--scale_after_phase_shift', type=float, help='Scaling amount after phase shift, set to 1 for no effect')
    parser.add_argument('--peer_influence_steps_per_sim', type=float, help='Number of synchronous update step for GDS threshold function, usually 1')
    parser.add_argument('--observe_neighbor_intention', help='whether intention is observable, so far we assume yes, but GDS-SBP No')
    
    # actual parametric variables, changes frequently, theoretically justified
    parser.add_argument('--region_name', required=True, type=str, help='Region to simulate, must be provided from CLI separately')
    parser.add_argument('--simulation_index', type=int, help='Simulation index, should stay same for a given set of parameters')
    parser.add_argument('--param_distance_decay', type=float, help='Distance decay parameter for spatial kernel (Gravity model currently)')
    parser.add_argument('--param_migration_bias', type=float, help='Reciprocal of Movement bias without observing events')
    parser.add_argument('--param_risk_growth_rate', type=float, help='Growth rate of movement likelihood as more events observed')
    parser.add_argument('--param_discount_rate', type=float, help='Exp Discounting% for every single timestep past events')
    parser.add_argument('--param_threshold_hi', type=float, help='If effective peer influnce above this, migration, set to +inf for no effect')
    parser.add_argument('--param_lambda1', type=float, help='Weight of inside peer influence/ Events')
    parser.add_argument('--param_lambda2', type=float, help='Weight of outside peer influence')
    
    # parameters that control resource and processing speed
    parser.add_argument('--S2_LEVEL', type=int, help='S2 Geometry level used to crete node partitions for threshold function')
    parser.add_argument('--group_region_file', type=str, help='File location with regions groupped to be simulated with single run')
    parser.add_argument('--cpu_core_usage', required=True, type=int, help='Number of cores to use, has to be passed externally')
    
    parser.add_argument('--log_file',help='Log file directory to store log')
    parser.add_argument('--info',help='general information about this config')
    return parser.parse_args(remaining_argv)

if __name__ == "__main__":
    args = parse_args()
    print(args)
    print(args.adult_male_beta[0])
    print(args.adult_male_beta[1])
    print(type(args.adult_male_beta[1]))