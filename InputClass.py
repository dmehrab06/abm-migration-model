from dataclasses import dataclass
from typing import Any

## fix to specific classes for the Any classes

@dataclass
class AgentConflictInputs:
    cc_data: Any = None
    hh_data: Any = None
    neighbor_data: Any = None
    cur_t: Any = None
    simtime: int = 0
    flag_a: bool = False
    flag_p: bool = False
    flag_s: bool = False
    del_col1: list[str] = None
    del_col2: list[str] = None
    use_core: int = 1
    ct_start: Any = None
    ct_end: Any = None
    
@dataclass
class AttitudePBCInputs:
    theta : float
    delta : float
    Q : float
    V : float
    e : float = 0.0001
    es : float = 1.0
    pbc_agent : bool = True 
    
@dataclass
class NetworkInputs:
    lambda_1 : float
    lambda_2 : float
    tau_lo : float
    tau_hi : float
    thresh_steps : int
    knows_neighbor : bool    
    phase_shift_day: int
    
@dataclass
class FinalActionInputs:
    refugee_ratio : float
    lo : float
    hi: float
    
@dataclass
class TimingCheckPoints:
    att_start : Any = None
    att_end : Any = None
    pcb_start : Any = None
    pcb_end : Any = None
    sn_start : Any = None
    sn_end : Any = None
    
@dataclass
class InputSizeInfo:
    num_agents : int
    num_households: int
    num_events: int
    num_edges_network : int
