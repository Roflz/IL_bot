class Plan:
    """
    Each plan exposes:
      - id: str
      - label: str
      - set_phase(phase: str, camera_setup: bool = True) -> str
      - loop(ui, payload) -> int
    
    Standardized Phase System:
      - Plans should use 'DONE' as their completion phase
      - When a plan reaches 'DONE', it should set_phase('DONE') and return SUCCESS
      - The GUI will detect 'phase → DONE' in the logs and automatically move to the next plan
    """
    id: str
    label: str
    
    def set_phase(self, phase: str, camera_setup: bool = True) -> str: ...
    def loop(self, ui) -> int: ...
