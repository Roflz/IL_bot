class Plan:
    """
    Each plan exposes:
      - id: str
      - label: str
      - set_phase(phase: str, camera_setup: bool = True) -> str
      - loop(ui, payload) -> int
    """
    id: str
    label: str
    
    def set_phase(self, phase: str, camera_setup: bool = True) -> str: ...
    def loop(self, ui) -> int: ...
