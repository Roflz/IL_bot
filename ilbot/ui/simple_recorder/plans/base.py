class Plan:
    """
    Each plan exposes:
      - id: str
      - label: str
      - compute_phase(payload: dict, craft_recent: bool) -> str
      - build_action_plan(payload: dict, phase: str) -> dict
    """
    id: str
    label: str
    def compute_phase(self, payload: dict, craft_recent: bool) -> str: ...
    def build_action_plan(self, payload: dict, phase: str) -> dict: ...
