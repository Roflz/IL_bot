# bot/plans/__init__.py
from .base import Plan

# Import concrete plans here
from .ring_craft import RingCraftPlan
from .ge_trade import GETradePlan
from .goto_rect import GoToRectPlan
from .romeo_and_juliet import RomeoAndJulietPlan

PLAN_REGISTRY = {
    RingCraftPlan.id: RingCraftPlan,
    GETradePlan.id: GETradePlan,
    GoToRectPlan.id: GoToRectPlan,
    RomeoAndJulietPlan.id: RomeoAndJulietPlan,
}
