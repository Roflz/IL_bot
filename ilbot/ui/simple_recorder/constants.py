# constants.py
from pathlib import Path

import pyautogui

# keep identical defaults/behavior
pyautogui.FAILSAFE = True  # moving mouse to a corner aborts

AUTO_REFRESH_MS      = 100
AUTO_RUN_TICK_MS     = 300    # scheduler tick for the UI/auto loop
PRE_ACTION_DELAY_MS  = 250    # delay before performing any action (click/key)
RULE_WAIT_TIMEOUT_MS = 10_000 # pre/post condition wait timeout

SESSIONS_DIR = Path(r"D:\repos\bot_runelite_IL\data\recording_sessions")

# Xmin, Xmax, Ymin, Ymax
BANK_REGIONS = {
    "EDGEVILLE_BANK": (3092, 3098, 3488, 3498),
    "GE": (3155, 3173, 3469, 3498),
    "VARROCK_WEST": (3180, 3190, 3433, 3447),
    "FALADOR_BANK": (3009, 3018, 3353, 3358),
}

REGIONS = {
    "VARROCK_SQUARE": (3200, 3227, 3406, 3442),
    "JULIET_MANSION": (3156, 3164, 3432, 3438),
    "VARROCK_CHURCH": (3252, 3259, 3471, 3488),
    "VARROCK_APOTHECARY": (3192, 3198, 3402, 3406),
    "GOBLIN_VILLAGE": (2953, 2960, 3495, 3515),
    "COOKING_TUTORIAL": (3074, 3078, 3083, 3086),
    "QUEST_TUTORIAL": (3080, 3089, 3118, 3125),
    "MINING_TUTORIAL": (3077, 3086, 9502, 9508),
    "COMBAT_TUTORIAL": (3105, 3110, 9507, 9513),
    "PRAYER_TUTORIAL": (3120, 3128, 3103, 3110),
    "MAGIC_TUTORIAL": (3137, 3144, 3082, 3091),
    "LUMBRIDGE_NEW_PLAYER_SPAWN": (3231, 3240, 3212, 3224),
    "FALADOR_COWS": (3021, 3044, 3297, 3313),
    "VARROCK_WEST_TREES": (3117, 3153, 3415, 3450)
}

# Grand Exchange region bounds (hardcoded)
GE_MIN_X = 3155
GE_MAX_X = 3173
GE_MIN_Y = 3479
GE_MAX_Y = 3498

# Edgeville Bank bounding box (world coordinates)
EDGE_BANK_MIN_X = 3092
EDGE_BANK_MAX_X = 3098
EDGE_BANK_MIN_Y = 3488
EDGE_BANK_MAX_Y = 3498

CAM_BUFFER_X = 200
CAM_BUFFER_Y_TOP = 50     # "too high" threshold -> press DOWN
CAM_BUFFER_Y_BOT = 200     # "too low" (near bottom) threshold for yaw-toward
CAM_YAW_HOLD_MS = 220

# MMO Name Generator - Organized by length for better combinations
# 3-Letter Words (Realistic/Common)
WORDS_3 = [
    "Max", "Sam", "Tom", "Dan", "Ben", "Jon", "Tim", "Rob", "Jim", "Bob", "Ted", "Joe", 
    "Roy", "Ray", "Jay", "Lee", "Kim", "Ann", "Amy", "Sue", "Meg", "Liz", "Jen", "Kim", 
    "Pat", "Kim", "Ron", "Don", "Ken", "Len", "Ian", "Eli", "Ace", "Zoe", "Rio", "Leo", 
    "Eve", "Ava", "Ivy", "Sky", "Fox", "Cat", "Dog", "Bat", "Rat", "Ant", "Bee", "Fly", 
    "Owl", "Eel", "Ray", "Sun", "Sea", "Sky", "Ice", "Oak", "Pine", "Maple", "Rose", "Lily"
]

# 4-Letter Words (Realistic/Common)
WORDS_4 = [
    "Alex", "Jake", "Luke", "Mark", "Paul", "John", "Mike", "Dave", "Ryan", "Eric", 
    "Adam", "Nick", "Sean", "Jack", "Will", "Matt", "Josh", "Tony", "Andy", "Carl", 
    "Kyle", "Troy", "Dean", "Glen", "Hank", "Ivan", "Jake", "Kane", "Lane", "Mick", 
    "Nate", "Owen", "Pete", "Rick", "Stan", "Todd", "Vic", "Wade", "Zack", "Ace", 
    "Blue", "Cool", "Fast", "Gold", "High", "Iron", "Jazz", "King", "Lion", "Moon", 
    "Navy", "Open", "Pure", "Rock", "Star", "True", "Wild", "Zero", "Bold", "Calm"
]

# 5-Letter Words (Realistic/Common)
WORDS_5 = [
    "Aaron", "Blake", "Chris", "David", "Ethan", "Frank", "Grant", "Henry", "Isaac", "James", 
    "Kevin", "Lucas", "Mason", "Noah", "Oscar", "Peter", "Quinn", "River", "Steve", "Tyler", 
    "Unity", "Victor", "Willy", "Xavier", "Young", "Zack", "Alpha", "Brave", "Cloud", "Dream", 
    "Eagle", "Flame", "Glory", "Happy", "Ideal", "Jolly", "Karma", "Light", "Magic", "Noble", 
    "Ocean", "Peace", "Quick", "Royal", "Smart", "Tiger", "Urban", "Vital", "Wise", "Zen"
]

# 6-Letter Words (Realistic/Common)
WORDS_6 = [
    "Andrew", "Brandon", "Carlos", "Daniel", "Edward", "Felix", "George", "Hunter", "Isaiah", "Jordan", 
    "Kyle", "Liam", "Marcus", "Nathan", "Oliver", "Parker", "Quinn", "Robert", "Samuel", "Thomas", 
    "Unique", "Vincent", "William", "Xavier", "Yusuf", "Zachary", "Advent", "Bright", "Crystal", "Dragon", 
    "Echo", "Forest", "Golden", "Humble", "Island", "Journey", "Knight", "Legend", "Mystic", "Nature", 
    "Oracle", "Phoenix", "Quest", "Ranger", "Spirit", "Thunder", "Unique", "Violet", "Warrior", "Zenith"
]

# Name generation patterns (length combinations that work well)
NAME_PATTERNS = [
    # Single words
    (6,),  # 6-letter word alone
    (5,),  # 5-letter word alone
    
    # Two word combinations
    (4, 3),  # 4 + 3 = 7 chars
    (3, 4),  # 3 + 4 = 7 chars
    (5, 3),  # 5 + 3 = 8 chars
    (3, 5),  # 3 + 5 = 8 chars
    (4, 4),  # 4 + 4 = 8 chars
    (6, 3),  # 6 + 3 = 9 chars
    (3, 6),  # 3 + 6 = 9 chars
    (5, 4),  # 5 + 4 = 9 chars
    (4, 5),  # 4 + 5 = 9 chars
    (6, 4),  # 6 + 4 = 10 chars
    (4, 6),  # 4 + 6 = 10 chars
    (5, 5),  # 5 + 5 = 10 chars
    (6, 5),  # 6 + 5 = 11 chars
    (5, 6),  # 5 + 6 = 11 chars
    (6, 6),  # 6 + 6 = 12 chars
    
    # Three word combinations
    (3, 3, 3),  # 3 + 3 + 3 = 9 chars
    (4, 3, 3),  # 4 + 3 + 3 = 10 chars
    (3, 4, 3),  # 3 + 4 + 3 = 10 chars
    (3, 3, 4),  # 3 + 3 + 4 = 10 chars
    (5, 3, 3),  # 5 + 3 + 3 = 11 chars
    (3, 5, 3),  # 3 + 5 + 3 = 11 chars
    (3, 3, 5),  # 3 + 3 + 5 = 11 chars
    (4, 4, 3),  # 4 + 4 + 3 = 11 chars
    (4, 3, 4),  # 4 + 3 + 4 = 11 chars
    (3, 4, 4),  # 3 + 4 + 4 = 11 chars
    (6, 3, 3),  # 6 + 3 + 3 = 12 chars
    (3, 6, 3),  # 3 + 6 + 3 = 12 chars
    (3, 3, 6),  # 3 + 3 + 6 = 12 chars
    (5, 4, 3),  # 5 + 4 + 3 = 12 chars
    (5, 3, 4),  # 5 + 3 + 4 = 12 chars
    (4, 5, 3),  # 4 + 5 + 3 = 12 chars
    (4, 3, 5),  # 4 + 3 + 5 = 12 chars
    (3, 5, 4),  # 3 + 5 + 4 = 12 chars
    (3, 4, 5),  # 3 + 4 + 5 = 12 chars
    (4, 4, 4),  # 4 + 4 + 4 = 12 chars
]

# Capitalization patterns for variety
CAPITALIZATION_PATTERNS = [
    "lower",      # all lowercase
    "title",      # Title Case
    "first_upper" # First letter uppercase, rest lowercase
]

# Name generation functions
def generate_player_name() -> str:
    """
    Generate a random player name using the MMO name generator system.
    Returns a name that's 12 characters or less.
    """
    import random
    
    # Choose a random pattern
    pattern = random.choice(NAME_PATTERNS)
    
    # Get the word lists
    word_lists = {
        3: WORDS_3,
        4: WORDS_4, 
        5: WORDS_5,
        6: WORDS_6
    }
    
    # Build the name by combining words from the pattern
    name_parts = []
    for length in pattern:
        word_list = word_lists[length]
        word = random.choice(word_list)
        name_parts.append(word)
    
    # Join the parts
    base_name = "".join(name_parts)
    
    # Apply capitalization pattern
    cap_pattern = random.choice(CAPITALIZATION_PATTERNS)
    name = apply_capitalization(base_name, cap_pattern)
    
    # Ensure it's 12 characters or less
    if len(name) > 12:
        # If too long, try a shorter pattern
        shorter_patterns = [p for p in NAME_PATTERNS if sum(p) <= 12]
        if shorter_patterns:
            pattern = random.choice(shorter_patterns)
            name_parts = []
            for length in pattern:
                word_list = word_lists[length]
                word = random.choice(word_list)
                name_parts.append(word)
            base_name = "".join(name_parts)
            name = apply_capitalization(base_name, cap_pattern)
    
    return name[:12]  # Truncate if still too long

def apply_capitalization(text: str, pattern: str) -> str:
    """Apply a capitalization pattern to text."""
    if pattern == "lower":
        return text.lower()
    elif pattern == "title":
        return text.title()
    elif pattern == "first_upper":
        if not text:
            return text
        return text[0].upper() + text[1:].lower()
    else:
        return text

def generate_multiple_names(count: int = 10) -> list[str]:
    """Generate multiple unique player names."""
    names = set()
    attempts = 0
    max_attempts = count * 3  # Prevent infinite loops
    
    while len(names) < count and attempts < max_attempts:
        name = generate_player_name()
        if name not in names:
            names.add(name)
        attempts += 1
    
    return list(names)

# Player animation IDs
PLAYER_ANIMATIONS = {
    621: "NETTING",
    879: "CHOPPING",  # Bronze axe
    877: "CHOPPING",  # Iron axe
    875: "CHOPPING",  # Steel axe
    873: "CHOPPING",  # Mithril axe
    871: "CHOPPING",  # Adamant axe
    869: "CHOPPING",  # Rune axe
    867: "CHOPPING",  # Dragon axe
    733: "FIREMAKING",
    897: "COOKING_ON_FIRE",
    899: "SMELTING",
    1249: "SEWING"
    # Add more animations as needed
}

# Woodcutting animation IDs (all chopping variants)
WOODCUTTING_ANIMATIONS = {879, 877, 875, 873, 871, 869, 867}

# Convenient access to regions
FALADOR_BANK = BANK_REGIONS["FALADOR_BANK"]
FALADOR_COWS = REGIONS["FALADOR_COWS"]

# Widget IDs Database
BANK_WIDGETS = {
    "SWAP": 786451,      # S 12.19 Bankmain.SWAP
    "INSERT": 786453,    # S 12.21 Bankmain.INSERT
    "ITEM": 786456,      # S 12.24 Bankmain.ITEM
    "NOTE": 786458,      # S 12.26 Bankmain.NOTE
    "QUANTITY1": 786462, # S 12.30 Bankmain.QUANTITY1
    "QUANTITY5": 786464, # S 12.32 Bankmain.QUANTITY5
    "QUANTITY10": 786466,# S 12.34 Bankmain.QUANTITY10
    "QUANTITYX": 786468, # S 12.36 Bankmain.QUANTITYX
    "QUANTITYALL": 786470,# S 12.38 Bankmain.QUANTITYALL
}

# GE Widget IDs
GE_WIDGETS = {
    "CONTENTS": 30474241,    # S 465.1 GeOffers.CONTENTS
    "INDEX": 30474247,       # S 465.5 GeOffers.INDEX (INDEX_0)
    "COLLECTALL": 30474248,  # S 465.6 GeOffers.COLLECTALL
    "INDEX_0": 30474247,     # S 465.7 GeOffers.INDEX_0
    "INDEX_1": 30474248,     # S 465.8 GeOffers.INDEX_1
    "INDEX_2": 30474249,     # S 465.9 GeOffers.INDEX_2
    "INDEX_3": 30474250,     # S 465.10 GeOffers.INDEX_3
    "INDEX_4": 30474251,     # S 465.11 GeOffers.INDEX_4
    "INDEX_5": 30474252,     # S 465.12 GeOffers.INDEX_5
    "INDEX_6": 30474253,     # S 465.13 GeOffers.INDEX_6
    "INDEX_7": 30474254,     # S 465.14 GeOffers.INDEX_7
    "INVENTORY": 30474255,   # S 465.15 GeOffers.INVENTORY
    "SETUP": 30474266,       # S 465.26 GeOffers.SETUP
    "CONFIRM": 30474270,     # S 465.30 GeOffers.CONFIRM
}

# Chat Widget IDs
CHAT_WIDGETS = {
    "CHATMODAL": 10617398,   # S 162.51 Chatbox.CHATMODAL
    "MES_LAYER_SCROLLCONTENTS": 10616883, # S 162.51 Chatbox.MES_LAYER_SCROLLCONTENTS
}

# GE Offer Screen Widget IDs (from widget children analysis)
GE_OFFER_WIDGETS = {
    "MAIN_CONTAINER": 30474242,    # Main offer screen container
    "CLOSE_BUTTON": 30474242,      # X button (index 12)
    "HISTORY_BUTTON": 30474243,    # History button
    "ITEM_SEARCH_ICON": 30474266,  # Item search icon (index 62)
    "QUANTITY_PLUS_1": 30474266,   # +1 quantity button (index 41)
    "QUANTITY_PLUS_10": 30474266,  # +10 quantity button (index 42)
    "QUANTITY_PLUS_100": 30474266, # +100 quantity button (index 43)
    "QUANTITY_PLUS_1K": 30474266,  # +1K quantity button (index 44)
    "QUANTITY_CUSTOM": 30474266,   # Custom quantity button (index 45)
    "QUANTITY_LEFT_ARROW": 30474266, # Quantity left arrow (index 39)
    "QUANTITY_RIGHT_ARROW": 30474266, # Quantity right arrow (index 40)
    "PRICE_MINUS_5_PERCENT": 30474266, # -5% price button (index 48)
    "PRICE_PLUS_5_PERCENT": 30474266,  # +5% price button (index 51)
    "PRICE_MINUS_X_PERCENT": 30474266, # -X% price button (index 52)
    "PRICE_PLUS_X_PERCENT": 30474266,  # +X% price button (index 53)
    "PRICE_LEFT_ARROW": 30474266,  # Price left arrow (index 46)
    "PRICE_RIGHT_ARROW": 30474266, # Price right arrow (index 47)
    "MARKET_PRICE_BUTTON": 30474266, # Market price button (index 49)
    "CONFIRM_BUTTON": 30474244,    # Confirm offer button
}

# RuneScape Experience Table (Level 1-99)
# Format: {level: total_experience_required}
EXPERIENCE_TABLE = {
    1: 0, 2: 83, 3: 174, 4: 276, 5: 388, 6: 512, 7: 650, 8: 801, 9: 969, 10: 1154,
    11: 1358, 12: 1584, 13: 1833, 14: 2107, 15: 2411, 16: 2746, 17: 3115, 18: 3523, 19: 3973, 20: 4470,
    21: 5018, 22: 5624, 23: 6291, 24: 7028, 25: 7842, 26: 8740, 27: 9730, 28: 10824, 29: 12031, 30: 13363,
    31: 14833, 32: 16456, 33: 18247, 34: 20224, 35: 22206, 36: 24815, 37: 27473, 38: 30408, 39: 33648, 40: 37224,
    41: 41171, 42: 45529, 43: 50339, 44: 55649, 45: 61512, 46: 67983, 47: 75127, 48: 83014, 49: 91721, 50: 101333,
    51: 111945, 52: 123660, 53: 136594, 54: 150872, 55: 166636, 56: 184040, 57: 203254, 58: 224466, 59: 247886, 60: 273742,
    61: 302288, 62: 333804, 63: 368599, 64: 407015, 65: 449428, 66: 496254, 67: 547953, 68: 605032, 69: 668051, 70: 737627,
    71: 814445, 72: 899257, 73: 992895, 74: 1096278, 75: 1210421, 76: 1336443, 77: 1475581, 78: 1629200, 79: 1798808, 80: 1986068,
    81: 2192818, 82: 2421087, 83: 2673114, 84: 2951373, 85: 3258594, 86: 3597792, 87: 3972294, 88: 4385776, 89: 4842295, 90: 5346332,
    91: 5902831, 92: 6517253, 93: 7195629, 94: 7944614, 95: 8771558, 96: 9684577, 97: 10692629, 98: 11805606, 99: 13034431
}

# Crafting experience values
CRAFTING_EXP = {
    "leather_gloves": 13.8,  # Experience per leather (leather gloves)
    "gold_ring": 15.0,       # Experience per gold ring
    "sapphire_ring": 40.0    # Experience per sapphire ring
}
