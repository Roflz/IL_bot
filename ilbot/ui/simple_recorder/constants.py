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
    "GE": (3155, 3173, 3479, 3498),
    "VARROCK_WEST": (3180, 3190, 3433, 3447),
    "FALADOR_BANK": (3009, 3018, 3353, 3358),
}

REGIONS = {
    "VARROCK_SQUARE": (3200, 3227, 3406, 3442),
    "JULIET_MANSION": (3156, 3164, 3432, 3438),
    "VARROCK_CHURCH": (3252, 3259, 3471, 3488),
    "VARROCK_APOTHECARY": (3192, 3198, 3402, 3406),
    "GOBLIN_VILLAGE": (2946, 2968, 3490, 3515),
    "COOKING_TUTORIAL": (3074, 3078, 3083, 3086),
    "QUEST_TUTORIAL": (3080, 3089, 3118, 3125),
    "MINING_TUTORIAL": (3077, 3086, 9502, 9508),
    "COMBAT_TUTORIAL": (3105, 3110, 9507, 9513),
    "PRAYER_TUTORIAL": (3120, 3128, 3103, 3110),
    "MAGIC_TUTORIAL": (3137, 3144, 3082, 3091),
    "LUMBRIDGE_NEW_PLAYER_SPAWN": (3231, 3240, 3212, 3224),
    "FALADOR_COWS": (3021, 3044, 3297, 3313),
    "VARROCK_WEST_TREES": (3117, 3148, 3423, 3444)
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
    879: "CHOPPING",
    733: "FIREMAKING",
    897: "COOKING_ON_FIRE",
    # Add more animations as needed
}

# Convenient access to regions
FALADOR_BANK = BANK_REGIONS["FALADOR_BANK"]
FALADOR_COWS = REGIONS["FALADOR_COWS"]
