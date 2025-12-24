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
    "DRAYNOR_BANK": (3090, 3095, 3240, 3246),
    "ALKHARID_BANK": (3265, 3272, 3161, 3173),
    "ARDOUGNE_EAST_NORTH_BANK": (2612, 2621, 3330, 3335),
    "ARDOUGNE_EAST_SOUTH_BANK": (2649, 2658, 3280, 3287),
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
    "LUMBRIDGE_NEW_PLAYER_SPAWN": (3210, 3230, 3205, 3235),
    "FALADOR_COWS": (3021, 3044, 3297, 3313),
    "VARROCK_WEST_TREES": (3117, 3153, 3415, 3450),
    "VARROCK_WEST_MINE": (3171, 3185, 3363, 3379),
    "AL_KHARID_MINE": (3298, 3301, 3308, 3316),
    "LUMBRIDGE_OCEAN_FISHING_AREA": (3234, 3244, 3145, 3156),
    "LUMBRIDGE_RIVER_FISHING_AREA": (3238, 3243, 3239, 3256),
    "LUMBRIDGE_GOBLINS": (3239, 3264, 3227, 3254),
    "GNOME_AGILITY_COURSE": (2469, 2490, 3414, 3440)
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
# 3-Letter Words (Gaming/MMO themed)
WORDS_3 = [
    "Ace", "Axe", "Arc", "Ash", "Bat", "Bee", "Boa", "Bow", "Box", "Bug",
    "Cat", "Cyp", "Dax", "Dog", "Eel", "Eve", "Eye", "Fae", "Fay", "Fen",
    "Fox", "Gag", "Gem", "God", "Hex", "Ice", "Ivy", "Jax", "Jay", "Jig",
    "Jot", "Kai", "Key", "Kix", "Lax", "Lee", "Leo", "Ley", "Lox", "Lux",
    "Lyn", "Map", "Meg", "Mob", "Nex", "Oak", "Ore", "Owl", "Pax", "Pit",
    "Pox", "Pyg", "Qua", "Ray", "Rat", "Red", "Rex", "Rio", "Rob", "Ron",
    "Rug", "Rye", "Sap", "Sea", "Sky", "Sun", "Tan", "Tar", "Tic", "Top",
    "Tub", "Ump", "Vex", "Vex", "Wax", "Woe", "Wok", "Xen", "Yak", "Yew",
    "Yin", "Zen", "Zed", "Zip", "Zoa"
]

# 4-Letter Words (Gaming/MMO themed with RuneScape references)
WORDS_4 = [
    "Apex", "Aster", "Bane", "Bark", "Beam", "Beast", "Blade", "Blue", "Bold", "Bolt",
    "Bone", "Book", "Boot", "Brew", "Cape", "Cast", "Cave", "Coal", "Cool", "Crab",
    "Craft", "Cron", "Cross", "Crow", "Cruz", "Cure", "Dark", "Dart", "Dawn", "Dead",
    "Demon", "Dice", "Disk", "Doss", "Dragon", "Dusk", "Dust", "Edge", "Elite", "Evil",
    "Fang", "Fast", "Fear", "Fire", "Fish", "Flag", "Flame", "Flax", "Flux", "Fury",
    "Fury", "Gale", "Gate", "Gaze", "Gem", "Giant", "Gift", "Gold", "Gray", "Grim",
    "Hack", "Haze", "Heat", "Hell", "Hero", "High", "Holy", "Hope", "Horn", "Hunt",
    "Ice", "Iris", "Iron", "Isle", "Jade", "Jazz", "Joke", "Jung", "Keon", "Key",
    "Kick", "Kill", "King", "Kite", "Knight", "Knob", "Know", "Lair", "Lake", "Lane",
    "Leaf", "Leap", "Legend", "Leon", "Less", "Life", "Light", "Lion", "Lock", "Long",
    "Loot", "Lord", "Lost", "Love", "Luck", "Lyra", "Mace", "Mage", "Magic", "Mail",
    "Main", "Major", "Make", "Mark", "Mars", "Mask", "Mast", "Math", "Maze", "Meat",
    "Meet", "Melt", "Metal", "Mind", "Mine", "Mist", "Moon", "More", "Moss", "Most",
    "Move", "Myth", "Navy", "Nest", "Net", "Next", "Night", "Node", "None", "Nova",
    "Oak", "Oath", "Ocean", "Ogre", "Once", "Ore", "Orin", "Over", "Owl", "Ox",
    "Pack", "Pain", "Palm", "Pan", "Pants", "Part", "Path", "Peak", "Peat", "Peer",
    "Pick", "Pier", "Pile", "Pink", "Pit", "Pity", "Pixy", "Plex", "Plot", "Plus",
    "Pool", "Poor", "Port", "Pot", "Pound", "Pour", "Pure", "Push", "Put", "Quay",
    "Quill", "Quit", "Race", "Rage", "Raid", "Rail", "Rain", "Rake", "Rank", "Rare",
    "Rate", "Raze", "Real", "Reap", "Reef", "Reel", "Rest", "Rice", "Rich", "Ride",
    "Rift", "Rig", "Right", "Rile", "Ring", "Ripe", "Rise", "Risk", "Rite", "Rive",
    "Road", "Roam", "Roar", "Rob", "Rock", "Roe", "Role", "Roll", "Rome", "Rope",
    "Rose", "Ross", "Rot", "Rota", "Rough", "Round", "Rout", "Rove", "Row", "Royal",
    "Rude", "Rue", "Rug", "Ruim", "Ruin", "Rule", "Rum", "Rune", "Rung", "Run",
    "Ruse", "Rush", "Rust", "Sage", "Salt", "Sand", "Sap", "Sat", "Save", "Saw",
    "Say", "Scab", "Scald", "Scale", "Scar", "Scare", "Scarf", "Scarlet", "Scat", "Scene",
    "Scent", "Scheme", "School", "Science", "Scissor", "Scold", "Scoop", "Scope", "Score", "Scorn",
    "Scout", "Scowl", "Scrap", "Scream", "Screen", "Screw", "Scribble", "Script", "Scratch", "Scrawl",
    "Scream", "Screen", "Screw", "Scribe", "Scroll", "Scrub", "Scuff", "Sculpt", "Scum", "Scuttle",
    "Star", "Tide", "True", "Voss", "Wild", "Wren", "Xeno", "Yara", "Zane", "Zero"
]

# 5-Letter Words (RuneScape themed)
WORDS_5 = [
    "Abyss", "Adder", "Aegis", "Agile", "Agnic", "Air", "Alder", "Alert", "Algae", "Alpha",
    "Altar", "Amaze", "Amber", "Ameth", "Anvil", "Apex", "Apple", "Aqua", "Arch", "Archa",
    "Arena", "Aris", "Armor", "Arrow", "Ash", "Ashen", "Ashur", "Aster", "Astral", "Atlas",
    "Auric", "Auto", "Axe", "Azure", "Bane", "Bank", "Bark", "Barley", "Barri", "Basalt",
    "Bass", "Bat", "Beach", "Beacon", "Bead", "Beak", "Beam", "Bean", "Bear", "Beast",
    "Beauty", "Beaver", "Beef", "Beer", "Beet", "Began", "Begin", "Being", "Bell", "Belly",
    "Below", "Bench", "Bend", "Bene", "Beryl", "Betta", "Better", "Between", "Bible", "Bike",
    "Bile", "Billy", "Bingo", "Birch", "Bird", "Birth", "Biscuit", "Bit", "Bite", "Bitter",
    "Black", "Blade", "Blame", "Blank", "Blast", "Blaze", "Bleak", "Bleed", "Bless", "Blind",
    "Blink", "Block", "Blood", "Bloom", "Blossom", "Blow", "Blue", "Bluff", "Blunt", "Blur",
    "Blush", "Boat", "Bobber", "Body", "Boil", "Bold", "Bolt", "Bomb", "Bone", "Bongo",
    "Bonus", "Bony", "Book", "Boom", "Boost", "Boot", "Booth", "Border", "Bore", "Born",
    "Borrow", "Boss", "Both", "Bother", "Bottle", "Bought", "Boulder", "Bounce", "Bound", "Bounty",
    "Bouquet", "Boutique", "Bouton", "Bow", "Bowl", "Box", "Boy", "Bracelet", "Brain", "Brake",
    "Brand", "Bran", "Brass", "Brave", "Brawny", "Bread", "Break", "Breath", "Breed", "Breeze",
    "Brew", "Brick", "Bridge", "Brief", "Bright", "Brilliant", "Bring", "Brink", "Brisk", "Brist",
    "Broad", "Broken", "Bronze", "Brook", "Broom", "Broth", "Brother", "Brought", "Brown", "Brush",
    "Bubble", "Buck", "Bucket", "Buckle", "Bud", "Budget", "Buff", "Bug", "Build", "Built",
    "Bulk", "Bull", "Bullet", "Bum", "Bump", "Bun", "Bunch", "Bundle", "Bunk", "Bunt",
    "Burden", "Bureau", "Burg", "Burial", "Burl", "Burn", "Burst", "Bury", "Bus", "Bush",
    "Bust", "Busy", "But", "Butcher", "Butt", "Butter", "Button", "Buy", "Buzz", "Bye",
    "Byte", "Cab", "Cabin", "Cabinet", "Cable", "Cactus", "Cage", "Cake", "Calam", "Calc",
    "Calm", "Camel", "Cameras", "Camp", "Can", "Canal", "Candy", "Cane", "Cannon", "Cant",
    "Canvas", "Canyon", "Cap", "Cape", "Capital", "Car", "Caramel", "Card", "Care", "Career",
    "Careful", "Cargo", "Carib", "Carn", "Carriage", "Carrier", "Carrot", "Carry", "Cart", "Carve",
    "Case", "Cash", "Cass", "Cast", "Casual", "Cat", "Cata", "Cata", "Catacomb", "Catch",
    "Cater", "Catfish", "Cathedral", "Cattle", "Caus", "Cause", "Caution", "Cave", "Caviar", "Cayenne",
    "Cease", "Cedar", "Celebr", "Celes", "Celest", "Cell", "Cellar", "Cement", "Cemetery", "Censor",
    "Census", "Cent", "Center", "Centra", "Centra", "Centur", "Cere", "Cereal", "Cert", "Cert",
    "Cha", "Chai", "Chair", "Chamber", "Champ", "Chance", "Change", "Channel", "Chant", "Chaos",
    "Chap", "Chap", "Chapt", "Char", "Char", "Char", "Charcoal", "Charge", "Charm", "Chase",
    "Chat", "Cheap", "Cheat", "Check", "Cheek", "Cheese", "Cheet", "Chef", "Cherry", "Chest",
    "Chev", "Chew", "Chic", "Chicken", "Chief", "Child", "Chill", "Chime", "Chin", "Chip",
    "Chit", "Choc", "Chocolate", "Choice", "Choke", "Chomp", "Chop", "Chord", "Chore", "Chr",
    "Christ", "Christian", "Chri", "Chronic", "Chron", "Chrome", "Chronic", "Chub", "Chuck", "Chum",
    "Chunky", "Church", "Cider", "Cigar", "Cinder", "Cinema", "Cinnamon", "Circle", "Circul", "Circum",
    "Circus", "Citadel", "Citrus", "City", "Civic", "Civil", "Civi", "Civil", "Civilian", "Clai",
    "Claim", "Clam", "Clamp", "Clan", "Clang", "Clank", "Clap", "Clar", "Clarif", "Clarinet",
    "Clarity", "Clash", "Clasp", "Class", "Classic", "Classif", "Classroom", "Clatter", "Clause", "Clav",
    "Claw", "Clay", "Clean", "Clear", "Clearance", "Clearing", "Cleave", "Clench", "Clerk", "Clever",
    "Click", "Client", "Cliff", "Climate", "Climb", "Clin", "Clinical", "Clinic", "Clink", "Clip",
    "Cloak", "Clock", "Clog", "Clone", "Close", "Clos", "Closet", "Cloth", "Clothe", "Cloth"
]

# 6-Letter Words (RuneScape themed)
WORDS_6 = [
    "Abyssal", "Acorn", "Adaman", "Advent", "Aether", "Agility", "Alchemy", "Alder", "Ancient", "Angel",
    "Anvil", "Apathy", "Apex", "Aqua", "Arcane", "Archer", "Arena", "Armor", "Arrow", "Artifact",
    "Astral", "Attack", "Aurora", "Avenge", "Azure", "Backpack", "Badge", "Bag", "Bait", "Balance",
    "Bandage", "Bandit", "Banjo", "Bank", "Bar", "Barb", "Barbar", "Bargain", "Bark", "Barn",
    "Barrel", "Barrier", "Barter", "Base", "Basin", "Bast", "Batche", "Bat", "Beacon", "Beak",
    "Beam", "Bear", "Beast", "Beat", "Beaver", "Became", "Because", "Become", "Bed", "Bee",
    "Beef", "Been", "Beer", "Beet", "Began", "Begin", "Begun", "Behalf", "Behave", "Behavior",
    "Being", "Belief", "Believe", "Bell", "Belong", "Below", "Belt", "Bench", "Beneath", "Benefit",
    "Bent", "Berry", "Beside", "Best", "Bet", "Better", "Between", "Beyond", "Bias", "Bible",
    "Bicycle", "Bid", "Big", "Bike", "Bill", "Billy", "Bind", "Bird", "Birth", "Bit",
    "Bite", "Bitter", "Black", "Blade", "Blame", "Blank", "Blanket", "Blast", "Blaze", "Bleach",
    "Bleak", "Bleed", "Blend", "Bless", "Blew", "Blind", "Blink", "Bliss", "Blitz", "Block",
    "Blond", "Blood", "Bloom", "Blossom", "Blow", "Blue", "Bluff", "Blunt", "Blur", "Blush",
    "Board", "Boat", "Body", "Boil", "Bold", "Bolt", "Bomb", "Bond", "Bone", "Bonus",
    "Book", "Boom", "Boost", "Boot", "Booth", "Border", "Bore", "Bore", "Born", "Borrow",
    "Boss", "Both", "Bother", "Bottle", "Bottom", "Bought", "Boulder", "Bounce", "Bound", "Bounty",
    "Bour", "Bout", "Boutique", "Bouton", "Bow", "Bow", "Bowel", "Bowl", "Box", "Boy",
    "Brace", "Brace", "Bracelet", "Bracket", "Brain", "Brake", "Branch", "Brand", "Brass", "Brave",
    "Brawny", "Breach", "Bread", "Break", "Breast", "Breath", "Breathe", "Bred", "Breed", "Breeze",
    "Brew", "Brick", "Bridge", "Brief", "Bright", "Brilliant", "Bring", "Brink", "Brisk", "Brist",
    "Broad", "Broad", "Broadcast", "Broken", "Bronze", "Brook", "Broom", "Broth", "Brother", "Brought",
    "Brown", "Browse", "Brush", "Brutal", "Brute", "Bubble", "Buck", "Bucket", "Buckle", "Bud",
    "Budget", "Buff", "Buffer", "Bug", "Build", "Built", "Bulk", "Bull", "Bullet", "Bump",
    "Bunch", "Bundle", "Bunk", "Burden", "Bureau", "Burial", "Burl", "Burn", "Burst", "Bury",
    "Bus", "Bush", "Bust", "Busy", "But", "Butcher", "Butt", "Butter", "Button", "Buy",
    "Buzz", "Byte"
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
    1249: "SEWING",
    632: "MINING",
    631: "MINING",
    630: "MINING",
    629: "MINING",
    628: "MINING",
    627: "MINING",
    623: "BAITING",
    622: "BAITING",
    896: "COOKING",
    6752: "MLM_MINING"
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
    "gold_bracelet": 11.0,   # Experience per gold bracelet
    "sapphire_ring": 40.0,   # Experience per sapphire ring
    "opal_ring": 30.0,       # Experience per opal ring
    "jade_ring": 32.0,       # Experience per jade ring
    "topaz_ring": 35.0,      # Experience per topaz ring
}

# Jewelry Crafting Widget IDs
JEWELRY_CRAFTING_WIDGETS = {
    # Ring crafting options (Id 29229064 - Id 29229071)
    "GOLD_RING": 29229064,        # S 446.8 CraftingGold.GOLD_RING
    "SAPPHIRE_RING": 29229065,    # S 446.9 CraftingGold.SAPPHIRE_RING
    "EMERALD_RING": 29229066,     # S 446.10 CraftingGold.EMERALD_RING
    "RUBY_RING": 29229067,        # S 446.11 CraftingGold.RUBY_RING
    "DIAMOND_RING": 29229068,     # S 446.12 CraftingGold.DIAMOND_RING
    "OPAL_RING": 393223,      # S 446.13 CraftingGold.DRAGON_RING
    "JADE_RING": 393224,        # S 446.14 CraftingGold.ONYX_RING
    "TOPAZ_RING": 393225,      # S 446.15 CraftingGold.ZENYTE_RING
    
    # Necklace crafting options (Id 29229079 - Id 29229086)
    "GOLD_NECKLACE": 29229079,    # S 446.23 CraftingGold.GOLD_NECKLACE
    "SAPPHIRE_NECKLACE": 29229080, # S 446.24 CraftingGold.SAPPHIRE_NECKLACE
    "EMERALD_NECKLACE": 29229081,  # S 446.25 CraftingGold.EMERALD_NECKLACE
    "RUBY_NECKLACE": 29229082,     # S 446.26 CraftingGold.RUBY_NECKLACE
    "DIAMOND_NECKLACE": 29229083,  # S 446.27 CraftingGold.DIAMOND_NECKLACE
    "OPAL_NECKLACE": 393227,   # S 446.28 CraftingGold.DRAGON_NECKLACE
    "JADE_NECKLACE": 393228,     # S 446.29 CraftingGold.ONYX_NECKLACE
    "TOPAZ_NECKLACE": 393229,   # S 446.30 CraftingGold.ZENYTE_NECKLACE
    
    # Bracelet crafting options (Id 29229072 - Id 29229078)
    "GOLD_BRACELET": 29229106,     # S 446.16 CraftingGold.GOLD_BRACELET
    "OPAL_BRACELET": 393235,     # S 446.17 CraftingGold.OPAL_BRACELET
    "JADE_BRACELET": 393236,     # S 446.18 CraftingGold.JADE_BRACELET
    "TOPAZ_BRACELET": 393237,    # S 446.19 CraftingGold.TOPAZ_BRACELET
    "SAPPHIRE_BRACELET": 29229108, # S 446.20 CraftingGold.SAPPHIRE_BRACELET
    "EMERALD_BRACELET": 29229109,  # S 446.21 CraftingGold.EMERALD_BRACELET
    "RUBY_BRACELET": 29229110,     # S 446.22 CraftingGold.RUBY_BRACELET

    # Bracelet crafting options (Id 29229072 - Id 29229078)
    "GOLD_AMULET": 29229093,     # S 446.16 CraftingGold.GOLD_AMULET
    "OPAL_AMULET": 393231,     # S 446.17 CraftingGold.OPAL_AMULET
    "JADE_AMULET": 393232,     # S 446.18 CraftingGold.JADE_AMULET
    "TOPAZ_AMULET": 393233,    # S 446.19 CraftingGold.TOPAZ_AMULET
    "SAPPHIRE_AMULET": 29229094, # S 446.20 CraftingGold.SAPPHIRE_AMULET
    "EMERALD_AMULET": 29229095,  # S 446.21 CraftingGold.EMERALD_AMULET
    "RUBY_AMULET": 29229096,     # S 446.22 CraftingGold.RUBY_AMULET
    
    # Opal/Jade/Topaz ring crafting options (TODO: Verify widget IDs - these may be in a different range)
    "OPAL_RING": 0,                # TODO: Find correct widget ID for opal ring
    "JADE_RING": 0,                # TODO: Find correct widget ID for jade ring
    "TOPAZ_RING": 0,               # TODO: Find correct widget ID for topaz ring
}
