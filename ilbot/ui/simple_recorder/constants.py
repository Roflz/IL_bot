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
}

REGIONS = {
    "VARROCK_SQUARE": (3200, 3227, 3416, 3442),
    "JULIET_MANSION": (3156, 3164, 3432, 3438),
    "VARROCK_CHURCH": (3252, 3259, 3471, 3488),
    "VARROCK_APOTHECARY": (3192, 3198, 3402, 3406),
    "GOBLIN_VILLAGE": (2946, 2968, 3490, 3515),
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

# Hilarious tutorial character name combinations
TUTORIAL_NAME_WORDS = [
    # Adjectives
    "Stinky", "Sweaty", "Crusty", "Moldy", "Rusty", "Dusty", "Musty", "Farty", 
    "Burpy", "Snotty", "Boogery", "Pimply", "Hairy", "Bald", "Chunky", "Slimy",
    "Gooey", "Sticky", "Greasy", "Oily", "Smelly", "Rotten", "Spoiled", "Moldy",
    "Wet", "Damp", "Moist", "Soggy", "Squishy", "Mushy", "Lumpy", "Bumpy",
    
    # Nouns
    "Butt", "Ass", "Poop", "Fart", "Burp", "Snot", "Booger", "Pimple", "Zit",
    "Toe", "Finger", "Nose", "Ear", "Belly", "Gut", "Chin", "Elbow", "Knee",
    "Sock", "Underwear", "Pants", "Shirt", "Hat", "Shoe", "Boot", "Sandwich",
    "Pizza", "Burger", "Taco", "Donut", "Cookie", "Cake", "Pie", "Soup",
    "Pickle", "Onion", "Garlic", "Cheese", "Bacon", "Sausage", "Hotdog",
    "Monkey", "Donkey", "Pig", "Cow", "Chicken", "Duck", "Goose", "Turkey",
    "Fish", "Shark", "Whale", "Dolphin", "Octopus", "Squid", "Lobster", "Crab",
    
    # Action words
    "Farting", "Burping", "Sneezing", "Coughing", "Snoring", "Drooling", "Sweating",
    "Crying", "Laughing", "Giggling", "Snorting", "Wheezing", "Panting", "Gasping",
    "Eating", "Drinking", "Sleeping", "Running", "Jumping", "Falling", "Tripping",
    "Slipping", "Sliding", "Rolling", "Spinning", "Dancing", "Singing", "Humming",
    
    # Silly suffixes
    "Face", "Head", "Nose", "Ears", "Toes", "Fingers", "Belly", "Butt", "Feet",
    "Pants", "Socks", "Shoes", "Hat", "Shirt", "Pants", "Underwear", "Gloves",
    "Muffin", "Pancake", "Waffle", "Bagel", "Pretzel", "Cracker", "Chip", "Nugget",
    "Master", "Lord", "King", "Queen", "Prince", "Princess", "Duke", "Duchess",
    "Wizard", "Witch", "Mage", "Sorcerer", "Knight", "Warrior", "Hero", "Villain"
]

# Pre-made hilarious combinations for guaranteed laughs (max 12 characters)
TUTORIAL_NAME_COMBOS = [
    "StinkyButt", "SweatySock", "CrustyToe", "MoldyPizza", "DustyFart", 
    "MustyBurp", "BoogeryFace", "PimplyNose", "HairyBelly", "BaldChin", 
    "ChunkyElbow", "SlimyKnee", "StickyDonut", "GreasyCookie", "OilyCake", 
    "SmellyPie", "RottenSoup", "WetOnion", "DampGarlic", "MoistCheese", 
    "SoggyBacon", "MushyHotdog", "LumpyMonkey", "BumpyDonkey", "FartyPig",
    "BurpyCow", "SnottyChicken", "BoogeryDuck", "PimplyGoose", "HairyTurkey",
    "BaldFish", "ChunkyShark", "SlimyWhale", "GooeyDolphin", "StickyOctopus",
    "GreasySquid", "OilyLobster", "SmellyCrab", "RottenMuffin", "SpoiledPancake",
    "WetWaffle", "DampBagel", "MoistPretzel", "SoggyCracker", "SquishyChip",
    "MushyNugget", "LumpyMaster", "BumpyLord", "FartingKing", "BurpingQueen",
    "SneezingPrince", "CoughingPrincess", "SnoringDuke", "DroolingDuchess",
    "SweatingWizard", "CryingWitch", "LaughingMage", "GigglingSorcerer",
    "SnortingKnight", "WheezingWarrior", "PantingHero", "GaspingVillain",
    "EatingFace", "DrinkingHead", "SleepingNose", "RunningEars", "JumpingToes",
    "FallingFingers", "TrippingBelly", "SlippingButt", "SlidingFeet", "RollingPants",
    "SpinningSocks", "DancingShoes", "SingingHat", "HummingShirt", "FartingUnderwear",
    "BurpingGloves", "SneezingMuffin", "CoughingPancake", "SnoringWaffle",
    "DroolingBagel", "SweatingPretzel", "CryingCracker", "LaughingChip",
    "GigglingNugget", "SnortingMaster", "WheezingLord", "PantingKing",
    "GaspingQueen", "EatingPrince", "DrinkingPrincess", "SleepingDuke",
    "RunningDuchess", "JumpingWizard", "FallingWitch", "TrippingMage",
    "SlippingSorcerer", "SlidingKnight", "RollingWarrior", "SpinningHero",
    "DancingVillain", "SingingFace", "HummingHead", "FartingNose", "BurpingEars",
    "SneezingToes", "CoughingFingers", "SnoringBelly", "DroolingButt",
    "SweatingFeet", "CryingPants", "LaughingSocks", "GigglingShoes",
    "SnortingHat", "WheezingShirt", "PantingUnderwear", "GaspingGloves"
]

# Filter to only include names 12 characters or less
TUTORIAL_NAME_COMBOS = [name for name in TUTORIAL_NAME_COMBOS if len(name) <= 12]
