#!/usr/bin/env python3
"""
Item ID Mappings for RuneScape

This module provides mappings from RuneScape item IDs to human-readable names,
specifically focused on items relevant to sapphire ring crafting.
"""

# Common crafting materials and tools
CRAFTING_MATERIALS = {
    # Gems
    1607: "Uncut sapphire",
    1609: "Uncut emerald", 
    1611: "Uncut ruby",
    1613: "Uncut diamond",
    1615: "Uncut dragonstone",
    1617: "Uncut onyx",
    
    # Cut gems
    1608: "Sapphire",
    1610: "Emerald",
    1612: "Ruby", 
    1614: "Diamond",
    1616: "Dragonstone",
    1618: "Onyx",
    
    # Metal bars
    2349: "Gold bar",
    2351: "Steel bar",
    2353: "Iron bar",
    2355: "Bronze bar",
    2357: "Silver bar",
    2359: "Mithril bar",
    2361: "Adamantite bar",
    2363: "Runite bar",
    
    # Ores
    436: "Tin ore",
    438: "Copper ore", 
    440: "Iron ore",
    442: "Silver ore",
    444: "Gold ore",
    447: "Mithril ore",
    449: "Adamantite ore",
    451: "Runite ore",
    
    # Logs
    1511: "Logs",
    1513: "Oak logs",
    1515: "Willow logs",
    1517: "Maple logs",
    1519: "Yew logs",
    1521: "Magic logs",
    
    # Crafting tools
    1735: "Chisel",
    1755: "Hammer",
    946: "Knife",
    590: "Tinderbox",
    
    # Molds
    1592: "Ring mould",
    1595: "Necklace mould",
    1597: "Amulet mould",
    11065: "Bracelet mould",
    11069: "Tiara mould",
}

# Jewelry items
JEWELRY_ITEMS = {
    # Rings
    1635: "Sapphire ring",
    1637: "Emerald ring",
    1639: "Ruby ring",
    1641: "Diamond ring",
    1643: "Dragonstone ring",
    1645: "Onyx ring",
    
    # Necklaces
    1654: "Sapphire necklace",
    1656: "Emerald necklace", 
    1658: "Ruby necklace",
    1660: "Diamond necklace",
    1662: "Dragonstone necklace",
    1664: "Onyx necklace",
    
    # Amulets
    1673: "Sapphire amulet",
    1675: "Emerald amulet",
    1677: "Ruby amulet", 
    1679: "Diamond amulet",
    1681: "Dragonstone amulet",
    1683: "Onyx amulet",
    
    # Bracelets
    11069: "Sapphire bracelet",
    11072: "Emerald bracelet",
    11074: "Ruby bracelet",
    11076: "Diamond bracelet",
    11079: "Dragonstone bracelet",
    11085: "Onyx bracelet",
    
    # Tiaras
    5525: "Sapphire tiara",
    5527: "Emerald tiara",
    5529: "Ruby tiara",
    5531: "Diamond tiara", 
    5533: "Dragonstone tiara",
    5535: "Onyx tiara",
}

# Bank interface items
BANK_ITEMS = {
    # Bank quantity modes
    1: "Quantity 1",
    5: "Quantity 5", 
    10: "Quantity 10",
    -1: "Quantity X",  # Custom quantity
    -2: "Quantity All",
}

# Common inventory items
COMMON_ITEMS = {
    # Food
    379: "Lobster",
    385: "Shark",
    391: "Manta ray",
    397: "Monkfish",
    403: "Trout",
    405: "Salmon",
    407: "Tuna",
    409: "Lobster",
    411: "Bass",
    413: "Swordfish",
    
    # Potions
    121: "Attack potion",
    123: "Antipoison",
    125: "Strength potion",
    127: "Restore potion",
    129: "Energy potion",
    131: "Defence potion",
    133: "Prayer potion",
    135: "Super attack",
    137: "Super strength",
    139: "Super defence",
    
    # Equipment
    1167: "Adamant full helm",
    1169: "Adamant platebody",
    1171: "Adamant platelegs",
    1173: "Adamant plateskirt",
    1175: "Adamant kiteshield",
    1177: "Adamant med helm",
    1179: "Adamant chainbody",
    1181: "Adamant sq shield",
    1183: "Adamant platelegs",
    1185: "Adamant plateskirt",
    
    # Quest items
    2434: "Silverlight",
    2436: "Darklight",
    2438: "Holy grail",
    2440: "Magic whistle",
    2442: "Ground guam",
    2444: "Ground marrentill",
    2446: "Ground tarromin",
    2448: "Ground harralander",
    2450: "Ground ranarr weed",
    2452: "Ground irit leaf",
    
    # Additional items found in your data
    -1: "Quantity X",  # Bank quantity mode
    1: "Quantity 1",   # Bank quantity mode
    5: "Quantity 5",   # Bank quantity mode
    10: "Quantity 10", # Bank quantity mode
    -2: "Quantity All", # Bank quantity mode
}

# Game objects (trees, rocks, etc.)
GAME_OBJECTS = {
    # Trees
    1276: "Tree",
    1277: "Dead tree",
    1278: "Evergreen",
    1279: "Jungle tree",
    1280: "Oak tree",
    1281: "Willow tree",
    1282: "Maple tree",
    1283: "Yew tree",
    1284: "Magic tree",
    1285: "Mahogany tree",
    
    # Rocks
    10943: "Rocks",
    10944: "Copper rocks",
    10945: "Tin rocks", 
    10946: "Iron rocks",
    10947: "Silver rocks",
    10948: "Gold rocks",
    10949: "Mithril rocks",
    10950: "Adamantite rocks",
    10951: "Runite rocks",
    
    # Furnaces and anvils
    164: "Furnace",
    165: "Anvil",
    166: "Range",
    167: "Fire",
    168: "Ladder",
    169: "Stairs",
    170: "Door",
    171: "Gate",
    172: "Wall",
    173: "Fence",
    
    # Banks and shops
    2213: "Bank booth",
    3045: "Bank chest",
    5276: "Bank deposit box",
    11402: "Bank booth (Grand Exchange)",
    11403: "Bank booth (Grand Exchange)",
    11404: "Bank booth (Grand Exchange)",
    11405: "Bank booth (Grand Exchange)",
    11406: "Bank booth (Grand Exchange)",
    11407: "Bank booth (Grand Exchange)",
    11408: "Bank booth (Grand Exchange)",
    
    # Additional objects found in your data
    10355: "Game object (10355)",  # Common object in your data
    3098: "Game object (3098)",    # Common object in your data
    2119: "Game object (2119)",    # Common object in your data
    16469: "Game object (16469)",  # Common object in your data
    3498: "Game object (3498)",    # Common object in your data
    10529: "Game object (10529)",  # Common object in your data
    16474: "Game object (16474)",  # Common object in your data
    3110: "Game object (3110)",    # Common object in your data
    85020: "Game object (85020)",  # Common object in your data
    2: "Game object (2)",          # Common object in your data
    88381: "Game object (88381)",  # Common object in your data
    1640: "Game object (1640)",    # Common object in your data
    3095: "Game object (3095)",    # Common object in your data
    3493: "Game object (3493)",    # Common object in your data
    3491: "Game object (3491)",    # Common object in your data
}

# Combine all mappings
ALL_ITEM_MAPPINGS = {
    **CRAFTING_MATERIALS,
    **JEWELRY_ITEMS,
    **BANK_ITEMS,
    **COMMON_ITEMS,
    **GAME_OBJECTS,
}

def get_item_name(item_id: int) -> str:
    """
    Get the human-readable name for an item ID.
    
    Args:
        item_id: The RuneScape item ID
        
    Returns:
        The item name if found, otherwise f"Unknown Item ({item_id})"
    """
    return ALL_ITEM_MAPPINGS.get(item_id, f"Unknown Item ({item_id})")

def get_item_category(item_id: int) -> str:
    """
    Get the category of an item based on its ID.
    
    Args:
        item_id: The RuneScape item ID
        
    Returns:
        The item category
    """
    if item_id in CRAFTING_MATERIALS:
        return "Crafting Material"
    elif item_id in JEWELRY_ITEMS:
        return "Jewelry"
    elif item_id in BANK_ITEMS:
        return "Bank Interface"
    elif item_id in COMMON_ITEMS:
        return "Common Item"
    elif item_id in GAME_OBJECTS:
        return "Game Object"
    else:
        return "Unknown"

def get_relevant_items_for_crafting() -> dict:
    """
    Get items specifically relevant to sapphire ring crafting.
    
    Returns:
        Dictionary mapping item IDs to names for crafting-related items
    """
    return {
        # Essential materials
        1607: "Uncut sapphire",
        1608: "Sapphire", 
        2349: "Gold bar",
        1592: "Ring mould",
        1735: "Chisel",
        
        # Final product
        1635: "Sapphire ring",
        
        # Related items
        1609: "Uncut emerald",
        1610: "Emerald",
        1637: "Emerald ring",
        1611: "Uncut ruby",
        1612: "Ruby", 
        1639: "Ruby ring",
    }

def format_inventory_slot(slot_id: int, slot_quantity: int) -> str:
    """
    Format an inventory slot for display.
    
    Args:
        slot_id: The item ID in the slot
        slot_quantity: The quantity of items
        
    Returns:
        Formatted string describing the slot contents
    """
    if slot_id == 0:
        return "Empty"
    
    item_name = get_item_name(slot_id)
    if slot_quantity > 1:
        return f"{item_name} x{slot_quantity}"
    else:
        return item_name

def format_game_object(obj_id: int, distance: float, x_pos: float) -> str:
    """
    Format a game object for display.
    
    Args:
        obj_id: The object ID
        distance: Distance to the object
        x_pos: X position of the object
        
    Returns:
        Formatted string describing the object
    """
    obj_name = get_item_name(obj_id)
    return f"{obj_name} (dist: {distance:.1f}, x: {x_pos:.1f})"
