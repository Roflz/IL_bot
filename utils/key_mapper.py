#!/usr/bin/env python3
"""
Comprehensive Keyboard Key Mapping System for OSRS Automation

This module provides a unified system for mapping keyboard keys to numerical values
without making assumptions about their game function. Covers the entire keyboard layout.
"""

class KeyboardKeyMapper:
    """Comprehensive keyboard key mapping system - no assumptions about game function"""
    
    # Standard keyboard layout mapping
    KEY_MAPPING = {
        # Letters (a-z) - map to 1-26
        'a': 1.0, 'b': 2.0, 'c': 3.0, 'd': 4.0, 'e': 5.0, 'f': 6.0, 'g': 7.0, 'h': 8.0,
        'i': 9.0, 'j': 10.0, 'k': 11.0, 'l': 12.0, 'm': 13.0, 'n': 14.0, 'o': 15.0, 'p': 16.0,
        'q': 17.0, 'r': 18.0, 's': 19.0, 't': 20.0, 'u': 21.0, 'v': 22.0, 'w': 23.0, 'x': 24.0,
        'y': 25.0, 'z': 26.0,
        
        # Numbers (0-9) - map to 30-39
        '0': 30.0, '1': 31.0, '2': 32.0, '3': 33.0, '4': 34.0,
        '5': 35.0, '6': 36.0, '7': 37.0, '8': 38.0, '9': 39.0,
        
        # Function keys (F1-F12) - map to 40-51
        'f1': 40.0, 'f2': 41.0, 'f3': 42.0, 'f4': 43.0, 'f5': 44.0, 'f6': 45.0,
        'f7': 46.0, 'f8': 47.0, 'f9': 48.0, 'f10': 49.0, 'f11': 50.0, 'f12': 51.0,
        
        # Arrow keys - map to 60-63
        'up': 60.0, 'down': 61.0, 'left': 62.0, 'right': 63.0,
        
        # Navigation keys - map to 70-79
        'home': 70.0, 'end': 71.0, 'pageup': 72.0, 'pagedown': 73.0,
        'insert': 74.0, 'delete': 75.0, 'backspace': 76.0, 'tab': 77.0,
        'enter': 78.0, 'escape': 79.0,
        
        # Modifier keys - map to 80-99
        'shift': 80.0, 'ctrl': 81.0, 'alt': 82.0, 'capslock': 83.0,
        'numlock': 84.0, 'scrolllock': 85.0, 'win': 86.0, 'menu': 87.0,
        'left_shift': 88.0, 'right_shift': 89.0, 'left_ctrl': 90.0, 'right_ctrl': 91.0,
        'left_alt': 92.0, 'right_alt': 93.0,
        
        # Punctuation and symbols - map to 100-199
        'space': 100.0, '`': 101.0, '-': 102.0, '=': 103.0, '[': 104.0, ']': 105.0,
        '\\': 106.0, ';': 107.0, "'": 108.0, ',': 109.0, '.': 110.0, '/': 111.0,
        '~': 112.0, '!': 113.0, '@': 114.0, '#': 115.0, '$': 116.0, '%': 117.0,
        '^': 118.0, '&': 119.0, '*': 120.0, '(': 121.0, ')': 122.0, '_': 123.0,
        '+': 124.0, '{': 125.0, '}': 126.0, '|': 127.0, ':': 128.0, '"': 129.0,
        '<': 130.0, '>': 131.0, '?': 132.0,
        
        # Numpad keys - map to 200-299
        'num0': 200.0, 'num1': 201.0, 'num2': 202.0, 'num3': 203.0, 'num4': 204.0,
        'num5': 205.0, 'num6': 206.0, 'num7': 207.0, 'num8': 208.0, 'num9': 209.0,
        'num+': 210.0, 'num-': 211.0, 'num*': 212.0, 'num/': 213.0, 'num.': 214.0,
        'numenter': 215.0,
        
        # Media keys - map to 300-399
        'volumeup': 300.0, 'volumedown': 301.0, 'volumemute': 302.0,
        'play': 303.0, 'pause': 304.0, 'stop': 305.0, 'next': 306.0, 'prev': 307.0,
        'browser_back': 308.0, 'browser_forward': 309.0, 'browser_refresh': 310.0,
        'browser_stop': 311.0, 'browser_search': 312.0, 'browser_favorites': 313.0,
        'browser_home': 314.0,
        
        # Special keys - map to 400-499
        'printscreen': 400.0, 'scrolllock': 401.0, 'pause': 402.0,
        'break': 403.0, 'sysrq': 404.0, 'app': 405.0
    }
    
    @classmethod
    def map_key_to_number(cls, key: str) -> float:
        """Map a key string to a unique numerical value"""
        if not key:
            return 0.0
        
        # Check if it's a known key in our mapping
        if key in cls.KEY_MAPPING:
            return cls.KEY_MAPPING[key]
        
        # Handle single characters not in our mapping
        if len(key) == 1:
            char_code = ord(key)
            # Map to 500+ range to avoid conflicts with known keys
            return 500.0 + char_code
        
        # Handle longer strings (unknown keys)
        # Use a hash-based approach for unknown keys
        hash_value = hash(key)
        # Map to 1000+ range to avoid conflicts with known keys
        return 1000.0 + (abs(hash_value) % 1000)
    
    @classmethod
    def get_key_mapping_info(cls) -> dict:
        """Get information about the key mapping system for debugging"""
        return {
            'total_mapped_keys': len(cls.KEY_MAPPING),
            'letter_keys': [chr(i) for i in range(ord('a'), ord('z')+1)],
            'number_keys': [str(i) for i in range(10)],
            'function_keys': [f'f{i}' for i in range(1, 13)],
            'arrow_keys': ['up', 'down', 'left', 'right'],
            'navigation_keys': ['home', 'end', 'pageup', 'pagedown', 'insert', 'delete', 'backspace', 'tab', 'enter', 'escape'],
            'modifier_keys': ['shift', 'ctrl', 'alt', 'capslock', 'numlock', 'scrolllock', 'win', 'menu'],
            'punctuation_keys': ['space', '`', '-', '=', '[', ']', '\\', ';', "'", ',', '.', '/'],
            'numpad_keys': ['num0', 'num1', 'num2', 'num3', 'num4', 'num5', 'num6', 'num7', 'num8', 'num9', 'num+', 'num-', 'num*', 'num/', 'num.', 'numenter'],
            'media_keys': ['volumeup', 'volumedown', 'volumemute', 'play', 'pause', 'stop', 'next', 'prev'],
            'browser_keys': ['browser_back', 'browser_forward', 'browser_refresh', 'browser_stop', 'browser_search', 'browser_favorites', 'browser_home'],
            'special_keys': ['printscreen', 'scrolllock', 'pause', 'break', 'sysrq', 'app']
        }
    
    @classmethod
    def get_key_name(cls, value: float) -> str:
        """Get the key name for a given numerical value (reverse lookup)"""
        for key, val in cls.KEY_MAPPING.items():
            if val == value:
                return key
        return f"Unknown Key ({value})"
    
    @classmethod
    def print_mapping_summary(cls):
        """Print a summary of the key mapping system"""
        info = cls.get_key_mapping_info()
        print("Keyboard Key Mapping System Summary:")
        print(f"Total mapped keys: {info['total_mapped_keys']}")
        print(f"Letter keys (a-z): {len(info['letter_keys'])} keys mapped to 1.0-26.0")
        print(f"Number keys (0-9): {len(info['number_keys'])} keys mapped to 30.0-39.0")
        print(f"Function keys (F1-F12): {len(info['function_keys'])} keys mapped to 40.0-51.0")
        print(f"Arrow keys: {len(info['arrow_keys'])} keys mapped to 60.0-63.0")
        print(f"Navigation keys: {len(info['navigation_keys'])} keys mapped to 70.0-79.0")
        print(f"Modifier keys: {len(info['modifier_keys'])} keys mapped to 80.0-99.0")
        print(f"Punctuation keys: {len(info['punctuation_keys'])} keys mapped to 100.0-199.0")
        print(f"Numpad keys: {len(info['numpad_keys'])} keys mapped to 200.0-299.0")
        print(f"Media keys: {len(info['media_keys'])} keys mapped to 300.0-399.0")
        print(f"Special keys: {len(info['special_keys'])} keys mapped to 400.0-499.0")
        print("Single characters not in mapping: mapped to 500.0+ range")
        print("Unknown longer strings: mapped to 1000.0+ range using hash")

if __name__ == "__main__":
    # Test the key mapper
    mapper = KeyboardKeyMapper()
    mapper.print_mapping_summary()
    
    # Test some key mappings
    test_keys = ['a', 'w', '1', 'f1', 'space', 'enter', 'shift', 'up']
    print("\nTest key mappings:")
    for key in test_keys:
        value = mapper.map_key_to_number(key)
        print(f"  '{key}' -> {value}")
    
    # Test reverse lookup
    print("\nTest reverse lookup:")
    for key in test_keys:
        value = mapper.map_key_to_number(key)
        name = mapper.get_key_name(value)
        print(f"  {value} -> '{name}'")
