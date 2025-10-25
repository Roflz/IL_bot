# Skill Icons Setup

To display skill icons in the GUI, you need to save the skill icon images in this `skill_icons/` directory.

## Required Icon Files

Save your skill icon images with these exact filenames:

- `attack_sword.png` - Attack skill (sword icon)
- `strength_fist.png` - Strength skill (fist icon)
- `defence_shield.png` - Defence skill (shield icon)
- `hitpoints_heart.png` - Hitpoints skill (heart icon)
- `ranged_bow.png` - Ranged skill (bow icon)
- `prayer_star.png` - Prayer skill (star icon)
- `magic_wizard_hat.png` - Magic skill (wizard hat icon)
- `cooking.png` - Cooking skill
- `woodcutting_tree.png` - Woodcutting skill (tree icon)
- `fletching_arrows.png` - Fletching skill (arrows icon)
- `fishing_fish.png` - Fishing skill (fish icon)
- `firemaking_fire.png` - Firemaking skill (fire icon)
- `crafting.png` - Crafting skill
- `smithing_anvil.png` - Smithing skill (anvil icon)
- `mining_pickaxe.png` - Mining skill (pickaxe icon)
- `herblore_leaf.png` - Herblore skill (leaf icon)
- `agility_running.png` - Agility skill (running figure icon)
- `thieving_mask.png` - Thieving skill (mask icon)
- `slayer_skull.png` - Slayer skill (skull icon)
- `farming_paw.png` - Farming skill (paw print icon)
- `runecraft_quiver.png` - Runecraft skill (quiver icon)
- `hunter_watering_can.png` - Hunter skill (watering can icon)
- `construction_saw.png` - Construction skill (saw icon)

## Image Requirements

- Format: PNG (recommended) or any format supported by PIL/Pillow
- Size: Any size (will be automatically resized to 20x20 pixels)
- Background: Transparent or solid color (both work)
- Style: Pixel art icons work best, matching the RuneScape style

## How to Save Icons

1. Extract or crop the skill icons from your screenshots/images
2. Save each icon as a separate PNG file with the filename listed above
3. Place all files in the `skill_icons/` directory
4. The GUI will automatically load and display them

## Fallback Behavior

If an icon file is missing, the GUI will display the first letter of the skill name as a fallback (e.g., "A" for Attack, "W" for Woodcutting).

## Notes

- Icons are loaded once when the GUI starts
- If you add new icons, restart the GUI to see them
- Icons are automatically resized to 20x20 pixels for consistent display


