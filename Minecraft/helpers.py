import numpy as np
import cv2

KEYWORDS = [
    'log',
    'planks',
    'stick',
    'crafting_table',
    'stone',
    'dirt',
    'cobblestone',
    'wooden_pickaxe',
    'stone_pickaxe',
    'iron_pickaxe',
    'coal',
    'iron_ore',
    'torch',
    'iron_ingot',
    'furnace'
]

def find_inventory_activity(inventory_list):
    activity_steps = []
    prev = inventory_list[0]
    prev_step = 0
    prev_resource = None
    first_change = True

    def matches_target(key):
        return any(keyword in key for keyword in KEYWORDS)

    for i in range(1, len(inventory_list)):
        curr = inventory_list[i]
        for key in curr:
            if matches_target(key) and curr[key] > prev.get(key, 0):
                if first_change:
                    activity_steps.extend([key] * i)
                    first_change = False
                else:
                    activity_steps.extend([prev_resource] * (i - prev_step))
                prev_resource = key
                prev_step = i
        prev = curr

    if prev_resource is not None:
        activity_steps.extend([prev_resource] * (len(inventory_list) - prev_step))
    
    # Use map_material to format the strigns
    activity_steps = [map_material(item) for item in activity_steps]

    return "\n".join(activity_steps)

def map_material(material):
    material = material.lower()
    
    if 'log' in material:
        return 'log'
    elif 'planks' in material:
        return 'planks'
    else:
        return material


def resize_and_format(obs, size=160):
    pov_bgr = cv2.cvtColor(obs, cv2.COLOR_RGB2BGR)
    pov_resized = cv2.resize(pov_bgr, (size, size), interpolation=cv2.INTER_LINEAR) #INTER_AREA or INTER_LINEAR(studio)
    pov_rgb = cv2.cvtColor(pov_resized, cv2.COLOR_BGR2RGB) 
    return pov_rgb
