import random

PART_A = [
    "Maple",
    "Cedar",
    "Aurora",
    "Nimbus",
    "Comet",
    "Jade",
    "Amber",
    "Echo",
    "Zephyr",
    "Quartz",
]
PART_B = [
    "Comet",
    "Spark",
    "Glide",
    "Quill",
    "Haven",
    "Ridge",
    "Reef",
    "Bloom",
    "Gale",
    "Shade",
]


def generate_name(used=None, numeric: int = None) -> str:
    used = used or set()
    for _ in range(100):
        name = f"{random.choice(PART_A)}-{random.choice(PART_B)}"
        if name not in used:
            break
    number = numeric if numeric is not None else random.randint(1, 9999)
    return f"{name}_{number:04d}"

