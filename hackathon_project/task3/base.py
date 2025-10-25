import pandas as pd
import numpy as np

# Resources
resources = ["Wood", "Brick", "Wheat", "Ore", "Sheep"]

# Actions / costs
actions = {
    "Settlement": {"Wood":1,"Brick":1,"Wheat":1,"Sheep":1,"Points":1},
    "Road": {"Wood":1,"Brick":1,"Points":0.5},
    "City": {"Wheat":2,"Ore":3,"Points":2},
    "Resource Card": {"Wheat":1,"Ore":1,"Sheep":1,"Points":1}
}

# Randomly generate resource production amounts for this "turn" or scenario
#np.random.seed(42)
resource_availability = {res: np.random.randint(1,5) for res in resources}

# Convert actions to DataFrame for convenience
df_actions = pd.DataFrame(actions).T.fillna(0)
df_actions = df_actions[resources + ["Points"]]
df_actions["Points"] = df_actions["Points"].astype(float)

print("Available Resources:")
print(resource_availability)
print("\nActions Table:")
print(df_actions)
