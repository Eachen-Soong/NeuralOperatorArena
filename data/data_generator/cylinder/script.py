from data_helper import single_data_gen
import json
import os

def save_config_to_json(config, output_dir='my_data'):
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, 'config.json')
    with open(output_file, 'w') as f:
        json.dump(config, f, indent=4)
    print(f"Config saved to {output_file}")
    
config = {
    "name": "flow_simulation",
    "nx": 201,
    "ny": 41,
    "nu": 0.01,
    "boundary": {
        "types": [0, 0, 1, 0],  # Dirichlet on left and right, Neumann on top and bottom.
        "values": [[0.1, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]]  # Velocity values for Dirichlet boundaries.
    },
    "mask": {
        "size": 10,
        "shape_counts": {"circle": 1}  # 3 circles and 2 squares on the mask.
    },
    "time": {
        "delta_t": 1.0,
        "interval": 20,
        "max_timestep": 10000
    }
}

# Save the config to the 'my_data' folder
save_config_to_json(config, output_dir='train_data')

# Define shape configurations
shape_configs = [
    {'circle': 1},
    {'square': 1},
    {'triangle': 1},
    {'circle': 1, 'square': 1},
    {'circle': 1, 'triangle': 1},
    {'square': 1, 'triangle': 1},
    {'circle': 1, 'square': 1, 'triangle': 1}
]

n_train = 64

n_test = 16

# Generate data for each shape configuration and seed
for shape_counts in shape_configs:
    config["mask"]["shape_counts"] = shape_counts
    for seed in range(0, 1024, 16):
        single_data_gen(config, output_dic='train_data', seed=seed)
    
for shape_counts in shape_configs:
    config["mask"]["shape_counts"] = shape_counts
    for seed in range(1024, 1280, 16):
        single_data_gen(config, output_dic='test_data', seed=seed)
    
