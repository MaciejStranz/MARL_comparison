import os
from pathlib import Path
from typing import List
import json
from benchmarl.eval_results import load_and_merge_json_dicts, Plotting

from matplotlib import pyplot as plt


experiments_json_files = [r"simple_spread_wyniki\1\qmix_simple_spread_mlp__071e685b_25_01_25-12_48_48\qmix_simple_spread_mlp__071e685b_25_01_25-12_48_48.json",
                          r"simple_spread_wyniki\1\qmix_simple_spread_mlp__ebf9fbf9_25_01_25-15_42_39\qmix_simple_spread_mlp__ebf9fbf9_25_01_25-15_42_39.json",
                          r"simple_spread_wyniki\1\vdn_simple_spread_mlp__8bef270b_25_01_25-10_09_54\vdn_simple_spread_mlp__8bef270b_25_01_25-10_09_54.json",
                          r"simple_spread_wyniki\1\vdn_simple_spread_mlp__12e2eb4a_25_01_25-07_30_42\vdn_simple_spread_mlp__12e2eb4a_25_01_25-07_30_42.json"
                          ]
raw_dict = load_and_merge_json_dicts(experiments_json_files)

output_dir = "wykresy/simple_spread/1"
os.makedirs(output_dir, exist_ok=True)  

processed_data = Plotting.process_data(raw_dict)
(
    environment_comparison_matrix,
    sample_efficiency_matrix,
) = Plotting.create_matrices(processed_data, env_name="pettingzoo")

# Plotting
Plotting.performance_profile_figure(
    environment_comparison_matrix=environment_comparison_matrix
)
plt.subplots_adjust(bottom=0.156)
plt.savefig(os.path.join(output_dir, "performance_profile.png"), dpi=300)

Plotting.aggregate_scores(
    environment_comparison_matrix=environment_comparison_matrix
)
fig = plt.gcf()                    
fig.set_size_inches(16,4)        
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "aggregate_scores.png"), dpi=300)

Plotting.environemnt_sample_efficiency_curves(
    sample_effeciency_matrix=sample_efficiency_matrix
)
plt.savefig(os.path.join(output_dir, "sample_efficiency.png"), dpi=300)

Plotting.task_sample_efficiency_curves(
    processed_data=processed_data, env="pettingzoo", task="simple_spread"
)
plt.savefig(os.path.join(output_dir, "task_sample_efficiency.png"), dpi=300)

Plotting.probability_of_improvement(
    environment_comparison_matrix,
    algorithms_to_compare=[["qmix", "vdn"]],
)
fig = plt.gcf()                  
fig.set_size_inches(16,10)         
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "probability_of_improvement.png"), dpi=300)

plt.show()