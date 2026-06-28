# main.py
from src.experiment import run_experiment, print_results_table

results = run_experiment()
print_results_table(results)