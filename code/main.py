from data_generation import *
from transformer import *
from training import *
from visualize import *
from first_experiment import *
from generalize_length import *


RANDOM_SEED = 42


if __name__ == "__main__":
    # visualize_ca_rules(
    #     rule_numbers=[30, 90, 110, 184],
    #     init_state=cpl.init_simple(32),
    #     steps=SHOW_STEPS,
    #     save_path=f"{RESULTS_DIR}/ca_rules_comparison.png"
    # )

    # first_experiment(
    #     save_models=False,
    #     load_models=False,
    #     save_history=False,
    #     show_ca=False,
    #     show_predictions=False,
    #     show_loss_history=False,
    #     show_eval_history=False,
    #     show_attention=False,
    #     print_eval=True,
    #     random_seed=RANDOM_SEED,
    # )

    # generalize_length(
    #     emb_type=PositionalEmbeddingType.ROPE,
    #     save_models=False,
    #     load_models=False,
    #     save_history=False,
    #     show_loss_history=False,
    #     show_eval_history=False,
    #     show_cell_accuracy=False,
    #     print_eval=True,
    #     random_seed=RANDOM_SEED,
    # )

    print("Uncomment the desired function calls in main.py to run the experiments and visualizations.")