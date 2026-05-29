import os
import torch

from data_generation import *
from transformer import *
from visualize import *


RANDOM_SEED = 42

SHOW_STEPS = 32
SHOW_N_STATES = 4

RESULTS_DIR = "results"
SAVED_MODELS_DIR = "saved_models"

RULES = [30, 90, 110, 184]

SEQ_LEN = 32
TRAIN_STEPS = 1
TRAIN_SIZE = 100000
TEST_SIZE = 10000
BATCH_SIZE = 128

D_MODEL = 64
N_HEADS = 4
N_LAYERS = 2
D_FF = 128
DROPOUT = 0.1

N_EPOCHS = 20
LR = 1e-3


def first_experiment(
    do_train: bool = True,
    save_models: bool = True,
    show_ca: bool = True,
    show_predictions: bool = True,
    show_loss_history: bool = True,
    show_eval_history: bool = True,
    print_eval: bool = True,
    random_seed: int | None = None,
):
    if random_seed is not None:
        torch.manual_seed(random_seed)

    save_dir = f"{RESULTS_DIR}/first_experiment"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    init_states = np.random.randint(0, 2, (SHOW_N_STATES, SEQ_LEN), dtype=np.uint8)

    if show_ca:
        for rule in RULES:
            visualize_ca_trajectories(
                rule_number=rule,
                init_states=init_states,
                steps=SHOW_STEPS,
                save_path=f"{save_dir}/ca_rule_{rule}.png"
            )
    
    if not do_train:
        return

    histories = []

    for rule in RULES:
        train_ds = CADataset(TRAIN_SIZE, SEQ_LEN, rule_number=rule, steps=TRAIN_STEPS)
        test_ds = CADataset(TEST_SIZE, SEQ_LEN, rule_number=rule, steps=TRAIN_STEPS)

        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
        test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE)

        model = CATransformer(
            seq_len=SEQ_LEN,
            d_model=D_MODEL,
            n_heads=N_HEADS,
            n_layers=N_LAYERS,
            d_ff=D_FF,
            dropout=DROPOUT,
        )

        history = train(
            model=model,
            data_loader=train_loader,
            device=device,
            n_epochs=N_EPOCHS,
            lr=LR,
        )
        histories.append(history)

        if save_models:
            save_model(
                model=model,
                save_path=f"{SAVED_MODELS_DIR}/rule_{rule}.pt"
            )

        if print_eval:
            train_metrics = evaluate(model, train_loader, device)
            test_metrics = evaluate(model, test_loader, device)
            print("=" * 40)
            print(f"RULE {rule}")
            print(f"Train cell accuracy: {train_metrics.cell_accuracy:.4f}, sequence accuracy: {train_metrics.sequence_accuracy:.4f}")
            print(f"Test cell accuracy: {test_metrics.cell_accuracy:.4f}, sequence accuracy: {test_metrics.sequence_accuracy:.4f}")
            print("=" * 40)

        if show_loss_history:
            visualize_loss_history(
                history=history,
                title=f"Loss history: rule {rule}",
                save_path=f"{save_dir}/loss_history_rule_{rule}.png"
            )
        
        if show_eval_history:
            visualize_metrics_history(
                history=history,
                title=f"Evaluation metrics history: rule {rule}",
                save_path=f"{save_dir}/eval_history_rule_{rule}.png"
            )

        if show_predictions:
            visualize_predictions(
                rule_number=rule,
                model=model.to("cpu"),
                init_states=init_states,
                steps=SHOW_STEPS,
                save_path=f"{save_dir}/predictions_rule_{rule}.png"
            )
    
    if show_loss_history:
        visualize_multiple_loss_histories(
            histories={f"Rule {rule}": h for rule, h in zip(RULES, histories)},
            title="Loss history comparison",
            save_path=f"{save_dir}/loss_history_comparison.png"
        )

    if show_eval_history:
        visualize_multiple_metrics_histories(
            histories={f"Rule {rule}": h for rule, h in zip(RULES, histories)},
            title="Evaluation metrics history comparison",
            save_path=f"{save_dir}/eval_history_comparison.png"
        )