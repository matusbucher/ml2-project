import os
import torch

from data_generation import *
from transformer import *
from training import *
from visualize import *


RESULTS_DIR = "results"
SAVED_MODELS_DIR = "saved_models"

# RULES = [30, 90, 110, 184]
RULES = [30]

TRAIN_SEQ_LEN = 32
TEST_SEQ_LENS = [16, 32, 64, 128]

TRAIN_STEPS = 1
TRAIN_SIZE = 10000
TEST_SIZE = 1000
BATCH_SIZE = 128

D_MODEL = 64
N_HEADS = 4
N_LAYERS = 2
D_FF = 128
DROPOUT = 0.0

N_EPOCHS = 20
LR = 1e-3


def test_generalization(
    save_models: bool = True,
    load_models: bool = False,
    save_history: bool = True,
    show_loss_history: bool = True,
    show_eval_history: bool = True,
    show_attention: bool = True,
    print_eval: bool = True,
    random_seed: int | None = None,
):
    if random_seed is not None:
        torch.manual_seed(random_seed)
        np.random.seed(random_seed)

    save_results_dir = f"{RESULTS_DIR}/generalization_length"
    if not os.path.exists(save_results_dir):
        os.makedirs(save_results_dir)
    
    save_models_dir = f"{SAVED_MODELS_DIR}/generalization_length"
    if save_models and not os.path.exists(save_models_dir):
        os.makedirs(save_models_dir)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"

    histories = []

    for rule in RULES:
        model = CATransformer(
            d_model=D_MODEL,
            n_heads=N_HEADS,
            n_layers=N_LAYERS,
            d_ff=D_FF,
            dropout=DROPOUT,
        )

        train_ds = CADataset(TRAIN_SIZE, TRAIN_SEQ_LEN, rule_number=rule, steps=TRAIN_STEPS)
        test_ds = {
            seq_len: CADataset(TEST_SIZE, seq_len, rule_number=rule, steps=TRAIN_STEPS)
            for seq_len in TEST_SEQ_LENS
        }

        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
        test_loaders = {
            seq_len: DataLoader(ds, batch_size=BATCH_SIZE)
            for seq_len, ds in test_ds.items()
        }

        if load_models:
            model = load_model(
                model=model,
                load_path=f"{save_models_dir}/rule_{rule}.pt",
                device=device,
            )
        else:
            history = train(
                model=model,
                train_loader=train_loader,
                device=device,
                n_epochs=N_EPOCHS,
                lr=LR,
            )
            histories.append(history)

            if save_models:
                save_model(
                    model=model,
                    save_path=f"{save_models_dir}/model_rule_{rule}.pt"
                )
            
            if save_history:
                history.save(
                    save_path=f"{save_models_dir}/history_rule_{rule}.pt"
                )
            
            if show_loss_history:
                visualize_loss_history(
                    history=history,
                    title=f"Loss history: rule {rule}",
                    save_path=f"{save_results_dir}/loss_history_rule_{rule}.png"
                )
            
            if show_eval_history:
                visualize_metrics_history(
                    history=history,
                    title=f"Evaluation metrics history: rule {rule}",
                    save_path=f"{save_results_dir}/eval_history_rule_{rule}.png"
                )

        if print_eval:
            train_metrics = evaluate(model, train_loader, device)
            test_metrics = {
                seq_len: evaluate(model, loader, device)
                for seq_len, loader in test_loaders.items()
            }
            print("=" * 40)
            print(f"RULE {rule}")
            print(f"Train cell accuracy: {train_metrics.cell_accuracy:.4f}, sequence accuracy: {train_metrics.sequence_accuracy:.4f}")
            for seq_len, metrics in test_metrics.items():
                print(f"Test (seq_len={seq_len}) cell accuracy: {metrics.cell_accuracy:.4f}, sequence accuracy: {metrics.sequence_accuracy:.4f}")
            print("=" * 40)
    
    if show_loss_history:
        visualize_multiple_loss_histories(
            histories={f"Rule {rule}": h for rule, h in zip(RULES, histories)},
            title="Loss history comparison",
            save_path=f"{save_results_dir}/loss_history_comparison.png"
        )

    if show_eval_history:
        visualize_multiple_metrics_histories(
            histories={f"Rule {rule}": h for rule, h in zip(RULES, histories)},
            title="Evaluation metrics history comparison",
            save_path=f"{save_results_dir}/eval_history_comparison.png"
        )