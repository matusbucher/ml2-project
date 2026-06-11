import os
import torch

from data_generation import *
from transformer import *
from training import *
from visualize import *


RESULTS_DIR = "results"
SAVED_MODELS_DIR = "saved_models"

RULES = [30, 90, 110, 184]

TRAIN_STEPS = [1, 2, 3, 4, 5, 6, 7, 8]

SEQ_LEN = 32
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


def load_history_generalize_steps() -> None:
    for steps in TRAIN_STEPS:
        histories = []
        for rule in RULES:
            history = TrainHistory.load(
                load_path=f"{SAVED_MODELS_DIR}/generalize_steps_{steps}/history_rule_{rule}.pt"
            )
            histories.append(history)
        
        visualize_multiple_loss_histories(
            histories={f"Rule {rule}": h for rule, h in zip(RULES, histories)},
            show_test_loss=False,
            title=f"Loss history comparison, steps={steps}",
            save_path=f"{RESULTS_DIR}/generalize_steps_{steps}/loss_history_comparison.png"
        )

        visualize_multiple_metrics_histories(
            histories={f"Rule {rule}": h for rule, h in zip(RULES, histories)},
            title=f"Evaluation metrics history comparison, steps={steps}",
            save_path=f"{RESULTS_DIR}/generalize_steps_{steps}/eval_history_comparison.png"
        )


def visualize_average_attention_generalize_steps() -> None:
    for steps in TRAIN_STEPS:
        for rule in RULES:
            model = CATransformer(
                emb_type=PositionalEmbeddingType.LEARNED,
                seq_len=SEQ_LEN,
                d_model=D_MODEL,
                n_heads=N_HEADS,
                n_layers=N_LAYERS,
                d_ff=D_FF,
                dropout=DROPOUT,
            )
            model = load_model(
                model=model,
                load_path=f"{SAVED_MODELS_DIR}/generalize_steps_{steps}/model_rule_{rule}.pt",
                device="cpu",
            )
            test_ds = CADataset(TEST_SIZE, SEQ_LEN, rule_number=rule, steps=steps)
            test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE)

            layer_attentions = model.get_average_attention(test_loader, device="cpu")
            stacked = torch.stack(layer_attentions)
            avg_attention = stacked.mean(dim=(0, 1))

            visualize_single_attention(
                attention_weight=avg_attention,
                title=f"Average attention weights, steps={steps}, rule {rule}",
                save_path=f"{RESULTS_DIR}/generalize_steps_{steps}/average_attention_rule_{rule}.png"
            )


def visualize_ca_generalize_steps() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    all_test_metrics = []

    for rule in RULES:
        test_metrics = {}
        for steps in TRAIN_STEPS:
            model = CATransformer(
                emb_type=PositionalEmbeddingType.LEARNED,
                seq_len=SEQ_LEN,
                d_model=D_MODEL,
                n_heads=N_HEADS,
                n_layers=N_LAYERS,
                d_ff=D_FF,
                dropout=DROPOUT,
            )
            model = load_model(
                model=model,
                load_path=f"{SAVED_MODELS_DIR}/generalize_steps_{steps}/model_rule_{rule}.pt",
                device="cpu",
            )
            test_ds = CADataset(TEST_SIZE, SEQ_LEN, rule_number=rule, steps=steps)
            test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE)

            test_metrics[steps] = evaluate(model, test_loader, device)
        
        visualize_cell_accuracy(
            eval_metrics=test_metrics,
            x_label="Number of steps",
            title=f"Cell accuracy for different number of steps: rule {rule}",
            save_path=f"{RESULTS_DIR}/generalize_steps/cell_accuracy_rule_{rule}.png"
        )
        all_test_metrics.append(test_metrics)
    
    visualize_multiple_cell_accuracies(
        all_eval_metrics={f"Rule {rule}": m for rule, m in zip(RULES, all_test_metrics)},
        x_label="Number of steps",
        title="Cell accuracy comparison for different number of steps",
        save_path=f"{RESULTS_DIR}/generalize_steps/cell_accuracy_comparison.png"
    )

def generalize_steps(
    save_models: bool = True,
    load_models: bool = False,
    save_history: bool = True,
    show_loss_history: bool = True,
    show_eval_history: bool = True,
    show_attention: bool = True,
    print_eval: bool = True,
    random_seed: int | None = None,
) -> None:
    if random_seed is not None:
        torch.manual_seed(random_seed)
        np.random.seed(random_seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    for steps in TRAIN_STEPS:
        save_results_dir = f"{RESULTS_DIR}/generalize_steps_{steps}"
        if not os.path.exists(save_results_dir):
            os.makedirs(save_results_dir)
        
        save_models_dir = f"{SAVED_MODELS_DIR}/generalize_steps_{steps}"
        if save_models and not os.path.exists(save_models_dir):
            os.makedirs(save_models_dir)
        
        histories = []

        for rule in RULES:
            model = CATransformer(
                emb_type=PositionalEmbeddingType.LEARNED,
                seq_len=SEQ_LEN,
                d_model=D_MODEL,
                n_heads=N_HEADS,
                n_layers=N_LAYERS,
                d_ff=D_FF,
                dropout=DROPOUT,
            )

            train_ds = CADataset(TRAIN_SIZE, SEQ_LEN, rule_number=rule, steps=steps)
            test_ds = CADataset(TEST_SIZE, SEQ_LEN, rule_number=rule, steps=steps)

            train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
            test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE)

            if load_models:
                model = load_model(
                    model=model,
                    load_path=f"{save_models_dir}/model_rule_{rule}.pt",
                    device=device,
                )
            else:
                history = train(
                    model=model,
                    train_loader=train_loader,
                    test_loader=test_loader,
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
                        show_test_loss=True,
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
                test_metrics = evaluate(model, test_loader, device)
                print("=" * 40)
                print(f"RULE {rule}")
                print(f"Train cell accuracy: {train_metrics.cell_accuracy:.4f}, sequence accuracy: {train_metrics.sequence_accuracy:.4f}")
                print(f"Test cell accuracy: {test_metrics.cell_accuracy:.4f}, sequence accuracy: {test_metrics.sequence_accuracy:.4f}")
                print("=" * 40)

            if show_attention:
                attention = model.get_average_attention(test_loader, device=device)
                visualize_attention(
                    attention_weights=attention,
                    layer_idx=0,
                    title=f"Average attention weights, first layer: rule {rule}",
                    save_path=f"{save_results_dir}/attention_layer_0_rule_{rule}.png"
                )
                visualize_attention(
                    attention_weights=attention,
                    layer_idx=1,
                    title=f"Average attention weights, second layer: rule {rule}",
                    save_path=f"{save_results_dir}/attention_layer_1_rule_{rule}.png"
                )

        if show_loss_history:
            visualize_multiple_loss_histories(
                histories={f"Rule {rule}": h for rule, h in zip(RULES, histories)},
                show_test_loss=True,
                title="Loss history comparison",
                save_path=f"{save_results_dir}/loss_history_comparison.png"
            )

        if show_eval_history:
            visualize_multiple_metrics_histories(
                histories={f"Rule {rule}": h for rule, h in zip(RULES, histories)},
                title="Evaluation metrics history comparison",
                save_path=f"{save_results_dir}/eval_history_comparison.png"
            )