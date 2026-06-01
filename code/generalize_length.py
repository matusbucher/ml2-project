import os
import torch

from data_generation import *
from transformer import *
from training import *
from visualize import *


RESULTS_DIR = "results"
SAVED_MODELS_DIR = "saved_models"

RULES = [30, 90, 110, 184]

TRAIN_SEQ_MIN_LEN = 16
TRAIN_SEQ_MAX_LEN = 128
TEST_SEQ_LENS = [16 * i for i in range(1, 17)]

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


def load_history_generalize_length(
    demb_type: PositionalEmbeddingType,
) -> None:
    histories = []
    for rule in RULES:
        history = TrainHistory.load(
            load_path=f"{SAVED_MODELS_DIR}/generalize_length_{demb_type.value}/history_rule_{rule}.pt"
        )
        histories.append(history)
    
    visualize_multiple_loss_histories(
        histories={f"Rule {rule}": h for rule, h in zip(RULES, histories)},
        show_test_loss=False,
        title="Loss history comparison",
        save_path=f"{RESULTS_DIR}/generalize_length_{demb_type.value}/loss_history_comparison.png"
    )

    visualize_multiple_metrics_histories(
        histories={f"Rule {rule}": h for rule, h in zip(RULES, histories)},
        title="Evaluation metrics history comparison",
        save_path=f"{RESULTS_DIR}/generalize_length_{demb_type.value}/eval_history_comparison.png"
    )


def trim_collate_fn(
    batch: list[tuple[torch.Tensor, torch.Tensor]],
    ca: CellularAutomaton,
) -> tuple[torch.Tensor, torch.Tensor]:
    seq_len = torch.randint(TRAIN_SEQ_MIN_LEN, TRAIN_SEQ_MAX_LEN + 1, (1,)).item()

    xs = []
    ys = []

    for x, y in batch:
        assert x.size(0) >= seq_len
        assert y.size(0) >= seq_len

        x_trim = x[:seq_len]
        y_trim = ca.evolve(x_trim.numpy(), steps=TRAIN_STEPS)

        xs.append(x_trim)
        ys.append(torch.from_numpy(y_trim).long())

    return torch.stack(xs), torch.stack(ys)


def generalize_length(
    emb_type: PositionalEmbeddingType = PositionalEmbeddingType.SINUSOIDAL,
    save_models: bool = True,
    load_models: bool = False,
    save_history: bool = True,
    show_loss_history: bool = True,
    show_eval_history: bool = True,
    show_cell_accuracy: bool = True,
    print_eval: bool = True,
    random_seed: int | None = None,
):
    if random_seed is not None:
        torch.manual_seed(random_seed)
        np.random.seed(random_seed)

    save_results_dir = f"{RESULTS_DIR}/generalize_length_{emb_type.value}"
    if not os.path.exists(save_results_dir):
        os.makedirs(save_results_dir)
    
    save_models_dir = f"{SAVED_MODELS_DIR}/generalize_length_{emb_type.value}"
    if save_models and not os.path.exists(save_models_dir):
        os.makedirs(save_models_dir)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"

    histories = []
    all_test_metrics = []

    for rule in RULES:
        model = CATransformer(
            emb_type=emb_type,
            d_model=D_MODEL,
            n_heads=N_HEADS,
            n_layers=N_LAYERS,
            d_ff=D_FF,
            dropout=DROPOUT,
        )

        train_ds = CADataset(TRAIN_SIZE, TRAIN_SEQ_MAX_LEN, rule_number=rule, steps=TRAIN_STEPS)
        test_ds = {
            seq_len: CADataset(TEST_SIZE, seq_len, rule_number=rule, steps=TRAIN_STEPS)
            for seq_len in TEST_SEQ_LENS
        }

        train_loader = DataLoader(
            dataset=train_ds,
            batch_size=BATCH_SIZE,
            shuffle=True,
            collate_fn=lambda batch: trim_collate_fn(
                batch,
                ca=train_ds.ca,
            )
        )
        test_loaders = {
            seq_len: DataLoader(ds, batch_size=BATCH_SIZE)
            for seq_len, ds in test_ds.items()
        }

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
                    show_test_loss=False,
                    title=f"Loss history: rule {rule}",
                    save_path=f"{save_results_dir}/loss_history_rule_{rule}.png"
                )
            
            if show_eval_history:
                visualize_metrics_history(
                    history=history,
                    title=f"Evaluation metrics history: rule {rule}",
                    save_path=f"{save_results_dir}/eval_history_rule_{rule}.png"
                )

        test_metrics = {
            seq_len: evaluate(model, loader, device)
            for seq_len, loader in test_loaders.items()
        }

        if show_cell_accuracy:
            visualize_cell_accuracy(
                eval_metrics=test_metrics,
                title=f"Cell accuracy for different sequence lengths: rule {rule}",
                save_path=f"{save_results_dir}/cell_accuracy_rule_{rule}.png"
            )
            all_test_metrics.append(test_metrics)

        if print_eval:
            train_metrics = evaluate(model, train_loader, device)

            print("=" * 40)
            print(f"RULE {rule}")
            print(f"Train cell accuracy: {train_metrics.cell_accuracy:.4f}, sequence accuracy: {train_metrics.sequence_accuracy:.4f}")
            for seq_len, metrics in test_metrics.items():
                print(f"Test (seq_len={seq_len}) cell accuracy: {metrics.cell_accuracy:.4f}, sequence accuracy: {metrics.sequence_accuracy:.4f}")
            print("=" * 40)
    
    if show_loss_history:
        visualize_multiple_loss_histories(
            histories={f"Rule {rule}": h for rule, h in zip(RULES, histories)},
            show_test_loss=False,
            title="Loss history comparison",
            save_path=f"{save_results_dir}/loss_history_comparison.png"
        )

    if show_eval_history:
        visualize_multiple_metrics_histories(
            histories={f"Rule {rule}": h for rule, h in zip(RULES, histories)},
            title="Evaluation metrics history comparison",
            save_path=f"{save_results_dir}/eval_history_comparison.png"
        )
    
    if show_cell_accuracy:
        visualize_multiple_cell_accuracies(
            all_eval_metrics={f"Rule {rule}": m for rule, m in zip(RULES, all_test_metrics)},
            title="Cell accuracy comparison for different sequence lengths",
            save_path=f"{save_results_dir}/cell_accuracy_comparison.png"
        )