import os
import torch

from data_generation import *
from transformer import *
from visualize import *
from constants import *


def first_experiment(
    save_models: bool = True,
    load_models: bool = False,
    show_ca: bool = True,
    show_predictions: bool = True,
    show_loss_history: bool = True,
    show_eval_history: bool = True,
    show_attention: bool = True,
    print_eval: bool = True,
    random_seed: int | None = None,
):
    if random_seed is not None:
        torch.manual_seed(random_seed)

    save_results_dir = f"{RESULTS_DIR}/first_experiment"
    if not os.path.exists(save_results_dir):
        os.makedirs(save_results_dir)
    
    save_models_dir = f"{SAVED_MODELS_DIR}/first_experiment"
    if save_models and not os.path.exists(save_models_dir):
        os.makedirs(save_models_dir)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    init_states = np.random.randint(0, 2, (SHOW_N_STATES, SEQ_LEN), dtype=np.uint8)

    if show_ca:
        for rule in RULES:
            visualize_ca_trajectories(
                rule_number=rule,
                init_states=init_states,
                steps=SHOW_STEPS,
                save_path=f"{save_results_dir}/ca_rule_{rule}.png"
            )
    
    histories = []

    for rule in RULES:
        model = CATransformer(
            seq_len=SEQ_LEN,
            d_model=D_MODEL,
            n_heads=N_HEADS,
            n_layers=N_LAYERS,
            d_ff=D_FF,
            dropout=DROPOUT,
        )

        train_ds = CADataset(TRAIN_SIZE, SEQ_LEN, rule_number=rule, steps=TRAIN_STEPS)
        test_ds = CADataset(TEST_SIZE, SEQ_LEN, rule_number=rule, steps=TRAIN_STEPS)

        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
        test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE)

        if load_models:
            model = load_model(
                model=model,
                load_path=f"{save_models_dir}/rule_{rule}.pt",
                device=device,
            )
        else:
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
                    save_path=f"{save_models_dir}/rule_{rule}.pt"
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
            test_metrics = evaluate(model, test_loader, device)
            print("=" * 40)
            print(f"RULE {rule}")
            print(f"Train cell accuracy: {train_metrics.cell_accuracy:.4f}, sequence accuracy: {train_metrics.sequence_accuracy:.4f}")
            print(f"Test cell accuracy: {test_metrics.cell_accuracy:.4f}, sequence accuracy: {test_metrics.sequence_accuracy:.4f}")
            print("=" * 40)

        if show_predictions:
            visualize_predictions(
                rule_number=rule,
                model=model.to("cpu"),
                init_states=init_states,
                steps=SHOW_STEPS,
                save_path=f"{save_results_dir}/predictions_rule_{rule}.png"
            )

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
            title="Loss history comparison",
            save_path=f"{save_results_dir}/loss_history_comparison.png"
        )

    if show_eval_history:
        visualize_multiple_metrics_histories(
            histories={f"Rule {rule}": h for rule, h in zip(RULES, histories)},
            title="Evaluation metrics history comparison",
            save_path=f"{save_results_dir}/eval_history_comparison.png"
        )