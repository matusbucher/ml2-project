from data_generation import *
from transformer import *
from visualize import *
from first_experiment import *


def test(
    load: bool = False,
    save_path: str | None = None,
):
    model = CATransformer(
        seq_len=SEQ_LEN,
        d_model=D_MODEL,
        n_heads=N_HEADS,
        n_layers=N_LAYERS,
        d_ff=D_FF,
        dropout=DROPOUT,
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if load and save_path is not None:
        model = load_model(model, load_path=save_path)
    else:
        rule = 110

        train_ds = CADataset(TRAIN_SIZE, SEQ_LEN, rule_number=rule, steps=TRAIN_STEPS)
        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)

        train(
            model=model,
            data_loader=train_loader,
            device=device,
            n_epochs=N_EPOCHS,
            lr=LR,
        )

        if save_path is not None:
            save_model(model, save_path=save_path)

    x = torch.randint(0, 2, (1, SEQ_LEN), dtype=torch.long)
    x = x.to(device)

    attention, _ = model.get_attention(x)
    visualize_attention(attention_weights=attention)


if __name__ == "__main__":
    # visualize_ca_rules(
    #     rule_numbers=[30, 90, 110, 184],
    #     init_state=cpl.init_simple(32),
    #     steps=SHOW_STEPS,
    #     save_path=f"{RESULTS_DIR}/ca_rules_comparison.png"
    # )

    # first_experiment(
    #     do_train=False,
    #     random_seed=RANDOM_SEED,
    # )

    test(
        load=False,
        save_path=f"{RESULTS_DIR}/catransformer_rule110.pth",
    )