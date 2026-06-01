"""Treina e salva o modelo final do Projeto M38.

Uso:
    python train_model.py --data credit_scoring.ftr --model model_final.pkl --sample-size 150000
"""
import argparse
import pickle

from credit_scoring_pipeline import (
    TARGET,
    build_model_pipeline,
    calculate_metrics,
    load_credit_data,
    prepare_xy,
    predict_proba_mau,
    sample_training_data,
    split_dev_oot,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="credit_scoring.ftr", help="Caminho da base .ftr/.csv")
    parser.add_argument("--model", default="model_final.pkl", help="Nome do pickle de saída")
    parser.add_argument("--sample-size", type=int, default=150000, help="Amostra do desenvolvimento; use 0 para base completa")
    args = parser.parse_args()

    print("Carregando dados...")
    df = load_credit_data(args.data)
    dev, oot = split_dev_oot(df, n_oot_months=3)
    print(f"Desenvolvimento: {dev.shape} | OOT: {oot.shape}")

    train_df = sample_training_data(dev, None if args.sample_size == 0 else args.sample_size)
    X_train, y_train = prepare_xy(train_df)

    print(f"Treinando modelo com {len(train_df):,} linhas...")
    model = build_model_pipeline()
    model.fit(X_train, y_train)

    for nome, base in [("Desenvolvimento", dev), ("OOT", oot)]:
        y = base[TARGET].astype(int)
        score = predict_proba_mau(model, base)
        metrics = calculate_metrics(y, score)
        print(f"\n{nome}")
        for k, v in metrics.items():
            print(f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}")

    with open(args.model, "wb") as f:
        pickle.dump(model, f)
    print(f"\nModelo salvo em: {args.model}")


if __name__ == "__main__":
    main()
