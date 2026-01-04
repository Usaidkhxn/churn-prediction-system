from src.ml_system.data import load_churn_csv
from src.ml_system.config import load_config


def main():
    cfg = load_config()
    X, y = load_churn_csv(cfg.data.path, cfg.data.target)

    print("Loaded dataset ✅")
    print(f"X shape: {X.shape}")
    print("Target distribution:")
    print(y.value_counts())


if __name__ == "__main__":
    main()
