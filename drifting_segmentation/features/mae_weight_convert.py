def main():
    raise NotImplementedError(
        "JAX/Flax MAE artifact conversion is intentionally separated from training. "
        "Export Flax params to a PyTorch state_dict and pass it as feature.mae_path."
    )


if __name__ == "__main__":
    main()
