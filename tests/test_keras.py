
def test_keras_can_memorize_tiny_batch():
    """Verify that Keras can overfit a small, learnable dataset."""
    keras.backend.clear_session()
    keras.utils.set_random_seed(2026)

    model = build_keras_model(input_size=8, hidden_sizes=(32, 16))
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.01),
        loss="mse",
    )

    rng = np.random.default_rng(2026)
    features = rng.normal(size=(4, 8)).astype(np.float32)

    targets = (
        2.0 * features[:, 0]
        - 1.5 * features[:, 1]
        + 0.5 * features[:, 2]
    ).astype(np.float32)

    initial_predictions = model(features, training=False).numpy().reshape(-1)
    initial_mse = float(np.mean((initial_predictions - targets) ** 2))

    model.fit(
        features,
        targets,
        epochs=300,
        batch_size=4,
        shuffle=False,
        verbose=0,
    )

    predictions = model(features, training=False).numpy().reshape(-1)
    final_mse = float(np.mean((predictions - targets) ** 2))

    assert final_mse < initial_mse
    assert final_mse < 0.1
