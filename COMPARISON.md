# PyTorch vs. Keras: Bike Rental Prediction

## Experimental Setup

I compared PyTorch and Keras using equivalent feedforward neural networks to predict bike rental demand. Both models used eight input features, two hidden layers containing 64 and 32 neurons with ReLU activation, and one linear output neuron. Both used the Adam optimizer, a learning rate of 0.001, a batch size of 64, and 100 training epochs.

I conducted six training runs using three random seeds: 2026, 2027, and 2028. For each seed, both frameworks used the same prepared training, validation, and test datasets. Model selection was based on validation mean squared error (MSE), while mean absolute error (MAE) measured prediction accuracy.

## Observed Differences

**1. Prediction accuracy:** PyTorch achieved an average validation MAE of 3.5231, compared with 3.5605 for Keras. However, Keras achieved a slightly lower average test MAE of 3.5364, compared with 3.5504 for PyTorch. These differences are small and do not establish a decisive accuracy advantage.

**2. Training speed:** PyTorch required approximately 18.92 seconds per run, while Keras required 30.38 seconds. PyTorch was approximately 38% faster based on these measurements. Both frameworks ran on the same Windows CPU environment.

**3. Variation across seeds:** PyTorch achieved the lower test MAE for seed 2026, while Keras performed better for seeds 2027 and 2028. This demonstrates that results vary with random initialization and training conditions, even when architecture and hyperparameters are equivalent.

## Recommendation

For a two-person team maintaining a weekly bike-share prediction workflow, I recommend **PyTorch** for the current implementation. It demonstrated comparable prediction accuracy with shorter training times, and the team already has a working PyTorch training pipeline.

Keras remains a strong alternative because its high-level training interface, built-in callbacks, and model-saving functionality simplify routine development.

If the team later deploys the model through Hugging Face, the decision should depend on the selected hosting method, deployment dependencies, and team experience rather than assuming one framework is universally better.

I would reconsider Keras if future experiments demonstrate a meaningful improvement in reliability, deployment simplicity, or predictive performance. Additional seeds and repeated timing measurements would strengthen this comparison.
