# Bulian Synthesizers

The synthetic data models found on this folder operate on single datasets, passed as python `pandas.DataFrame` and column list broken by numeric and categorical columns passes as python `list`.

Implemented models:

* Twin Synthesizer: GAN models that generate non-privacy preserving synthetic datasets given an input python `pandas.DataFrame` and column list broken by numeric and categorical columns passes as python `list`..
    * `discrete_columns (list-like)`: List of discrete columns to be used to generate the Conditional Vector. This list should contatin the column names.
    * `embedding_dim (int)`: Size of the random sample passed to the Generator. Defaults to `128`.
    * `generator_dim (tuple or list of ints)`: Size of the output samples for each one of the Residuals. Defaults to `(256, 256)`.
    * `discriminator_dim (tuple or list of ints)`: Size of the output samples for each one of the Discriminator Layers. A Linear Layer will be created for each one of the  values provided. Defaults to (256, 256).
    * `generator_lr (float)`: Learning rate for the generator. Defaults to 2e-4.
    * `generator_decay (float)`: Generator weight decay for the Adam Optimizer. Defaults to 1e-6.
    * `discriminator_lr (float)`: Learning rate for the discriminator. Defaults to 2e-4.
    * `discriminator_decay (float)`: Discriminator weight decay for the Adam Optimizer. Defaults to 1e-6.
    * `batch_size (int)`: of data samples to process in each step.
    * `discriminator_steps (int)`: Number of discriminator updates to do for each generator update. WGAN paper default is 5. Here we use 1 to match original CTGAN implementation.
    * `log_frequency (boolean)`: Whether to use log frequency of categorical levels in conditional sampling. Defaults to ``True``.
    * `epochs (int)`: Number of training epochs. Defaults to 300.
    * `pac (int)`: Number of samples to group together when applying the discriminator. Defaults to 10.
    * `device (torch.device or str)`: Device to use.


