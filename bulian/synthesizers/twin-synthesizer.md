# Twin Synthesizer

## Bulian Synthesizers

The synthetic data models found on this folder operate on single datasets, passed as python `pandas.DataFrame` and column list specified by Numeric and Categorical columns passes as python `list`.

### Twin Synthesizer

***

GAN models inherits from `BaseSynthesizer` class; generate non-privacy preserving synthetic datasets given an input python `pandas.DataFrame` and column list broken by numeric and categorical columns passes as python `list`.

***

* **Suported arguments**
  * `discrete_columns (list-like)` : List of discrete columns to be used to generate the Conditional Vector. This list should contatin the column names.
  * `embedding_dim (int)` : Size of the random sample passed to the Generator. Defaults to `128`.
  * `generator_dim (tuple or list of ints)` : Size of the output samples for each one of the Residuals. Defaults to `(256, 256)`.
  * `discriminator_dim (tuple or list of ints)` : Size of the output samples for each one of the Discriminator Layers. A Linear Layer will be created for each one of the values provided. Defaults to `(256, 256)`.
  * `generator_lr (float)` : Learning rate for the generator. Defaults to 2e-4.
  * `generator_decay (float)` : Generator weight decay for the Adam Optimizer. Defaults to `1e-6`.
  * `discriminator_lr (float)` : Learning rate for the discriminator. Defaults to `2e-4`.
  * `discriminator_decay (float)` : Discriminator weight decay for the Adam Optimizer. Defaults to `1e-6`.
  * `batch_size (int)` : Number of data samples to process in each step. Defaults to `500`.
  * `discriminator_steps (int)` : Number of discriminator updates to do for each generator update. WGAN paper default is 5. Here we use `1` to match original CTGAN implementation.
  * `log_frequency (boolean)` : Whether to use log frequency of categorical levels in conditional sampling. Defaults to `True`.
  * `epochs (int)` : Number of training epochs. Defaults to `300`.
  * `pac (int)` : Number of samples to group together when applying the discriminator. Defaults to `10`.
  * `device (torch.device or str)` : Device to use. Defaults to `cpu`.

***

* **Suported fit methods**
  * `model.fit` : Vanilla fit method over Twin Synthesize models.
  * `model.fit_adversarial` : Adversarial fit API which has an additional parameter called `test_pct` defaults to `0.2`. The distilled samples will mimic these `test_pct` samples from data closely.

***

*   **Suported sampling mechanisms**

    * `model.sample` : Vanilla sample generator on the fit model object. Takes in number of samples `k` to be generated.
    * `model.sample_adversarial` : Adversarial sampling API for adversial fit models. Additional arguments:

    > * `upsample_frac` : Ratio of samples to generate from which `k` distilled samples will be returned based on adversarial random forest model. Defaults to `4`.
    > * `rf_params` : Parameters of adversarial `Random Forest` model. Defaults to `sklearn's` Random Forest model default params.

***

* **Example init and fit code snapshot**

```
In [1]: synth = TwinSynthesizer(batch_size=200,device='cpu')
In [2]: synth.fit(data=data,epochs=2,discrete_columns=discrete_columns)
Out [2]: 
  Epoch: [0]  [  0/161]  eta: 0:00:21  loss_g: 2.0557 (2.0557)  loss_d: 0.0199 (0.0199)  loss: 2.0756 (2.0756)  time: 0.1307  data: 0.0000  max mem: 0
  Epoch: [0]  [ 50/161]  eta: 0:00:06  loss_g: 1.5708 (1.8150)  loss_d: -0.5792 (-0.5329)  loss: 0.9937 (1.2820)  time: 0.0534  data: 0.0000  max mem: 0
  Epoch: [0]  [100/161]  eta: 0:00:03  loss_g: 1.5334 (1.7166)  loss_d: -0.1319 (-0.3837)  loss: 1.5359 (1.3329)  time: 0.0521  data: 0.0000  max mem: 0
  Epoch: [0]  [150/161]  eta: 0:00:00  loss_g: 1.3369 (1.5969)  loss_d: 0.0850 (-0.2217)  loss: 1.4005 (1.3752)  time: 0.0537  data: 0.0000  max mem: 0
  Epoch: [0]  [161/161]  eta: 0:00:00  loss_g: 1.2277 (1.5699)  loss_d: 0.1289 (-0.1908)  loss: 1.4107 (1.3791)  time: 0.0552  data: 0.0000  max mem: 0
  Epoch: [0] Total time: 0:00:08
  Epoch: [1]  [  0/161]  eta: 0:00:08  loss_g: 0.9840 (0.9840)  loss_d: 0.4220 (0.4220)  loss: 1.4060 (1.4060)  time: 0.0537  data: 0.0000  max mem: 0
  Epoch: [1]  [ 50/161]  eta: 0:00:06  loss_g: 0.7299 (0.9327)  loss_d: 0.0064 (0.0797)  loss: 0.7498 (1.0124)  time: 0.0532  data: 0.0000  max mem: 0
  Epoch: [1]  [100/161]  eta: 0:00:03  loss_g: 0.8318 (0.8499)  loss_d: 0.0925 (0.0817)  loss: 0.9715 (0.9315)  time: 0.0583  data: 0.0000  max mem: 0
  Epoch: [1]  [150/161]  eta: 0:00:00  loss_g: 0.7053 (0.8766)  loss_d: 0.0312 (0.0602)  loss: 0.7088 (0.9368)  time: 0.0546  data: 0.0000  max mem: 0
  Epoch: [1]  [161/161]  eta: 0:00:00  loss_g: 0.6291 (0.8618)  loss_d: 0.0254 (0.0568)  loss: 0.6772 (0.9186)  time: 0.0546  data: 0.0000  max mem: 0
  Epoch: [1] Total time: 0:00:08

In [3]: samples = synth.sample(500)
```

```
In [4]: synth = TwinSynthesizer(batch_size=200,device='cpu')
In [5]: synth.fit_adversarial(data=data,epochs=2,discrete_columns=discrete_columns)
Out [5]: 
  Generating train and test splits ...
  TRAIN SAMPLES: n=21815
  TEST SAMPLES: n=10746
  Epoch: [0]  [  0/108]  eta: 0:00:06  loss_g: 2.2097 (2.2097)  loss_d: 0.0046 (0.0046)  loss: 2.2144 (2.2144)  time: 0.0574  data: 0.0000  max mem: 0
  Epoch: [0]  [ 50/108]  eta: 0:00:03  loss_g: 1.7613 (1.9363)  loss_d: -0.7787 (-0.5969)  loss: 0.9821 (1.3395)  time: 0.0666  data: 0.0000  max mem: 0
  Epoch: [0]  [100/108]  eta: 0:00:00  loss_g: 1.8145 (1.8498)  loss_d: -0.1251 (-0.4208)  loss: 1.6819 (1.4290)  time: 0.0630  data: 0.0000  max mem: 0
  Epoch: [0]  [108/108]  eta: 0:00:00  loss_g: 1.8142 (1.8446)  loss_d: 0.0725 (-0.3797)  loss: 1.8088 (1.4649)  time: 0.0591  data: 0.0000  max mem: 0
  Epoch: [0] Total time: 0:00:06
  Epoch: [1]  [  0/108]  eta: 0:00:06  loss_g: 1.8160 (1.8160)  loss_d: 0.3067 (0.3067)  loss: 2.1227 (2.1227)  time: 0.0597  data: 0.0000  max mem: 0
  Epoch: [1]  [ 50/108]  eta: 0:00:03  loss_g: 1.7233 (1.7885)  loss_d: 0.0969 (0.1402)  loss: 1.8434 (1.9287)  time: 0.0566  data: 0.0000  max mem: 0
  Epoch: [1]  [100/108]  eta: 0:00:00  loss_g: 1.6994 (1.7161)  loss_d: 0.0392 (0.1198)  loss: 1.6980 (1.8358)  time: 0.0537  data: 0.0000  max mem: 0
  Epoch: [1]  [108/108]  eta: 0:00:00  loss_g: 1.7424 (1.7199)  loss_d: 0.0552 (0.1224)  loss: 1.8577 (1.8424)  time: 0.0543  data: 0.0000  max mem: 0
  Epoch: [1] Total time: 0:00:06
In [6]: synth.sample_adversarial(data,1000)
Out [6]:
  [Parallel(n_jobs=-1)]: Using backend ThreadingBackend with 8 concurrent workers.
  [Parallel(n_jobs=-1)]: Done  34 tasks      | elapsed:    0.2s
  [Parallel(n_jobs=-1)]: Done 100 out of 100 | elapsed:    0.7s finished
  [Parallel(n_jobs=8)]: Using backend ThreadingBackend with 8 concurrent workers.
  [Parallel(n_jobs=8)]: Done  34 tasks      | elapsed:    0.0s
  [Parallel(n_jobs=8)]: Done 100 out of 100 | elapsed:    0.0s finished
```
