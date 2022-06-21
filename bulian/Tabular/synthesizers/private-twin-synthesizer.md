# Private Twin Synthesizer

Differtially private GAN models inherits from `BaseSynthesizerPrivate` class and generate privacy preserving synthetic datasets given an input python `pandas.DataFrame` and column list broken by numeric and categorical columns passes as python `list`.

***

*   **Suported arguments**

    * `discrete_columns (list-like)` : List of discrete columns to be used to generate the Conditional Vector. This list should contain the column names.
    * `epsilon :` We perform generator iterations until our privacy constraint, has been reached. Generally, high-utilizty datasets need higher privacy budgets. Defaults to `1`.
    * `latent_dim (int)` : Size of the latent embedding layer in the Generator. Defaults to `64`.
    * `binary (boolean)` : Boolean to decide activation in Generator, `True` corresponds to `tanh` activation, else, `leakyRelu(0.2)`. Defaults to `True`.
    * `batch_size (int)` : Number of data samples to process in each step. Defaults to `64`.
    * `teacher_iters (int)` : Number of update steps for teacher models. Defaults to `5`.
    * `student_iters (int)` : Number of update steps for student models. Defaults to `5`.
    * `device (torch.device or str)` : Device to use. Defaults to `cpu`.


*   **Suported fit methods**

    * `model.fit` : Vanilla fit method over Private Twin Synthesize models. User can update value of `epsilon` here as well using the `update_epsilon` parameter.&#x20;
    * `model.fit_adversarial` : Adversarial fit API which has an additional parameter called `test_pct` defaults to `0.2`. The distilled samples will mimic these `test_pct` samples from data closely. User can update value of `epsilon` here as well using the `update_epsilon` parameter.


*   **Suported sampling mechanisms**

    * `model.sample` : Vanilla sample generator on the fit model object. Takes in number of samples `k` to be generated.
    * `model.sample_adversarial` : Adversarial sampling API for adversarial fit models. Additional arguments:

    > * `upsample_frac` : Ratio of samples to generate from which `k` distilled samples will be returned based on adversarial random forest model. Defaults to `4`.
    > * `rf_params` : Parameters of adversarial `Random Forest` model. Defaults to `sklearn's` Random Forest model default params.

***

* **Code starter example**

```
In [1]: synth = PrivateTwinSynthesizer(epsilon=0.1,batch_size=64,device='cuda')

In [2]: synth.fit(data=data,update_epsilon=1,discrete_columns=discrete_columns)

Out [2]: 
    Iteration: [1]  [  0/159]  eta: 4:09:26  loss_t_fake: 0.4787 (0.6327)  loss_t_real: 0.6836 (0.7031)  time: 93.5428  data: 0.0000  max mem: 62
    Iteration: [1]  [ 50/159]  eta: 0:04:24  loss_t_fake: 0.0080 (0.0875)  loss_t_real: 0.0478 (0.2935)  time: 0.5846  data: 0.0000  max mem: 62
    Iteration: [1]  [100/159]  eta: 0:01:30  loss_t_fake: 0.0016 (0.0460)  loss_t_real: 0.0092 (0.1599)  time: 0.5757  data: 0.0000  max mem: 62
    Iteration: [1]  [150/159]  eta: 0:00:11  loss_t_fake: 0.0006 (0.0311)  loss_t_real: 0.0032 (0.1089)  time: 0.5751  data: 0.0000  max mem: 62
    Iteration: [1]  [159/159]  eta: 0:00:01  loss_t_fake: 0.0005 (0.0294)  loss_t_real: 0.0030 (0.1030)  time: 0.5680  data: 0.0000  max mem: 62
    Iteration: [1] Total time: 0:03:05
    Iteration: [1]  [0/4]  eta: 0:00:00  loss_t_fake: 0.0005 (0.0294)  loss_t_real: 0.0030 (0.1030)  loss_s: 0.8012 (0.8048)  time: 0.1262  data: 0.0000  max mem: 63
    Iteration: [1]  [1/4]  eta: 0:00:00  loss_t_fake: 0.0005 (0.0294)  loss_t_real: 0.0030 (0.1030)  loss_s: 0.7539 (0.7665)  time: 0.1187  data: 0.0000  max mem: 63
    Iteration: [1]  [2/4]  eta: 0:00:00  loss_t_fake: 0.0005 (0.0294)  loss_t_real: 0.0030 (0.1030)  loss_s: 0.7420 (0.7499)  time: 0.1172  data: 0.0000  max mem: 63
    Iteration: [1]  [3/4]  eta: 0:00:00  loss_t_fake: 0.0005 (0.0294)  loss_t_real: 0.0030 (0.1030)  loss_s: 0.7199 (0.7391)  time: 0.1163  data: 0.0000  max mem: 63
    Iteration: [1]  [4/4]  eta: 0:00:00  loss_t_fake: 0.0005 (0.0294)  loss_t_real: 0.0030 (0.1030)  loss_s: 0.7154 (0.7323)  time: 0.1164  data: 0.0000  max mem: 63
    Iteration: [1] Total time: 0:00:00
    Iteration: [1]  [  0/507]  eta: 0:00:01  loss_t_fake: 0.0005 (0.0294)  loss_t_real: 0.0030 (0.1030)  loss_s: 0.7154 (0.7323)  loss_g: 0.8145 (0.8145)  time: 0.0030  data: 0.0000  max mem: 63
    Iteration: [1]  [ 50/507]  eta: 0:00:01  loss_t_fake: 0.0005 (0.0294)  loss_t_real: 0.0030 (0.1030)  loss_s: 0.7154 (0.7323)  loss_g: 0.1951 (0.3748)  time: 0.0022  data: 0.0000  max mem: 63
    Iteration: [1]  [100/507]  eta: 0:00:01  loss_t_fake: 0.0005 (0.0294)  loss_t_real: 0.0030 (0.1030)  loss_s: 0.7154 (0.7323)  loss_g: 0.0613 (0.2326)  time: 0.0031  data: 0.0000  max mem: 63
    Iteration: [1]  [150/507]  eta: 0:00:00  loss_t_fake: 0.0005 (0.0294)  loss_t_real: 0.0030 (0.1030)  loss_s: 0.7154 (0.7323)  loss_g: 0.0279 (0.1674)  time: 0.0027  data: 0.0000  max mem: 63
    Iteration: [1]  [200/507]  eta: 0:00:00  loss_t_fake: 0.0005 (0.0294)  loss_t_real: 0.0030 (0.1030)  loss_s: 0.7154 (0.7323)  loss_g: 0.0152 (0.1304)  time: 0.0024  data: 0.0000  max mem: 63
    Iteration: [1]  [250/507]  eta: 0:00:00  loss_t_fake: 0.0005 (0.0294)  loss_t_real: 0.0030 (0.1030)  loss_s: 0.7154 (0.7323)  loss_g: 0.0102 (0.1067)  time: 0.0023  data: 0.0000  max mem: 63
    Iteration: [1]  [300/507]  eta: 0:00:00  loss_t_fake: 0.0005 (0.0294)  loss_t_real: 0.0030 (0.1030)  loss_s: 0.7154 (0.7323)  loss_g: 0.0074 (0.0903)  time: 0.0025  data: 0.0000  max mem: 63
    Iteration: [1]  [350/507]  eta: 0:00:00  loss_t_fake: 0.0005 (0.0294)  loss_t_real: 0.0030 (0.1030)  loss_s: 0.7154 (0.7323)  loss_g: 0.0056 (0.0783)  time: 0.0026  data: 0.0000  max mem: 63
    Iteration: [1]  [400/507]  eta: 0:00:00  loss_t_fake: 0.0005 (0.0294)  loss_t_real: 0.0030 (0.1030)  loss_s: 0.7154 (0.7323)  loss_g: 0.0046 (0.0691)  time: 0.0025  data: 0.0000  max mem: 63
    Iteration: [1]  [450/507]  eta: 0:00:00  loss_t_fake: 0.0005 (0.0294)  loss_t_real: 0.0030 (0.1030)  loss_s: 0.7154 (0.7323)  loss_g: 0.0037 (0.0619)  time: 0.0025  data: 0.0000  max mem: 63
    Iteration: [1]  [500/507]  eta: 0:00:00  loss_t_fake: 0.0005 (0.0294)  loss_t_real: 0.0030 (0.1030)  loss_s: 0.7154 (0.7323)  loss_g: 0.0032 (0.0561)  time: 0.0024  data: 0.0000  max mem: 63
    Iteration: [1]  [507/507]  eta: 0:00:00  loss_t_fake: 0.0005 (0.0294)  loss_t_real: 0.0030 (0.1030)  loss_s: 0.7154 (0.7323)  loss_g: 0.0031 (0.0553)  time: 0.0025  data: 0.0000  max mem: 63

In [3]: samples = synth.sample(500)
```

```
In [4]: synth = PrivateTwinSynthesizer(epsilon=0.1,batch_size=64,device='cuda')

In [5]: synth.fit_adversarial(data=data,update_epsilon=1,discrete_columns=discrete_columns)

Out [5]: 
  Generating train and test splits ...
  TRAIN SAMPLES: n=26048
  TEST SAMPLES: n=6513
  Iteration: [1]  [  0/129]  eta: 0:01:01  loss_t_fake: 0.5879 (0.7448)  loss_t_real: 0.6553 (0.7024)  time: 0.4757  data: 0.0000  max mem: 63
  Iteration: [1]  [ 50/129]  eta: 0:00:30  loss_t_fake: 0.0074 (0.1032)  loss_t_real: 0.0542 (0.2888)  time: 0.3806  data: 0.0000  max mem: 63
  Iteration: [1]  [100/129]  eta: 0:00:11  loss_t_fake: 0.0016 (0.0539)  loss_t_real: 0.0097 (0.1579)  time: 0.3771  data: 0.0000  max mem: 63
  Iteration: [1]  [129/129]  eta: 0:00:00  loss_t_fake: 0.0009 (0.0422)  loss_t_real: 0.0053 (0.1244)  time: 0.3823  data: 0.0000  max mem: 63
  Iteration: [1] Total time: 0:00:49
  Iteration: [1]  [0/4]  eta: 0:00:00  loss_t_fake: 0.0009 (0.0422)  loss_t_real: 0.0053 (0.1244)  loss_s: 0.9174 (0.9082)  time: 0.0997  data: 0.0000  max mem: 63
  Iteration: [1]  [1/4]  eta: 0:00:00  loss_t_fake: 0.0009 (0.0422)  loss_t_real: 0.0053 (0.1244)  loss_s: 0.8291 (0.8523)  time: 0.0992  data: 0.0000  max mem: 63
  Iteration: [1]  [2/4]  eta: 0:00:00  loss_t_fake: 0.0009 (0.0422)  loss_t_real: 0.0053 (0.1244)  loss_s: 0.8227 (0.8322)  time: 0.0997  data: 0.0000  max mem: 63
  Iteration: [1]  [3/4]  eta: 0:00:00  loss_t_fake: 0.0009 (0.0422)  loss_t_real: 0.0053 (0.1244)  loss_s: 0.7994 (0.8083)  time: 0.1000  data: 0.0000  max mem: 63
  Iteration: [1]  [4/4]  eta: 0:00:00  loss_t_fake: 0.0009 (0.0422)  loss_t_real: 0.0053 (0.1244)  loss_s: 0.7399 (0.7870)  time: 0.0999  data: 0.0000  max mem: 63
  Iteration: [1] Total time: 0:00:00
  Iteration: [1]  [  0/406]  eta: 0:00:00  loss_t_fake: 0.0009 (0.0422)  loss_t_real: 0.0053 (0.1244)  loss_s: 0.7399 (0.7870)  loss_g: 0.6212 (0.6212)  time: 0.0020  data: 0.0000  max mem: 63
  Iteration: [1]  [ 50/406]  eta: 0:00:00  loss_t_fake: 0.0009 (0.0422)  loss_t_real: 0.0053 (0.1244)  loss_s: 0.7399 (0.7870)  loss_g: 0.1222 (0.2688)  time: 0.0021  data: 0.0000  max mem: 63
  Iteration: [1]  [100/406]  eta: 0:00:00  loss_t_fake: 0.0009 (0.0422)  loss_t_real: 0.0053 (0.1244)  loss_s: 0.7399 (0.7870)  loss_g: 0.0466 (0.1658)  time: 0.0021  data: 0.0000  max mem: 63
  Iteration: [1]  [150/406]  eta: 0:00:00  loss_t_fake: 0.0009 (0.0422)  loss_t_real: 0.0053 (0.1244)  loss_s: 0.7399 (0.7870)  loss_g: 0.0273 (0.1215)  time: 0.0022  data: 0.0000  max mem: 63
  Iteration: [1]  [200/406]  eta: 0:00:00  loss_t_fake: 0.0009 (0.0422)  loss_t_real: 0.0053 (0.1244)  loss_s: 0.7399 (0.7870)  loss_g: 0.0186 (0.0964)  time: 0.0021  data: 0.0000  max mem: 63
  Iteration: [1]  [250/406]  eta: 0:00:00  loss_t_fake: 0.0009 (0.0422)  loss_t_real: 0.0053 (0.1244)  loss_s: 0.7399 (0.7870)  loss_g: 0.0138 (0.0802)  time: 0.0021  data: 0.0000  max mem: 63
  Iteration: [1]  [300/406]  eta: 0:00:00  loss_t_fake: 0.0009 (0.0422)  loss_t_real: 0.0053 (0.1244)  loss_s: 0.7399 (0.7870)  loss_g: 0.0104 (0.0687)  time: 0.0021  data: 0.0000  max mem: 63
  Iteration: [1]  [350/406]  eta: 0:00:00  loss_t_fake: 0.0009 (0.0422)  loss_t_real: 0.0053 (0.1244)  loss_s: 0.7399 (0.7870)  loss_g: 0.0083 (0.0602)  time: 0.0020  data: 0.0000  max mem: 63
  Iteration: [1]  [400/406]  eta: 0:00:00  loss_t_fake: 0.0009 (0.0422)  loss_t_real: 0.0053 (0.1244)  loss_s: 0.7399 (0.7870)  loss_g: 0.0066 (0.0536)  time: 0.0021  data: 0.0000  max mem: 63
  Iteration: [1]  [406/406]  eta: 0:00:00  loss_t_fake: 0.0009 (0.0422)  loss_t_real: 0.0053 (0.1244)  loss_s: 0.7399 (0.7870)  loss_g: 0.0064 (0.0529)  time: 0.0021  data: 0.0000  max mem: 63

In [6]: synth.sample_adversarial(data,1000)

Out [6]:
  [Parallel(n_jobs=-1)]: Using backend ThreadingBackend with 8 concurrent workers.
  [Parallel(n_jobs=-1)]: Done  34 tasks      | elapsed:    0.2s
  [Parallel(n_jobs=-1)]: Done 100 out of 100 | elapsed:    0.7s finished
  [Parallel(n_jobs=8)]: Using backend ThreadingBackend with 8 concurrent workers.
  [Parallel(n_jobs=8)]: Done  34 tasks      | elapsed:    0.0s
  [Parallel(n_jobs=8)]: Done 100 out of 100 | elapsed:    0.0s finished
```
