# demo\_single\_table

### Videha AI: Single Table Demo

```python
import os,sys,torch
import pandas as pd
```

```python
from videha.synthesizers import TwinSynthesizer,PrivateTwinSynthesizer
```

```python
from videha.metrics import *
from videha.metrics.reports import *
from videha.metrics import compute_metrics
from videha.metrics.single_table import SingleTableMetric
from videha.metrics.single_table import *
```

```python
import pandas as pd
import warnings
warnings.filterwarnings('ignore')
```

```python
discrete_columns = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
    "income",
]
```

```python
torch.cuda.is_available()
```

```
True
```

```python
data = pd.read_csv("examples/csv/adult.csv")
```

### Normal API: Non Privately Differentable Synthesizer

```python
synth = TwinSynthesizer(batch_size=200,device='cpu')   ### cpu else cuda
```

```python
synth.fit(data=data,epochs=2,discrete_columns=discrete_columns)
```

```
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
```

```python
sample = synth.sample(1000)
```

```python
metrics = SingleTableMetric.get_subclasses()
numeric_features = ['capital-gain','capital-loss','hours-per-week']
discrete_columns = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
    "income",
]
```

**Report with privacy metrics**

```python
get_full_report(data, sample,discrete_columns,numeric_features, key_fields=['age','workclass','education'],sensitive_fields = ['income'])
```

![png](../Example%20Notebooks/demo\_single\_table\_files/demo\_single\_table\_14\_2%20\(6\).png)

![png](../Example%20Notebooks/demo\_single\_table\_files/demo\_single\_table\_14\_3%20\(6\).png)

![png](../Example%20Notebooks/demo\_single\_table\_files/demo\_single\_table\_14\_4%20\(6\).png)

![png](../Example%20Notebooks/demo\_single\_table\_files/demo\_single\_table\_14\_5%20\(6\).png)

![png](../Example%20Notebooks/demo\_single\_table\_files/demo\_single\_table\_14\_6%20\(5\).png)

![png](../Example%20Notebooks/demo\_single\_table\_files/demo\_single\_table\_14\_7%20\(6\).png)

![png](../Example%20Notebooks/demo\_single\_table\_files/demo\_single\_table\_14\_8%20\(5\).png)

![png](../Example%20Notebooks/demo\_single\_table\_files/demo\_single\_table\_14\_9%20\(6\).png)

![png](../Example%20Notebooks/demo\_single\_table\_files/demo\_single\_table\_14\_10%20\(6\).png)

![png](../Example%20Notebooks/demo\_single\_table\_files/demo\_single\_table\_14\_11%20\(6\).png)

![png](../Example%20Notebooks/demo\_single\_table\_files/demo\_single\_table\_14\_12%20\(4\).png)

![png](../Example%20Notebooks/demo\_single\_table\_files/demo\_single\_table\_14\_13%20\(6\).png)

![png](../Example%20Notebooks/demo\_single\_table\_files/demo\_single\_table\_14\_14%20\(5\).png)

**Report without privacy metrics, but includes ML efficacy stuff**

```python
get_full_report(data, sample,discrete_columns,numeric_features,target='income')
```

![png](../Example%20Notebooks/demo\_single\_table\_files/demo\_single\_table\_16\_2%20\(1\).png)

![png](../Example%20Notebooks/demo\_single\_table\_files/demo\_single\_table\_16\_3.png)

![png](../Example%20Notebooks/demo\_single\_table\_files/demo\_single\_table\_16\_4%20\(1\).png)

![png](../Example%20Notebooks/demo\_single\_table\_files/demo\_single\_table\_16\_5.png)

![png](../Example%20Notebooks/demo\_single\_table\_files/demo\_single\_table\_16\_6.png)

![png](../Example%20Notebooks/demo\_single\_table\_files/demo\_single\_table\_16\_7%20\(1\).png)

![png](../Example%20Notebooks/demo\_single\_table\_files/demo\_single\_table\_16\_8.png)

![png](../Example%20Notebooks/demo\_single\_table\_files/demo\_single\_table\_16\_9.png)

![png](../Example%20Notebooks/demo\_single\_table\_files/demo\_single\_table\_16\_10.png)

![png](../Example%20Notebooks/demo\_single\_table\_files/demo\_single\_table\_16\_11.png)

![png](../Example%20Notebooks/demo\_single\_table\_files/demo\_single\_table\_16\_12.png)

![png](../Example%20Notebooks/demo\_single\_table\_files/demo\_single\_table\_16\_13%20\(1\).png)

![png](../Example%20Notebooks/demo\_single\_table\_files/demo\_single\_table\_16\_14%20\(1\).png)

**Report without privacy metrics and without ML efficacy stuff**

```python
get_full_report(data, sample,discrete_columns,numeric_features)
```

![png](../Example%20Notebooks/demo\_single\_table\_files/demo\_single\_table\_16\_2.png)

![png](../Example%20Notebooks/demo\_single\_table\_files/demo\_single\_table\_16\_3%20\(1\).png)

![png](../Example%20Notebooks/demo\_single\_table\_files/demo\_single\_table\_16\_4.png)

![png](../Example%20Notebooks/demo\_single\_table\_files/demo\_single\_table\_16\_5%20\(1\).png)

![png](../Example%20Notebooks/demo\_single\_table\_files/demo\_single\_table\_16\_6%20\(1\).png)

![png](../Example%20Notebooks/demo\_single\_table\_files/demo\_single\_table\_16\_7.png)

![png](../Example%20Notebooks/demo\_single\_table\_files/demo\_single\_table\_16\_8%20\(1\).png)

![png](../Example%20Notebooks/demo\_single\_table\_files/demo\_single\_table\_16\_9%20\(1\).png)

![png](../Example%20Notebooks/demo\_single\_table\_files/demo\_single\_table\_16\_10%20\(1\).png)

![png](../Example%20Notebooks/demo\_single\_table\_files/demo\_single\_table\_16\_11%20\(1\).png)

![png](../Example%20Notebooks/demo\_single\_table\_files/demo\_single\_table\_16\_12%20\(1\).png)

![png](../Example%20Notebooks/demo\_single\_table\_files/demo\_single\_table\_16\_13.png)

![png](../Example%20Notebooks/demo\_single\_table\_files/demo\_single\_table\_16\_14.png)

**Save model to disk**

```python
synth.save('NormalAPI.pth')
```

### Adversarial API: Non-privately differentiable synthesizer

```python
synth = TwinSynthesizer(batch_size=200,device='cpu')   ### cpu else cuda
```

```python
synth.fit_adversarial(data=data,epochs=2,discrete_columns=discrete_columns,test_pct=0.33)
```

```
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
```

```python
sample = synth.sample_adversarial(data,1000)
```

```
[Parallel(n_jobs=-1)]: Using backend ThreadingBackend with 8 concurrent workers.
[Parallel(n_jobs=-1)]: Done  34 tasks      | elapsed:    0.2s
[Parallel(n_jobs=-1)]: Done 100 out of 100 | elapsed:    0.7s finished
[Parallel(n_jobs=8)]: Using backend ThreadingBackend with 8 concurrent workers.
[Parallel(n_jobs=8)]: Done  34 tasks      | elapsed:    0.0s
[Parallel(n_jobs=8)]: Done 100 out of 100 | elapsed:    0.0s finished
```

```python
metrics = SingleTableMetric.get_subclasses()
numeric_features = ['capital-gain','capital-loss','hours-per-week']
discrete_columns = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
    "income",
]
```

**Report with privacy metrics**

```python
get_full_report(data, sample,discrete_columns,numeric_features, key_fields=['age','workclass','education'],sensitive_fields = ['income'])
```

![png](../Example%20Notebooks/demo\_single\_table\_files/demo\_single\_table\_28\_2%20\(5\).png)

![png](../Example%20Notebooks/demo\_single\_table\_files/demo\_single\_table\_28\_3%20\(4\).png)

![png](../Example%20Notebooks/demo\_single\_table\_files/demo\_single\_table\_28\_4%20\(6\).png)

![png](../Example%20Notebooks/demo\_single\_table\_files/demo\_single\_table\_28\_5%20\(6\).png)

![png](../Example%20Notebooks/demo\_single\_table\_files/demo\_single\_table\_28\_6%20\(6\).png)

![png](../Example%20Notebooks/demo\_single\_table\_files/demo\_single\_table\_28\_7%20\(4\).png)

![png](../Example%20Notebooks/demo\_single\_table\_files/demo\_single\_table\_28\_8%20\(6\).png)

![png](../Example%20Notebooks/demo\_single\_table\_files/demo\_single\_table\_28\_9%20\(5\).png)

![png](../Example%20Notebooks/demo\_single\_table\_files/demo\_single\_table\_28\_10%20\(6\).png)

![png](../Example%20Notebooks/demo\_single\_table\_files/demo\_single\_table\_28\_11%20\(6\).png)

![png](../Example%20Notebooks/demo\_single\_table\_files/demo\_single\_table\_28\_12%20\(5\).png)

![png](../Example%20Notebooks/demo\_single\_table\_files/demo\_single\_table\_28\_13%20\(5\).png)

![png](../Example%20Notebooks/demo\_single\_table\_files/demo\_single\_table\_28\_14%20\(4\).png)

**Report without privacy metrics, but includes ML efficacy stuff**

```python
get_full_report(data, sample,discrete_columns,numeric_features,target='income')
```

![png](../Example%20Notebooks/demo\_single\_table\_files/demo\_single\_table\_30\_2%20\(1\).png)

![png](../Example%20Notebooks/demo\_single\_table\_files/demo\_single\_table\_30\_3.png)

![png](../Example%20Notebooks/demo\_single\_table\_files/demo\_single\_table\_30\_4.png)

![png](../Example%20Notebooks/demo\_single\_table\_files/demo\_single\_table\_30\_5.png)

![png](../Example%20Notebooks/demo\_single\_table\_files/demo\_single\_table\_30\_6.png)

![png](../Example%20Notebooks/demo\_single\_table\_files/demo\_single\_table\_30\_7%20\(1\).png)

![png](../Example%20Notebooks/demo\_single\_table\_files/demo\_single\_table\_30\_8.png)

![png](../Example%20Notebooks/demo\_single\_table\_files/demo\_single\_table\_30\_9%20\(1\).png)

![png](../Example%20Notebooks/demo\_single\_table\_files/demo\_single\_table\_30\_10%20\(1\).png)

![png](../Example%20Notebooks/demo\_single\_table\_files/demo\_single\_table\_30\_11.png)

![png](../Example%20Notebooks/demo\_single\_table\_files/demo\_single\_table\_30\_12.png)

![png](../Example%20Notebooks/demo\_single\_table\_files/demo\_single\_table\_30\_13%20\(1\).png)

![png](../Example%20Notebooks/demo\_single\_table\_files/demo\_single\_table\_30\_14%20\(1\).png)

**Report without privacy metrics and without ML efficacy stuff**

```python
get_full_report(data, sample,discrete_columns,numeric_features)
```

![png](../Example%20Notebooks/demo\_single\_table\_files/demo\_single\_table\_30\_2.png)

![png](../Example%20Notebooks/demo\_single\_table\_files/demo\_single\_table\_30\_3%20\(1\).png)

![png](../Example%20Notebooks/demo\_single\_table\_files/demo\_single\_table\_30\_4%20\(1\).png)

![png](../Example%20Notebooks/demo\_single\_table\_files/demo\_single\_table\_30\_5%20\(1\).png)

![png](../Example%20Notebooks/demo\_single\_table\_files/demo\_single\_table\_30\_6%20\(1\).png)

![png](../Example%20Notebooks/demo\_single\_table\_files/demo\_single\_table\_30\_7.png)

![png](../Example%20Notebooks/demo\_single\_table\_files/demo\_single\_table\_30\_8%20\(1\).png)

![png](../Example%20Notebooks/demo\_single\_table\_files/demo\_single\_table\_30\_9.png)

![png](../Example%20Notebooks/demo\_single\_table\_files/demo\_single\_table\_30\_10.png)

![png](../Example%20Notebooks/demo\_single\_table\_files/demo\_single\_table\_30\_11%20\(1\).png)

![png](../Example%20Notebooks/demo\_single\_table\_files/demo\_single\_table\_30\_12%20\(1\).png)

![png](../Example%20Notebooks/demo\_single\_table\_files/demo\_single\_table\_30\_13.png)

![png](../Example%20Notebooks/demo\_single\_table\_files/demo\_single\_table\_30\_14.png)

**Save model to disk**

```python
synth.save('AdversarialAPI.pth')
```

### Normal API: Privately differentiable synthesizer

```python
synth = PrivateTwinSynthesizer(epsilon=0.1,batch_size=64,device='cuda')   ### cpu else cuda
```

```python
synth.fit(data=data,discrete_columns=discrete_columns,update_epsilon=1)
```

```
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
Iteration: [1] Total time: 0:00:01
Iteration: [2]  [  0/159]  eta: 0:01:33  loss_t_fake: 0.0402 (0.1429)  loss_t_real: 0.0031 (0.0033)  time: 0.5825  data: 0.0000  max mem: 63
Iteration: [2]  [ 50/159]  eta: 0:01:03  loss_t_fake: 0.0003 (0.0042)  loss_t_real: 0.0020 (0.0029)  time: 0.5717  data: 0.0000  max mem: 63
Iteration: [2]  [100/159]  eta: 0:00:34  loss_t_fake: 0.0002 (0.0022)  loss_t_real: 0.0012 (0.0022)  time: 0.5615  data: 0.0000  max mem: 63
Iteration: [2]  [150/159]  eta: 0:00:05  loss_t_fake: 0.0001 (0.0015)  loss_t_real: 0.0007 (0.0018)  time: 0.5563  data: 0.0000  max mem: 63
Iteration: [2]  [159/159]  eta: 0:00:00  loss_t_fake: 0.0001 (0.0015)  loss_t_real: 0.0007 (0.0017)  time: 0.5601  data: 0.0000  max mem: 63
Iteration: [2] Total time: 0:01:30
Iteration: [2]  [0/4]  eta: 0:00:00  loss_t_fake: 0.0001 (0.0015)  loss_t_real: 0.0007 (0.0017)  loss_s: 2.9493 (2.9323)  time: 0.1247  data: 0.0000  max mem: 63
Iteration: [2]  [1/4]  eta: 0:00:00  loss_t_fake: 0.0001 (0.0015)  loss_t_real: 0.0007 (0.0017)  loss_s: 2.7469 (2.7971)  time: 0.1356  data: 0.0000  max mem: 63
Iteration: [2]  [2/4]  eta: 0:00:00  loss_t_fake: 0.0001 (0.0015)  loss_t_real: 0.0007 (0.0017)  loss_s: 2.7469 (2.7363)  time: 0.1333  data: 0.0000  max mem: 63
Iteration: [2]  [3/4]  eta: 0:00:00  loss_t_fake: 0.0001 (0.0015)  loss_t_real: 0.0007 (0.0017)  loss_s: 2.5675 (2.5371)  time: 0.1325  data: 0.0000  max mem: 63
Iteration: [2]  [4/4]  eta: 0:00:00  loss_t_fake: 0.0001 (0.0015)  loss_t_real: 0.0007 (0.0017)  loss_s: 2.1635 (2.3490)  time: 0.1315  data: 0.0000  max mem: 63
Iteration: [2] Total time: 0:00:00
Iteration: [2]  [  0/507]  eta: 0:00:01  loss_t_fake: 0.0001 (0.0015)  loss_t_real: 0.0007 (0.0017)  loss_s: 2.1635 (2.3490)  loss_g: 0.0330 (0.0330)  time: 0.0030  data: 0.0000  max mem: 63
Iteration: [2]  [ 50/507]  eta: 0:00:01  loss_t_fake: 0.0001 (0.0015)  loss_t_real: 0.0007 (0.0017)  loss_s: 2.1635 (2.3490)  loss_g: 0.0158 (0.0218)  time: 0.0029  data: 0.0000  max mem: 63
Iteration: [2]  [100/507]  eta: 0:00:01  loss_t_fake: 0.0001 (0.0015)  loss_t_real: 0.0007 (0.0017)  loss_s: 2.1635 (2.3490)  loss_g: 0.0088 (0.0162)  time: 0.0024  data: 0.0000  max mem: 63
Iteration: [2]  [150/507]  eta: 0:00:00  loss_t_fake: 0.0001 (0.0015)  loss_t_real: 0.0007 (0.0017)  loss_s: 2.1635 (2.3490)  loss_g: 0.0063 (0.0132)  time: 0.0022  data: 0.0000  max mem: 63
Iteration: [2]  [200/507]  eta: 0:00:00  loss_t_fake: 0.0001 (0.0015)  loss_t_real: 0.0007 (0.0017)  loss_s: 2.1635 (2.3490)  loss_g: 0.0050 (0.0112)  time: 0.0025  data: 0.0000  max mem: 63
Iteration: [2]  [250/507]  eta: 0:00:00  loss_t_fake: 0.0001 (0.0015)  loss_t_real: 0.0007 (0.0017)  loss_s: 2.1635 (2.3490)  loss_g: 0.0043 (0.0099)  time: 0.0025  data: 0.0000  max mem: 63
Iteration: [2]  [300/507]  eta: 0:00:00  loss_t_fake: 0.0001 (0.0015)  loss_t_real: 0.0007 (0.0017)  loss_s: 2.1635 (2.3490)  loss_g: 0.0037 (0.0089)  time: 0.0025  data: 0.0000  max mem: 63
Iteration: [2]  [350/507]  eta: 0:00:00  loss_t_fake: 0.0001 (0.0015)  loss_t_real: 0.0007 (0.0017)  loss_s: 2.1635 (2.3490)  loss_g: 0.0034 (0.0081)  time: 0.0025  data: 0.0000  max mem: 63
Iteration: [2]  [400/507]  eta: 0:00:00  loss_t_fake: 0.0001 (0.0015)  loss_t_real: 0.0007 (0.0017)  loss_s: 2.1635 (2.3490)  loss_g: 0.0030 (0.0075)  time: 0.0024  data: 0.0000  max mem: 63
Iteration: [2]  [450/507]  eta: 0:00:00  loss_t_fake: 0.0001 (0.0015)  loss_t_real: 0.0007 (0.0017)  loss_s: 2.1635 (2.3490)  loss_g: 0.0028 (0.0070)  time: 0.0026  data: 0.0000  max mem: 63
Iteration: [2]  [500/507]  eta: 0:00:00  loss_t_fake: 0.0001 (0.0015)  loss_t_real: 0.0007 (0.0017)  loss_s: 2.1635 (2.3490)  loss_g: 0.0026 (0.0066)  time: 0.0025  data: 0.0000  max mem: 63
Iteration: [2]  [507/507]  eta: 0:00:00  loss_t_fake: 0.0001 (0.0015)  loss_t_real: 0.0007 (0.0017)  loss_s: 2.1635 (2.3490)  loss_g: 0.0026 (0.0065)  time: 0.0026  data: 0.0000  max mem: 63
Iteration: [2] Total time: 0:00:01
Iteration: [3]  [  0/159]  eta: 0:01:34  loss_t_fake: 0.0272 (0.1460)  loss_t_real: 0.0007 (0.0007)  time: 0.5919  data: 0.0000  max mem: 63
Iteration: [3]  [ 50/159]  eta: 0:01:01  loss_t_fake: 0.0001 (0.0035)  loss_t_real: 0.0006 (0.0008)  time: 0.5586  data: 0.0000  max mem: 63
Iteration: [3]  [100/159]  eta: 0:00:33  loss_t_fake: 0.0001 (0.0018)  loss_t_real: 0.0004 (0.0006)  time: 0.5624  data: 0.0000  max mem: 63
Iteration: [3]  [150/159]  eta: 0:00:05  loss_t_fake: 0.0000 (0.0012)  loss_t_real: 0.0003 (0.0005)  time: 0.5497  data: 0.0000  max mem: 63
Iteration: [3]  [159/159]  eta: 0:00:00  loss_t_fake: 0.0001 (0.0012)  loss_t_real: 0.0003 (0.0005)  time: 0.5559  data: 0.0000  max mem: 63
Iteration: [3] Total time: 0:01:29
Iteration: [3]  [0/4]  eta: 0:00:00  loss_t_fake: 0.0001 (0.0012)  loss_t_real: 0.0003 (0.0005)  loss_s: 3.2282 (3.3129)  time: 0.1346  data: 0.0000  max mem: 63
Iteration: [3]  [1/4]  eta: 0:00:00  loss_t_fake: 0.0001 (0.0012)  loss_t_real: 0.0003 (0.0005)  loss_s: 2.6533 (2.7659)  time: 0.1307  data: 0.0000  max mem: 63
Iteration: [3]  [2/4]  eta: 0:00:00  loss_t_fake: 0.0001 (0.0012)  loss_t_real: 0.0003 (0.0005)  loss_s: 2.3995 (2.5927)  time: 0.1292  data: 0.0000  max mem: 63
Iteration: [3]  [3/4]  eta: 0:00:00  loss_t_fake: 0.0001 (0.0012)  loss_t_real: 0.0003 (0.0005)  loss_s: 2.2632 (2.4963)  time: 0.1302  data: 0.0000  max mem: 63
Iteration: [3]  [4/4]  eta: 0:00:00  loss_t_fake: 0.0001 (0.0012)  loss_t_real: 0.0003 (0.0005)  loss_s: 2.1577 (2.3862)  time: 0.1320  data: 0.0000  max mem: 63
Iteration: [3] Total time: 0:00:00
Iteration: [3]  [  0/507]  eta: 0:00:01  loss_t_fake: 0.0001 (0.0012)  loss_t_real: 0.0003 (0.0005)  loss_s: 2.1577 (2.3862)  loss_g: 0.0332 (0.0332)  time: 0.0030  data: 0.0000  max mem: 63
Iteration: [3]  [ 50/507]  eta: 0:00:01  loss_t_fake: 0.0001 (0.0012)  loss_t_real: 0.0003 (0.0005)  loss_s: 2.1577 (2.3862)  loss_g: 0.0184 (0.0240)  time: 0.0033  data: 0.0000  max mem: 63
Iteration: [3]  [100/507]  eta: 0:00:01  loss_t_fake: 0.0001 (0.0012)  loss_t_real: 0.0003 (0.0005)  loss_s: 2.1577 (2.3862)  loss_g: 0.0099 (0.0180)  time: 0.0024  data: 0.0000  max mem: 63
Iteration: [3]  [150/507]  eta: 0:00:00  loss_t_fake: 0.0001 (0.0012)  loss_t_real: 0.0003 (0.0005)  loss_s: 2.1577 (2.3862)  loss_g: 0.0074 (0.0147)  time: 0.0025  data: 0.0000  max mem: 63
Iteration: [3]  [200/507]  eta: 0:00:00  loss_t_fake: 0.0001 (0.0012)  loss_t_real: 0.0003 (0.0005)  loss_s: 2.1577 (2.3862)  loss_g: 0.0060 (0.0126)  time: 0.0025  data: 0.0000  max mem: 63
Iteration: [3]  [250/507]  eta: 0:00:00  loss_t_fake: 0.0001 (0.0012)  loss_t_real: 0.0003 (0.0005)  loss_s: 2.1577 (2.3862)  loss_g: 0.0051 (0.0112)  time: 0.0026  data: 0.0000  max mem: 63
Iteration: [3]  [300/507]  eta: 0:00:00  loss_t_fake: 0.0001 (0.0012)  loss_t_real: 0.0003 (0.0005)  loss_s: 2.1577 (2.3862)  loss_g: 0.0045 (0.0101)  time: 0.0022  data: 0.0000  max mem: 63
Iteration: [3]  [350/507]  eta: 0:00:00  loss_t_fake: 0.0001 (0.0012)  loss_t_real: 0.0003 (0.0005)  loss_s: 2.1577 (2.3862)  loss_g: 0.0040 (0.0092)  time: 0.0024  data: 0.0000  max mem: 63
Iteration: [3]  [400/507]  eta: 0:00:00  loss_t_fake: 0.0001 (0.0012)  loss_t_real: 0.0003 (0.0005)  loss_s: 2.1577 (2.3862)  loss_g: 0.0037 (0.0086)  time: 0.0028  data: 0.0000  max mem: 63
Iteration: [3]  [450/507]  eta: 0:00:00  loss_t_fake: 0.0001 (0.0012)  loss_t_real: 0.0003 (0.0005)  loss_s: 2.1577 (2.3862)  loss_g: 0.0034 (0.0080)  time: 0.0027  data: 0.0000  max mem: 63
Iteration: [3]  [500/507]  eta: 0:00:00  loss_t_fake: 0.0001 (0.0012)  loss_t_real: 0.0003 (0.0005)  loss_s: 2.1577 (2.3862)  loss_g: 0.0032 (0.0075)  time: 0.0023  data: 0.0000  max mem: 63
Iteration: [3]  [507/507]  eta: 0:00:00  loss_t_fake: 0.0001 (0.0012)  loss_t_real: 0.0003 (0.0005)  loss_s: 2.1577 (2.3862)  loss_g: 0.0032 (0.0075)  time: 0.0023  data: 0.0000  max mem: 63
Iteration: [3] Total time: 0:00:01
Iteration: [4]  [  0/159]  eta: 0:01:28  loss_t_fake: 0.1292 (0.8385)  loss_t_real: 0.0003 (0.0003)  time: 0.5540  data: 0.0000  max mem: 63
Iteration: [4]  [ 50/159]  eta: 0:01:00  loss_t_fake: 0.0001 (0.0176)  loss_t_real: 0.0004 (0.0005)  time: 0.5464  data: 0.0000  max mem: 63
Iteration: [4]  [100/159]  eta: 0:00:33  loss_t_fake: 0.0000 (0.0089)  loss_t_real: 0.0003 (0.0004)  time: 0.5562  data: 0.0000  max mem: 63
Iteration: [4]  [150/159]  eta: 0:00:05  loss_t_fake: 0.0000 (0.0060)  loss_t_real: 0.0002 (0.0004)  time: 0.5533  data: 0.0000  max mem: 63
Iteration: [4]  [159/159]  eta: 0:00:00  loss_t_fake: 0.0000 (0.0056)  loss_t_real: 0.0002 (0.0004)  time: 0.5537  data: 0.0000  max mem: 63
Iteration: [4] Total time: 0:01:28
Iteration: [4]  [0/4]  eta: 0:00:00  loss_t_fake: 0.0000 (0.0056)  loss_t_real: 0.0002 (0.0004)  loss_s: 3.0088 (2.9071)  time: 0.1132  data: 0.0000  max mem: 63
Iteration: [4]  [1/4]  eta: 0:00:00  loss_t_fake: 0.0000 (0.0056)  loss_t_real: 0.0002 (0.0004)  loss_s: 2.4297 (2.6382)  time: 0.1264  data: 0.0000  max mem: 63
Iteration: [4]  [2/4]  eta: 0:00:00  loss_t_fake: 0.0000 (0.0056)  loss_t_real: 0.0002 (0.0004)  loss_s: 2.4297 (2.5435)  time: 0.1302  data: 0.0000  max mem: 63
Iteration: [4]  [3/4]  eta: 0:00:00  loss_t_fake: 0.0000 (0.0056)  loss_t_real: 0.0002 (0.0004)  loss_s: 2.3699 (2.4267)  time: 0.1310  data: 0.0000  max mem: 63
Iteration: [4]  [4/4]  eta: 0:00:00  loss_t_fake: 0.0000 (0.0056)  loss_t_real: 0.0002 (0.0004)  loss_s: 2.1661 (2.3231)  time: 0.1330  data: 0.0000  max mem: 63
Iteration: [4] Total time: 0:00:00
Iteration: [4]  [  0/507]  eta: 0:00:01  loss_t_fake: 0.0000 (0.0056)  loss_t_real: 0.0002 (0.0004)  loss_s: 2.1661 (2.3231)  loss_g: 0.0514 (0.0514)  time: 0.0020  data: 0.0000  max mem: 63
Iteration: [4]  [ 50/507]  eta: 0:00:01  loss_t_fake: 0.0000 (0.0056)  loss_t_real: 0.0002 (0.0004)  loss_s: 2.1661 (2.3231)  loss_g: 0.0272 (0.0354)  time: 0.0033  data: 0.0000  max mem: 63
Iteration: [4]  [100/507]  eta: 0:00:01  loss_t_fake: 0.0000 (0.0056)  loss_t_real: 0.0002 (0.0004)  loss_s: 2.1661 (2.3231)  loss_g: 0.0176 (0.0276)  time: 0.0024  data: 0.0000  max mem: 63
Iteration: [4]  [150/507]  eta: 0:00:01  loss_t_fake: 0.0000 (0.0056)  loss_t_real: 0.0002 (0.0004)  loss_s: 2.1661 (2.3231)  loss_g: 0.0141 (0.0234)  time: 0.0025  data: 0.0000  max mem: 63
Iteration: [4]  [200/507]  eta: 0:00:00  loss_t_fake: 0.0000 (0.0056)  loss_t_real: 0.0002 (0.0004)  loss_s: 2.1661 (2.3231)  loss_g: 0.0121 (0.0207)  time: 0.0024  data: 0.0000  max mem: 63
Iteration: [4]  [250/507]  eta: 0:00:00  loss_t_fake: 0.0000 (0.0056)  loss_t_real: 0.0002 (0.0004)  loss_s: 2.1661 (2.3231)  loss_g: 0.0108 (0.0188)  time: 0.0025  data: 0.0000  max mem: 63
Iteration: [4]  [300/507]  eta: 0:00:00  loss_t_fake: 0.0000 (0.0056)  loss_t_real: 0.0002 (0.0004)  loss_s: 2.1661 (2.3231)  loss_g: 0.0098 (0.0174)  time: 0.0024  data: 0.0000  max mem: 63
Iteration: [4]  [350/507]  eta: 0:00:00  loss_t_fake: 0.0000 (0.0056)  loss_t_real: 0.0002 (0.0004)  loss_s: 2.1661 (2.3231)  loss_g: 0.0089 (0.0162)  time: 0.0026  data: 0.0000  max mem: 63
Iteration: [4]  [400/507]  eta: 0:00:00  loss_t_fake: 0.0000 (0.0056)  loss_t_real: 0.0002 (0.0004)  loss_s: 2.1661 (2.3231)  loss_g: 0.0078 (0.0152)  time: 0.0025  data: 0.0000  max mem: 63
Iteration: [4]  [450/507]  eta: 0:00:00  loss_t_fake: 0.0000 (0.0056)  loss_t_real: 0.0002 (0.0004)  loss_s: 2.1661 (2.3231)  loss_g: 0.0068 (0.0143)  time: 0.0025  data: 0.0000  max mem: 63
Iteration: [4]  [500/507]  eta: 0:00:00  loss_t_fake: 0.0000 (0.0056)  loss_t_real: 0.0002 (0.0004)  loss_s: 2.1661 (2.3231)  loss_g: 0.0061 (0.0135)  time: 0.0022  data: 0.0000  max mem: 63
Iteration: [4]  [507/507]  eta: 0:00:00  loss_t_fake: 0.0000 (0.0056)  loss_t_real: 0.0002 (0.0004)  loss_s: 2.1661 (2.3231)  loss_g: 0.0060 (0.0134)  time: 0.0023  data: 0.0000  max mem: 63
Iteration: [4] Total time: 0:00:01
Iteration: [5]  [  0/159]  eta: 0:01:28  loss_t_fake: 0.0679 (0.2958)  loss_t_real: 0.0002 (0.0003)  time: 0.5530  data: 0.0000  max mem: 63
Iteration: [5]  [ 50/159]  eta: 0:01:01  loss_t_fake: 0.0001 (0.0063)  loss_t_real: 0.0003 (0.0003)  time: 0.5567  data: 0.0000  max mem: 63
Iteration: [5]  [100/159]  eta: 0:00:33  loss_t_fake: 0.0001 (0.0032)  loss_t_real: 0.0002 (0.0003)  time: 0.5527  data: 0.0000  max mem: 63
Iteration: [5]  [150/159]  eta: 0:00:05  loss_t_fake: 0.0000 (0.0022)  loss_t_real: 0.0002 (0.0003)  time: 0.5631  data: 0.0000  max mem: 63
Iteration: [5]  [159/159]  eta: 0:00:00  loss_t_fake: 0.0000 (0.0021)  loss_t_real: 0.0002 (0.0003)  time: 0.5734  data: 0.0000  max mem: 63
Iteration: [5] Total time: 0:01:29
Iteration: [5]  [0/4]  eta: 0:00:00  loss_t_fake: 0.0000 (0.0021)  loss_t_real: 0.0002 (0.0003)  loss_s: 2.9214 (2.8119)  time: 0.1332  data: 0.0000  max mem: 63
Iteration: [5]  [1/4]  eta: 0:00:00  loss_t_fake: 0.0000 (0.0021)  loss_t_real: 0.0002 (0.0003)  loss_s: 2.5288 (2.5705)  time: 0.1277  data: 0.0000  max mem: 63
Iteration: [5]  [2/4]  eta: 0:00:00  loss_t_fake: 0.0000 (0.0021)  loss_t_real: 0.0002 (0.0003)  loss_s: 2.4572 (2.3596)  time: 0.1240  data: 0.0000  max mem: 63
Iteration: [5]  [3/4]  eta: 0:00:00  loss_t_fake: 0.0000 (0.0021)  loss_t_real: 0.0002 (0.0003)  loss_s: 2.0153 (2.1572)  time: 0.1219  data: 0.0000  max mem: 63
Iteration: [5]  [4/4]  eta: 0:00:00  loss_t_fake: 0.0000 (0.0021)  loss_t_real: 0.0002 (0.0003)  loss_s: 1.7323 (1.9761)  time: 0.1239  data: 0.0000  max mem: 63
Iteration: [5] Total time: 0:00:00
Iteration: [5]  [  0/507]  eta: 0:00:02  loss_t_fake: 0.0000 (0.0021)  loss_t_real: 0.0002 (0.0003)  loss_s: 1.7323 (1.9761)  loss_g: 0.1233 (0.1233)  time: 0.0050  data: 0.0000  max mem: 63
Iteration: [5]  [ 50/507]  eta: 0:00:01  loss_t_fake: 0.0000 (0.0021)  loss_t_real: 0.0002 (0.0003)  loss_s: 1.7323 (1.9761)  loss_g: 0.0587 (0.0755)  time: 0.0026  data: 0.0000  max mem: 63
Iteration: [5]  [100/507]  eta: 0:00:01  loss_t_fake: 0.0000 (0.0021)  loss_t_real: 0.0002 (0.0003)  loss_s: 1.7323 (1.9761)  loss_g: 0.0459 (0.0623)  time: 0.0023  data: 0.0000  max mem: 63
Iteration: [5]  [150/507]  eta: 0:00:00  loss_t_fake: 0.0000 (0.0021)  loss_t_real: 0.0002 (0.0003)  loss_s: 1.7323 (1.9761)  loss_g: 0.0322 (0.0537)  time: 0.0026  data: 0.0000  max mem: 63
Iteration: [5]  [200/507]  eta: 0:00:00  loss_t_fake: 0.0000 (0.0021)  loss_t_real: 0.0002 (0.0003)  loss_s: 1.7323 (1.9761)  loss_g: 0.0273 (0.0474)  time: 0.0025  data: 0.0000  max mem: 63
Iteration: [5]  [250/507]  eta: 0:00:00  loss_t_fake: 0.0000 (0.0021)  loss_t_real: 0.0002 (0.0003)  loss_s: 1.7323 (1.9761)  loss_g: 0.0246 (0.0430)  time: 0.0025  data: 0.0000  max mem: 63
Iteration: [5]  [300/507]  eta: 0:00:00  loss_t_fake: 0.0000 (0.0021)  loss_t_real: 0.0002 (0.0003)  loss_s: 1.7323 (1.9761)  loss_g: 0.0224 (0.0397)  time: 0.0025  data: 0.0000  max mem: 63
Iteration: [5]  [350/507]  eta: 0:00:00  loss_t_fake: 0.0000 (0.0021)  loss_t_real: 0.0002 (0.0003)  loss_s: 1.7323 (1.9761)  loss_g: 0.0202 (0.0370)  time: 0.0024  data: 0.0000  max mem: 63
Iteration: [5]  [400/507]  eta: 0:00:00  loss_t_fake: 0.0000 (0.0021)  loss_t_real: 0.0002 (0.0003)  loss_s: 1.7323 (1.9761)  loss_g: 0.0182 (0.0348)  time: 0.0026  data: 0.0000  max mem: 63
Iteration: [5]  [450/507]  eta: 0:00:00  loss_t_fake: 0.0000 (0.0021)  loss_t_real: 0.0002 (0.0003)  loss_s: 1.7323 (1.9761)  loss_g: 0.0162 (0.0328)  time: 0.0026  data: 0.0000  max mem: 63
Iteration: [5]  [500/507]  eta: 0:00:00  loss_t_fake: 0.0000 (0.0021)  loss_t_real: 0.0002 (0.0003)  loss_s: 1.7323 (1.9761)  loss_g: 0.0150 (0.0310)  time: 0.0025  data: 0.0000  max mem: 63
Iteration: [5]  [507/507]  eta: 0:00:00  loss_t_fake: 0.0000 (0.0021)  loss_t_real: 0.0002 (0.0003)  loss_s: 1.7323 (1.9761)  loss_g: 0.0149 (0.0308)  time: 0.0025  data: 0.0000  max mem: 63
Iteration: [5] Total time: 0:00:01
```

```python
sample = synth.sample(1000)
```

```python
metrics = SingleTableMetric.get_subclasses()
numeric_features = ['capital-gain','capital-loss','hours-per-week']
discrete_columns = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
    "income",
]
```

**Report with privacy metrics**

```python
get_full_report(data, sample,discrete_columns,numeric_features, key_fields=['age','workclass','education'],sensitive_fields = ['income'])
```

![png](../Example%20Notebooks/demo\_single\_table\_files/demo\_single\_table\_42\_2%20\(5\).png)

![png](../Example%20Notebooks/demo\_single\_table\_files/demo\_single\_table\_42\_3%20\(4\).png)

![png](../Example%20Notebooks/demo\_single\_table\_files/demo\_single\_table\_42\_4%20\(6\).png)

![png](../Example%20Notebooks/demo\_single\_table\_files/demo\_single\_table\_42\_5%20\(5\).png)

![png](../Example%20Notebooks/demo\_single\_table\_files/demo\_single\_table\_42\_6%20\(6\).png)

![png](../Example%20Notebooks/demo\_single\_table\_files/demo\_single\_table\_42\_7%20\(5\).png)

![png](../Example%20Notebooks/demo\_single\_table\_files/demo\_single\_table\_42\_8%20\(3\).png)

![png](../Example%20Notebooks/demo\_single\_table\_files/demo\_single\_table\_42\_9%20\(6\).png)

![png](../Example%20Notebooks/demo\_single\_table\_files/demo\_single\_table\_42\_10%20\(6\).png)

![png](../Example%20Notebooks/demo\_single\_table\_files/demo\_single\_table\_42\_11%20\(6\).png)

![png](../Example%20Notebooks/demo\_single\_table\_files/demo\_single\_table\_42\_12%20\(4\).png)

![png](../Example%20Notebooks/demo\_single\_table\_files/demo\_single\_table\_42\_13%20\(6\).png)

![png](../Example%20Notebooks/demo\_single\_table\_files/demo\_single\_table\_42\_14%20\(4\).png)

**Report without privacy metrics, but includes ML efficacy stuff**

```python
get_full_report(data, sample,discrete_columns,numeric_features,target='income')
```

![png](../Example%20Notebooks/demo\_single\_table\_files/demo\_single\_table\_44\_2%20\(1\).png)

![png](../Example%20Notebooks/demo\_single\_table\_files/demo\_single\_table\_44\_3%20\(1\).png)

![png](../Example%20Notebooks/demo\_single\_table\_files/demo\_single\_table\_44\_4%20\(1\).png)

![png](../Example%20Notebooks/demo\_single\_table\_files/demo\_single\_table\_44\_5.png)

![png](../Example%20Notebooks/demo\_single\_table\_files/demo\_single\_table\_44\_6%20\(1\).png)

![png](../Example%20Notebooks/demo\_single\_table\_files/demo\_single\_table\_44\_7%20\(1\).png)

![png](../Example%20Notebooks/demo\_single\_table\_files/demo\_single\_table\_44\_8.png)

![png](../Example%20Notebooks/demo\_single\_table\_files/demo\_single\_table\_44\_9.png)

![png](../Example%20Notebooks/demo\_single\_table\_files/demo\_single\_table\_44\_10.png)

![png](../Example%20Notebooks/demo\_single\_table\_files/demo\_single\_table\_44\_11%20\(1\).png)

![png](../Example%20Notebooks/demo\_single\_table\_files/demo\_single\_table\_44\_12.png)

![png](../Example%20Notebooks/demo\_single\_table\_files/demo\_single\_table\_44\_13.png)

![png](../Example%20Notebooks/demo\_single\_table\_files/demo\_single\_table\_44\_14.png)

**Report without privacy metrics and without ML efficacy stuff**

```python
get_full_report(data, sample,discrete_columns,numeric_features)
```

![png](../Example%20Notebooks/demo\_single\_table\_files/demo\_single\_table\_44\_2.png)

![png](../Example%20Notebooks/demo\_single\_table\_files/demo\_single\_table\_44\_3.png)

![png](../Example%20Notebooks/demo\_single\_table\_files/demo\_single\_table\_44\_4.png)

![png](../Example%20Notebooks/demo\_single\_table\_files/demo\_single\_table\_44\_5%20\(1\).png)

![png](../Example%20Notebooks/demo\_single\_table\_files/demo\_single\_table\_44\_6.png)

![png](../Example%20Notebooks/demo\_single\_table\_files/demo\_single\_table\_44\_7.png)

![png](../Example%20Notebooks/demo\_single\_table\_files/demo\_single\_table\_44\_8%20\(1\).png)

![png](../Example%20Notebooks/demo\_single\_table\_files/demo\_single\_table\_44\_9%20\(1\).png)

![png](../Example%20Notebooks/demo\_single\_table\_files/demo\_single\_table\_44\_10%20\(1\).png)

![png](../Example%20Notebooks/demo\_single\_table\_files/demo\_single\_table\_44\_11.png)

![png](../Example%20Notebooks/demo\_single\_table\_files/demo\_single\_table\_44\_12%20\(1\).png)

![png](../Example%20Notebooks/demo\_single\_table\_files/demo\_single\_table\_44\_13%20\(1\).png)

![png](../Example%20Notebooks/demo\_single\_table\_files/demo\_single\_table\_44\_14%20\(1\).png)

**Save model to disk**

```python
synth.save('PrivateModelNormalAPI.pth')
```

### Adversarial API: Privately differentiable synthesizer

```python
synth = PrivateTwinSynthesizer(epsilon=0.1,batch_size=64,device='cuda')   ### cpu else cuda
```

```python
synth.fit_adversarial(data=data,discrete_columns=discrete_columns,update_epsilon=1)
```

```
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
Iteration: [1] Total time: 0:00:00
Iteration: [2]  [  0/129]  eta: 0:00:48  loss_t_fake: 0.1730 (0.5114)  loss_t_real: 0.0059 (0.0058)  time: 0.3710  data: 0.0000  max mem: 63
Iteration: [2]  [ 50/129]  eta: 0:00:29  loss_t_fake: 0.0007 (0.0145)  loss_t_real: 0.0033 (0.0057)  time: 0.3688  data: 0.0000  max mem: 63
Iteration: [2]  [100/129]  eta: 0:00:11  loss_t_fake: 0.0004 (0.0076)  loss_t_real: 0.0018 (0.0042)  time: 0.3665  data: 0.0000  max mem: 63
Iteration: [2]  [129/129]  eta: 0:00:00  loss_t_fake: 0.0003 (0.0060)  loss_t_real: 0.0013 (0.0037)  time: 0.3709  data: 0.0000  max mem: 63
Iteration: [2] Total time: 0:00:48
Iteration: [2]  [0/4]  eta: 0:00:00  loss_t_fake: 0.0003 (0.0060)  loss_t_real: 0.0013 (0.0037)  loss_s: 2.6770 (2.6076)  time: 0.0997  data: 0.0000  max mem: 63
Iteration: [2]  [1/4]  eta: 0:00:00  loss_t_fake: 0.0003 (0.0060)  loss_t_real: 0.0013 (0.0037)  loss_s: 2.4557 (2.4655)  time: 0.0997  data: 0.0000  max mem: 63
Iteration: [2]  [2/4]  eta: 0:00:00  loss_t_fake: 0.0003 (0.0060)  loss_t_real: 0.0013 (0.0037)  loss_s: 2.4130 (2.3959)  time: 0.0997  data: 0.0000  max mem: 63
Iteration: [2]  [3/4]  eta: 0:00:00  loss_t_fake: 0.0003 (0.0060)  loss_t_real: 0.0013 (0.0037)  loss_s: 2.2592 (2.2764)  time: 0.1002  data: 0.0000  max mem: 63
Iteration: [2]  [4/4]  eta: 0:00:00  loss_t_fake: 0.0003 (0.0060)  loss_t_real: 0.0013 (0.0037)  loss_s: 2.0317 (2.1671)  time: 0.1043  data: 0.0000  max mem: 63
Iteration: [2] Total time: 0:00:00
Iteration: [2]  [  0/406]  eta: 0:00:00  loss_t_fake: 0.0003 (0.0060)  loss_t_real: 0.0013 (0.0037)  loss_s: 2.0317 (2.1671)  loss_g: 0.0541 (0.0541)  time: 0.0020  data: 0.0000  max mem: 63
Iteration: [2]  [ 50/406]  eta: 0:00:00  loss_t_fake: 0.0003 (0.0060)  loss_t_real: 0.0013 (0.0037)  loss_s: 2.0317 (2.1671)  loss_g: 0.0277 (0.0375)  time: 0.0020  data: 0.0000  max mem: 63
Iteration: [2]  [100/406]  eta: 0:00:00  loss_t_fake: 0.0003 (0.0060)  loss_t_real: 0.0013 (0.0037)  loss_s: 2.0317 (2.1671)  loss_g: 0.0166 (0.0284)  time: 0.0021  data: 0.0000  max mem: 63
Iteration: [2]  [150/406]  eta: 0:00:00  loss_t_fake: 0.0003 (0.0060)  loss_t_real: 0.0013 (0.0037)  loss_s: 2.0317 (2.1671)  loss_g: 0.0118 (0.0234)  time: 0.0020  data: 0.0000  max mem: 63
Iteration: [2]  [200/406]  eta: 0:00:00  loss_t_fake: 0.0003 (0.0060)  loss_t_real: 0.0013 (0.0037)  loss_s: 2.0317 (2.1671)  loss_g: 0.0088 (0.0200)  time: 0.0020  data: 0.0000  max mem: 63
Iteration: [2]  [250/406]  eta: 0:00:00  loss_t_fake: 0.0003 (0.0060)  loss_t_real: 0.0013 (0.0037)  loss_s: 2.0317 (2.1671)  loss_g: 0.0066 (0.0174)  time: 0.0021  data: 0.0000  max mem: 63
Iteration: [2]  [300/406]  eta: 0:00:00  loss_t_fake: 0.0003 (0.0060)  loss_t_real: 0.0013 (0.0037)  loss_s: 2.0317 (2.1671)  loss_g: 0.0052 (0.0155)  time: 0.0020  data: 0.0000  max mem: 63
Iteration: [2]  [350/406]  eta: 0:00:00  loss_t_fake: 0.0003 (0.0060)  loss_t_real: 0.0013 (0.0037)  loss_s: 2.0317 (2.1671)  loss_g: 0.0041 (0.0139)  time: 0.0022  data: 0.0000  max mem: 63
Iteration: [2]  [400/406]  eta: 0:00:00  loss_t_fake: 0.0003 (0.0060)  loss_t_real: 0.0013 (0.0037)  loss_s: 2.0317 (2.1671)  loss_g: 0.0035 (0.0126)  time: 0.0022  data: 0.0000  max mem: 63
Iteration: [2]  [406/406]  eta: 0:00:00  loss_t_fake: 0.0003 (0.0060)  loss_t_real: 0.0013 (0.0037)  loss_s: 2.0317 (2.1671)  loss_g: 0.0035 (0.0125)  time: 0.0022  data: 0.0000  max mem: 63
Iteration: [2] Total time: 0:00:00
Iteration: [3]  [  0/129]  eta: 0:00:50  loss_t_fake: 0.1923 (0.6392)  loss_t_real: 0.0015 (0.0016)  time: 0.3890  data: 0.0000  max mem: 63
Iteration: [3]  [ 50/129]  eta: 0:00:30  loss_t_fake: 0.0003 (0.0150)  loss_t_real: 0.0016 (0.0024)  time: 0.3686  data: 0.0000  max mem: 63
Iteration: [3]  [100/129]  eta: 0:00:11  loss_t_fake: 0.0002 (0.0077)  loss_t_real: 0.0010 (0.0019)  time: 0.3740  data: 0.0000  max mem: 63
Iteration: [3]  [129/129]  eta: 0:00:00  loss_t_fake: 0.0002 (0.0060)  loss_t_real: 0.0008 (0.0017)  time: 0.3678  data: 0.0000  max mem: 63
Iteration: [3] Total time: 0:00:48
Iteration: [3]  [0/4]  eta: 0:00:00  loss_t_fake: 0.0002 (0.0060)  loss_t_real: 0.0008 (0.0017)  loss_s: 2.8905 (2.8601)  time: 0.1027  data: 0.0000  max mem: 63
Iteration: [3]  [1/4]  eta: 0:00:00  loss_t_fake: 0.0002 (0.0060)  loss_t_real: 0.0008 (0.0017)  loss_s: 2.6492 (2.7154)  time: 0.1007  data: 0.0000  max mem: 63
Iteration: [3]  [2/4]  eta: 0:00:00  loss_t_fake: 0.0002 (0.0060)  loss_t_real: 0.0008 (0.0017)  loss_s: 2.4404 (2.5976)  time: 0.1007  data: 0.0000  max mem: 63
Iteration: [3]  [3/4]  eta: 0:00:00  loss_t_fake: 0.0002 (0.0060)  loss_t_real: 0.0008 (0.0017)  loss_s: 2.3743 (2.4482)  time: 0.1015  data: 0.0000  max mem: 63
Iteration: [3]  [4/4]  eta: 0:00:00  loss_t_fake: 0.0002 (0.0060)  loss_t_real: 0.0008 (0.0017)  loss_s: 2.1485 (2.2820)  time: 0.1013  data: 0.0000  max mem: 63
Iteration: [3] Total time: 0:00:00
Iteration: [3]  [  0/406]  eta: 0:00:01  loss_t_fake: 0.0002 (0.0060)  loss_t_real: 0.0008 (0.0017)  loss_s: 2.1485 (2.2820)  loss_g: 0.0490 (0.0490)  time: 0.0040  data: 0.0000  max mem: 63
Iteration: [3]  [ 50/406]  eta: 0:00:00  loss_t_fake: 0.0002 (0.0060)  loss_t_real: 0.0008 (0.0017)  loss_s: 2.1485 (2.2820)  loss_g: 0.0210 (0.0308)  time: 0.0024  data: 0.0000  max mem: 63
Iteration: [3]  [100/406]  eta: 0:00:00  loss_t_fake: 0.0002 (0.0060)  loss_t_real: 0.0008 (0.0017)  loss_s: 2.1485 (2.2820)  loss_g: 0.0125 (0.0229)  time: 0.0021  data: 0.0000  max mem: 63
Iteration: [3]  [150/406]  eta: 0:00:00  loss_t_fake: 0.0002 (0.0060)  loss_t_real: 0.0008 (0.0017)  loss_s: 2.1485 (2.2820)  loss_g: 0.0094 (0.0187)  time: 0.0021  data: 0.0000  max mem: 63
Iteration: [3]  [200/406]  eta: 0:00:00  loss_t_fake: 0.0002 (0.0060)  loss_t_real: 0.0008 (0.0017)  loss_s: 2.1485 (2.2820)  loss_g: 0.0075 (0.0160)  time: 0.0021  data: 0.0000  max mem: 63
Iteration: [3]  [250/406]  eta: 0:00:00  loss_t_fake: 0.0002 (0.0060)  loss_t_real: 0.0008 (0.0017)  loss_s: 2.1485 (2.2820)  loss_g: 0.0064 (0.0142)  time: 0.0021  data: 0.0000  max mem: 63
Iteration: [3]  [300/406]  eta: 0:00:00  loss_t_fake: 0.0002 (0.0060)  loss_t_real: 0.0008 (0.0017)  loss_s: 2.1485 (2.2820)  loss_g: 0.0058 (0.0128)  time: 0.0021  data: 0.0000  max mem: 63
Iteration: [3]  [350/406]  eta: 0:00:00  loss_t_fake: 0.0002 (0.0060)  loss_t_real: 0.0008 (0.0017)  loss_s: 2.1485 (2.2820)  loss_g: 0.0052 (0.0118)  time: 0.0021  data: 0.0000  max mem: 63
Iteration: [3]  [400/406]  eta: 0:00:00  loss_t_fake: 0.0002 (0.0060)  loss_t_real: 0.0008 (0.0017)  loss_s: 2.1485 (2.2820)  loss_g: 0.0048 (0.0109)  time: 0.0020  data: 0.0000  max mem: 63
Iteration: [3]  [406/406]  eta: 0:00:00  loss_t_fake: 0.0002 (0.0060)  loss_t_real: 0.0008 (0.0017)  loss_s: 2.1485 (2.2820)  loss_g: 0.0047 (0.0108)  time: 0.0020  data: 0.0000  max mem: 63
Iteration: [3] Total time: 0:00:00
Iteration: [4]  [  0/129]  eta: 0:00:49  loss_t_fake: 0.3875 (1.2377)  loss_t_real: 0.0010 (0.0010)  time: 0.3770  data: 0.0000  max mem: 63
Iteration: [4]  [ 50/129]  eta: 0:00:30  loss_t_fake: 0.0001 (0.0269)  loss_t_real: 0.0014 (0.0021)  time: 0.3801  data: 0.0000  max mem: 63
Iteration: [4]  [100/129]  eta: 0:00:11  loss_t_fake: 0.0001 (0.0137)  loss_t_real: 0.0011 (0.0017)  time: 0.3934  data: 0.0000  max mem: 63
Iteration: [4]  [129/129]  eta: 0:00:00  loss_t_fake: 0.0001 (0.0106)  loss_t_real: 0.0009 (0.0016)  time: 0.4011  data: 0.0000  max mem: 63
Iteration: [4] Total time: 0:00:51
Iteration: [4]  [0/4]  eta: 0:00:00  loss_t_fake: 0.0001 (0.0106)  loss_t_real: 0.0009 (0.0016)  loss_s: 2.7158 (2.7590)  time: 0.1027  data: 0.0000  max mem: 63
Iteration: [4]  [1/4]  eta: 0:00:00  loss_t_fake: 0.0001 (0.0106)  loss_t_real: 0.0009 (0.0016)  loss_s: 2.5139 (2.4838)  time: 0.1027  data: 0.0000  max mem: 63
Iteration: [4]  [2/4]  eta: 0:00:00  loss_t_fake: 0.0001 (0.0106)  loss_t_real: 0.0009 (0.0016)  loss_s: 2.3989 (2.3807)  time: 0.1070  data: 0.0000  max mem: 63
Iteration: [4]  [3/4]  eta: 0:00:00  loss_t_fake: 0.0001 (0.0106)  loss_t_real: 0.0009 (0.0016)  loss_s: 2.0983 (2.2751)  time: 0.1065  data: 0.0000  max mem: 63
Iteration: [4]  [4/4]  eta: 0:00:00  loss_t_fake: 0.0001 (0.0106)  loss_t_real: 0.0009 (0.0016)  loss_s: 1.9317 (2.1118)  time: 0.1061  data: 0.0000  max mem: 63
Iteration: [4] Total time: 0:00:00
Iteration: [4]  [  0/406]  eta: 0:00:00  loss_t_fake: 0.0001 (0.0106)  loss_t_real: 0.0009 (0.0016)  loss_s: 1.9317 (2.1118)  loss_g: 0.0874 (0.0874)  time: 0.0020  data: 0.0000  max mem: 63
Iteration: [4]  [ 50/406]  eta: 0:00:00  loss_t_fake: 0.0001 (0.0106)  loss_t_real: 0.0009 (0.0016)  loss_s: 1.9317 (2.1118)  loss_g: 0.0412 (0.0544)  time: 0.0023  data: 0.0000  max mem: 63
Iteration: [4]  [100/406]  eta: 0:00:00  loss_t_fake: 0.0001 (0.0106)  loss_t_real: 0.0009 (0.0016)  loss_s: 1.9317 (2.1118)  loss_g: 0.0283 (0.0429)  time: 0.0024  data: 0.0000  max mem: 63
Iteration: [4]  [150/406]  eta: 0:00:00  loss_t_fake: 0.0001 (0.0106)  loss_t_real: 0.0009 (0.0016)  loss_s: 1.9317 (2.1118)  loss_g: 0.0209 (0.0362)  time: 0.0022  data: 0.0000  max mem: 63
Iteration: [4]  [200/406]  eta: 0:00:00  loss_t_fake: 0.0001 (0.0106)  loss_t_real: 0.0009 (0.0016)  loss_s: 1.9317 (2.1118)  loss_g: 0.0174 (0.0318)  time: 0.0023  data: 0.0000  max mem: 63
Iteration: [4]  [250/406]  eta: 0:00:00  loss_t_fake: 0.0001 (0.0106)  loss_t_real: 0.0009 (0.0016)  loss_s: 1.9317 (2.1118)  loss_g: 0.0153 (0.0286)  time: 0.0021  data: 0.0000  max mem: 63
Iteration: [4]  [300/406]  eta: 0:00:00  loss_t_fake: 0.0001 (0.0106)  loss_t_real: 0.0009 (0.0016)  loss_s: 1.9317 (2.1118)  loss_g: 0.0138 (0.0262)  time: 0.0021  data: 0.0000  max mem: 63
Iteration: [4]  [350/406]  eta: 0:00:00  loss_t_fake: 0.0001 (0.0106)  loss_t_real: 0.0009 (0.0016)  loss_s: 1.9317 (2.1118)  loss_g: 0.0128 (0.0244)  time: 0.0021  data: 0.0000  max mem: 63
Iteration: [4]  [400/406]  eta: 0:00:00  loss_t_fake: 0.0001 (0.0106)  loss_t_real: 0.0009 (0.0016)  loss_s: 1.9317 (2.1118)  loss_g: 0.0118 (0.0228)  time: 0.0021  data: 0.0000  max mem: 63
Iteration: [4]  [406/406]  eta: 0:00:00  loss_t_fake: 0.0001 (0.0106)  loss_t_real: 0.0009 (0.0016)  loss_s: 1.9317 (2.1118)  loss_g: 0.0116 (0.0227)  time: 0.0023  data: 0.0000  max mem: 63
Iteration: [4] Total time: 0:00:00
Iteration: [5]  [  0/129]  eta: 0:00:52  loss_t_fake: 0.0701 (0.2647)  loss_t_real: 0.0010 (0.0009)  time: 0.4029  data: 0.0000  max mem: 63
Iteration: [5]  [ 50/129]  eta: 0:00:32  loss_t_fake: 0.0002 (0.0062)  loss_t_real: 0.0010 (0.0014)  time: 0.4009  data: 0.0000  max mem: 63
Iteration: [5]  [100/129]  eta: 0:00:11  loss_t_fake: 0.0001 (0.0032)  loss_t_real: 0.0006 (0.0011)  time: 0.3935  data: 0.0000  max mem: 63
Iteration: [5]  [129/129]  eta: 0:00:00  loss_t_fake: 0.0001 (0.0025)  loss_t_real: 0.0005 (0.0010)  time: 0.3969  data: 0.0000  max mem: 63
Iteration: [5] Total time: 0:00:51
Iteration: [5]  [0/4]  eta: 0:00:00  loss_t_fake: 0.0001 (0.0025)  loss_t_real: 0.0005 (0.0010)  loss_s: 2.1768 (2.1855)  time: 0.1007  data: 0.0000  max mem: 63
Iteration: [5]  [1/4]  eta: 0:00:00  loss_t_fake: 0.0001 (0.0025)  loss_t_real: 0.0005 (0.0010)  loss_s: 2.1295 (2.0597)  time: 0.1017  data: 0.0000  max mem: 63
Iteration: [5]  [2/4]  eta: 0:00:00  loss_t_fake: 0.0001 (0.0025)  loss_t_real: 0.0005 (0.0010)  loss_s: 1.9087 (1.9015)  time: 0.1027  data: 0.0000  max mem: 63
Iteration: [5]  [3/4]  eta: 0:00:00  loss_t_fake: 0.0001 (0.0025)  loss_t_real: 0.0005 (0.0010)  loss_s: 1.7018 (1.7505)  time: 0.1027  data: 0.0000  max mem: 63
Iteration: [5]  [4/4]  eta: 0:00:00  loss_t_fake: 0.0001 (0.0025)  loss_t_real: 0.0005 (0.0010)  loss_s: 1.4207 (1.6208)  time: 0.1031  data: 0.0000  max mem: 63
Iteration: [5] Total time: 0:00:00
Iteration: [5]  [  0/406]  eta: 0:00:01  loss_t_fake: 0.0001 (0.0025)  loss_t_real: 0.0005 (0.0010)  loss_s: 1.4207 (1.6208)  loss_g: 0.1619 (0.1619)  time: 0.0030  data: 0.0000  max mem: 63
Iteration: [5]  [ 50/406]  eta: 0:00:00  loss_t_fake: 0.0001 (0.0025)  loss_t_real: 0.0005 (0.0010)  loss_s: 1.4207 (1.6208)  loss_g: 0.0890 (0.1097)  time: 0.0023  data: 0.0000  max mem: 63
Iteration: [5]  [100/406]  eta: 0:00:00  loss_t_fake: 0.0001 (0.0025)  loss_t_real: 0.0005 (0.0010)  loss_s: 1.4207 (1.6208)  loss_g: 0.0699 (0.0922)  time: 0.0024  data: 0.0000  max mem: 63
Iteration: [5]  [150/406]  eta: 0:00:00  loss_t_fake: 0.0001 (0.0025)  loss_t_real: 0.0005 (0.0010)  loss_s: 1.4207 (1.6208)  loss_g: 0.0582 (0.0822)  time: 0.0022  data: 0.0000  max mem: 63
Iteration: [5]  [200/406]  eta: 0:00:00  loss_t_fake: 0.0001 (0.0025)  loss_t_real: 0.0005 (0.0010)  loss_s: 1.4207 (1.6208)  loss_g: 0.0510 (0.0749)  time: 0.0021  data: 0.0000  max mem: 63
Iteration: [5]  [250/406]  eta: 0:00:00  loss_t_fake: 0.0001 (0.0025)  loss_t_real: 0.0005 (0.0010)  loss_s: 1.4207 (1.6208)  loss_g: 0.0471 (0.0696)  time: 0.0023  data: 0.0000  max mem: 63
Iteration: [5]  [300/406]  eta: 0:00:00  loss_t_fake: 0.0001 (0.0025)  loss_t_real: 0.0005 (0.0010)  loss_s: 1.4207 (1.6208)  loss_g: 0.0444 (0.0655)  time: 0.0021  data: 0.0000  max mem: 63
Iteration: [5]  [350/406]  eta: 0:00:00  loss_t_fake: 0.0001 (0.0025)  loss_t_real: 0.0005 (0.0010)  loss_s: 1.4207 (1.6208)  loss_g: 0.0418 (0.0623)  time: 0.0021  data: 0.0000  max mem: 63
Iteration: [5]  [400/406]  eta: 0:00:00  loss_t_fake: 0.0001 (0.0025)  loss_t_real: 0.0005 (0.0010)  loss_s: 1.4207 (1.6208)  loss_g: 0.0381 (0.0594)  time: 0.0022  data: 0.0000  max mem: 63
Iteration: [5]  [406/406]  eta: 0:00:00  loss_t_fake: 0.0001 (0.0025)  loss_t_real: 0.0005 (0.0010)  loss_s: 1.4207 (1.6208)  loss_g: 0.0377 (0.0591)  time: 0.0023  data: 0.0000  max mem: 63
Iteration: [5] Total time: 0:00:00
```

```python
sample = synth.sample_adversarial(data,1000)
```

```
[Parallel(n_jobs=-1)]: Using backend ThreadingBackend with 8 concurrent workers.
[Parallel(n_jobs=-1)]: Done  34 tasks      | elapsed:    0.2s
[Parallel(n_jobs=-1)]: Done 100 out of 100 | elapsed:    0.6s finished
[Parallel(n_jobs=8)]: Using backend ThreadingBackend with 8 concurrent workers.
[Parallel(n_jobs=8)]: Done  34 tasks      | elapsed:    0.0s
[Parallel(n_jobs=8)]: Done 100 out of 100 | elapsed:    0.0s finished
```

```python
metrics = SingleTableMetric.get_subclasses()
numeric_features = ['capital-gain','capital-loss','hours-per-week']
discrete_columns = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
    "income",
]
```

**Report with privacy metrics**

```python
get_full_report(data, sample,discrete_columns,numeric_features, key_fields=['age','workclass','education'],sensitive_fields = ['income'])
```

![png](../Example%20Notebooks/demo\_single\_table\_files/demo\_single\_table\_56\_2%20\(6\).png)

![png](../Example%20Notebooks/demo\_single\_table\_files/demo\_single\_table\_56\_3%20\(5\).png)

![png](../Example%20Notebooks/demo\_single\_table\_files/demo\_single\_table\_56\_4%20\(6\).png)

![png](../Example%20Notebooks/demo\_single\_table\_files/demo\_single\_table\_56\_5%20\(6\).png)

![png](../Example%20Notebooks/demo\_single\_table\_files/demo\_single\_table\_56\_6%20\(4\).png)

![png](../Example%20Notebooks/demo\_single\_table\_files/demo\_single\_table\_56\_7%20\(6\).png)

![png](../Example%20Notebooks/demo\_single\_table\_files/demo\_single\_table\_56\_8%20\(6\).png)

![png](../Example%20Notebooks/demo\_single\_table\_files/demo\_single\_table\_56\_9%20\(6\).png)

![png](../Example%20Notebooks/demo\_single\_table\_files/demo\_single\_table\_56\_10%20\(6\).png)

![png](../Example%20Notebooks/demo\_single\_table\_files/demo\_single\_table\_56\_11%20\(6\).png)

![png](../Example%20Notebooks/demo\_single\_table\_files/demo\_single\_table\_56\_12%20\(4\).png)

![png](../Example%20Notebooks/demo\_single\_table\_files/demo\_single\_table\_56\_13%20\(5\).png)

![png](../Example%20Notebooks/demo\_single\_table\_files/demo\_single\_table\_56\_14%20\(6\).png)

**Report without privacy metrics, but includes ML efficacy stuff**

```python
get_full_report(data, sample,discrete_columns,numeric_features,target='income')
```

![png](../Example%20Notebooks/demo\_single\_table\_files/demo\_single\_table\_58\_2.png)

![png](../Example%20Notebooks/demo\_single\_table\_files/demo\_single\_table\_58\_3.png)

![png](../Example%20Notebooks/demo\_single\_table\_files/demo\_single\_table\_58\_4%20\(1\).png)

![png](../Example%20Notebooks/demo\_single\_table\_files/demo\_single\_table\_58\_5%20\(1\).png)

![png](../Example%20Notebooks/demo\_single\_table\_files/demo\_single\_table\_58\_6.png)

![png](../Example%20Notebooks/demo\_single\_table\_files/demo\_single\_table\_58\_7%20\(1\).png)

![png](../Example%20Notebooks/demo\_single\_table\_files/demo\_single\_table\_58\_8.png)

![png](../Example%20Notebooks/demo\_single\_table\_files/demo\_single\_table\_58\_9.png)

![png](../Example%20Notebooks/demo\_single\_table\_files/demo\_single\_table\_58\_10.png)

![png](../Example%20Notebooks/demo\_single\_table\_files/demo\_single\_table\_58\_11%20\(1\).png)

![png](../Example%20Notebooks/demo\_single\_table\_files/demo\_single\_table\_58\_12%20\(1\).png)

![png](../Example%20Notebooks/demo\_single\_table\_files/demo\_single\_table\_58\_13%20\(1\).png)

![png](../Example%20Notebooks/demo\_single\_table\_files/demo\_single\_table\_58\_14%20\(1\).png)

**Report without privacy metrics and without ML efficacy stuff**

```python
get_full_report(data, sample,discrete_columns,numeric_features)
```

![png](../Example%20Notebooks/demo\_single\_table\_files/demo\_single\_table\_58\_2%20\(1\).png)

![png](../Example%20Notebooks/demo\_single\_table\_files/demo\_single\_table\_58\_3%20\(1\).png)

![png](../Example%20Notebooks/demo\_single\_table\_files/demo\_single\_table\_58\_4.png)

![png](../Example%20Notebooks/demo\_single\_table\_files/demo\_single\_table\_58\_5.png)

![png](../Example%20Notebooks/demo\_single\_table\_files/demo\_single\_table\_58\_6%20\(1\).png)

![png](../Example%20Notebooks/demo\_single\_table\_files/demo\_single\_table\_58\_7.png)

![png](../Example%20Notebooks/demo\_single\_table\_files/demo\_single\_table\_58\_8%20\(1\).png)

![png](../Example%20Notebooks/demo\_single\_table\_files/demo\_single\_table\_58\_9%20\(1\).png)

![png](../Example%20Notebooks/demo\_single\_table\_files/demo\_single\_table\_58\_10%20\(1\).png)

![png](../Example%20Notebooks/demo\_single\_table\_files/demo\_single\_table\_58\_11.png)

![png](../Example%20Notebooks/demo\_single\_table\_files/demo\_single\_table\_58\_12.png)

![png](../Example%20Notebooks/demo\_single\_table\_files/demo\_single\_table\_58\_13.png)

![png](../Example%20Notebooks/demo\_single\_table\_files/demo\_single\_table\_58\_14.png)

**Save model to disk**

```python
synth.save('PrivateModelAdvAPI.pth')
```

#### Fin

```python
```
