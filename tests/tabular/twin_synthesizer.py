import unittest
import pandas as pd
from bulian.Tabular.synthesizers import TwinSynthesizer, PrivateTwinSynthesizer


class BaseTestClass(unittest.TestCase):
    def setUp(self):
        self.data = pd.DataFrame(
            {
                '1':[0.6417366765311319, 0.4995232533994396, 0.7048236572174155, 0.01483747235772337, 0.8785052529521522, 0.7899419323434513, 0.6491719454951946, 0.8280835960643933, 0.5778327106751368, 0.309529058151795, 0.8033292868913439, 0.45642783362372885, 0.07560689463557335, 0.7325691884959649, 0.4736925521335209, 0.3611024877244906, 0.9803347507946942, 0.08558737496794311, 0.888512878083493, 0.1306618189016644],
                '2':[True, False, False, False, True, False, False, False, False, True, False, True, False, False, True, False, False, True, True, False],
                '3':['1', '1', '1', '1', '2', '2', '1', '2', '1', '2', '1', '1', '2', '2', '2', '2', '2', '2', '1', '2'],
                '4':[1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 1, 1, 0, 0, 1, 0, 1, 0]

            }
        )


class TwinSynthesizerTest(BaseTestClass):
    def test_fit(self):
        synth = TwinSynthesizer(batch_size=200,device='cpu')
        synth.fit(data=self.data,epochs=5,discrete_columns=['2', '3', '4'])
        sample = synth.sample(10)
        self.assertEqual(len(sample), 10)


class PrivateTwinSynthesizerTest(BaseTestClass):
    def test_fit(self):
        synth = PrivateTwinSynthesizer(device='cpu')
        synth.fit(data=self.data, update_epsilon=0.1, discrete_columns=[])
        sample = synth.sample(10)
        self.assertEqual(len(sample), 10)