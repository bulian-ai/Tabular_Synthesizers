from abc import ABCMeta, abstractmethod
from typing import *

import pandas as pd
import torch
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from torch import nn

from ..utils import is_main_process, save_on_master



class BaseSynthesizerPrivate(nn.Module, metaclass=ABCMeta):
    """Base class for all default synthesizers"""

    device: torch.device = torch.device("cpu")
    data: pd.DataFrame = None
    discrete_columns: Union[List, Tuple] = []    
    
    def forward(self, input):
        RuntimeWarning(
            "Forward method is not implemented. Use .fit and .sample methods"
        )
        return input

    @abstractmethod
    def fit(self, *args, **kwargs):
        pass

    @abstractmethod
    def sample(self, *args, **kwargs):
        pass


    def set_device(self, device):
        self.device = device
        if self.generator is not None:
            self.generator.to(self.device)

    def save_checkpoint(self, path):
        """Save model checkpoint"""
        device_backup = self.device
        self.set_device(torch.device("cpu"))
        state_dict = {
            "model": self.state_dict(),
        }
        save_on_master(state_dict, f=path)
        self.set_device(device_backup)
        return path

    @classmethod
    def load_from_checkpoint(cls, path, **cls_kwargs):
        """Load model from checkpoint"""
        print(f"Loading pretrained from {path} ...")
        checkpoint = torch.load(path, map_location="cpu")
        model = checkpoint['model']
        # model = cls(**cls_kwargs)
        # print(model.state_dict())
        # model.load_state_dict(checkpoint["model"])
        return model

class BaseSynthesizer(nn.Module, metaclass=ABCMeta):
    """Base class for all default synthesizers"""

    device: torch.device = torch.device("cpu")
    # data: pd.DataFrame = None
    discrete_columns: Union[List, Tuple] = []
    train_idxs: List = None
    test_idxs: List = None

    def forward(self, input):
        RuntimeWarning(
            "Forward method is not implemented. Use .fit and .sample methods"
        )
        return input

    @abstractmethod
    def fit(self, *args, **kwargs):
        pass

    @abstractmethod
    def sample(self, *args, **kwargs):
        pass

    def fit_adversarial(self, data,test_pct=0.15, epochs=300, discrete_columns=(), print_freq=50,random_state=42):

        if not isinstance(data, pd.DataFrame):
            raise TypeError(".{}_adversarial currently supports only pandas.DataFrame")

        if is_main_process() and (self.train_idxs is None and self.test_idxs is None):
            print("Generating train and test splits ...")
            train_data, test_data = train_test_split(data, test_size=test_pct,random_state=random_state)
            self.train_idxs = train_data.index.values.astype("int")
            self.test_idxs = test_data.index.values.astype("int")
        else:
            train_data = self.data.iloc[self.train_idxs]
            test_data = self.data.iloc[self.test_idxs]

        print(f"TRAIN SAMPLES: n={len(train_data)}")
        print(f"TEST SAMPLES: n={len(test_data)}")

        self.fit(train_data,epochs,discrete_columns, print_freq)

    def sample_adversarial(
        self,
        data,
        n=500,
        upsampleFrac=4,
        condition_column=None,
        condition_value=None,
        rf_params=dict(verbose=1, n_jobs=-1),
    ):

        if not isinstance(data, pd.DataFrame):
            raise TypeError(".{}_adversarial currently supports only pandas.DataFrame")

        if self.train_idxs is None:
            raise RuntimeError(
                "Please call .fit_adversarial before .sample_adversarial"
            )

        encoders = {}
        indices = []
        
        if len(self.discrete_columns):
            for cn in self.discrete_columns:
                e = LabelEncoder()
                data[cn] = e.fit_transform(data[cn].values)
                encoders[cn] = e
                indices.append(data.columns.get_loc(cn))

        train_data = data.iloc[self.train_idxs]
        test_data = data.iloc[self.test_idxs]

        train_data["adv_target"] = 0
        test_data["adv_target"] = 1

        sampled_data = self.sample(upsampleFrac * n, condition_column, condition_value)
        sampled_data["adv_target"] = 0

        if len(self.discrete_columns):
            for cn in self.discrete_columns:
                sampled_data[cn] = encoders[cn].transform(sampled_data[cn].values)

        adv_train_data = pd.concat([train_data, sampled_data, test_data])

        X_train = adv_train_data.drop(columns=["adv_target"], inplace=False)
        Y_train = adv_train_data["adv_target"]

        classifier = RandomForestClassifier(**rf_params)
        classifier.fit(X_train, Y_train)

        sampled_data.drop(columns="adv_target", inplace=True)
        sampled_data["predictions"] = classifier.predict_proba(sampled_data)[:, 1]

        sampled_data = sampled_data.sort_values(by=["predictions"], ascending=False)
        sampled_data.reset_index(inplace=True, drop=True)

        if len(self.discrete_columns):
            for cn in self.discrete_columns:
                vals = sampled_data[cn].values
                sampled_data[cn] = encoders[cn].inverse_transform(vals)
                vals = data[cn].values
                data[cn] =  encoders[cn].inverse_transform(vals)
        sampled_data.drop(columns="predictions", inplace=True)
        return sampled_data[:n]

    def set_device(self, device):
        self.device = device
        if self.generator is not None:
            self.generator.to(self.device)

    def save(self, path):
        """Save the model in the passed `path`."""
        device_backup = self.device
        self.set_device(torch.device('cpu'))
        torch.save(self, path)
        self.set_device(device_backup)

    @classmethod
    def load(cls, path):
        """Load the model stored in the passed `path`."""
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        model = torch.load(path)
        model.set_device(device)
        return model

    def save_advarsarial(self, path):
        """Save model checkpoint"""
        device_backup = self.device
        self.set_device(torch.device("cpu"))
        state_dict = {
            "train_idxs": self.train_idxs,
            "test_idxs": self.test_idxs,
            "discrete_columns": self.discrete_columns,
            "model": self,
            'state_dict':self.state_dict()
        }
        torch.save(state_dict,path)
        # save_on_master(state_dict, f=path)
        self.set_device(device_backup)
        return path

    @classmethod
    def load_advarsarial(self, path):
        """Load model from checkpoint"""
        print(f"Loading pretrained from {path} ...")
        # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        checkpoint = torch.load(path)
        model = checkpoint['model']
        model.load_state_dict(checkpoint["state_dict"])        
        self.train_idxs = checkpoint["train_idxs"]
        self.test_idxs = checkpoint["test_idxs"]
        self.discrete_columns = checkpoint["discrete_columns"]
        model.set_device(device)
        return model
