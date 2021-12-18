from typing import *

import numpy as np
import pandas as pd
import torch
from packaging import version
from torch import nn, optim
import torch.nn.functional as F

from ..data_sampler import DataSampler
from ..data_transformer import DataTransformer
from .base import BaseSynthesizer
from ..utils import MetricLogger, SmoothedValue


class Discriminator(nn.Module):
    def __init__(self, input_dim, discriminator_dim, pac=10):
        super(Discriminator, self).__init__()
        dim = input_dim * pac
        self.pac = pac
        self.pacdim = dim
        seq = []
        for item in list(discriminator_dim):
            seq += [nn.Linear(dim, item), nn.LeakyReLU(0.2), nn.Dropout(0.5)]
            dim = item

        seq += [nn.Linear(dim, 1)]
        self.seq = nn.Sequential(*seq)

    def calc_gradient_penalty(
        self, real_data, fake_data, device="cpu", pac=10, lambda_=10
    ):
        alpha = torch.rand(real_data.size(0) // pac, 1, 1, device=device)
        alpha = alpha.repeat(1, pac, real_data.size(1))
        alpha = alpha.view(-1, real_data.size(1))

        interpolates = alpha * real_data + ((1 - alpha) * fake_data)

        disc_interpolates = self(interpolates)

        gradients = torch.autograd.grad(
            outputs=disc_interpolates,
            inputs=interpolates,
            grad_outputs=torch.ones(disc_interpolates.size(), device=device),
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]

        gradient_penalty = (
            (gradients.view(-1, pac * real_data.size(1)).norm(2, dim=1) - 1) ** 2
        ).mean() * lambda_

        return gradient_penalty

    def forward(self, input):
        assert input.size()[0] % self.pac == 0
        return self.seq(input.view(-1, self.pacdim))


class Residual(nn.Module):
    def __init__(self, i, o):
        super(Residual, self).__init__()
        self.fc = nn.Linear(i, o)
        self.bn = nn.BatchNorm1d(o)
        self.relu = nn.ReLU()

    def forward(self, input):
        out = self.fc(input)
        out = self.bn(out)
        out = self.relu(out)
        return torch.cat([out, input], dim=1)


class Generator(nn.Module):
    def __init__(self, embedding_dim, generator_dim, data_dim):
        super(Generator, self).__init__()
        dim = embedding_dim
        seq = []
        for item in list(generator_dim):
            seq += [Residual(dim, item)]
            dim += item
        seq.append(nn.Linear(dim, data_dim))
        self.seq = nn.Sequential(*seq)

    def forward(self, input):
        data = self.seq(input)
        return data


class TwinSynthesizer(BaseSynthesizer):
    """Conditional Table GAN Synthesizer.
    For more details about the process, please check the [Modeling Tabular data using
    Conditional GAN](https://arxiv.org/abs/1907.00503) paper.
    Args:
            data (pandas.DataFrame):
                    Training Data
            discrete_columns (list-like):
                    List of discrete columns to be used to generate the Conditional
                    Vector. This list should contatin the column names.
            embedding_dim (int):
                    Size of the random sample passed to the Generator. Defaults to 128.
            generator_dim (tuple or list of ints):
                    Size of the output samples for each one of the Residuals. A Residual Layer
                    will be created for each one of the values provided. Defaults to (256, 256).
            discriminator_dim (tuple or list of ints):
                    Size of the output samples for each one of the Discriminator Layers. A Linear Layer
                    will be created for each one of the values provided. Defaults to (256, 256).
            generator_lr (float):
                    Learning rate for the generator. Defaults to 2e-4.
            generator_decay (float):
                    Generator weight decay for the Adam Optimizer. Defaults to 1e-6.
            discriminator_lr (float):
                    Learning rate for the discriminator. Defaults to 2e-4.
            discriminator_decay (float):
                    Discriminator weight decay for the Adam Optimizer. Defaults to 1e-6.
            batch_size (int):
                    Number of data samples to process in each step.
            discriminator_steps (int):
                    Number of discriminator updates to do for each generator update.
                    From the WGAN paper: https://arxiv.org/abs/1701.07875. WGAN paper
                    default is 5. Default used is 1 to match original CTGAN implementation.
            log_frequency (boolean):
                    Whether to use log frequency of categorical levels in conditional
                    sampling. Defaults to ``True``.
            epochs (int):
                    Number of training epochs. Defaults to 300.
            pac (int):
                    Number of samples to group together when applying the discriminator.
                    Defaults to 10.
            device (torch.device or str):
                    Device to use.
    """

    def __init__(
        self,
        embedding_dim: int = 128,
        generator_dim: Union[Tuple[int], List[int]] = (256, 256),
        discriminator_dim: Union[Tuple[int], List[int]] = (256, 256),
        generator_lr: float = 2e-4,
        generator_decay: float = 1e-6,
        discriminator_lr: float = 2e-4,
        discriminator_decay: float = 1e-6,
        batch_size: int = 500,
        discriminator_steps: int = 1,
        log_frequency: bool = True,
        pac: int = 10,
        device: Union[str, torch.device] = "cpu",
    ):
        super(TwinSynthesizer, self).__init__()
        assert batch_size % 2 == 0


        self.embedding_dim = embedding_dim
        self.generator_dim = generator_dim
        self.discriminator_dim = discriminator_dim
        self.generator_lr = generator_lr
        self.generator_decay = generator_decay
        self.discriminator_lr = discriminator_lr
        self.discriminator_decay = discriminator_decay
        self.batch_size = batch_size
        self.discriminator_steps = discriminator_steps
        self.log_frequency = log_frequency
        self.pac = pac
        self.device = torch.device(device) if isinstance(device, str) else device
        self.transformer = None
        self.sampler = None
        self.generator = None


    @staticmethod
    def gumbel_softmax(logits, tau=1, hard=False, eps=1e-10, dim=-1):
        """Deals with the instability of the gumbel_softmax for older versions of torch.

        For more details about the issue:
        https://drive.google.com/file/d/1AA5wPfZ1kquaRtVruCd6BiYZGcDeNxyP/view?usp=sharing
        Args:
                logits:
                        […, num_features] unnormalized log probabilities
                tau:
                        non-negative scalar temperature
                hard:
                        if True, the returned samples will be discretized as one-hot vectors,
                        but will be differentiated as if it is the soft sample in autograd
                dim (int):
                        a dimension along which softmax will be computed. Default: -1.
        Returns:
                Sampled tensor of same shape as logits from the Gumbel-Softmax distribution.
        """
        if version.parse(torch.__version__) < version.parse("1.2.0"):
            for _ in range(10):
                transformed = F.functional.gumbel_softmax(
                    logits, tau=tau, hard=hard, eps=eps, dim=dim
                )
                if not torch.isnan(transformed).any():
                    return transformed
            raise ValueError("gumbel_softmax returning NaN.")

        return F.gumbel_softmax(logits, tau=tau, hard=hard, eps=eps, dim=dim)

    def apply_activate(self, data):
        """Apply proper activation function to the output of the generator."""
        data_t = []
        st = 0
        for column_info in self.transformer.output_info_list:
            for span_info in column_info:
                if span_info.activation_fn == "tanh":
                    ed = st + span_info.dim
                    data_t.append(torch.tanh(data[:, st:ed]))
                    st = ed
                elif span_info.activation_fn == "softmax":
                    ed = st + span_info.dim
                    transformed = self.gumbel_softmax(data[:, st:ed], tau=0.2)
                    data_t.append(transformed)
                    st = ed
                else:
                    assert 0

        return torch.cat(data_t, dim=1)

    def cond_loss(self, data, c, m):
        """Compute the cross entropy loss on the fixed discrete column."""
        loss = []
        st = 0
        st_c = 0
        for column_info in self.transformer.output_info_list:
            for span_info in column_info:
                if len(column_info) != 1 or span_info.activation_fn != "softmax":
                    # not discrete column
                    st += span_info.dim
                else:
                    ed = st + span_info.dim
                    ed_c = st_c + span_info.dim
                    tmp = F.cross_entropy(
                        data[:, st:ed],
                        torch.argmax(c[:, st_c:ed_c], dim=1),
                        reduction="none",
                    )
                    loss.append(tmp)
                    st = ed
                    st_c = ed_c

        loss = torch.stack(loss, dim=1)

        return (loss * m).sum() / data.size()[0]

    def validate_discrete_columns(self, train_data, discrete_columns):
        """Check whether ``discrete_columns`` exists in ``train_data``.

        Args:
                train_data (numpy.ndarray or pandas.DataFrame):
                        Training Data. It must be a 2-dimensional numpy array or a pandas.DataFrame.
                discrete_columns (list-like):
                        List of discrete columns to be used to generate the Conditional
                        Vector. If ``train_data`` is a Numpy array, this list should
                        contain the integer indices of the columns. Otherwise, if it is
                        a ``pandas.DataFrame``, this list should contain the column names.
        """
        if isinstance(train_data, pd.DataFrame):
            invalid_columns = set(discrete_columns) - set(train_data.columns)
        elif isinstance(train_data, np.ndarray):
            invalid_columns = []
            for column in discrete_columns:
                if column < 0 or column >= train_data.shape[1]:
                    invalid_columns.append(column)
        else:
            raise TypeError("``train_data`` should be either pd.DataFrame or np.array.")

        if invalid_columns:
            raise ValueError("Invalid columns found: {}".format(invalid_columns))

    def fit(self, data, epochs=300, discrete_columns=(),print_freq=50):
        """Fit the CTGAN Synthesizer models to the training data.

        Args:
                train_data (numpy.ndarray or pandas.DataFrame):
                        Training Data. It must be a 2-dimensional numpy array or a pandas.DataFrame.
                discrete_columns (list-like):
                        List of discrete columns to be used to generate the Conditional
                        Vector. If ``train_data`` is a Numpy array, this list should
                        contain the integer indices of the columns. Otherwise, if it is
                        a ``pandas.DataFrame``, this list should contain the column names.
        """

        self.discrete_columns = discrete_columns
        self.validate_discrete_columns(data, self.discrete_columns)
        self.transformer = DataTransformer()
        self.transformer.fit(data, self.discrete_columns)

        transformed_data = self.transformer.transform(data)

        self.sampler = DataSampler(
            transformed_data,
            self.transformer.output_info_list,
            self.log_frequency,
        )

        data_dim = self.transformer.output_dimensions

        self.generator = Generator(
            self.embedding_dim + self.sampler.dim_cond_vec(),
            self.generator_dim,
            data_dim,
        )

        self.discriminator = Discriminator(
            data_dim + self.sampler.dim_cond_vec(),
            self.discriminator_dim,
            pac=self.pac,
        )

        self.steps_per_epoch = max(len(transformed_data) // self.batch_size, 1)

        # device placement
        self.generator.to(self.device)
        self.discriminator.to(self.device)

        # Generate Optimizers
        optimizerG = optim.Adam(
            self.generator.parameters(),
            self.generator_lr,
            betas=(0.5, 0.9),
            weight_decay=self.generator_decay,
        )
        optimizerD = optim.Adam(
            self.discriminator.parameters(),
            self.discriminator_lr,
            betas=(0.5, 0.9),
            weight_decay=self.discriminator_lr,
        )
        mean = torch.zeros(self.batch_size, self.embedding_dim, device=self.device)
        std = mean + 1
        for i in range(
            epochs
        ):  # epoch ------------------------------------------------------------

            metric_logger = MetricLogger(delimiter="  ")
            metric_logger.add_meter("loss_g", SmoothedValue())
            metric_logger.add_meter("loss_d", SmoothedValue())

            header = f"Epoch: [{i}]"
            steps = range(self.steps_per_epoch)

            for _ in metric_logger.log_every(
                steps, print_freq, header
            ):  # batch -------------------------------------------------------------

                for _ in range(self.discriminator_steps):
                    fakez = torch.normal(mean=mean, std=std)
                    condvec = self.sampler.sample_condvec(self.batch_size)

                    if condvec is None:
                        c1, m1, col, opt = None, None, None, None
                        real = self.sampler.sample_data(self.batch_size, col, opt)
                    else:
                        c1, m1, col, opt = condvec
                        c1 = torch.from_numpy(c1).to(self.device)
                        m1 = torch.from_numpy(m1).to(self.device)
                        fakez = torch.cat([fakez, c1], dim=1)

                        perm = np.arange(self.batch_size)
                        np.random.shuffle(perm)
                        real = self.sampler.sample_data(
                            self.batch_size, col[perm], opt[perm]
                        )
                        c2 = c1[perm]

                    fake = self.generator(fakez)
                    fakeact = self.apply_activate(fake)

                    real = torch.from_numpy(real.astype("float32")).to(self.device)

                    if c1 is not None:
                        fake_cat = torch.cat([fakeact, c1], dim=1)
                        real_cat = torch.cat([real, c2], dim=1)
                    else:
                        real_cat = real
                        fake_cat = fakeact

                    y_fake = self.discriminator(fake_cat)
                    y_real = self.discriminator(real_cat)

                    pen = self.discriminator.calc_gradient_penalty(
                        real_cat, fake_cat, self.device, self.pac
                    )
                    loss_d = -(torch.mean(y_real) - torch.mean(y_fake))

                    optimizerD.zero_grad()
                    pen.backward(retain_graph=True)
                    loss_d.backward()
                    optimizerD.step()

                    metric_logger.update(loss_d=loss_d.item())

                fakez = torch.normal(mean=mean, std=std)
                condvec = self.sampler.sample_condvec(self.batch_size)

                if condvec is None:
                    c1, m1, col, opt = None, None, None, None
                else:
                    c1, m1, col, opt = condvec
                    c1 = torch.from_numpy(c1).to(self.device)
                    m1 = torch.from_numpy(m1).to(self.device)
                    fakez = torch.cat([fakez, c1], dim=1)

                fake = self.generator(fakez)
                fakeact = self.apply_activate(fake)

                if c1 is not None:
                    y_fake = self.discriminator(torch.cat([fakeact, c1], dim=1))
                else:
                    y_fake = self.discriminator(fakeact)

                if condvec is None:
                    cross_entropy = 0
                else:
                    cross_entropy = self.cond_loss(fake, c1, m1)

                loss_g = -torch.mean(y_fake) + cross_entropy

                optimizerG.zero_grad()
                loss_g.backward()
                optimizerG.step()
                # print(f"Loss:{loss_g.item()}")
                metric_logger.update(loss_g=loss_g.item())
                metric_logger.update(loss=loss_d.item() + loss_g.item())

    def sample(self, n, condition_column=None, condition_value=None):
        """Sample data similar to the training data.

        Choosing a condition_column and condition_value will increase the probability of the
        discrete condition_value happening in the condition_column.
        Args:
                n (int):
                        Number of rows to sample.
                condition_column (string):
                        Name of a discrete column.
                condition_value (string):
                        Name of the category in the condition_column which we wish to increase the
                        probability of happening.
        Returns:
                numpy.ndarray or pandas.DataFrame
        """
        if condition_column is not None and condition_value is not None:
            condition_info = self.transformer.convert_column_name_value_to_id(
                condition_column, condition_value
            )
            global_condition_vec = (
                self.sampler.generate_cond_from_condition_column_info(
                    condition_info, self.batch_size
                )
            )
        else:
            global_condition_vec = None

        steps = n // self.batch_size + 1
        data = []

        for _ in range(steps):
            mean = torch.zeros(self.batch_size, self.embedding_dim)
            std = mean + 1
            fakez = torch.normal(mean=mean, std=std).to(self.device)

            if global_condition_vec is not None:
                condvec = global_condition_vec.copy()
            else:
                condvec = self.sampler.sample_original_condvec(self.batch_size)

            if condvec is None:
                pass
            else:
                c1 = condvec
                c1 = torch.from_numpy(c1).to(self.device)
                fakez = torch.cat([fakez, c1], dim=1)

            fake = self.generator(fakez)
            fakeact = self.apply_activate(fake)
            data.append(fakeact.detach().cpu().numpy())

        data = np.concatenate(data, axis=0)
        data = data[:n]

        return self.transformer.inverse_transform(data)
