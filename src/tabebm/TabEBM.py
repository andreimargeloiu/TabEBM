import os
import random
from collections import defaultdict
from functools import partial

import numpy as np
import pandas as pd
import scipy
import torch
from sklearn.model_selection import train_test_split
from tabpfn import TabPFNClassifier
from tabpfn.config import ModelInterfaceConfig, PreprocessorConfig
from tabpfn.utils import meta_dataset_collator
from torch.utils.data import DataLoader


def to_numpy(X):
    match type(X):
        case np.ndarray:
            return X
        case torch.Tensor:
            return X.detach().cpu().numpy()
        case pd.DataFrame:
            return X.to_numpy()
        case None:
            return None
        case _:
            raise ValueError("X must be either a np.ndarray or a torch.Tensor")


def seed_everything(seed):
    os.environ["PL_GLOBAL_SEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ========================================================================
#                               TabEBM CLASS
# ========================================================================


class TabEBM:
    def __init__(self):
        no_preprocessing_inference_config = ModelInterfaceConfig(
            FINGERPRINT_FEATURE=False,
            FEATURE_SHIFT_METHOD=None,
            CLASS_SHIFT_METHOD=None,
            PREPROCESS_TRANSFORMS=[PreprocessorConfig(name="none")],
        )

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = TabPFNClassifier(
            # In TabPFN-v2, the preprocessing is deeply coupled with the model for faster training and inference.
            # As a result, if we want to get the gradients to update input samples during generation (SGLD sampling),
            # we have to disable all potential preprocessing in the TabPFN-v2 model, including the feature shift etc.
            # Thus, internal ensemble of TabPFN-v2 is no longer effective, and TabEBM only supports `n_estimators=1`.
            # If you would like to use multiple TabPFN models for TabEBM, please switch back to TabPFN-v1.
            # This is because TabPFN-v1 supports a pure "no-preprocessing" fitting mode.
            n_estimators=1,
            fit_mode="batched",
            inference_config=no_preprocessing_inference_config,
            device=self.device,
        )

    def generate(
        self,
        X,
        y,  # The data must have been processed using TabEBM.add_surrogate_negative_samples()
        num_samples,  # Number of samples to generate (per class)
        starting_point_noise_std=0.01,  # SGLD noise standard deviation to initialise the starting points
        sgld_step_size=0.1,  # SGLD step size
        sgld_noise_std=0.01,  # SGLD noise standard deviation
        sgld_steps=200,  # Number of SGLD steps
        distance_negative_class=5,  # Distance of the "negative samples" created to have a different class
        seed=42,
        debug=False,  # if True,  print debug information
    ):
        if not (isinstance(X, torch.Tensor) or isinstance(X, np.ndarray)):
            X = to_numpy(X)
        if not (isinstance(y, torch.Tensor) or isinstance(y, np.ndarray)):
            y = to_numpy(y).reshape(-1)

        res = self._sampling_internal(
            X=X,
            y=y,
            num_samples=num_samples,
            starting_point_noise_std=starting_point_noise_std,
            sgld_step_size=sgld_step_size,
            sgld_noise_std=sgld_noise_std,
            sgld_steps=sgld_steps,
            distance_negative_class=distance_negative_class,
            seed=seed,
            debug=debug,
        )

        augmented_data = defaultdict(list)
        for target_class in range(len(np.unique(to_numpy(y)))):
            augmented_data[f"class_{int(target_class)}"] = res[f"class_{int(target_class)}"]["sampling_paths"]

        return augmented_data

    def _sampling_internal(
        self,
        X,
        y,  # The data must have been processed using add_surrogate_negative_samples()
        num_samples,  # number of samples to generate
        starting_point_noise_std=0.01,  # Noise std to compute the starting points for the sampling
        sgld_step_size=0.1,
        sgld_noise_std=0.01,
        sgld_steps=200,
        distance_negative_class=5,  # Distance of the "negative samples" created to have a different class
        seed=42,
        debug=False,  # if True, print debug information
    ):
        if debug:
            print("Inside TabEBM sampling")
            print(f"sgld_step_size = {sgld_step_size}")
            print(f"sgld_noise_std = {sgld_noise_std}")
            print(f"sgld_steps = {sgld_steps}")
            print(f"distance_negative_class = {distance_negative_class}")
            print(f"starting_point_noise_std = {starting_point_noise_std}")

        # ===== Sampling for each class =====
        synthetic_data_per_class = defaultdict(list)
        for target_class in np.unique(to_numpy(y)):
            # === Create ebm dataset with surrogate negative samples for a specific class ===
            ebm_dict = self.get_ebm_datatset(X, y, target_class, distance_negative_class)
            X_ebm = ebm_dict["X_ebm"]
            y_ebm = ebm_dict["y_ebm"]

            # === Fit the predictor ===
            self.fit_predictor(X_ebm, y_ebm)

            # === Create starting points for SGLD ===
            start_dict = self.initialise_sgld_starting_points(X_ebm, y_ebm, num_samples, starting_point_noise_std, seed)
            X_sgld = start_dict["X_start"]
            y_sgld = start_dict["y_start"]
            batch_dict = self.prepare_tabpfn_batch_data(X_sgld, y_sgld)
            # Note that no ensemble is used here, so we will be able to use the first element directly
            X_sgld_list = [x_batch.to(self.device).requires_grad_(True) for x_batch in batch_dict["X_train"]]

            # === SGLD sampling ===
            # Create noise for SGLD steps
            noise = self.compute_sgld_noise(X_sgld, sgld_steps, seed)

            # Create the SGLD sampling path
            for t in range(sgld_steps):
                # Compute the class-specific energy
                logits = self.model.forward(X_sgld_list, return_logits=True)
                energy = TabEBM.compute_energy(logits.reshape(logits.shape[-1], -1))
                # Use the first element for SGLD sampling
                total_energy = energy.sum() / X_sgld_list[0].shape[1]
                total_energy.backward()

                # Get the first element of the list
                X_sgld = X_sgld_list[0]
                if debug:
                    print(
                        f"Step {t} has energy {total_energy.item():.3f} with gradient norm {X_sgld.grad.norm().item():.4f}"
                    )

                # Update the sample with gradients
                X_sgld = X_sgld - sgld_step_size * X_sgld.grad + sgld_noise_std * noise[t]
                X_sgld = X_sgld.detach()

                # Update the first element with the new sample
                X_sgld_list[0] = X_sgld.requires_grad_(True)

            res = {
                "sampling_paths": X_sgld_list[0].detach().cpu().squeeze(0).numpy()
            }  # shape = (num_samples, num_features)
            synthetic_data_per_class[f"class_{int(target_class)}"] = res

        return synthetic_data_per_class

    def get_ebm_datatset(self, X, y, target_class, distance_negative_class):
        X_one_class = X[y == target_class]
        X_ebm, y_ebm = TabEBM.add_surrogate_negative_samples(
            X_one_class, distance_negative_class=distance_negative_class
        )
        X_ebm = torch.from_numpy(X_ebm).float()
        y_ebm = torch.from_numpy(y_ebm).long()

        return {"X_ebm": X_ebm, "y_ebm": y_ebm}

    def fit_predictor(self, X_ebm, y_ebm):
        # === Prepare batch data for TabPFN ===
        batch_dict = self.prepare_tabpfn_batch_data(X_ebm, y_ebm)
        X_ebm_list = [X_batch.to(self.device) for X_batch in batch_dict["X_train"]]
        y_ebm_list = [y_batch.to(self.device) for y_batch in batch_dict["y_train"]]
        cat_ixs = batch_dict["cat_ixs"]
        confs = batch_dict["confs"]

        # === Train the model ===
        self.model.fit_from_preprocessed(
            X_ebm_list,
            y_ebm_list,
            cat_ix=cat_ixs,
            configs=confs,
        )

    def initialise_sgld_starting_points(self, X_ebm, y_ebm, num_samples, starting_point_noise_std, seed):
        seed_everything(seed)

        # Select random samples from the training set
        # The convention is that the target class is always 0
        real_sample_idx = y_ebm == 0
        start_sample_idx = np.random.choice(real_sample_idx.sum(), size=num_samples)
        X_start = X_ebm[real_sample_idx][start_sample_idx]
        y_start = y_ebm[real_sample_idx][start_sample_idx]

        # Add noise to the starting points
        X_start = X_start + (starting_point_noise_std * torch.randn(X_start.shape))

        return {"X_start": X_start, "y_start": y_start}

    def compute_sgld_noise(self, X_sgld, sgld_steps, seed):
        seed_everything(seed)
        noise = torch.randn(sgld_steps, *X_sgld.shape)
        noise = noise.to(self.device)

        return noise

    def prepare_tabpfn_batch_data(self, X, y):
        splitter = partial(TabEBM.train_test_split_allow_full_train, test_size=0, random_state=42, shuffle=False)
        batched_datasets = self.model.get_preprocessed_datasets(X, y, splitter, max_data_size=10000)

        batch_dataloader = DataLoader(
            batched_datasets,
            batch_size=1,
            collate_fn=meta_dataset_collator,
        )

        # Get only the first batch from the dataloader
        X_train, X_val, y_train, y_val, cat_ixs, confs = next(iter(batch_dataloader))

        return {
            "X_train": X_train,
            "X_val": X_val,
            "y_train": y_train,
            "y_val": y_val,
            "cat_ixs": cat_ixs,
            "confs": confs,
        }

    @staticmethod
    def compute_energy(
        logits,  # Model's logit (unnormalized) for each class (shape = (num_samples, num_classes))
        return_unnormalized_prob=False,  # Whether to compute the unnormalized probability p(x) instead of the energy
    ):
        # Compute the proposed TabEBM class-specific energy E_c(x) = -log(exp(f^c(\x)[0]) + exp(f^c(x)[1]))
        if type(logits) is torch.Tensor:
            # === Assert the logits are unnormlized (and not probabilities)
            assert (logits.sum(dim=1) - 1).abs().max() > 1e-5, "Logits must be unnormalized"

            energy = -torch.logsumexp(logits, dim=1)
            if return_unnormalized_prob:
                return torch.exp(-energy)
            else:
                return energy
        elif type(logits) is np.ndarray:
            # === Assert the logits are unnormlized (and not probabilities)
            assert (logits.sum(axis=1) - 1).max() > 1e-5, "Logits must be unnormalized)"

            energy = -1 * scipy.special.logsumexp(logits, axis=1)
            if return_unnormalized_prob:
                return np.exp(-energy)
            else:
                return energy
        else:
            raise ValueError("Logits must be either a torch.Tensor or a np.ndarray")

    @staticmethod
    def add_surrogate_negative_samples(
        X,  # The data (expected to be standardized to have zero mean and unit variance)
        distance_negative_class=5,  # The distance of the surrogate "negative samples" from the data
    ):
        """
        Create the surrogate negative samples for TabEBM's proposed surrogate task (for each class)
        """
        if X.shape[1] == 2:
            true_negatives = [
                [-distance_negative_class, -distance_negative_class],
                [distance_negative_class, distance_negative_class],
                [-distance_negative_class, distance_negative_class],
                [distance_negative_class, -distance_negative_class],
            ]
        else:
            # === Generate "true negative" samples ===
            true_negatives = set()
            while len(true_negatives) < 4:
                point = np.random.choice([-distance_negative_class, distance_negative_class], X.shape[1])
                point = tuple(point)
                if point not in true_negatives:
                    true_negatives.add(point)
                    true_negatives.add(tuple(-np.array(point)))
            true_negatives = list(true_negatives)
        num_true_negatives = len(true_negatives)

        if type(X) is np.ndarray:
            X_ebm = np.array(true_negatives)
            X_ebm = np.concatenate([X, X_ebm], axis=0)
            y_ebm = np.concatenate([np.zeros(X.shape[0]), np.ones(num_true_negatives)], axis=0)

            return X_ebm, y_ebm
        elif type(X) is torch.Tensor:
            X_ebm = torch.tensor(true_negatives).float().to(X.device)
            X_ebm = torch.cat([X, X_ebm], dim=0)
            y_ebm = torch.cat([torch.zeros(X.shape[0]), torch.ones(num_true_negatives)], dim=0).long().to(X.device)

            return X_ebm, y_ebm
        else:
            raise ValueError("X must be either a np.ndarray or a torch.Tensor")

    @staticmethod
    def train_test_split_allow_full_train(
        X,
        y,
        test_size=None,
        train_size=None,
        random_state=None,
        shuffle=True,
        stratify=None,
    ):
        # === Full Train Mode ===
        full_train = False
        if test_size == 0:
            test_size = None
            full_train = True

        # === Train/Validation Split ===
        X_train, X_val, y_train, y_val = train_test_split(
            X,
            y,
            test_size=test_size,
            train_size=train_size,
            random_state=random_state,
            shuffle=shuffle,
            stratify=stratify,
        )

        # === Return full data when in full train mode ===
        if full_train:
            X_train = X
            y_train = y

        return X_train, X_val, y_train, y_val
