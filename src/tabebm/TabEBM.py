import os
import random
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import scipy
import torch
from tabpfn import TabPFNClassifier


def to_numpy(X):
    match type(X):
        case np.ndarray:
            return X
        case torch.Tensor:
            return X.detach().cpu().numpy()
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
    def __init__(self, tabpfn=None, plotting=False):
        if tabpfn is not None:
            self.tabpfn = tabpfn
        else:
            if plotting is False:
                self.tabpfn = TabPFNClassifier(
                    device="cuda",
                    N_ensemble_configurations=3,
                    only_inference=True,
                    no_grad=False,
                    no_preprocess_mode=True,
                )
            else:
                self.tabpfn = TabPFNClassifier(
                    device="cpu",
                    N_ensemble_configurations=3,
                    only_inference=True,
                    no_grad=True,
                    no_preprocess_mode=True,
                )

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
        return_trajectory=False,  # If False, then return only save the last point of the sampling path
        return_energy_values=False,  # If True, then return the energy values of the samples
        return_gradients_energy_surface=False,  # If True, then return the gradients of the energy surface as part of the final output
        debug=False,  # if True, print debug information
    ):
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
        return_trajectory=False,  # If False, then return only save the last point of the sampling path
        return_energy_values=False,  # If True, then return the energy values of the samples
        return_gradients_energy_surface=False,  # If True, then return the gradients of the energy surface as part of the final output
        debug=False,  # if True, print debug information
    ):
        if debug:
            print("Inside TabEBM sampling")
            print(f"sgld_step_size = {sgld_step_size}")
            print(f"sgld_noise_std = {sgld_noise_std}")
            print(f"sgld_steps = {sgld_steps}")
            print(f"distance_negative_class = {distance_negative_class}")
            print(f"starting_point_noise_std = {starting_point_noise_std}")

        if return_gradients_energy_surface:
            assert (
                return_trajectory
            ), "If return_gradients_energy_surface is True, then return_trajectory must be True to get the trajectory of the gradients"

        # === Sampling for each class ===
        synthetic_data_per_class = defaultdict(list)
        for target_class in np.unique(to_numpy(y)):
            X_one_class = X[y == target_class]
            X_ebm, y_ebm = TabEBM.add_surrogate_negative_samples(
                X_one_class, distance_negative_class=distance_negative_class
            )

            X_ebm = torch.from_numpy(X_ebm).float()
            y_ebm = torch.from_numpy(y_ebm).long()
            self.tabpfn.fit(X_ebm, y_ebm)

            # ======= CREATE THE STARTING POINTS FOR RUNNING SGLD =======
            seed_everything(seed)
            X_one_class = X_ebm[y_ebm == 0]  # The convention is that the target class is always 0
            x = X_one_class[
                np.random.choice(len(X_one_class), size=num_samples)
            ]  # Select random samples from the training set
            x += starting_point_noise_std * np.random.randn(*x.shape)  # Add noise to the starting points
            x.requires_grad = True
            x.to(self.tabpfn.device)

            if return_trajectory:
                sgld_sampling_paths = [x.detach().cpu().numpy()]
                gradients_energy_surface = []
                energy_values = []

            seed_everything(seed)
            noise = torch.randn(sgld_steps, *x.shape)
            noise.to(self.tabpfn.device)
            for t in range(sgld_steps):
                if x.grad is not None:
                    x.grad.zero_()

                # === Compute the class-specific energy ===
                logits = self.tabpfn.predict_proba(
                    x, return_logits=True
                )  # ====== NOTE: tabpfn.predict_proba() internally sets the seed to 0
                energy = TabEBM.compute_energy(logits)
                total_energy = energy.sum() / len(x)
                total_energy.backward()

                if debug:
                    print(
                        f"Step {t} has energy {total_energy.item():.3f} with gradient norm {x.grad.norm().item():.4f}"
                    )

                if return_gradients_energy_surface and return_trajectory:
                    gradients_energy_surface.append(x.grad.detach().cpu().numpy())
                if return_energy_values and return_trajectory:
                    energy_values.append(energy.detach().cpu().numpy())

                x = x.detach() - sgld_step_size * x.grad + sgld_noise_std * noise[t]
                x.requires_grad = True

                if return_trajectory:
                    sgld_sampling_paths.append(x.detach().cpu().numpy())

            if return_trajectory:
                res = {
                    "sampling_paths": np.array(sgld_sampling_paths).transpose(1, 0, 2)
                }  # shape = (num_samples, num_steps, num_features)

                if return_gradients_energy_surface:
                    res["gradients_energy_surface"] = np.array(gradients_energy_surface).transpose(
                        1, 0, 2
                    )  # shape = (num_samples, num_steps, num_features)
                if return_energy_values:
                    res["energy_values"] = np.array(energy_values).transpose(1, 0)  # shape = (num_samples, num_steps)
                return res
            else:
                res = {"sampling_paths": x.detach().cpu().numpy()}  # shape = (num_samples, num_features)

            synthetic_data_per_class[f"class_{int(target_class)}"] = res

        return synthetic_data_per_class


# ========================================================
#               PLOTTING ENERGY CONTOURS
# ========================================================
def plot_TabEBM_energy_contour(
    tabebm,
    X,
    y,  # Point to compute the energy contours
    target_class,  # The class for which to compute the energy contours
    ax=None,  # If None, a new figure is created. Otherwise, the plot is overlaid on this axis.
    prefix_title="",  # A prefix to add to the title
    full_title=None,  # If not None, the title of the plot
    x_min_user=None,
    x_max_user=None,
    y_min_user=None,
    y_max_user=None,  # The limits of the plot provided by the user
    show_unnormalized_prob=False,  # Whether to show the unnormalized probability instead of the energy
    show_scatter_points=True,  # Whether to show the scatter points
    show_legend=False,  # Whether to show the legend
    show_ticks=False,  # Whether to show the ticks on the axes
    show_colorbar=False,  # Whether to show the colorbar
    show_vector_field=False,  # Whether to show the vector field
    show_contours=True,  # Whether to show the contours
    number_contours=20,  # The number of contours to show
    alpha_contourf=0.8,  # The alpha of the filled contours
    cmap_scatter="Paired",
    color_all_scatter_points_in=None,
    cmap="Blues",
    s=20,  # Marker size
    h=0.1,  # The step size in the mesh
    figsize=(5, 5),
    debug=False,
):
    """
    Plot the energy contours of the TabEBM model

    Returns
    - the axis on which the plot was made
    """
    # ==== Sanity checks ====
    if type(X) is torch.Tensor:
        X = X.detach().cpu().numpy()
        y = y.detach().cpu().numpy()

        if debug:
            print("DEBUG: X is a torch.Tensor and has been converted to a numpy.ndarray")

    X = X[y == target_class]
    y = y[y == target_class]
    X, y = TabEBM.add_surrogate_negative_samples(X)

    # ==== Fit the surrogate binary classifier ====
    tabpfn = tabebm.tabpfn
    tabpfn.model[2].train()
    tabpfn.fit(X, y, overwrite_warning=True)
    tabpfn.model[2].eval()

    # === Create a meshgrid of points ===
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5

    # === Adjust the limits of the plot with the limits ===
    x_min = x_min if x_min_user is None else x_min_user
    x_max = x_max if x_max_user is None else x_max_user
    y_min = y_min if y_min_user is None else y_min_user
    y_max = y_max if y_max_user is None else y_max_user

    # === PLOT CONTOURS ===
    if ax is None:
        print("Creating a new figure because 'ax' is None")
        _, ax = plt.subplots(1, 1, figsize=figsize)

    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)

    ax.set_aspect("equal")
    title = ""

    if show_unnormalized_prob is False:
        cmap += "_r"  # because we want color to mean low energy (thus high density)

    if show_contours:
        # === Create a meshgrid of points ===
        xx, yy = np.meshgrid(np.arange(x_min, x_max + h, h), np.arange(y_min, y_max + h, h), indexing="ij")
        mesh_inputs = np.c_[xx.ravel(), yy.ravel()]
        mesh_inputs = torch.from_numpy(mesh_inputs).float()

        # === Compute the energy on the meshgrid ===
        all_logits = tabpfn.predict_proba(mesh_inputs, return_logits=True)
        if type(all_logits) is torch.Tensor:
            all_logits = all_logits.detach().cpu().numpy()

        energy = TabEBM.compute_energy(all_logits, return_unnormalized_prob=show_unnormalized_prob)
        if show_unnormalized_prob:
            title = "The unnormalized $p(x)$"
        else:
            title = "The energy of $p(x)$"

        ax.contourf(xx, yy, energy.reshape(xx.shape), cmap=cmap, alpha=alpha_contourf, levels=number_contours)

        if show_colorbar:
            cbar = plt.colorbar(
                ax.contourf(xx, yy, energy.reshape(xx.shape), cmap=cmap, alpha=0.8), ax=ax, fraction=0.046, pad=0.04
            )

            if show_unnormalized_prob:
                cbar.ax.set_ylabel("Unnormalized density (blue regions have high density)")
            else:
                cbar.ax.set_ylabel("Energy (red regions have high density)")

    if full_title is None:
        ax.set_title(prefix_title + title)
    else:
        ax.set_title(full_title)

    if show_scatter_points:
        if color_all_scatter_points_in is None:
            ax.scatter(X[:, 0], X[:, 1], c=y, cmap=cmap_scatter, s=s, edgecolors="black")
        else:
            ax.scatter(X[:, 0], X[:, 1], c=color_all_scatter_points_in, s=s, edgecolors="black")

        if show_legend:
            # Creating legend
            unique_labels = np.unique(y)
            # Get the color list from the colormap
            colors = plt.cm.Paired(np.linspace(0, 1, len(unique_labels)))
            # Create legend handles
            legend_handles = [
                plt.Line2D(
                    [0],
                    [0],
                    marker="o",
                    color="w",
                    label=label,
                    markerfacecolor=color,
                    markersize=np.sqrt(s),
                    markeredgewidth=2,
                    markeredgecolor="black",
                )
                for label, color in zip(unique_labels, colors)
            ]

            ax.legend(handles=legend_handles, title="Labels")

    if show_ticks is False:
        ax.set_xticks(())
        ax.set_yticks(())

    # === PLOT VECTOR FIELD ===
    if show_vector_field:
        x = np.linspace(x_min, x_max, 20)
        y = np.linspace(y_min, y_max, 20)
        X_mesh, Y_mesh = np.meshgrid(x, y)
        grid_inputs = np.c_[X_mesh.ravel(), Y_mesh.ravel()]
        grid_inputs = torch.from_numpy(grid_inputs).float()
        grid_inputs.requires_grad = True
        all_logits = tabpfn.predict_proba(grid_inputs, return_logits=True)

        energy = TabEBM.compute_energy(all_logits, return_unnormalized_prob=show_unnormalized_prob)

        total_energy = energy.sum()
        total_energy.backward(retain_graph=True)
        grad = grid_inputs.grad.cpu().numpy()

        if show_unnormalized_prob:
            ax.quiver(X_mesh, Y_mesh, grad[:, 0], grad[:, 1], color="black", alpha=0.5)
        else:
            ax.quiver(X_mesh, Y_mesh, -grad[:, 0], -grad[:, 1], color="black", alpha=0.5)

    return ax
