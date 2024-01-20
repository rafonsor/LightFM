import typing as t
from math import floor, log as lognat

import torch as pt

from lightfm.optimizer import LightFMOptimizer, AdagradOptimizer, AdadeltaOptimizer
from lightfm.utils import SparseCOOTensorT, SparseCSRTensorT, SparseTensorDataset, sparse_identity

MAX_LOSS = 10.0
MAX_REG_SCALE = 1000000.0


class LightFM(pt.nn.Module):
    """LightFM recommender model

    Parameters
    ----------
    n_user_features: int
        Number of user features in the model
    n_item_features: int
        Number of item features in the model
    n_components: int
        Dimensionality of latent embeddings
    """
    def __init__(
        self,
        n_user_features: int,
        n_item_features: int,
        n_components: int,
    ):
        super().__init__()
        self.sparse = True  # Enable sparse gradients
        self.n_user_features = n_user_features
        self.n_item_features = n_item_features
        self.n_components = n_components

        self.item_embeddings: pt.nn.Parameter  # n_item_features x n_components
        self.user_embeddings: pt.nn.Parameter  # n_user_features x n_components
        self.item_biases: pt.nn.Parameter  # n_item_features x 1
        self.user_biases: pt.nn.Parameter  # n_user_features x 1
        self.user_scale: pt.nn.Parameter  # 1
        self.item_scale: pt.nn.Parameter  # 1
        self._init_parameters()

    def _init_parameters(self):
        """Initialise model parameters"""
        def scaled_zero_mean(shape):
            """Uniform initialisation of embeddings with zero mean and dimension scaling"""
            return (pt.rand(shape, dtype=pt.float) - 0.5) / self.n_components

        self.item_embeddings = pt.nn.Parameter(scaled_zero_mean((self.n_item_features,
                                                                 self.n_components)))
        self.user_embeddings = pt.nn.Parameter(scaled_zero_mean((self.n_user_features,
                                                                 self.n_components)))
        self.item_biases = pt.nn.Parameter(pt.zeros(self.n_item_features, 1))
        self.user_biases = pt.nn.Parameter(pt.zeros(self.n_user_features, 1))
        self.user_scale = pt.nn.Parameter(pt.tensor(1.0), requires_grad=False)
        self.item_scale = pt.nn.Parameter(pt.tensor(1.0), requires_grad=False)

    def forward(
            self,
            users_features: pt.Tensor,
            items_features: pt.Tensor,
            return_repr: bool = False
    ) -> t.Union[pt.Tensor, t.Tuple[pt.Tensor, ...]]:
        """Predict scores for user-item pairs

        Parameters
        ----------
        users_features: pt.Tensor
            Feature vectors for a group of users
        items_features: pt.Tensor
            Feature vectors for a group of items
        return_repr: t.Tuple
            Return latent representations
        Returns
        -------
        scores: pt.Tensor
            Predicted scores for user-item pairs
        """
        assert len(users_features.shape) == 2
        assert len(items_features.shape) == 2
        # feature vectors: n_entities x n_features
        scaled_users_features = users_features * self.user_scale
        scaled_items_features = items_features * self.item_scale
        # latent representation vector: p x n_components
        users_repr = pt.sparse.mm(scaled_users_features, self.user_embeddings)
        items_repr = pt.sparse.mm(scaled_items_features, self.item_embeddings)
        user_biases = pt.sparse.mm(scaled_users_features, self.user_biases)
        item_biases = pt.sparse.mm(scaled_items_features, self.item_biases)
        scores = pt.mm(users_repr, items_repr.T) + user_biases[:, 0, None] + item_biases[None, :, 0]
        if return_repr:
            return scores, users_repr, items_repr
        return scores

    def predict(
            self,
            user_features: pt.Tensor,
            item_features: pt.Tensor,
            return_repr: bool = False
    ) -> t.Union[float, t.Tuple[pt.Tensor, ...]]:
        """Predict scores for a single user-item pair

        Parameters
        ----------
        user_features: pt.Tensor
            Feature vectors for a group of users
        item_features: pt.Tensor
            Feature vectors for a group of items
        return_repr: t.Tuple
            Return latent representations
        Returns
        -------
        scores: pt.Tensor
            Predicted scores for user-item pairs
        """
        output = self.forward(user_features.unsqueeze(0), item_features.unsqueeze(0),
                              return_repr=return_repr)
        if return_repr:
            return output[0], *output[1:]
        return output.item()

    def recommend(self, k: int, users_features: pt.Tensor, items_features: pt.Tensor) -> pt.Tensor:
        """Recommend Top-K items for the provided users

        Parameters
        ----------
        k: int
            Number of items to recommend
        users_features: pt.Tensor (n_users, n_user_features)
            Feature vectors for a group of users
        items_features: pt.Tensor (n_items, n_item_features)
            Feature vectors for a group of items

        Returns
        -------
        recommendations: pt.Tensor (n_users, k)
            Top items for the requested users
        """
        assert k > 0, "k must be a positive integer"
        return self.forward(users_features, items_features).argsort(descending=True)[..., -k:]


class LightFMTrainer:
    """LightFM trainer

    Notes:
    - Optional user and item features must be sparse matrices in CSR format.
    - Interactions and optional sample weights must be sparse matrices in COO format.
    - LightFM model's embeddings are sometimes labelled as "features" in the original
      implementation, in particular within the C structure FastLightFM.
    """
    def __init__(
        self,
        n_user_features: int,
        n_item_features: int,
        n_components: int = 10,
        kos_k: int = 5,  # k-th positive sample to used during each k-OS update
        kos_max: int = 10,  # maximum number of interactions to sample each k-OS update
        learning_schedule: str = "adagrad",
        builtin_optimizer: bool = False,
        loss: str = "warp",
        learning_rate: float = 0.05,
        rho: float = 0.95,
        epsilon: float = 1e-6,
        alpha: float = 0.0,  # Common alpha for user/item feature parameters (i.e. weight decay)
        max_sampled: int = 10,
        random_state: t.Optional[int] = None,
    ):
        assert alpha >= 0.0
        assert n_components > 0
        assert kos_k > 0
        assert kos_max > 0
        assert 0 < rho < 1
        assert epsilon >= 0
        assert learning_schedule in ("adagrad", "adadelta"), "unsupported learning schedule"
        assert loss in ("logistic", "warp", "bpr", "warp-kos"), "unsupported loss function"
        assert max_sampled > 0, "max_sampled must be a positive integer"

        self.n_user_features = n_user_features
        self.n_item_features = n_item_features
        self.n_components = n_components
        self.learning_rate = learning_rate

        self.kos_k = kos_k
        self.kos_max = kos_max
        self.learning_rate = learning_rate
        self.rho = rho
        self.eps = epsilon
        self.alpha = alpha

        self.item_scale = 1.0
        self.user_scale = 1.0
        self.max_sampled = max_sampled

        self.learning_schedule = learning_schedule
        self.builtin_optimizer = builtin_optimizer
        self.loss = loss

        if random_state is not None:
            pt.manual_seed(random_state)

        self._model: LightFM
        self._optimizer: t.Union[pt.optim.Optimizer, LightFMOptimizer]
        self._trained = False
        self._reset_state()

    @property
    def model(self) -> LightFM:
        return self._model

    def _reset_state(self):
        """Reset model and optimizer state"""
        self._trained = False
        self._model = self._init_model()
        self._optimizer = self._init_optimizer()

    def _init_model(self) -> LightFM:
        return LightFM(self.n_user_features, self.n_item_features, self.n_components)

    def _init_optimizer(self) -> t.Union[pt.optim.Optimizer, LightFMOptimizer]:
        if self.learning_schedule == 'adagrad':
            kwargs = {
                'lr': self.learning_rate,
                'lr_decay': self.rho,
                'weight_decay': self.alpha,
                'initial_accumulator_value': 1.0,
                'eps': self.eps
            }
            builtin_cls = pt.optim.Adagrad
            lightfm_cls = AdagradOptimizer
        elif self.learning_schedule == 'adadelta':
            kwargs = {
                'lr': self.learning_rate,
                'rho': self.rho,
                'weight_decay': self.alpha,
                'eps': self.eps
            }
            builtin_cls = pt.optim.Adadelta
            lightfm_cls = AdadeltaOptimizer
        else:
            raise ValueError('Unknown learning schedule: {}'.format(self.learning_schedule))

        if self.builtin_optimizer:
            return builtin_cls(self._model.parameters(), **kwargs)

        return lightfm_cls(
            dict(self._model.named_parameters()), n_components=self.n_components, **kwargs)

    @staticmethod
    def _progress(seq, verbose, desc: str = 'Epoch', level: int = 0):
        if not verbose:
            return seq
        try:
            from tqdm import tqdm
            return tqdm(seq, desc=desc, leave=not level, position=level)
        except ImportError:
            def verbose_iteration():
                for i, elem in enumerate(seq):
                    print(f"{desc} {i}")
                    yield elem
            return verbose_iteration()

    @staticmethod
    def _check_input_finite(data):
        if not pt.isfinite(data.sum()):
            raise ValueError(
                "Not all input values are finite. Check the input for NaNs and infinite values.")

    def _check_parameters_finite(self):
        if self._model is None:
            return
        for parameter in (
            self._model.item_embeddings,
            self._model.item_biases,
            self._model.user_embeddings,
            self._model.user_biases,
        ):
            if not pt.isfinite(parameter.sum()):
                raise ValueError(
                    "Not all estimated parameters are finite,"
                    " your model may have diverged. Try decreasing"
                    " the learning rate or normalising feature values"
                    " and sample weights"
                )

    def _construct_feature_matrices(self, n_users, n_items, user_features, item_features):

        if user_features is None:
            user_features = sparse_identity(n_users, layout="csr", requires_grad=False)
        else:
            user_features = user_features.to_sparse_csr()
            if n_users > user_features.shape[0]:
                raise Exception("Number of user feature rows does not equal the number of users")

        if item_features is None:
            item_features = sparse_identity(n_items, layout="csr", requires_grad=False)
        else:
            item_features = item_features.to_sparse_csr()
            if n_items > item_features.shape[0]:
                raise Exception("Number of item feature rows does not equal the number of items")

        # If we already have embeddings, verify that we have them for all the supplied features
        if self._model is not None:
            if user_features.shape[1] > self._model.user_embeddings.shape[0]:
                raise ValueError(
                    "The user feature matrix specifies more features than there are estimated "
                    "feature embeddings: {} vs {}.".format(
                        self._model.user_embeddings.shape[0], user_features.shape[1]
                    )
                )
            if item_features.shape[1] > self._model.item_embeddings.shape[0]:
                raise ValueError(
                    "The item feature matrix specifies more features than there are estimated "
                    "feature embeddings: {} vs {}.".format(
                        self._model.item_embeddings.shape[0], item_features.shape[1]
                    )
                )
        return user_features, item_features

    def _process_sample_weights(
            self, interactions: SparseCOOTensorT, sample_weights: SparseCOOTensorT
    ) -> pt.Tensor:
        if sample_weights is not None:
            if self.loss == "warp-kos":
                raise NotImplementedError("k-OS loss with sample weights not implemented.")

            if sample_weights.layout != pt.sparse_coo:
                raise ValueError("Sample weight must be a sparse COO matrix.")

            if sample_weights.shape != interactions.shape:
                raise ValueError("Sample weight and interactions matrices must be the same shape")

            if not pt.equal(sample_weights.indices(), interactions.indices()):
                raise ValueError("Sample weight and interaction entries must be in the same order")

            return sample_weights.values().type(pt.float)

        if interactions.values().max() == 1.0 and interactions.values().min() == 1.0:
            # Re-use interactions data if they are all ones
            return interactions.values()

        # Otherwise allocate a new array of ones
        return pt.ones_like(interactions.values(), dtype=pt.float)

    @pt.no_grad()
    def _regularize(self):
        """
        Apply accumulated L2 regularization to all latent space features.
        """
        # Scale down latent space features
        self._model.item_embeddings /= self._model.item_scale
        self._model.item_biases /= self._model.item_scale
        self._model.user_embeddings /= self._model.user_scale
        self._model.user_biases /= self._model.user_scale
        # Reset scaling factors
        self._model.item_scale.fill_(1.0)
        self._model.user_scale.fill_(1.0)

    @staticmethod
    def _warp_loss(rank: int, weight: pt.Tensor, n_items: int) -> pt.Tensor:
        return weight * lognat(max(1.0, floor((n_items - 1) / rank)))

    def _run_epoch(
            self,
            data_loader: pt.utils.data.DataLoader,
            interactions: SparseCOOTensorT,
            user_features: t.Optional[SparseCSRTensorT],
            item_features: t.Optional[SparseCSRTensorT],
            verbose: bool = False,
    ):
        """Fit the model

        Parameters
        ----------
        interactions: pt.Tensor of layout sparse_coo and shape [n_users, n_items]
             the matrix containing
             user-item interactions. Will be converted to
             numpy.float32 dtype if it is not of that type.
        user_features: pt.Tensor of layout sparse_csr and shape [n_users, n_user_features], optional
             Each row contains that user's weights over features.
        item_features: pt.Tensor of layout sparse_csr and shape [n_items, n_item_features], optional
             Each row contains that item's weights over features.
        sample_weights: pt.Tensor of layout sparse_coo and shape [n_users, n_items], optional
             matrix with entries expressing weights of individual interactions from the interactions
            matrix. Its row and col arrays must be the same as those of the interactions matrix. For
            memory efficiency its possible to use the same arrays for both weights and interaction
            matrices. Defaults to weight 1.0 for all interactions. Not implemented for k-OS loss.
        verbose: bool
            Whether to print training progress
        """
        self._model.train()
        if self.loss == 'warp':
            fit_sample_fn = self._fit_sample_warp
        else:
            raise NotImplementedError(f"Loss {self.loss} not yet implemented. Use 'warp'.")

        epoch_loss = sum(
            fit_sample_fn(positive_sample, interactions, user_features, item_features)
            for positive_sample
            in self._progress(data_loader, verbose=verbose, desc='Sample', level=1)
        )
        self._regularize()  # L2 regularisation
        return epoch_loss

    def _fit_sample_warp(
            self,
            positive_sample: t.Tuple[pt.Tensor, ...],
            interactions: SparseCOOTensorT,
            user_features: SparseCSRTensorT,
            item_features: SparseCSRTensorT,
    ) -> float:
        """
        Fit the model using the WARP loss.

        Weighted Approximate-Rank Pairwise (WARP) loss: Maximises the rank of positive examples by
        repeatedly sampling negative examples until a rank-violating one is found. Useful when only
        positive interactions are present and optimising the top of the recommendation list
        (precision@k) is desired.
        

        WARP Training follows adaptive stochastic gradient descent with contrastive learning:
            for all interactions, we individually and iteratively search for situations where the
            model assigns higher scores to items a specific user has not interacted with compare to
            the item at hand. If such occurs within at most `max_sampled` tries, this leads to
            tweaking model parameters such that users with similar features get closer in the latent
            space to items with observed interactions.

        Returns
        -------
        Sample loss
        """
        user_id, item_id, value, weight = positive_sample
        n_items = item_features.shape[0]

        if value <= 0:
            return 0.0 # Skip negative-values samples

        user_feature_vector = user_features[user_id]
        item_feature_vector = item_features[item_id]
        positive_score, user_repr, item_repr = self._model.predict(
            user_feature_vector, item_feature_vector, return_repr=True)

        for rank in range(self.max_sampled):
            negative_item_id = pt.randint(0, n_items, (1,)).squeeze()

            if pt.is_nonzero(interactions[user_id, negative_item_id]):
                continue  # Skip item with existing interactions from this user

            negative_item_feature_vector = item_features[negative_item_id]
            negative_score, _, neg_item_repr = self._model.predict(user_feature_vector, negative_item_feature_vector, return_repr=True)

            if negative_score < positive_score:
                continue  # No violation

            # Compute WARP loss, clipped for numerical stability
            loss = self._warp_loss(rank + 1, weight, n_items).clip_(max=MAX_LOSS)

            if isinstance(self._optimizer, pt.optim.Optimizer):
                self._optimizer.zero_grad()
                loss.backward()
                self._optimizer.step()
            else:
                # Apply Adaptive update rule with WARP loss
                self._optimizer.step(
                    loss,
                    user_features,
                    item_features,
                    user_id,
                    item_id,
                    negative_item_id,
                    user_repr,
                    item_repr,
                    neg_item_repr,
                )

            if self.item_scale > MAX_REG_SCALE or self.user_scale > MAX_REG_SCALE:
                self._regularize()
            return loss.item()
        return 0.0

    def fit(
        self,
        interactions: SparseCOOTensorT,
        user_features: t.Optional[SparseCSRTensorT] = None,
        item_features: t.Optional[SparseCSRTensorT] = None,
        sample_weights: t.Optional[SparseCOOTensorT] = None,
        epochs: int = 10,
        verbose: bool = False,
    ) -> t.List[float]:
        """
        Fit the model.

        For details on how to use feature matrices, see the documentation
        on the :class:`lightfm.LightFM` class.

        Arguments
        ---------
        interactions: pt.Tensor of layout sparse_coo and shape [n_users, n_items]
             the matrix containing
             user-item interactions. Will be converted to
             numpy.float32 dtype if it is not of that type.
        user_features: pt.Tensor of layout sparse_csr and shape [n_users, n_user_features], optional
             Each row contains that user's weights over features.
        item_features: pt.Tensor of layout sparse_csr and shape [n_items, n_item_features], optional
             Each row contains that item's weights over features.
        sample_weights: pt.Tensor of layout sparse_coo and shape [n_users, n_items], optional
             matrix with entries expressing weights of individual interactions from the interactions
            matrix. Its row and col arrays must be the same as those of the interactions matrix. For
            memory efficiency its possible to use the same arrays for both weights and interaction
            matrices. Defaults to weight 1.0 for all interactions. Not implemented for k-OS loss.
        epochs: int, optional
             number of epochs to run
        verbose: bool, optional
             whether to print progress messages.
             If `tqdm` is installed, a progress bar will be displayed instead.

        Returns
        -------
        List[float]
            A list of epoch losses (sum of all sample losses)
        """
        # Discard old results, if any
        self._reset_state()

        return self.fit_partial(
            interactions,
            user_features=user_features,
            item_features=item_features,
            sample_weights=sample_weights,
            epochs=epochs,
            verbose=verbose,
        )

    def fit_partial(
        self,
        interactions: SparseCOOTensorT,
        user_features: t.Optional[SparseCSRTensorT] = None,
        item_features: t.Optional[SparseCSRTensorT] = None,
        sample_weights: t.Optional[SparseCOOTensorT] = None,
        epochs: int = 1,
        verbose: bool = False,
    ) -> t.List[float]:
        """
        Fit the model.

        Unlike fit, repeated calls to this method will cause training to resume from the current
        model state.

        For details on how to use feature matrices, see the documentation on the
        :class:`lightfm.LightFM` class.

        Arguments
        ---------
        interactions: pt.Tensor of layout sparse_coo and shape [n_users, n_items]
             the matrix containing
             user-item interactions. Will be converted to
             numpy.float32 dtype if it is not of that type.
        user_features: pt.Tensor of layout sparse_csr and shape [n_users, n_user_features], optional
             Each row contains that user's weights over features.
        item_features: pt.Tensor of layout sparse_csr and shape [n_items, n_item_features], optional
             Each row contains that item's weights over features.
        sample_weights: pt.Tensor of layout sparse_coo and shape [n_users, n_items], optional
             matrix with entries expressing weights of individual interactions from the interactions
            matrix. Its row and col arrays must be the same as those of the interactions matrix. For
            memory efficiency its possible to use the same arrays for both weights and interaction
            matrices. Defaults to weight 1.0 for all interactions. Not implemented for k-OS loss.
        epochs: int, optional
             number of epochs to run
        verbose: bool, optional
             Report on training progress using `tqdm` if available, otherwise uses iterative prints.

        Returns
        -------
        List[float]
            A list of epoch losses (sum of all sample losses)
        """
        if len(interactions.shape) != 2:
            raise ValueError("Incorrect interactions dimension, expected (n_users, n_items).")

        n_users, n_items = interactions.shape
        interactions = interactions.to_sparse_coo().type(pt.float)

        sample_weight_data = self._process_sample_weights(interactions, sample_weights)

        (user_features, item_features) = self._construct_feature_matrices(
            n_users, n_items, user_features, item_features)

        for input_data in (user_features, item_features, interactions, sample_weight_data):
            self._check_input_finite(input_data)

        # Check that the dimensionality of the feature matrices has not changed between runs.
        if not item_features.shape[1] == self._model.item_embeddings.shape[0]:
            raise ValueError("Incorrect number of features in item_features")
        if not user_features.shape[1] == self._model.user_embeddings.shape[0]:
            raise ValueError("Incorrect number of features in user_features")

        # Feed-in single samples directly
        trainset = SparseTensorDataset(interactions, sample_weight_data)
        data_loader = pt.utils.data.DataLoader(
            trainset, batch_size=1, shuffle=True, collate_fn=lambda x: x[0])

        epoch_losses = []
        for _ in self._progress(range(epochs), verbose=verbose):
            loss = self._run_epoch(
                data_loader, interactions, user_features, item_features, verbose)
            self._check_parameters_finite()
            epoch_losses.append(loss)
        self._trained = True
        return epoch_losses


class MiniBatchTrainer(LightFMTrainer):
    def __init__(
        self,
        n_user_features: int,
        n_item_features: int,
        n_components: int = 10,
        batch_size: int = 1024,
        kos_k: int = 5,  # k-th positive sample to used during each k-OS update
        kos_max: int = 10,  # maximum number of interactions to sample each k-OS update
        learning_schedule: str = "adagrad",
        builtin_optimizer: bool = False,
        loss: str = "warp",
        learning_rate: float = 0.05,
        rho: float = 0.95,
        epsilon: float = 1e-6,
        alpha: float = 0.0,  # Common alpha for user/item feature parameters (i.e. weight decay)
        max_sampled: int = 10,
        random_state: t.Optional[int] = None,
    ):
        assert batch_size > 0
        super().__init__(n_user_features, n_item_features, n_components, kos_k, kos_max,
                         learning_schedule, builtin_optimizer, loss, learning_rate, rho, epsilon,
                         alpha, max_sampled, random_state)
        self.batch_size = batch_size

    def fit_partial(
        self,
        interactions: SparseCOOTensorT,
        user_features: t.Optional[SparseCSRTensorT] = None,
        item_features: t.Optional[SparseCSRTensorT] = None,
        sample_weights: t.Optional[SparseCOOTensorT] = None,
        epochs: int = 1,
        verbose: bool = False,
    ) -> t.List[float]:
        """
        Fit the model.

        Unlike fit, repeated calls to this method will cause training to resume from the current
        model state.

        For details on how to use feature matrices, see the documentation on the
        :class:`lightfm.LightFM` class.

        Arguments
        ---------
        interactions: pt.Tensor of layout sparse_coo and shape [n_users, n_items]
             the matrix containing
             user-item interactions. Will be converted to
             numpy.float32 dtype if it is not of that type.
        user_features: pt.Tensor of layout sparse_csr and shape [n_users, n_user_features], optional
             Each row contains that user's weights over features.
        item_features: pt.Tensor of layout sparse_csr and shape [n_items, n_item_features], optional
             Each row contains that item's weights over features.
        sample_weights: pt.Tensor of layout sparse_coo and shape [n_users, n_items], optional
             matrix with entries expressing weights of individual interactions from the interactions
            matrix. Its row and col arrays must be the same as those of the interactions matrix. For
            memory efficiency its possible to use the same arrays for both weights and interaction
            matrices. Defaults to weight 1.0 for all interactions. Not implemented for k-OS loss.
        epochs: int, optional
             number of epochs to run
        verbose: bool, optional
             Report on training progress using `tqdm` if available, otherwise uses iterative prints.

        Returns
        -------
        List[float]
            A list of epoch losses (sum of all sample losses)
        """
        if len(interactions.shape) != 2:
            raise ValueError("Incorrect interactions dimension, expected (n_users, n_items).")

        n_users, n_items = interactions.shape
        interactions = interactions.to_sparse_coo().type(pt.float)

        sample_weight_data = self._process_sample_weights(interactions, sample_weights)

        (user_features, item_features) = self._construct_feature_matrices(
            n_users, n_items, user_features, item_features)

        for input_data in (user_features, item_features, interactions, sample_weight_data):
            self._check_input_finite(input_data)

        # Check that the dimensionality of the feature matrices has not changed between runs.
        if not item_features.shape[1] == self._model.item_embeddings.shape[0]:
            raise ValueError("Incorrect number of features in item_features")
        if not user_features.shape[1] == self._model.user_embeddings.shape[0]:
            raise ValueError("Incorrect number of features in user_features")

        trainset = SparseTensorDataset(interactions, sample_weight_data)
        data_loader = pt.utils.data.DataLoader(
            trainset, batch_size=self.batch_size, shuffle=True, drop_last=True)

        epoch_losses = []
        for _ in self._progress(range(epochs), verbose=verbose):
            # with pt.profiler.profile(profile_memory=True) as prof:
                loss = self._run_epoch(
                    data_loader, interactions, user_features, item_features, verbose)
                self._check_parameters_finite()
                epoch_losses.append(loss)
            # print(prof.key_averages().table(sort_by="self_cpu_time_total", row_limit=-1))
            # breakpoint()
        self._trained = True
        return epoch_losses

    def _run_epoch(
            self,
            data_loader: pt.utils.data.DataLoader,
            interactions: SparseCOOTensorT,
            user_features: t.Optional[SparseCSRTensorT],
            item_features: t.Optional[SparseCSRTensorT],
            verbose: bool = False,
    ):
        """Fit the model

        Parameters
        ----------
        data_loader: pt.utils.data.DataLoader
            Minibatch sampler and feeder for weighted interactions.
        interactions: pt.Tensor of layout sparse_coo and shape [n_users, n_items]
            the matrix containing user-item interactions.
        user_features: pt.Tensor of layout sparse_csr and shape [n_users, n_user_features], optional
            Each row contains that user's weights over features.
        item_features: pt.Tensor of layout sparse_csr and shape [n_items, n_item_features], optional
            Each row contains that item's weights over features.
        verbose: bool
            Whether to print training progress
        """
        self._model.train()
        if self.loss == 'warp':
            fit_batch_fn = self._fit_batch_warp
        else:
            raise NotImplementedError(f"Loss {self.loss} not yet implemented. Use 'warp'.")

        epoch_loss = sum(
            fit_batch_fn(positive_batch, interactions, user_features, item_features)
            for positive_batch
            in self._progress(data_loader, verbose=verbose, desc='Sample', level=1)
        )
        self._regularize()  # L2 regularisation
        return epoch_loss

    def _fit_batch_warp(
            self,
            positive_batch: t.Tuple[pt.Tensor, ...],
            interactions: SparseCOOTensorT,
            user_features: SparseCSRTensorT,
            item_features: SparseCSRTensorT,
    ) -> float:
        """
        Fit the model using the WARP loss.

        Weighted Approximate-Rank Pairwise (WARP) loss: Maximises the rank of positive examples by
        repeatedly sampling negative examples until a rank-violating one is found. Useful when only
        positive interactions are present and optimising the top of the recommendation list
        (precision@k) is desired.


        WARP Training follows adaptive stochastic gradient descent with contrastive learning:
            for all interactions, we individually and iteratively search for situations where the
            model assigns higher scores to items a specific user has not interacted with compare to
            the item at hand. If such occurs within at most `max_sampled` tries, this leads to
            tweaking model parameters such that users with similar features get closer in the latent
            space to items with observed interactions.

        Returns
        -------
        Sample loss
        """
        user_ids, item_ids, values, weights = positive_batch
        n_items = item_features.shape[0]

        user_feature_vectors = pt.stack([user_features[uid] for uid in user_ids])
        all_scores, user_reprs, all_item_reprs = self._model(
            user_feature_vectors, item_features, return_repr=True
        )

        sampled_items = pt.randint(0, n_items, (self.batch_size, self.max_sampled))
        losses = pt.zeros(self.batch_size, dtype=pt.float)
        found = pt.zeros(self.batch_size, dtype=pt.bool)
        neg_item_ids = pt.empty_like(item_ids)

        # Search for rank infringements across user batch
        for idx, (uid, positive_item_id) in enumerate(zip(user_ids, item_ids)):
            for rank in range(self.max_sampled):
                if found[idx]:
                    continue

                negative_item_id = sampled_items[idx, rank]

                if pt.is_nonzero(interactions[uid, negative_item_id]):
                    continue  # Skip item with existing interactions from this user

                if all_scores[uid, negative_item_id] < all_scores[idx, positive_item_id]:
                    continue  # No violation

                # Compute WARP loss
                losses[idx] = self._warp_loss(rank + 1, weights[idx], n_items).clip_(max=MAX_LOSS)

                # Save conflicting item
                neg_item_ids[idx] = negative_item_id
                found[idx] = True
                break

        # Condense losses to apply. That is, only retain batch samples where found[:] is True.
        idcs = pt.nonzero(found).squeeze()
        user_ids = user_ids[idcs]
        pos_item_ids = item_ids[idcs]
        neg_item_ids = neg_item_ids[idcs]
        user_reprs = user_reprs[idcs]
        pos_item_reprs = all_item_reprs[pos_item_ids]
        neg_item_reprs = all_item_reprs[neg_item_ids]
        losses = losses[idcs]

        # Apply Adaptive update rule with WARP losses
        if isinstance(self._optimizer, pt.optim.Optimizer):
            self._optimizer.zero_grad()
            losses.sum().backward()
            self._optimizer.step()
        else:
            self._optimizer.step_batch(
                losses,
                user_features,
                item_features,
                user_ids,
                pos_item_ids,
                neg_item_ids,
                user_reprs,
                pos_item_reprs,
                neg_item_reprs,
            )

        if self.item_scale > MAX_REG_SCALE or self.user_scale > MAX_REG_SCALE:
            self._regularize()
        return losses.sum().item()


class DistributedTrainer:
    pass


if __name__ == '__main__':
    n_user_features = 20
    n_item_features = 15
    # trainer = LightFMTrainer(
    trainer = MiniBatchTrainer(
        n_user_features=n_user_features,
        n_item_features=n_item_features,
        n_components=100,
        learning_schedule='adagrad',
        loss='warp',
        learning_rate=0.05,
        random_state=42
    )

    n_users = 1000
    n_items = 1000
    interactions = (pt.randint(0, 55, (n_users, n_items))/100).round().to_sparse_coo()
    user_features = pt.rand(n_users, n_user_features).round().to_sparse_csr()
    item_features = pt.rand(n_items, n_item_features).round().to_sparse_csr()

    # pt.set_num_threads(10)
    # pt.set_num_interop_threads(10)
    epoch_losses = trainer.fit(
        interactions,
        user_features,
        item_features,
        epochs=20,
        verbose=True
    )
