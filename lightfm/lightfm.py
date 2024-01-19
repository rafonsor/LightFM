import torch as pt


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
        self.item_biases: pt.nn.Parameter  # n_item_features
        self.user_biases: pt.nn.Parameter  # n_user_features
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
        self.item_biases = pt.nn.Parameter(pt.zeros(self.n_item_features))
        self.user_biases = pt.nn.Parameter(pt.zeros(self.n_user_features))
        self.user_scale = pt.nn.Parameter(pt.tensor(1.0), requires_grad=False)
        self.item_scale = pt.nn.Parameter(pt.tensor(1.0), requires_grad=False)

    def forward(self, users_features: pt.Tensor, items_features: pt.Tensor) -> pt.Tensor:
        """Predict scores for user-item pairs

        Parameters
        ----------
        users_features: pt.Tensor
            Feature vectors for a group of users
        items_features: pt.Tensor
            Feature vectors for a group of items

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
        return pt.mm(users_repr, items_repr.T) + user_biases[:, None] + item_biases[None, :]

    def predict(self, user_features: pt.Tensor, item_features: pt.Tensor) -> float:
        """Predict scores for a single user-item pair

        Parameters
        ----------
        user_features: pt.Tensor
            Feature vectors for a group of users
        item_features: pt.Tensor
            Feature vectors for a group of items

        Returns
        -------
        scores: pt.Tensor
            Predicted scores for user-item pairs
        """
        return self.forward(user_features.unsqueeze(0), item_features.unsqueeze(0)).item()

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
