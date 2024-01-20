import typing as t
from collections import defaultdict

import torch as pt

from lightfm.utils import SparseCSRTensorT

__all__ = [
    'AdagradOptimizer',
    'LightFMOptimizer',
]


def get_representation(
        feature_vector: pt.sparse.Tensor, embeddings: pt.nn.Parameter, scale: pt.nn.Parameter
) -> pt.sparse.Tensor:
    return (feature_vector * scale) @ embeddings


class LightFMOptimizer:
    @pt.no_grad()
    def step(
            self,
            loss: pt.Tensor,
            user_features: SparseCSRTensorT,
            item_features: SparseCSRTensorT,
            user_id: int,
            positive_item_id: int,
            negative_item_id: int
    ) -> None:
        raise NotImplementedError('@pt.no_grad() def step(...) -> None')

    @pt.no_grad()
    def step_batch(
            self,
            losses: pt.Tensor,
            user_features: SparseCSRTensorT,
            item_features: SparseCSRTensorT,
            user_ids: int,
            positive_item_ids: int,
            negative_item_ids: int,
            user_reprs: pt.Tensor,
            pos_item_reprs: pt.Tensor,
            neg_item_reprs: pt.Tensor,
    ) -> None:
        raise NotImplementedError('@pt.no_grad() def step_batch(...) -> None')

    def zero_grad(self):
        pass  # no-op, parameters do not require gradient backpropagation


class AdagradOptimizer(LightFMOptimizer):
    """Stochastic Adagrad learning rule.

    Note: this is a like-for-like re-implementation of the Adagrad variant used in LightFM's library
    """
    def __init__(
            self,
            params: t.Dict[str, pt.nn.Parameter],
            n_components: int,
            lr: float = 1e-2,
            lr_decay: float = 0.0,
            weight_decay: float = 0.0,
            initial_accumulator_value: float = 0.0,
            eps: float = 1e-6,
    ):
        self.latent_dim = n_components + 1  # include bias term
        self.params = params
        self.defaults = {
            'lr': lr,
            'lr_decay': lr_decay,
            'weight_decay': weight_decay,
            'initial_accumulator_value': initial_accumulator_value,
            'eps': eps,
        }
        self.state = defaultdict(dict)
        self._reset_state()

    def _reset_state(self):
        for name, param in self.params.items():
            self.state[name] = defaultdict(dict)
            state = self.state[name]
            state['step'] = 0
            state['gradients'] = pt.full_like(
                param,
                fill_value=self.defaults['initial_accumulator_value'],
                memory_format=pt.preserve_format)

    def _update_parameter(
            self,
            param_name: str,
            feature_index: pt.Tensor,
            feature_weights: pt.Tensor,
            loss: pt.Tensor
    ) -> float:
        # Update first gradient moments
        gradients = self.state[param_name]['gradients'][feature_index]
        local_learning_rates = self.defaults['lr'] / gradients.sqrt()
        gradients += (loss * feature_weights) ** 2
        # Update parameter and scale using regularization parameter
        param = self.params[param_name][feature_index]
        param -= local_learning_rates * loss * feature_weights
        param *= 1.0 + self.defaults['weight_decay'] * local_learning_rates
        return local_learning_rates.sum().item()

    def _step_warp(
            self,
            loss: pt.Tensor,
            user_features: SparseCSRTensorT,
            item_features: SparseCSRTensorT,
            user_id: int,
            positive_item_id: int,
            negative_item_id: int,
            user_repr: SparseCSRTensorT,
            pos_item_repr: SparseCSRTensorT,
            neg_item_repr: SparseCSRTensorT,
    ):
        """
        Apply the gradient step.
        """
        pos_item_feature_vector = item_features[positive_item_id]
        neg_item_feature_vector = item_features[negative_item_id]
        user_feature_vector = user_features[user_id]

        pos_item_feature_index = pos_item_feature_vector.indices().view(-1)
        pos_item_feature_weights = pos_item_feature_vector.values().view(-1, 1)
        neg_item_feature_index = neg_item_feature_vector.indices().view(-1)
        neg_item_feature_weights = neg_item_feature_vector.values().view(-1, 1)
        user_feature_index = user_feature_vector.indices().view(-1)
        user_feature_weights = user_feature_vector.values().view(-1, 1)
        del pos_item_feature_vector, neg_item_feature_vector, user_feature_vector

        # Update latent bias terms
        avg_learning_rate = self._update_parameter(
            'item_biases', pos_item_feature_index, pos_item_feature_weights, -loss)
        avg_learning_rate += self._update_parameter(
            'item_biases', neg_item_feature_index, neg_item_feature_weights, loss)
        avg_learning_rate += self._update_parameter(
            'user_biases', user_feature_index, user_feature_weights, -loss)

        # Update latent embeddings
        # Note: Item embeddings use the user latent representation. Whereas user embeddings use the
        # difference between the latent representations of the two items.
        avg_learning_rate += self._update_parameter(
            'item_embeddings',
            pos_item_feature_index,
            pos_item_feature_weights,
            -loss * user_repr)
        avg_learning_rate += self._update_parameter(
            'item_embeddings',
            neg_item_feature_index,
            neg_item_feature_weights,
            loss * user_repr)
        avg_learning_rate += self._update_parameter(
            'user_embeddings',
            user_feature_index,
            user_feature_weights,
            loss * (neg_item_repr - pos_item_repr))

        # Average the local learning rates
        avg_learning_rate /= self.latent_dim * (
            user_feature_index[-1] + pos_item_feature_index[-1] + neg_item_feature_index[-1]
            - user_feature_index[0] - pos_item_feature_index[0] - neg_item_feature_index[0]
        )

        # Update the scaling factors for lazy regularization, using the average learning rate
        # of features updated for this example.
        self.params['item_scale'] *= 1.0 + self.defaults['weight_decay'] * avg_learning_rate
        self.params['user_scale'] *= 1.0 + self.defaults['weight_decay'] * avg_learning_rate

    @pt.no_grad()
    def step(
            self,
            loss: pt.Tensor,
            user_features: SparseCSRTensorT,
            item_features: SparseCSRTensorT,
            user_id: int,
            positive_item_id: int,
            negative_item_id: int,
            user_repr: SparseCSRTensorT,
            pos_item_repr: SparseCSRTensorT,
            neg_item_repr: SparseCSRTensorT,
    ):
        """Apply Adagrad learning rule"""
        # For now only WARP updates are supported
        self._step_warp(loss, user_features, item_features, user_id, positive_item_id,
                        negative_item_id, user_repr, pos_item_repr, neg_item_repr)

    @pt.no_grad()
    def step_batch(
            self,
            losses: pt.Tensor,
            user_features: SparseCSRTensorT,
            item_features: SparseCSRTensorT,
            user_ids: int,
            positive_item_ids: int,
            negative_item_ids: int,
            user_reprs: pt.Tensor,
            pos_item_reprs: pt.Tensor,
            neg_item_reprs: pt.Tensor,
    ):
        """Apply Adagrad learning rule"""
        # For now only WARP updates are supported
        for idx in range(losses.shape[0]):
            self._step_warp(
                losses[idx], user_features, item_features,
                user_ids[idx], positive_item_ids[idx], negative_item_ids[idx],
                user_reprs[idx], pos_item_reprs[idx], neg_item_reprs[idx],)
