import torch
from torch.nn.modules.loss import _Loss

from graphphysics.utils.nodetype import NodeType


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _prepare_mask_for_loss(
    network_output: torch.Tensor,
    node_type: torch.Tensor,
    masks: list[NodeType],
    selected_indexes: torch.Tensor = None,
):
    mask = node_type == masks[0]
    for i in range(1, len(masks)):
        mask = torch.logical_or(mask, node_type == masks[i])

    if selected_indexes is not None:
        n, _ = network_output.shape
        nodes_mask = ~torch.isin(torch.arange(n), selected_indexes).to(device)
        mask = torch.logical_and(nodes_mask, mask)

    return mask


class L2Loss(_Loss):
    def __init__(self, cv_weight: float = 0.0, **kwargs):
        """
        Initializes the L2Loss with optional CV regularization.

        Args:
            cv_weight (float, optional): Weight for the CV (coefficient of variation) regularization term.
                Defaults to 0.0 (no CV regularization).
            **kwargs: Additional keyword arguments for the parent class.
        """
        super().__init__(**kwargs)
        self.cv_weight = cv_weight

    @property
    def __name__(self):
        return "MSE"

    def forward(
        self,
        target: torch.Tensor,
        network_output: torch.Tensor,
        node_type: torch.Tensor,
        masks: list[NodeType],
        selected_indexes: torch.Tensor = None,
        cv_terms: list = None,
        **kwargs
    ) -> torch.Tensor:
        """
        Computes L2 loss for nodes of specific types with optional CV regularization.

        Args:
            target (torch.Tensor): The target values.
            network_output (torch.Tensor): The predicted values from the network.
            node_type (torch.Tensor): Tensor containing the type of each node.
            masks (list[NodeType]): List of NodeTypes to include in the loss calculation.
            selected_indexes (torch.Tensor, optional): Indexes of nodes to exclude from the loss calculation.
            cv_terms (list, optional): List of CV terms from MoE blocks to include in regularization.

        Returns:
            torch.Tensor: The mean squared error for the specified node types plus CV regularization term.

        Note:
            This method calculates the L2 loss only for nodes of the types specified in 'masks'.
            If 'selected_indexes' is provided, those nodes are excluded from the loss calculation.
            If 'cv_terms' and cv_weight > 0, adds a CV regularization term to encourage load balancing.
        """
        mask = _prepare_mask_for_loss(
            network_output, node_type, masks, selected_indexes
        )
        errors = ((network_output - target) ** 2)[mask]
        mse_loss = torch.mean(errors)

        # Add CV regularization term if cv_terms are provided
        if cv_terms is not None and self.cv_weight > 0.0 and len(cv_terms) > 0:
            cv_loss = 0.0
            for cv_dict in cv_terms:
                if "edge_cv" in cv_dict:
                    cv_loss += cv_dict["edge_cv"]
                if "node_cv" in cv_dict:
                    cv_loss += cv_dict["node_cv"]
            cv_loss = cv_loss / (2 * len(cv_terms))  # Average CV across all blocks
            return mse_loss + self.cv_weight * cv_loss**2
        
        return mse_loss
