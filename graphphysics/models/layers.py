import math
from typing import Any, Dict, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn import MessagePassing
from torch_geometric.utils import softmax


try:
    import dgl.sparse as dglsp
    from dgl.sparse import SparseMatrix
    HAS_DGL_SPARSE = True
except ImportError:
    HAS_DGL_SPARSE = False
    dglsp = None
    SparseMatrix = Any

class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization.

    This module applies RMS normalization over the last dimension of the input tensor.
    """

    def __init__(self, d: int, p: float = -1.0, eps: float = 1e-8, bias: bool = False):
        """
        Initializes the RMSNorm module.

        Args:
            d (int): The dimension of the input tensor.
            p (float, optional): Partial RMSNorm. Valid values are in [0, 1].
                Default is -1.0 (disabled).
            eps (float, optional): A small value to avoid division by zero.
                Default is 1e-8.
            bias (bool, optional): Whether to include a bias term. Default is False.
        """
        super().__init__()

        self.d = d
        self.p = p
        self.eps = eps
        self.bias = bias

        self.scale = nn.Parameter(torch.ones(d))

        if self.bias:
            self.offset = nn.Parameter(torch.zeros(d))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of RMSNorm.

        Args:
            x (torch.Tensor): Input tensor of shape (..., d).

        Returns:
            torch.Tensor: Normalized tensor of the same shape as input.
        """
        if self.p < 0.0 or self.p > 1.0:
            norm_x = x.norm(2, dim=-1, keepdim=True)
            d_x = self.d
        else:
            partial_size = int(self.d * self.p)
            partial_x, _ = torch.split(x, [partial_size, self.d - partial_size], dim=-1)
            norm_x = partial_x.norm(2, dim=-1, keepdim=True)
            d_x = partial_size

        rms_x = norm_x / math.sqrt(d_x)
        x_normed = x / (rms_x + self.eps)

        if self.bias:
            return self.scale * x_normed + self.offset

        return self.scale * x_normed


ACTIVATION = {
    "relu": nn.ReLU,
    "gelu": nn.GELU,
}


def build_mlp(
    in_size: int,
    hidden_size: int,
    out_size: int,
    nb_of_layers: int = 4,
    layer_norm: bool = True,
    act: str = "relu",
) -> nn.Module:
    """
    Builds a Multilayer Perceptron.

    Args:
        in_size (int): Size of the input features.
        hidden_size (int): Size of the hidden layers.
        out_size (int): Size of the output features.
        nb_of_layers (int, optional): Total number of linear layers in the MLP.
            Must be at least 2. Defaults to 4.
        layer_norm (bool, optional): Whether to apply RMS normalization to the
            output layer. Defaults to True.
        act (str, optional): Activation function to use ('relu' or 'gelu'). Defaults to 'relu'.

    Returns:
        nn.Module: The constructed MLP model.
    """
    assert nb_of_layers >= 2, "The MLP must have at least 2 layers (input and output)."

    if act not in ACTIVATION:
        raise NotImplementedError(f"Activation '{act}' not supported.")
    activation = ACTIVATION[act]

    layers = [nn.Linear(in_size, hidden_size), activation()]

    # Add hidden layers
    for _ in range(nb_of_layers - 2):
        layers.extend([nn.Linear(hidden_size, hidden_size), activation()])

    # Add output layer
    layers.append(nn.Linear(hidden_size, out_size))

    if layer_norm:
        layers.append(RMSNorm(out_size))

    return nn.Sequential(*layers)


class GatedMLP(nn.Module):
    """
    A Gated Multilayer Perceptron.

    This layer applies a gated activation to the input features.
    """

    def __init__(self, in_size: int, hidden_size: int, expansion_factor: int):
        """
        Initializes the GatedMLP layer.

        Args:
            in_size (int): Size of the input features.
            hidden_size (int): Size of the hidden layer.
            expansion_factor (int): Expansion factor for the hidden layer size.
        """
        super().__init__()

        self.linear1 = nn.Linear(in_size, expansion_factor * hidden_size)
        self.linear2 = nn.Linear(in_size, expansion_factor * hidden_size)

        self.activation = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the GatedMLP layer.

        Args:
            x (torch.Tensor): Input tensor of shape (..., in_size).

        Returns:
            torch.Tensor: Output tensor of shape (..., expansion_factor * hidden_size).
        """
        left = self.activation(self.linear1(x))
        right = self.linear2(x)
        return left * right


def build_gated_mlp(
    in_size: int,
    hidden_size: int,
    out_size: int,
    expansion_factor: int = 3,
) -> nn.Module:
    """
    Builds a Gated MLP.

    Args:
        in_size (int): Size of the input features.
        hidden_size (int): Size of the hidden layer.
        out_size (int): Size of the output features.
        expansion_factor (int, optional): Expansion factor for the hidden layer size.
            Defaults to 3.

    Returns:
        nn.Module: The constructed Gated MLP model.
    """
    layers = [
        RMSNorm(in_size),
        GatedMLP(
            in_size=in_size, hidden_size=hidden_size, expansion_factor=expansion_factor
        ),
        nn.Linear(hidden_size * expansion_factor, out_size),
    ]
    return nn.Sequential(*layers)


class Normalizer(nn.Module):
    """
    A module for normalizing data during training.

    This module maintains running statistics to normalize input data.
    """

    def __init__(
        self,
        size: int,
        max_accumulations: int = 10**5,
        std_epsilon: float = 1e-8,
        name: str = "Normalizer",
        device: Optional[Union[str, torch.device]] = "cuda",
    ):
        """
        Initializes the Normalizer module.

        Args:
            size (int): Size of the input data.
            max_accumulations (int, optional): Maximum number of accumulations allowed.
                Defaults to 1e5.
            std_epsilon (float, optional): Epsilon value to avoid division by zero in
                standard deviation. Defaults to 1e-8.
            name (str, optional): Name of the Normalizer. Defaults to "Normalizer".
            device (str or torch.device, optional): Device to run the Normalizer on.
                Defaults to "cuda".
        """
        super().__init__()
        self.name = name
        self.device = device
        self._max_accumulations = max_accumulations
        self._std_epsilon = torch.tensor(
            std_epsilon, dtype=torch.float32, requires_grad=False, device=device
        )
        self.register_buffer("_acc_count", torch.tensor(0.0, device=device))
        self.register_buffer("_num_accumulations", torch.tensor(0.0, device=device))
        self.register_buffer(
            "_acc_sum",
            torch.zeros(
                (1, size), dtype=torch.float32, requires_grad=False, device=device
            ),
        )
        self.register_buffer(
            "_acc_sum_squared",
            torch.zeros(
                (1, size), dtype=torch.float32, requires_grad=False, device=device
            ),
        )

    def forward(
        self, batched_data: torch.Tensor, accumulate: bool = True
    ) -> torch.Tensor:
        """
        Normalizes input data and accumulates statistics.

        Args:
            batched_data (torch.Tensor): Input data of shape (batch_size, size).
            accumulate (bool, optional): Whether to accumulate statistics.
                Defaults to True.

        Returns:
            torch.Tensor: Normalized data of the same shape as input.
        """
        if accumulate:
            # Stop accumulating after reaching max_accumulations to prevent numerical issues
            if self._num_accumulations < self._max_accumulations:
                self._accumulate(batched_data.detach())
        return (batched_data - self._mean()) / self._std_with_epsilon()

    def inverse(self, normalized_batch_data: torch.Tensor) -> torch.Tensor:
        """
        Inverse transformation of the normalizer.

        Args:
            normalized_batch_data (torch.Tensor): Normalized data.

        Returns:
            torch.Tensor: Denormalized data.
        """
        return normalized_batch_data * self._std_with_epsilon() + self._mean()

    def _accumulate(self, batched_data: torch.Tensor):
        """
        Accumulates the statistics of the batched data.

        Args:
            batched_data (torch.Tensor): Input data of shape (batch_size, size).
        """
        count = batched_data.shape[0]
        data_sum = torch.sum(batched_data, dim=0, keepdim=True)
        squared_data_sum = torch.sum(batched_data**2, dim=0, keepdim=True)

        self._acc_sum += data_sum
        self._acc_sum_squared += squared_data_sum
        self._acc_count += count
        self._num_accumulations += 1

    def _mean(self) -> torch.Tensor:
        safe_count = torch.max(
            self._acc_count, torch.tensor(1.0, device=self._acc_count.device)
        )
        return self._acc_sum / safe_count

    def _std_with_epsilon(self) -> torch.Tensor:
        safe_count = torch.max(
            self._acc_count, torch.tensor(1.0, device=self._acc_count.device)
        )
        variance = self._acc_sum_squared / safe_count - self._mean() ** 2
        std = torch.sqrt(torch.clamp(variance, min=0.0))
        return torch.max(std, self._std_epsilon)

    def get_variable(self) -> Dict[str, Any]:
        """
        Returns the internal variables of the normalizer.

        Returns:
            Dict[str, Any]: A dictionary containing the normalizer's variables.
        """
        return {
            "_max_accumulations": self._max_accumulations,
            "_std_epsilon": self._std_epsilon,
            "_acc_count": self._acc_count,
            "_num_accumulations": self._num_accumulations,
            "_acc_sum": self._acc_sum,
            "_acc_sum_squared": self._acc_sum_squared,
            "name": self.name,
        }
    



class GraphNetBlock(MessagePassing):
    """
    Graph Network Block implementing the message passing mechanism.
    This block updates both node and edge features.
    """

    def __init__(
        self, hidden_size: int, nb_of_layers: int = 4, layer_norm: bool = True
    ):
        """
        Initializes the GraphNetBlock.

        Args:
            hidden_size (int): The size of the hidden representations.
            nb_of_layers (int, optional): The number of layers in the MLPs.
                Defaults to 4.
            layer_norm (bool, optional): Whether to use layer normalization in the MLPs.
                Defaults to True.
        """
        super().__init__(aggr="add", flow="source_to_target")
        edge_input_dim = 3 * hidden_size
        node_input_dim = 2 * hidden_size
        self.edge_block = build_mlp(
            in_size=edge_input_dim,
            hidden_size=hidden_size,
            out_size=hidden_size,
            nb_of_layers=nb_of_layers,
            layer_norm=layer_norm,
        )
        self.node_block = build_mlp(
            in_size=node_input_dim,
            hidden_size=hidden_size,
            out_size=hidden_size,
            nb_of_layers=nb_of_layers,
            layer_norm=layer_norm,
        )

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        size: int = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass of the GraphNetBlock.

        Args:
            x (torch.Tensor): Node features of shape [num_nodes, hidden_size].
            edge_index (torch.Tensor): Edge indices of shape [2, num_edges].
            edge_attr (torch.Tensor): Edge features of shape [num_edges, hidden_size].
            size (Size, optional): The size of the source and target nodes.
                Defaults to None.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Updated node features and edge features.
        """
        # Update edge attributes
        row, col = edge_index
        x_i = x[col]  # Target node features
        x_j = x[row]  # Source node features
        edge_attr_ = self.edge_update(edge_attr, x_i, x_j)

        # Perform message passing and update node features
        x_ = self.propagate(
            edge_index, x=x, edge_attr=edge_attr_, size=(x.size(0), x.size(0))
        )

        edge_attr = edge_attr + edge_attr_
        x = x + x_

        return x, edge_attr

    def edge_update(
        self, edge_attr: torch.Tensor, x_i: torch.Tensor, x_j: torch.Tensor
    ) -> torch.Tensor:
        """
        Updates edge features.

        Args:
            edge_attr (torch.Tensor): Edge features [num_edges, hidden_size].
            x_i (torch.Tensor): Target node features [num_edges, hidden_size].
            x_j (torch.Tensor): Source node features [num_edges, hidden_size].

        Returns:
            torch.Tensor: Updated edge features [num_edges, hidden_size].
        """
        edge_input = torch.cat([edge_attr, x_i, x_j], dim=-1)
        edge_attr = self.edge_block(edge_input)
        return edge_attr

    def message(self, edge_attr: torch.Tensor) -> torch.Tensor:
        """
        Constructs messages to be aggregated.

        Args:
            edge_attr (torch.Tensor): Edge features [num_edges, hidden_size].

        Returns:
            torch.Tensor: Messages [num_edges, hidden_size].
        """
        return edge_attr

    def update(self, aggr_out: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """
        Updates node features after aggregation.

        Args:
            aggr_out (torch.Tensor): Aggregated messages [num_nodes, hidden_size].
            x (torch.Tensor): Node features [num_nodes, hidden_size].

        Returns:
            torch.Tensor: Updated node features [num_nodes, hidden_size].
        """
        node_input = torch.cat([x, aggr_out], dim=-1)
        x = self.node_block(node_input)
        return x



class SparseEdgeAttentionBlock(MessagePassing):
    """
    Sparse attention over existing edges in the graph.
    Same interface as GraphNetBlock: returns (x, edge_attr).
    """

    def __init__(
        self,
        hidden_size: int,
        nb_of_layers: int = 4,
        layer_norm: bool = True,
        edge_bias: bool = True,
        dropout: float = 0.0,
    ):
        super().__init__(aggr="add", flow="source_to_target")
        self.hidden_size = hidden_size
        self.scale = 1.0 / math.sqrt(hidden_size)

        self.W_q = nn.Linear(hidden_size, hidden_size, bias=False)
        self.W_k = nn.Linear(hidden_size, hidden_size, bias=False)
        self.W_v = nn.Linear(hidden_size, hidden_size, bias=False)

        self.edge_bias = edge_bias
        self.W_e = nn.Linear(hidden_size, 1, bias=False) if edge_bias else None
        self.attn_drop = nn.Dropout(dropout)

        self.node_block = build_mlp(
            in_size=2 * hidden_size,
            hidden_size=hidden_size,
            out_size=hidden_size,
            nb_of_layers=nb_of_layers,
            layer_norm=layer_norm,
        )

        # edge update résiduel (comme GraphNetBlock)
        self.edge_block = build_mlp(
            in_size=3 * hidden_size,
            hidden_size=hidden_size,
            out_size=hidden_size,
            nb_of_layers=nb_of_layers,
            layer_norm=layer_norm,
        )

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor):
        # 1) Edge update avec les residus
        row, col = edge_index  # row=src (j), col=dst (i)
        edge_update = self.edge_block(torch.cat([edge_attr, x[col], x[row]], dim=-1))
        edge_attr = edge_attr + edge_update

        # 2) Propagate attention-weighted messages using PyG aggregation
        out = self.propagate(edge_index, x=x, edge_attr=edge_attr, size=(x.size(0), x.size(0)))

        # 3) Node update avec les résidus
        x_update = self.node_block(torch.cat([x, out], dim=-1))
        x = x + x_update

        return x, edge_attr

    def message(self, x_i: torch.Tensor, x_j: torch.Tensor, edge_attr: torch.Tensor, index: torch.Tensor):
        """
        x_i: features of target nodes for each edge [E, H]
        x_j: features of source nodes for each edge [E, H]
        index: target node indices (same as col) [E]
        """
        q_i = self.W_q(x_i)
        k_j = self.W_k(x_j)
        v_j = self.W_v(x_j)

        logits = (q_i * k_j).sum(dim=-1) * self.scale  # [E]
        if self.edge_bias:
            logits = logits + self.W_e(edge_attr).squeeze(-1)  # [E]

        alpha = softmax(logits, index)  # softmax per target node
        alpha = self.attn_drop(alpha)

        return v_j * alpha.unsqueeze(-1)  # [E, H] Correction



class SparseNodeAttentionBlock(nn.Module):
    """
    Sparse Node Attention Block.
    
    Flow:
    1. Node Attention (calculée via Q*K sur les nœuds).
    2. Node Update (Aggregation -> MLP).
    3. Edge Update (utilise les NOUVEAUX features des nœuds).
    
    Complexité : O(|E|) grâce à dgl.sparse.
    """

    def __init__(
        self,
        hidden_size: int,
        nb_of_layers: int = 4,
        layer_norm: bool = True,
        edge_bias: bool = True,
        dropout: float = 0.0,
    ):
        super().__init__()
        
        if not HAS_DGL_SPARSE:
            raise ImportError("dgl.sparse is required for this block.")

        self.hidden_size = hidden_size
        self.edge_bias = edge_bias
        # Scaling factor 1/sqrt(d)
        self.scale = 1.0 / math.sqrt(hidden_size)

        # Projections Q, K, V (Node-based)
        self.W_q = nn.Linear(hidden_size, hidden_size, bias=False)
        self.W_k = nn.Linear(hidden_size, hidden_size, bias=False)
        self.W_v = nn.Linear(hidden_size, hidden_size, bias=False)

        # Projection pour le biais d'arête
        if self.edge_bias:
            self.W_e = nn.Linear(hidden_size, 1, bias=False)

        self.attn_dropout = nn.Dropout(dropout)

        # Node Update MLP
        self.node_block = build_mlp(
            in_size=2 * hidden_size,
            hidden_size=hidden_size,
            out_size=hidden_size,
            nb_of_layers=nb_of_layers,
            layer_norm=layer_norm,
        )

        # Edge Update MLP
        self.edge_block = build_mlp(
            in_size=3 * hidden_size,
            hidden_size=hidden_size,
            out_size=hidden_size,
            nb_of_layers=nb_of_layers,
            layer_norm=layer_norm,
        )

    def forward(
        self, 
        x: torch.Tensor, 
        edge_index: torch.Tensor, 
        edge_attr: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: Node features [N, H]
            edge_index: PyG Edge Index [2, E] (Source, Target)
            edge_attr: Edge features [E, H]
        """
        N = x.size(0)

        # -----------------------------------------------------------
        # 0. Conversion PyG edge_index -> DGL SparseMatrix
        # -----------------------------------------------------------
        # PyG edge_index est [Source, Target].
        # Pour l'agrégation (Target <- Source), la matrice d'adjacence A doit avoir :
        # Rows = Target, Cols = Source.
        # Donc on stack [Target, Source] -> [edge_index[1], edge_index[0]]
        indices = torch.stack([edge_index[1], edge_index[0]])
        adj = dglsp.spmatrix(indices, shape=(N, N))

        # -----------------------------------------------------------
        # 1. Calcul de l'Attention (Node-Driven)
        # -----------------------------------------------------------
        q = self.W_q(x) # [N, H]
        k = self.W_k(x) # [N, H]
        v = self.W_v(x) # [N, H]

        # SDDMM : Sampled Dense-Dense Matrix Multiplication
        # Calcule le score pour chaque arête existante.
        # A_ij = q_i * k_j^T
        # Shape inputs: adj=[N, N], q=[N, H], k.T=[H, N]
        attn_score_mat = dglsp.sddmm(adj, q, k.transpose(0, 1))

        # Scaling (1/sqrt(d))
        # On multiplie directement les valeurs (.val)
        attn_score_mat = dglsp.val_like(attn_score_mat, attn_score_mat.val * self.scale)

        # Edge Bias
        if self.edge_bias:
            # On projette l'attribut d'arête vers un scalaire
            e_bias = self.W_e(edge_attr).view(-1)
            # On ajoute ce biais aux scores d'attention
            new_val = attn_score_mat.val + e_bias
            attn_score_mat = dglsp.val_like(attn_score_mat, new_val)

        # Softmax (sur la dimension des voisins entrants, dim=1)
        attn_weights = attn_score_mat.softmax(dim=1)
        
        # Dropout
        if self.training and self.attn_dropout.p > 0:
            attn_weights = dglsp.val_like(attn_weights, self.attn_dropout(attn_weights.val))

        # -----------------------------------------------------------
        # 2. Agrégation & Node Update
        # -----------------------------------------------------------
        # SpMM : Sparse-Dense Matrix Multiplication
        # Aggregate: Somme pondérée des valeurs v des voisins
        aggr_out = dglsp.spmm(attn_weights, v) # [N, H]

        # Update des noeuds
        node_input = torch.cat([x, aggr_out], dim=-1)
        x_new = x + self.node_block(node_input)

        # -----------------------------------------------------------
        # 3. Edge Update (Node First -> Edge Second)
        # -----------------------------------------------------------
        # On récupère les features des nœuds MIS A JOUR
        # On utilise l'edge_index original [Source, Target]
        src_idx, dst_idx = edge_index[0], edge_index[1]
        
        x_src_new = x_new[src_idx]
        x_dst_new = x_new[dst_idx]

        edge_input = torch.cat([edge_attr, x_src_new, x_dst_new], dim=-1)
        edge_attr_new = edge_attr + self.edge_block(edge_input)

        return x_new, edge_attr_new
    

# -------------------------------------------------------------------------
# 1. Core Logic : Kimi Delta Attention (RNN Scan)
# -------------------------------------------------------------------------
class KimiDeltaAttention(nn.Module):
    def __init__(
        self, 
        hidden_size: int, 
        num_heads: int = 4, 
        head_dim: int = 64, 
        chunk_size: int = 32,
        use_short_conv: bool = True
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.chunk_size = chunk_size
        self.v_dim = num_heads * head_dim
        self.use_short_conv = use_short_conv
        
        # 1. Linear Projections
        self.q_proj = nn.Linear(hidden_size, self.v_dim, bias=False)
        self.k_proj = nn.Linear(hidden_size, self.v_dim, bias=False)
        self.v_proj = nn.Linear(hidden_size, self.v_dim, bias=False)
        
        # 2. Short Convolution (Depthwise 1D Conv)
        # Le papier utilise kernel_size=3 pour capturer le contexte local
        if self.use_short_conv:
            self.conv_q = nn.Conv1d(self.v_dim, self.v_dim, kernel_size=3, padding=1, groups=self.v_dim)
            self.conv_k = nn.Conv1d(self.v_dim, self.v_dim, kernel_size=3, padding=1, groups=self.v_dim)
            self.conv_v = nn.Conv1d(self.v_dim, self.v_dim, kernel_size=3, padding=1, groups=self.v_dim)

        # 3. Gate Projection (Log-space decay)
        self.g_proj = nn.Linear(hidden_size, self.v_dim, bias=True)

        # 4. Output
        self.out_norm = RMSNorm(self.v_dim)
        self.out_proj = nn.Linear(self.v_dim, hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [Batch, Seq_Len, Hidden]
        """
        B, L, _ = x.shape
        H = self.num_heads
        D = self.head_dim
        
        # --- A. Projections ---
        # [B, L, V_dim]
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        
        # --- B. ShortConv + Swish (SiLU) ---
        # Conv1d attend [Batch, Channel, Length], donc on transpose
        if self.use_short_conv:
            q = q.transpose(1, 2) # [B, V_dim, L]
            k = k.transpose(1, 2)
            v = v.transpose(1, 2)
            
            # Application Conv + Swish (SiLU) comme dans le papier
            q = F.silu(self.conv_q(q)).transpose(1, 2) # Retour en [B, L, V_dim]
            k = F.silu(self.conv_k(k)).transpose(1, 2)
            v = F.silu(self.conv_v(v)).transpose(1, 2)
        else:
            # Même sans conv, le Swish est appliqué
            q = F.silu(q)
            k = F.silu(k)
            v = F.silu(v)

        # --- C. Reshape & L2 Norm ---
        q = q.view(B, L, H, D).transpose(1, 2) # [B, H, L, D]
        k = k.view(B, L, H, D).transpose(1, 2)
        v = v.view(B, L, H, D).transpose(1, 2)

        # L2 Norm pour Q et K (Eigenvalue stability)
        q = F.normalize(q, p=2, dim=-1)
        k = F.normalize(k, p=2, dim=-1)
        
        # --- D. Gate (Decay) ---
        # g = LogSigmoid(Linear(x))
        # Note: on pourrait aussi mettre une ShortConv sur g si on voulait être puriste, 
        # mais le papier insiste surtout sur q,k,v.
        g = F.logsigmoid(self.g_proj(x)).view(B, L, H, D).transpose(1, 2)
        
        # --- E. Fast Chunk Kernel ---
        out = chunk_kimi_kda(q, k, v, g, chunk_size=self.chunk_size)
        
        # --- F. Output ---
        out = out.reshape(B, L, -1) # Flatten Heads
        out = self.out_norm(out)
        
        return self.out_proj(out)

# -------------------------------------------------------------------------
# 2. Wrapper compatible avec Processor (Node + Edge Update)
# -------------------------------------------------------------------------
class KimiSpatialBlock(nn.Module):
    """
    Bloc Spatial utilisant Kimi Linear Attention.
    Remplace SparseNodeAttentionBlock.
    
    Particularité : Traite le graphe comme une séquence globale.
    Bidirectionnel : Scan Forward + Scan Backward pour capturer tout le contexte spatial.
    """
    def __init__(self, hidden_size, bidirectional=True, **kwargs):
        super().__init__()
        self.bidirectional = bidirectional
        
        # Le coeur : Kimi Attention
        self.kimi_layer = KimiDeltaAttention(hidden_size, **kwargs)
        
        # Node Update MLP
        self.node_block = build_mlp(
            in_size=2 * hidden_size, # [x_old, x_kimi_out]
            hidden_size=hidden_size,
            out_size=hidden_size
        )

        # Edge Update MLP (Standard residuel)
        self.edge_block = build_mlp(
            in_size=3 * hidden_size,
            hidden_size=hidden_size,
            out_size=hidden_size
        )

    def forward(self, x, edge_index, edge_attr):
        # x: [N, H] -> On reshape en [1, N, H] pour le traiter comme une séquence
        # (On suppose ici 1 graphe par forward, ou Batch PyG concaténé)
        x_seq = x.unsqueeze(0) 

        # 1. Forward Scan
        out_fwd = self.kimi_layer(x_seq) # [1, N, H]

        if self.bidirectional:
            # 2. Backward Scan (Flip sequence, apply Kimi, Flip back)
            x_bwd = torch.flip(x_seq, dims=[1])
            out_bwd = self.kimi_layer(x_bwd)
            out_bwd = torch.flip(out_bwd, dims=[1])
            
            # Combine (Somme ou Moyenne)
            out_kimi = out_fwd + out_bwd
        else:
            out_kimi = out_fwd

        # Retour en format node [N, H]
        out_kimi = out_kimi.squeeze(0)

        # 3. Node Update (Residual)
        node_in = torch.cat([x, out_kimi], dim=-1)
        x_new = x + self.node_block(node_in)

        # 4. Edge Update (Utilise les nouveaux noeuds)
        row, col = edge_index
        x_src = x_new[row]
        x_dst = x_new[col]
        
        edge_in = torch.cat([edge_attr, x_src, x_dst], dim=-1)
        edge_attr_new = edge_attr + self.edge_block(edge_in)

        return x_new, edge_attr_new
    
    
def chunk_kimi_kda(q, k, v, g, chunk_size=64, initial_state=None):
    """
    Implémentation Chunk-wise de Kimi Delta Attention (KDA).
    Correspond au Listing 8b du papier technique.
    
    Args:
        q, k, v: [Batch, Heads, Seq_Len, Head_Dim]
        g: Log-space decay (log(alpha)). [Batch, Heads, Seq_Len, Head_Dim]
        chunk_size: Taille du bloc (BT).
    
    Returns:
        o: Output [Batch, Heads, Seq_Len, Head_Dim]
    """
    B, H, T, D = q.shape
    BT = chunk_size
    
    # Padding si T n'est pas divisible par BT
    if T % BT != 0:
        pad_len = BT - (T % BT)
        q = F.pad(q, (0, 0, 0, pad_len))
        k = F.pad(k, (0, 0, 0, pad_len))
        v = F.pad(v, (0, 0, 0, pad_len))
        g = F.pad(g, (0, 0, 0, pad_len))
    
    # 1. Chunking: [B, H, Num_Chunks, BT, D]
    # On utilise view/permute pour remplacer einops 'b h (n c) d -> b h n c d'
    q = q.view(B, H, -1, BT, D)
    k = k.view(B, H, -1, BT, D)
    v = v.view(B, H, -1, BT, D)
    g = g.view(B, H, -1, BT, D)
    
    NT = q.shape[2] # Nombre de chunks

    # 2. Intra-Chunk Computation (Parallel)
    # CumSum du decay en log-space
    gc = g.cumsum(dim=-2) # Somme sur la dimension temporelle du chunk (BT)
    
    # Préparation des matrices d'interaction locale
    # On doit calculer Aqk et Akk pour chaque chunk
    # C'est du broadcast [..., BT, 1, D] * [..., 1, BT, D] -> [..., BT, BT, D] -> sum(-1) -> [..., BT, BT]
    
    # Astuce: Pour éviter de stocker [NT, BT, BT] qui est lourd, on le fait à la volée ou vectorisé.
    # Ici, implémentation vectorisée pour la clarté.
    
    # Masque causal local (i >= j)
    mask_causal = torch.tril(torch.ones(BT, BT, device=q.device, dtype=torch.bool))
    
    # Termes de decay relatifs: exp(g_i - g_j)
    # g: [..., BT, D] -> gc
    # gc_i: [..., BT, 1, D], gc_j: [..., 1, BT, D]
    # decay_matrix = (gc.unsqueeze(-2) - gc.unsqueeze(-3)).exp() # [..., BT, BT, D]
    # Note: Le snippet original fait des boucles manuelles sur BT pour économiser la mémoire.
    # Pour PyTorch, vectoriser est souvent mieux sauf si BT est grand.
    # On va suivre la logique "Block-Parallel" du snippet mais vectorisée sur B et H.

    # --- Phase A: Pré-calcul des matrices locales (Aqk, Akk) ---
    # Pour respecter le snippet exactement, on itère sur i (colonnes du chunk)
    # Mais vectoriser Aqk et Akk est plus rapide sur GPU moderne que la boucle Python.
    
    # Matrice des différences de gate: G[i, j] = gc[i] - gc[j]
    # Attention: le snippet utilise g_i (cumulé) et g (cumulé).
    # s1_i = (gc[i] - gc).exp() pour j <= i
    
    gc_i = gc.unsqueeze(-2) # [..., BT, 1, D]
    gc_j = gc.unsqueeze(-3) # [..., 1, BT, D]
    decay_rel = (gc_i - gc_j) # Log-diff
    
    # On masque les positions futures pour respecter la causalité
    # mask: True si j > i (interdit)
    mask_future = ~mask_causal
    decay_rel = decay_rel.masked_fill(mask_future.unsqueeze(-1), -float('inf'))
    decay_term = decay_rel.exp() # [..., BT, BT, D]

    # Aqk[i, j] = sum_d (q[i,d] * k[j,d] * decay[i,j,d])
    # [..., BT, 1, D] * [..., 1, BT, D] * [..., BT, BT, D] -> sum(-1)
    Aqk = (q.unsqueeze(-2) * k.unsqueeze(-3) * decay_term).sum(dim=-1) # [B, H, NT, BT, BT]
    
    # Akk[i, j] = sum_d (k[i,d] * k[j,d] * decay[i,j,d]) (Pour le terme delta)
    # Note: Dans le snippet, s2_i utilise (gc - g_i).exp(). C'est l'inverse ?
    # Vérif Listing 8b line 12: s2_i = (gc - g_i).exp(). 
    # Ah, c'est pour la construction de la matrice d'inversion.
    
    # Recalcul précis de Akk pour l'inversion (Listing 8b lines 10-12)
    # Le snippet fait une boucle. Reproduisons la boucle pour l'exactitude mathématique.
    # C'est la partie critique "Inverse Iterative".
    
    A = torch.zeros(B, H, NT, BT, BT, device=q.device, dtype=q.dtype)
    
    # On calcule Akk "spécial" pour l'inversion M
    # Akk[i,j] (masked) = k_i * k_j * exp(gc_j - gc_i)  <-- Attention au sens
    # Le snippet ligne 16: A = -Akk
    
    # Version vectorisée de la boucle lines 8-13 du snippet :
    # s2_i = (gc - g_i).exp() -> decay "futur" local
    # Akk[..., i] = (k_i * k * s2_i).sum(-1)
    # Cela remplit la colonne i.
    
    # Pour vectoriser proprement la construction de A (Matrice de transition) :
    # A_base[j, i] = - (k[j] * k[i] * exp(gc[j] - gc[i])).sum() pour j > i
    # C'est une Strict Lower Triangular.
    
    decay_inv = (gc_j - gc_i).exp() # exp(gc[j] - gc[i])
    A_base = -(k.unsqueeze(-2) * k.unsqueeze(-3) * decay_inv).sum(dim=-1)
    
    # On ne garde que la partie Strict Lower (j > i selon les indices du snippet, ou j < i ?)
    # Snippet ligne 14: mask = triu(diagonal=0). A = -Akk.masked_fill(mask, 0)
    # Donc on garde la partie STRICT LOWER.
    A = A_base.tril(-1) # Strict lower part
    
    # --- Phase B: Inversion Itérative (Le "Forward Substitution") ---
    # C'est la ligne 15-16 du snippet : A[..., i, :i] += ...
    # C'est O(BT) séquentiel. Sur BT=64, c'est acceptable.
    for i in range(1, BT):
        # A[..., i, :i] += (A[..., i, :, None] * A[..., :, :i]).sum(-2)
        # On met à jour la ligne i en utilisant les lignes précédentes.
        # Vectorisation sur B, H, NT
        row_i = A[..., i, :i] # [..., i]
        # Produit scalaire "batché" des termes précédents
        # A[..., i, :i] est [..., 1, i] (une partie de ligne)
        # A[..., :i, :i] est le bloc carré déjà calculé
        # Le snippet est subtil. Il fait:
        # A[i, :i] = A[i, :i] + sum_k (A[i, k] * A[k, :i])
        
        # Slice views
        # A_prev: [B, H, NT, i, i]
        # A_curr_row_part: [B, H, NT, i] -> A[..., i, :i]
        
        # update = A[..., i, :i].unsqueeze(-2) @ A[..., :i, :i]
        # Non, la boucle snippet est: A[i, :i] += (A[i, :, None] * A[:, :i]).sum(-2)
        # Mais le mask triu a mis des 0.
        # C'est une résolution de système triangulaire.
        # Faisons-le simplement :
        vec = A[..., i, :i].clone()
        mat = A[..., :i, :i]
        # update = vec @ mat
        update = torch.matmul(vec.unsqueeze(-2), mat).squeeze(-2)
        A[..., i, :i] = vec + update

    # A devient (I + Lower)^-1. On ajoute I.
    A = A + torch.eye(BT, device=q.device)

    # --- Phase C: Calcul de u et w (Auxiliary Vectors) ---
    # w = A @ (exp(gc) * k)
    # u = A @ v
    
    # exp(gc) * k
    k_scaled = k * gc.exp()
    
    # Matmul [..., BT, BT] @ [..., BT, D]
    w = torch.matmul(A, k_scaled)
    u = torch.matmul(A, v)

    # --- Phase D: Inter-Chunk Recurrence (Scan) ---
    # S: [B, H, D, D]
    if initial_state is None:
        S = torch.zeros(B, H, D, D, device=q.device, dtype=q.dtype)
    else:
        S = initial_state

    # Output container
    o = torch.zeros_like(v)
    
    # Recurrence loop over NT chunks
    for i in range(NT):
        q_i = q[:, :, i] # [B, H, BT, D]
        g_i = gc[:, :, i]
        u_i = u[:, :, i]
        w_i = w[:, :, i]
        k_i = k[:, :, i]
        v_i = v[:, :, i]
        
        # 1. Output intra-chunk (Partie venant de l'état passé S)
        # o_recurrent = (q_i * exp(g_i)) @ S
        term_S = torch.matmul(q_i * g_i.exp(), S)
        
        # 2. Output intra-chunk (Partie locale venant de Aqk)
        # o_local = Aqk @ (u_i - w_i @ S)
        # w_i @ S -> [..., BT, D] @ [..., D, D] -> [..., BT, D]
        term_correction = u_i - torch.matmul(w_i, S)
        
        # Aqk pour ce chunk:
        # Aqk_i = Aqk[:, :, i] # [..., BT, BT]
        # Attention: Aqk calculé plus haut (Phase A) utilisait decay_term causal.
        # C'est bien ce qu'il faut.
        # term_local = Aqk_i @ term_correction
        
        # Note: Aqk n'a pas été stocké entièrement pour économiser la mémoire ? 
        # Si NT est grand, stocker Aqk [NT, BT, BT] est OK (BT=64).
        # On recalcul Aqk ici si on veut, ou on l'utilise.
        # On l'a calculé ligne 66.
        Aqk_i = Aqk[:, :, i]
        term_local = torch.matmul(Aqk_i, term_correction)
        
        o[:, :, i] = term_S + term_local
        
        # 3. State Update pour le prochain chunk
        # Decay total du chunk: exp(gc_last - gc)
        # g_i_last: [..., 1, D] (Dernier token du chunk)
        decay_chunk = (g_i[:, :, -1:, :] - g_i).exp()
        
        # S = S * exp(g_i_last)
        S = S * g_i[:, :, -1, :].unsqueeze(-2).exp()
        
        # S += (k_i * decay).T @ v_i
        # [..., BT, D].T @ [..., BT, D] -> [..., D, D]
        # Attention aux dimensions batch [B, H, D, BT] @ [B, H, BT, D]
        k_decayed = k_i * decay_chunk
        S = S + torch.matmul(k_decayed.transpose(-1, -2), v_i)
        
    # Flatten structure
    o = o.view(B, H, T, D)
    if T != q.shape[2]*BT: # Si padding
        o = o[:, :, :T, :]
        
    return o.permute(0, 2, 1, 3) # [B, T, H, D]