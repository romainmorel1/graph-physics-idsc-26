import torch
import torch.nn as nn
from torch_geometric.data import Data

from graphphysics.models.layers import (
    GraphNetBlock,
    build_mlp,
)


class EncodeProcessDecode(nn.Module):
    """
    An Encode-Process-Decode model for graph neural networks.

    This model architecture is designed for processing graph-structured data. It consists of three main components:
    an encoder, a processor, and a decoder. The encoder maps input graph features to a latent space, the processor
    performs message passing and updates node and edge representations, and the decoder generates the final output from
    the processed graph.
    """

    def __init__(
        self,
        message_passing_num: int,
        node_input_size: int,
        edge_input_size: int,
        output_size: int,
        hidden_size: int = 128,
        only_processor: bool = False,
    ):
        """
        Initializes the EncodeProcessDecode model.

        Args:
            message_passing_num (int): Number of message passing steps.
            node_input_size (int): Size of the node input features.
            edge_input_size (int): Size of the edge input features.
            output_size (int): Size of the output features.
            hidden_size (int, optional): Size of the hidden representations. Defaults to 128.
            only_processor (bool, optional): If True, only the processor is used (no encoding or decoding).
            Defaults to False.
        """
        super().__init__()
        self.only_processor = only_processor
        self.hidden_size = hidden_size
        self.d = output_size

        if not self.only_processor:
            self.nodes_encoder = build_mlp(
                in_size=node_input_size,
                hidden_size=hidden_size,
                out_size=hidden_size,
            )

            self.edges_encoder = build_mlp(
                in_size=edge_input_size,
                hidden_size=hidden_size,
                out_size=hidden_size,
            )

            self.decode_module = build_mlp(
                in_size=hidden_size,
                hidden_size=hidden_size,
                out_size=output_size,
                layer_norm=False,
            )

        self.processor_list = nn.ModuleList(
            [GraphNetBlock(hidden_size=hidden_size) for _ in range(message_passing_num)]
        )

    def forward(self, graph: Data) -> tuple[torch.Tensor, list]:
        """
        Forward pass of the EncodeProcessDecode model.

        Args:
            graph (Data): Input graph data containing 'x' (node features), 'edge_index', and 'edge_attr'.

        Returns:
            tuple[torch.Tensor, list]: The output tensor and a list of CV terms from each block.
                If 'only_processor' is False, the node features are passed through the decoder before returning.
        """
        edge_index = graph.edge_index

        if self.only_processor:
            x, edge_attr = graph.x, graph.edge_attr
        else:
            x = self.nodes_encoder(graph.x)
            edge_attr = self.edges_encoder(graph.edge_attr)

        cv_terms = []
        for block in self.processor_list:
            x, edge_attr, cv_dict = block(x, edge_index, edge_attr)
            cv_terms.append(cv_dict)

        if self.only_processor:
            return x, cv_terms
        else:
            x_decoded = self.decode_module(x)
            return x_decoded, cv_terms
