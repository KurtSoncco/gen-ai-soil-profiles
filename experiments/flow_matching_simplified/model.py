"""Transformer Model for Flow Matching with Variable-Length Paired Token Breakpoints.

This module contains the transformer model for flow matching with variable-length paired token breakpoints.
"""

import math
import os

import torch
import torch.nn as nn
from torchinfo import summary

# Disable nested tensor support for torchinfo compatibility
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"


class SinusoidalTimeEmbedding(nn.Module):
    """Sinusoidal time embedding for flow matching."""

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            t: Time tensor of shape (batch_size, 1) or (batch_size,)
        Returns:
            Time embeddings of shape (batch_size, dim)
        """
        device = t.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = t.unsqueeze(-1) * embeddings.unsqueeze(0)
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings


class PositionalEncoding(nn.Module):
    """Positional encoding for transformer."""

    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape (batch_size, seq_len, d_model)
        Returns:
            Tensor with positional encoding added
        """
        seq_len = x.size(1)
        pe = self.pe[:seq_len, :].unsqueeze(0)  # type: ignore[index] # (1, seq_len, d_model)
        x = x + pe
        return self.dropout(x)


class TransformerModel(nn.Module):
    """Transformer Model for Flow Matching with Variable-Length Paired Token Breakpoints.

    Input:
        tokens: (batch_size, max_length, 2) - [ts, depth] pairs
        attention_mask: (batch_size, max_length) - boolean mask (True for real tokens)
        t: (batch_size, 1) - time for flow matching
    Output:
        predicted_vector_field: (batch_size, max_length, 2) - predicted flow vector
    """

    def __init__(
        self,
        input_dim: int = 2,
        output_dim: int = 2,
        hidden_dim: int = 256,
        num_layers: int = 4,
        num_heads: int = 8,
        dropout: float = 0.1,
        max_length: int = 20,
        time_emb_dim: int = 128,
        use_sequence_stats: bool = True,
        stats_dim: int = 4,  # ts_mean, ts_std, depth_mean, depth_std
    ):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dropout = dropout
        self.max_length = max_length
        self.time_emb_dim = time_emb_dim
        self.use_sequence_stats = use_sequence_stats
        self.stats_dim = stats_dim

        # Input projection: (batch_size, max_length, 2) -> (batch_size, max_length, hidden_dim)
        self.input_projection = nn.Linear(input_dim, hidden_dim)

        # Sequence statistics conditioning (to preserve absolute scale information)
        if use_sequence_stats:
            self.stats_projection = nn.Linear(stats_dim, hidden_dim)

        # Time embedding
        self.time_embedding = nn.Sequential(
            SinusoidalTimeEmbedding(time_emb_dim),
            nn.Linear(time_emb_dim, hidden_dim),
            nn.SiLU(),
        )

        # Positional encoding
        self.pos_encoder = PositionalEncoding(
            hidden_dim, max_len=max_length, dropout=dropout
        )

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,  # Use (batch, seq_len, features) format for better performance
        )
        # Disable nested tensor for compatibility with torchinfo
        transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers
        )
        # Disable nested tensor support
        transformer_encoder.use_nested_tensor = False
        self.transformer_encoder = transformer_encoder

        # Output projection: (batch_size, max_length, hidden_dim) -> (batch_size, max_length, 2)
        self.output_projection = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, output_dim),
        )

    def forward(
        self,
        tokens: torch.Tensor,
        attention_mask: torch.Tensor,
        t: torch.Tensor,
        sequence_stats: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Forward pass.

        Args:
            tokens: Token sequences of shape (batch_size, max_length, 2)
            attention_mask: Boolean mask of shape (batch_size, max_length)
                           True for real tokens, False for padding
            t: Time tensor of shape (batch_size, 1)
            sequence_stats: Optional tensor of shape (batch_size, stats_dim) containing
                          per-sequence statistics [ts_mean, ts_std, depth_mean, depth_std]
                          to preserve absolute scale information

        Returns:
            Predicted vector field of shape (batch_size, max_length, 2)
        """
        batch_size, seq_len, _ = tokens.shape

        # Project input tokens to hidden dimension
        # (batch_size, max_length, 2) -> (batch_size, max_length, hidden_dim)
        x = self.input_projection(tokens)

        # Add time embedding
        # t: (batch_size, 1) -> time_emb: (batch_size, hidden_dim)
        time_emb = self.time_embedding(t.squeeze(-1))  # (batch_size, hidden_dim)
        # Broadcast time embedding to all positions
        # (batch_size, hidden_dim) -> (batch_size, max_length, hidden_dim)
        time_emb = time_emb.unsqueeze(1).expand(-1, seq_len, -1)
        x = x + time_emb

        # Add sequence statistics conditioning to preserve absolute scale information
        if self.use_sequence_stats and sequence_stats is not None:
            # Project statistics to hidden dimension
            stats_emb = self.stats_projection(
                sequence_stats
            )  # (batch_size, hidden_dim)
            # Broadcast to all positions
            stats_emb = stats_emb.unsqueeze(1).expand(
                -1, seq_len, -1
            )  # (batch_size, max_length, hidden_dim)
            x = x + stats_emb

        # Convert attention mask from boolean to float mask for transformer
        # Transformer expects: False/0 for tokens to attend to, True/1 for tokens to mask out
        # Our mask: True for real tokens, False for padding
        # So we need to invert: padding_mask = ~attention_mask
        if attention_mask.dtype == torch.bool:
            padding_mask = ~attention_mask  # (batch_size, max_length)
        else:
            # Handle non-boolean masks (e.g., from torchinfo)
            padding_mask = attention_mask == 0  # (batch_size, max_length)

        # Add positional encoding (x is already in batch_first format)
        x = self.pos_encoder(x)  # (batch_size, seq_len, hidden_dim)

        # Apply transformer encoder with attention mask to ignore padded tokens
        # src_key_padding_mask: (batch_size, seq_len) - True for positions to mask out (padding)
        # This ensures the transformer ignores padded tokens during attention
        x = self.transformer_encoder(x, src_key_padding_mask=padding_mask)

        # Project to output dimension
        # (batch_size, max_length, hidden_dim) -> (batch_size, max_length, 2)
        output = self.output_projection(x)

        return output


if __name__ == "__main__":
    import os
    import sys
    from pathlib import Path

    import torch

    # Add parent directory to path for imports
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))

    from experiments.flow_matching_simplified.data import FlowMatchingDataLoader

    os.chdir(Path(__file__).parent)

    # Load data
    data_loader = FlowMatchingDataLoader(
        data_path=Path(__file__).parent / "data" / "breakpoints.parquet"
    )
    data_loader.load_data()

    assert data_loader.sequences is not None, "Sequences must be loaded"

    # Create train/val split
    n_total = len(data_loader.sequences)
    all_indices = torch.randperm(n_total)
    n_train = int(0.8 * n_total)
    train_indices = all_indices[:n_train].tolist()
    val_indices = all_indices[n_train:].tolist()

    # Get datasets
    train_dataset, val_dataset = data_loader.get_dataset(
        train_indices=train_indices, val_indices=val_indices
    )

    # Create model with sequence statistics conditioning
    model = TransformerModel(
        input_dim=2,
        output_dim=2,
        hidden_dim=256,
        num_layers=4,
        num_heads=8,
        dropout=0.1,
        max_length=20,
        time_emb_dim=128,
        use_sequence_stats=True,  # Enable per-sequence statistics conditioning
        stats_dim=4,  # ts_mean, ts_std, depth_mean, depth_std
    )

    # Print model summary with torchinfo
    print("=" * 80)
    print("Model Summary (torchinfo):")
    print("=" * 80)
    try:
        model_summary = summary(
            model,
            input_data=(
                torch.randn(1, 20, 2),
                torch.ones(1, 20, dtype=torch.bool),
                torch.rand(1, 1),
                torch.randn(1, 4),  # sequence_stats
            ),
            verbose=1,
            col_names=["input_size", "output_size", "num_params", "trainable"],
        )
        print(model_summary)
    except Exception as e:
        print(f"Warning: Could not generate torchinfo summary: {e}")
        print("\nFalling back to basic model structure:")
        print(model)
        print(f"\nTotal parameters: {sum(p.numel() for p in model.parameters()):,}")
    print("=" * 80)

    # Test forward pass
    sample = train_dataset[0]  # type: ignore[index] # Returns dict with 'tokens', 'attention_mask', 'length', 'sequence_stats'
    tokens = sample["tokens"]  # type: ignore[index] # (max_length, 2)
    attention_mask = sample["attention_mask"]  # type: ignore[index] # (max_length,)
    sequence_stats = sample["sequence_stats"]  # type: ignore[index] # (4,)
    t = torch.rand(1, 1)  # Random time for flow matching

    # Add batch dimension
    tokens = tokens.unsqueeze(0)  # (1, max_length, 2)
    attention_mask = attention_mask.unsqueeze(0)  # (1, max_length)
    sequence_stats = sequence_stats.unsqueeze(0)  # (1, 4)

    # Forward pass with sequence statistics
    output = model(tokens, attention_mask, t, sequence_stats=sequence_stats)
    print(f"Success! Input shape: {tokens.shape}, Output shape: {output.shape}")
    print(f"Sequence stats shape: {sequence_stats.shape}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print("\nNote: The model now uses sequence statistics conditioning to preserve")
    print("absolute scale information despite per-profile normalization.")
