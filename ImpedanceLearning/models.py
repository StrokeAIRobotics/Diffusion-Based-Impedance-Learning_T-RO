import torch
import torch.nn as nn


class NoisePredictorTransformerWithCrossAttentionTime(nn.Module):
    def __init__(self, seq_length, hidden_dim, num_heads=8, num_layers=4, num_timesteps=20, use_forces=True):
        super(NoisePredictorTransformerWithCrossAttentionTime, self).__init__()
        self.use_forces = use_forces
        self.seq_length = seq_length

        # Input projection for trajectory (pos + quat)
        self.traj_embedding = nn.Linear(7, hidden_dim)
        self.positional_encoding = nn.Parameter(torch.zeros(1, seq_length, hidden_dim))

        # Learnable time embedding
        self.time_embedding = nn.Embedding(num_timesteps, hidden_dim)

        if self.use_forces:
            # Conditioning projection for forces + moments
            self.cond_proj = nn.Linear(6, hidden_dim)
            self.cross_attention = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=num_heads, batch_first=True)

        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=num_heads, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Output layers
        self.fc1 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim // 2, 7)

    def forward(self, noisy_pos, noisy_q, t, forces=None, moment=None):
        batch_size, seq_length, _ = noisy_pos.shape

        # Embed trajectory
        x = torch.cat((noisy_pos, noisy_q), dim=-1)  # [B, T, 7]
        x = self.traj_embedding(x) + self.positional_encoding  # [B, T, H]

        # Add time embedding
        t_embed = self.time_embedding(torch.tensor([t], device=x.device))  # [1, H]
        t_embed = t_embed.expand(x.size(0), self.seq_length, -1)  # [B, T, H]
        x = x + t_embed

        # Optional cross-attention with force/moment
        if self.use_forces:
            cond = torch.cat((forces, moment), dim=-1)  # [B, T, 6]
            cond = self.cond_proj(cond)  # [B, T, H]

            # Cross-attention fusion
            attn_output, _ = self.cross_attention(query=x, key=cond, value=cond)
            x = x + attn_output  # residual connection

        # Transformer Encoder
        x = self.transformer(x)  # [B, T, H]

        # Decode to output
        x = self.fc1(x)
        x = self.relu(x)
        predicted_trajectory = self.fc2(x)  # [B, T, 7]

        return predicted_trajectory