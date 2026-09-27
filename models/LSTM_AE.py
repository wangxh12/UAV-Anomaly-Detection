import torch
import torch.nn as nn


class Model(nn.Module):
    """
    LSTM Autoencoder for multivariate time-series anomaly detection.

    Input:
        x_enc: [B, T, C]

    Output:
        reconstruction: [B, T, C]
    """

    def __init__(self, configs):
        super().__init__()

        self.input_dim = configs.enc_in

        # Compatible with your custom arguments,
        # while retaining fallback support for standard TSLib arguments.
        self.hidden_dim = getattr(
            configs, "hidden_dim", getattr(configs, "d_model", 64)
        )

        self.num_layers = getattr(configs, "depth", getattr(configs, "e_layers", 2))

        self.dropout = getattr(configs, "dropout", 0.0)

        lstm_dropout = self.dropout if self.num_layers > 1 else 0.0

        # -----------------------------------------------------
        # Encoder
        # -----------------------------------------------------
        self.encoder = nn.LSTM(
            input_size=self.input_dim,
            hidden_size=self.hidden_dim,
            num_layers=self.num_layers,
            batch_first=True,
            dropout=lstm_dropout,
        )

        # -----------------------------------------------------
        # Decoder
        # -----------------------------------------------------
        self.decoder = nn.LSTM(
            input_size=self.hidden_dim,
            hidden_size=self.hidden_dim,
            num_layers=self.num_layers,
            batch_first=True,
            dropout=lstm_dropout,
        )

        # -----------------------------------------------------
        # Reconstruction head
        # -----------------------------------------------------
        self.projection = nn.Linear(self.hidden_dim, self.input_dim)

    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None, mask=None):
        """
        Args:
            x_enc: [B, T, C]

        Returns:
            output: [B, T, C]
        """

        # x_enc:
        # [B, T, C]
        _, (hidden, cell) = self.encoder(x_enc)

        # Last encoder layer:
        # [B, H]
        context = hidden[-1]

        # Repeat latent representation over time:
        # [B, H]
        #    ->
        # [B, T, H]
        decoder_input = context.unsqueeze(1).repeat(1, x_enc.size(1), 1)

        # Use encoder hidden states to initialize decoder
        decoder_output, _ = self.decoder(decoder_input, (hidden, cell))

        # [B, T, H]
        #    ->
        # [B, T, C]
        output = self.projection(decoder_output)

        return output
