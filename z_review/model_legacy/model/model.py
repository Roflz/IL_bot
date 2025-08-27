import torch
import torch.nn as nn
import torchvision.models as models

class ImitationHybridModel(nn.Module):
    """
    Hybrid model for OSRS imitation learning:
    - Visual encoder (ResNet18) for screenshots
    - MLP for structured features
    - LSTM for sequence modeling
    - Output head for action prediction
    """
    def __init__(self, structured_input_dim, action_output_dim, visual_embedding_dim=256, structured_embedding_dim=64, lstm_hidden_dim=256, lstm_layers=1):
        super().__init__()
        # Visual encoder: ResNet18 backbone (remove final fc)
        resnet = models.resnet18(weights=None)
        self.visual_encoder = nn.Sequential(*list(resnet.children())[:-1])  # Output: (B, 512, 1, 1)
        self.visual_fc = nn.Linear(512, visual_embedding_dim)
        # Structured encoder: simple MLP
        self.structured_encoder = nn.Sequential(
            nn.Linear(structured_input_dim, structured_embedding_dim),
            nn.ReLU(),
            nn.Linear(structured_embedding_dim, structured_embedding_dim),
            nn.ReLU()
        )
        # LSTM for sequence modeling
        self.lstm = nn.LSTM(
            input_size=visual_embedding_dim + structured_embedding_dim,
            hidden_size=lstm_hidden_dim,
            num_layers=lstm_layers,
            batch_first=True
        )
        # Output head: predict action (regression or classification)
        self.output_head = nn.Sequential(
            nn.Linear(lstm_hidden_dim, lstm_hidden_dim),
            nn.ReLU(),
            nn.Linear(lstm_hidden_dim, action_output_dim)
        )

    def forward(self, images, structured_inputs, seq_lens=None, hidden=None):
        """
        Args:
            images: (B, T, C, H, W) tensor of screenshots
            structured_inputs: (B, T, F) tensor of structured features
            seq_lens: Optional list of sequence lengths (for packing)
            hidden: Optional initial LSTM state
        Returns:
            action_preds: (B, T, action_output_dim)
        """
        B, T, C, H, W = images.shape
        # Visual encoding
        images_flat = images.view(B * T, C, H, W)
        visual_emb = self.visual_encoder(images_flat).view(B * T, -1)  # (B*T, 512)
        visual_emb = self.visual_fc(visual_emb)  # (B*T, visual_embedding_dim)
        visual_emb = visual_emb.view(B, T, -1)
        # Structured encoding
        struct_emb = self.structured_encoder(structured_inputs)  # (B, T, structured_embedding_dim)
        # Concatenate
        x = torch.cat([visual_emb, struct_emb], dim=-1)  # (B, T, D)
        # LSTM
        lstm_out, _ = self.lstm(x, hidden)
        # Output head
        action_preds = self.output_head(lstm_out)  # (B, T, action_output_dim)
        return action_preds

# TODO: Add support for variable-length sequences (packing/padding)
# TODO: Add option to use pretrained ResNet weights
# TODO: Customize output head for specific action types (mouse, key, etc.) 