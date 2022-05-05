import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel, AutoConfig


__all__ = ['Deberta_V3_Large']

# ====================================================
# Model - deberta_v3_large
# ====================================================
class Deberta_V3_Large(nn.Module):

    def __init__(self, args):
        super().__init__()
        self.args = args

        self.config = AutoConfig.from_pretrained(args.pretrained_model, output_hidden_states=True)
        self.model = AutoModel.from_pretrained(args.pretrained_model, config=self.config)

        self.fc_dropout = nn.Dropout(args.fc_dropout)
        self.fc = nn.Linear(self.config.hidden_size, args.target_size)
        self._init_weights(self.fc)
        self.attention = nn.Sequential(
            nn.Linear(self.config.hidden_size, 512),
            nn.Tanh(),
            nn.Linear(512, 1),
            nn.Softmax(dim=1)
        )
        self._init_weights(self.attention)
        
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        
    def feature(self, inputs):
        outputs = self.model(**inputs)
        last_hidden_states = outputs[0]
        # feature = torch.mean(last_hidden_states, 1)
        weights = self.attention(last_hidden_states)
        feature = torch.sum(weights * last_hidden_states, dim=1)
        return feature

    def forward(self, inputs):
        feature = self.feature(inputs)
        output = torch.sigmoid(self.fc(self.fc_dropout(feature)))
        return output