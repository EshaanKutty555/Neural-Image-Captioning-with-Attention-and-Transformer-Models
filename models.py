import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import math
from nltk.translate.bleu_score import corpus_bleu

# -----------------------------------------------------------------------------
# 1. VGG-16 Encoder
# -----------------------------------------------------------------------------
class VGG16Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        vgg = models.vgg16(pretrained=True)
        # keep only convolutional layers
        self.feature_extractor = vgg.features
        # ensure output is 7×7 spatially
        self.adaptive_pool = nn.AdaptiveAvgPool2d((7,7))

    def forward(self, images):
        """
        images: (B, 3, 224, 224)
        returns: (B, 49, 512)  — 49 spatial locations × 512‐dim features
        """
        feats = self.feature_extractor(images)        # (B,512,H,W)
        feats = self.adaptive_pool(feats)             # (B,512,7,7)
        B, C, H, W = feats.shape
        # flatten spatial dims and move channels last
        feats = feats.view(B, C, H*W).permute(0,2,1)  # (B,49,512)
        return feats

# -----------------------------------------------------------------------------
# 2. Baseline LSTM Decoder (no attention)
# -----------------------------------------------------------------------------
class DecoderLSTM(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, feature_dim=512):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        # at each time step we feed [word_embed; global_image_feat]
        self.lstm = nn.LSTM(embed_dim + feature_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, features, captions):
        """
        features: (B,49,512)
        captions: (B, T)  (with <sos> at captions[:,0])
        returns: logits (B, T-1, vocab_size)
        """
        # collapse spatial features to one global vector
        global_feat = features.mean(dim=1)                           # (B,512)
        embeds = self.embedding(captions[:,:-1])                     # (B,T-1,embed_dim)
        # repeat global vector at each time step
        gf = global_feat.unsqueeze(1).repeat(1, embeds.size(1), 1)   # (B,T-1,512)
        lstm_input = torch.cat([embeds, gf], dim=2)                  # (B,T-1,embed+512)
        outputs, _ = self.lstm(lstm_input)                           # (B,T-1,hidden)
        logits = self.fc(outputs)                                    # (B,T-1,vocab)
        return logits

# -----------------------------------------------------------------------------
# 3. Bahdanau Attention Mechanism
# -----------------------------------------------------------------------------
class BahdanauAttention(nn.Module):
    def __init__(self, hidden_dim, feature_dim, attn_dim):
        super().__init__()
        self.W_h = nn.Linear(hidden_dim, attn_dim)
        self.W_f = nn.Linear(feature_dim, attn_dim)
        self.v   = nn.Linear(attn_dim, 1)

    def forward(self, hidden, features):
        """
        hidden:   (B, hidden_dim)  — previous LSTM hidden state
        features: (B, 49, feature_dim)
        returns:
          context: (B, feature_dim)  — weighted sum of features
          alpha:   (B, 49)           — attention weights
        """
        # project hidden and features
        h = self.W_h(hidden).unsqueeze(1)         # (B,1,attn_dim)
        f = self.W_f(features)                    # (B,49,attn_dim)
        # score each spatial location
        scores = self.v(torch.tanh(h + f)).squeeze(2)  # (B,49)
        alpha  = F.softmax(scores, dim=1)             # (B,49)
        # compute context vector
        context = (alpha.unsqueeze(2) * features).sum(dim=1)  # (B,feature_dim)
        return context, alpha

# -----------------------------------------------------------------------------
# 3b. Attention-augmented LSTM Decoder
# -----------------------------------------------------------------------------
class AttentionDecoder(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, feature_dim=512, attn_dim=256):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.attention = BahdanauAttention(hidden_dim, feature_dim, attn_dim)
        self.lstm_cell = nn.LSTMCell(embed_dim + feature_dim, hidden_dim)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, features, captions):
        """
        features: (B,49,512)
        captions: (B, T)
        returns: logits (B, T-1, vocab_size)
        """
        B, T = captions.size()
        device = features.device
        # init LSTMCell states
        hx = torch.zeros(B, hidden_dim, device=device)
        cx = torch.zeros(B, hidden_dim, device=device)
        outputs = []
        for t in range(T-1):
            word_embed = self.embedding(captions[:,t])        # (B, embed_dim)
            context, _ = self.attention(hx, features)         # (B, feature_dim)
            lstm_in = torch.cat([word_embed, context], dim=1) # (B, embed+feature)
            hx, cx = self.lstm_cell(lstm_in, (hx, cx))        # (B, hidden)
            logit = self.fc(hx)                               # (B, vocab)
            outputs.append(logit.unsqueeze(1))
        outputs = torch.cat(outputs, dim=1)                   # (B, T-1, vocab)
        return outputs

# -----------------------------------------------------------------------------
# 4. Transformer Decoder
# -----------------------------------------------------------------------------
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0)/d_model))
        pe[:,0::2] = torch.sin(pos * div)
        pe[:,1::2] = torch.cos(pos * div)
        self.pe = pe.unsqueeze(0)  # (1, max_len, d_model)

    def forward(self, x):
        # x: (T, B, d_model)
        x = x + self.pe[:,:x.size(0),:].to(x.device)
        return x

class TransformerDecoderModel(nn.Module):
    def __init__(self, vocab_size, d_model=512, nhead=8, num_layers=6, dim_feedforward=2048):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.pos_enc = PositionalEncoding(d_model)
        dec_layer = nn.TransformerDecoderLayer(d_model, nhead, dim_feedforward)
        self.transformer_decoder = nn.TransformerDecoder(dec_layer, num_layers)
        self.fc_out = nn.Linear(d_model, vocab_size)

    def forward(self, features, captions):
        """
        features: (B,49,512)  → memory for cross-attention
        captions: (B, T)
        returns: logits (B, T-1, vocab_size)
        """
        # prepare memory: (49, B, 512)
        memory = features.permute(1,0,2)
        # prepare target: (T-1, B, d_model)
        tgt = self.embed(captions[:,:-1]).permute(1,0,2)
        tgt = self.pos_enc(tgt)
        # subsequent mask
        mask = nn.Transformer.generate_square_subsequent_mask(tgt.size(0)).to(features.device)
        out = self.transformer_decoder(tgt, memory, tgt_mask=mask)
        out = out.permute(1,0,2)  # (B, T-1, d_model)
        logits = self.fc_out(out)
        return logits

# -----------------------------------------------------------------------------
# 5. BLEU Evaluation Function
# -----------------------------------------------------------------------------
def evaluate_bleu(encoder, decoder, dataloader, idx2word, device):
    """
    Runs inference on dataloader, collects references & candidates,
    and returns BLEU-1 and BLEU-4 scores.
    """
    encoder.eval()
    decoder.eval()
    refs, hyps = [], []
    with torch.no_grad():
        for imgs, caps in dataloader:
            imgs, caps = imgs.to(device), caps.to(device)
            feats = encoder(imgs)
            logits = decoder(feats, caps)
            preds = logits.argmax(dim=2)
            for ref_seq, pred_seq in zip(caps[:,1:], preds):
                ref = [idx2word[i.item()] for i in ref_seq if i.item() not in {0,1}]
                hyp = [idx2word[i.item()] for i in pred_seq if i.item() not in {0,1}]
                refs.append([ref])
                hyps.append(hyp)
    bleu1 = corpus_bleu(refs, hyps, weights=(1,0,0,0))
    bleu4 = corpus_bleu(refs, hyps, weights=(0.25,0.25,0.25,0.25))
    return bleu1, bleu4

# -----------------------------------------------------------------------------
# Usage notes:
# - Define `hidden_dim` in your script before instantiating AttentionDecoder.
# - Create your Dataset + DataLoader to yield (image_tensor, caption_tensor).
# - Instantiate:
#       encoder = VGG16Encoder().to(device)
#       decoder = DecoderLSTM(vocab_size, embed_dim, hidden_dim).to(device)
#   Or swap in AttentionDecoder or TransformerDecoderModel.
# - Write your usual training loop (optimizer, loss = CrossEntropy, backprop, etc.)
# - After training, call `evaluate_bleu(...)` for before/after comparisons.
