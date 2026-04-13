#!/usr/bin/env python3
"""
train.py

Train image-captioning on Flickr8k using:
  • VGG-16 encoder (frozen)
  • Decoder: baseline LSTM, LSTM+Bahdanau Attn, or Transformer
Usage:
  python train.py --model attn --epochs 10 --batch_size 32 --lr 1e-3
"""

import argparse
import torch
import torch.nn as nn
import torch.optim as optim

from data_loader import get_loaders
from models import (
    VGG16Encoder,
    DecoderLSTM,
    AttentionDecoder,
    TransformerDecoderModel,
    evaluate_bleu,
)

def train_one_epoch(encoder, decoder, dataloader, criterion, optimizer, device):
    decoder.train()
    encoder.eval()  # keep encoder frozen
    total_loss = 0.0
    for images, captions, _ in dataloader:
        images, captions = images.to(device), captions.to(device)
        feats = encoder(images)                       # (B,49,512)
        outputs = decoder(feats, captions)            # (B,T-1,V)
        # shift targets
        targets = captions[:, 1:].reshape(-1)         # (B*(T-1))
        logits  = outputs.reshape(-1, outputs.size(2))# (B*(T-1),V)
        loss = criterion(logits, targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(dataloader)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["baseline","attn","transformer"], default="baseline")
    parser.add_argument("--epochs",     type=int,   default=5)
    parser.add_argument("--batch_size", type=int,   default=32)
    parser.add_argument("--lr",         type=float, default=1e-3)
    parser.add_argument("--data_root",  type=str,   default="data/Flickr8k_Dataset")
    parser.add_argument("--cap_file",   type=str,   default="data/Flickr8k_text/Flickr8k.token.txt")
    parser.add_argument("--train_ids",  type=str,   default="data/Flickr8k_text/Flickr_8k.trainImages.txt")
    parser.add_argument("--val_ids",    type=str,   default="data/Flickr8k_text/Flickr_8k.devImages.txt")
    parser.add_argument("--test_ids",   type=str,   default="data/Flickr8k_text/Flickr_8k.testImages.txt")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1) Data loaders + vocab
    train_loader, val_loader, test_loader, vocab = get_loaders(
        root_dir=args.data_root,
        captions_file=args.cap_file,
        train_ids=args.train_ids,
        val_ids=args.val_ids,
        test_ids=args.test_ids,
        batch_size=args.batch_size,
    )
    pad_idx = vocab.stoi["<pad>"]

    # 2) Encoder (frozen)
    encoder = VGG16Encoder().to(device)
    for p in encoder.parameters():
        p.requires_grad = False

    # 3) Decoder selection
    if args.model == "baseline":
        decoder = DecoderLSTM(
            vocab_size=len(vocab),
            embed_dim=256,
            hidden_dim=512,
        ).to(device)

    elif args.model == "attn":
        decoder = AttentionDecoder(
            vocab_size=len(vocab),
            embed_dim=256,
            hidden_dim=512,
            feature_dim=512,
            attn_dim=256,
        ).to(device)

    else:  # transformer
        decoder = TransformerDecoderModel(
            vocab_size=len(vocab),
            d_model=512,
            nhead=8,
            num_layers=6,
            dim_feedforward=2048,
        ).to(device)

    # 4) Loss & optimizer
    criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)
    optimizer = optim.Adam(decoder.parameters(), lr=args.lr)

    # 5) Training + validation loop
    best_bleu4 = 0.0
    for epoch in range(1, args.epochs + 1):
        avg_loss = train_one_epoch(
            encoder, decoder, train_loader, criterion, optimizer, device
        )
        print(f"[Epoch {epoch}/{args.epochs}] Train Loss: {avg_loss:.4f}")

        bleu1, bleu4 = evaluate_bleu(
            encoder, decoder, val_loader, vocab.itos, device
        )
        print(f"[Epoch {epoch}] Val BLEU-1: {bleu1:.4f}, BLEU-4: {bleu4:.4f}")

        # save best
        if bleu4 > best_bleu4:
            best_bleu4 = bleu4
            torch.save(decoder.state_dict(), f"{args.model}_best.pth")
            print(f"→ New best BLEU-4: {bleu4:.4f}, model saved")

    # 6) Final test evaluation
    decoder.load_state_dict(torch.load(f"{args.model}_best.pth"))
    test_bleu1, test_bleu4 = evaluate_bleu(
        encoder, decoder, test_loader, vocab.itos, device
    )
    print(f"Test BLEU-1: {test_bleu1:.4f}, BLEU-4: {test_bleu4:.4f}")

if __name__ == "__main__":
    main()
