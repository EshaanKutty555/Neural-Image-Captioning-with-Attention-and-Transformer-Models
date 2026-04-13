import os
import nltk
from collections import Counter

import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as transforms

# -----------------------------------------------------------------------------
# 1. Vocabulary
# -----------------------------------------------------------------------------
class Vocabulary:
    """Simple word‐to‐index mapping."""
    def __init__(self, freq_threshold=5):
        self.freq_threshold = freq_threshold
        self.itos = {0: "<pad>", 1: "<sos>", 2: "<eos>", 3: "<unk>"}
        self.stoi = {v:k for k,v in self.itos.items()}

    def __len__(self):
        return len(self.itos)

    @staticmethod
    def tokenizer(text):
        return [tok.lower() for tok in nltk.word_tokenize(text)]

    def build_vocab(self, captions_file):
        counter = Counter()
        with open(captions_file, 'r', encoding='utf-8') as f:
            # skip header if present
            first = f.readline()
            if 'caption' in first.lower():
                pass
            else:
                f.seek(0)
            for line in f:
                parts = line.strip().split(',', 1)  # split on first comma
                if len(parts) != 2:
                    continue
                _, caption = parts
                tokens = Vocabulary.tokenizer(caption)
                counter.update(tokens)

        idx = len(self.itos)
        for word, cnt in counter.items():
            if cnt >= self.freq_threshold:
                self.stoi[word] = idx
                self.itos[idx] = word
                idx += 1

    def numericalize(self, text):
        tokens = Vocabulary.tokenizer(text)
        nums = [self.stoi.get(tok, self.stoi["<unk>"]) for tok in tokens]
        return [self.stoi["<sos>"]] + nums + [self.stoi["<eos>"]]


# -----------------------------------------------------------------------------
# 2. Flickr8k Dataset
# -----------------------------------------------------------------------------
class Flickr8kDataset(Dataset):
    def __init__(self, images_dir, captions_file, ids, vocab, transform=None):
        """
        ids: either a path to a text file listing image names, or a list of image filenames.
        """
        self.images_dir = images_dir
        self.vocab = vocab
        self.transform = transform

        # Load split IDs
        if isinstance(ids, (list, tuple)):
            self.ids = ids
        else:
            with open(ids, 'r', encoding='utf-8') as f:
                self.ids = [line.strip() for line in f]

        # Load captions (CSV: image,caption)
        self.captions = {}
        with open(captions_file, 'r', encoding='utf-8') as f:
            header = f.readline()
            if 'caption' not in header.lower():
                f.seek(0)
            for line in f:
                parts = line.strip().split(',', 1)
                if len(parts) != 2:
                    continue
                img, caption = parts
                if img in self.ids:
                    self.captions.setdefault(img, []).append(caption.strip())

        # Flatten to (img, caption) pairs
        self.samples = []
        for img, caps in self.captions.items():
            for cap in caps:
                self.samples.append((img, cap))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_name, caption = self.samples[idx]
        img_path = os.path.join(self.images_dir, img_name)
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        cap_idxs = self.vocab.numericalize(caption)
        cap_tensor = torch.tensor(cap_idxs, dtype=torch.long)
        return image, cap_tensor


# -----------------------------------------------------------------------------
# 3. Collate function for padding
# -----------------------------------------------------------------------------
def collate_fn(batch):
    images, captions = zip(*batch)
    images = torch.stack(images, 0)
    lengths = [len(c) for c in captions]
    max_len = max(lengths)
    padded = torch.zeros(len(captions), max_len, dtype=torch.long)
    for i, cap in enumerate(captions):
        padded[i, :lengths[i]] = cap
    return images, padded, lengths


# -----------------------------------------------------------------------------
# 4. DataLoader factory
# -----------------------------------------------------------------------------
def get_loaders(
    root_dir="data/Images",
    captions_file="data/captions.txt",
    train_ids=None,
    val_ids=None,
    test_ids=None,
    freq_threshold=5,
    batch_size=64,
    num_workers=4
):
    # Transforms
    transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485,0.456,0.406],
            std =[0.229,0.224,0.225]
        )
    ])

    # Build vocab
    vocab = Vocabulary(freq_threshold)
    vocab.build_vocab(captions_file)

    # Helper to list all image files if no IDs provided
    def load_ids(ids_arg):
        if ids_arg and os.path.isfile(ids_arg):
            with open(ids_arg, 'r', encoding='utf-8') as f:
                return [line.strip() for line in f]
        return [
            fn for fn in os.listdir(root_dir)
            if fn.lower().endswith(('.jpg','jpeg','png'))
        ]

    train_list = load_ids(train_ids)
    val_list   = load_ids(val_ids)
    test_list  = load_ids(test_ids)

    # Create datasets
    train_ds = Flickr8kDataset(root_dir, captions_file, train_list, vocab, transform)
    val_ds   = Flickr8kDataset(root_dir, captions_file, val_list,   vocab, transform)
    test_ds  = Flickr8kDataset(root_dir, captions_file, test_list,  vocab, transform)

    # DataLoaders
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, collate_fn=collate_fn)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, collate_fn=collate_fn)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, collate_fn=collate_fn)

    return train_loader, val_loader, test_loader, vocab


# -----------------------------------------------------------------------------
# 5. Quick test (optional)
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    train_loader, val_loader, test_loader, vocab = get_loaders(batch_size=16)
    print("Vocab size:", len(vocab))
    imgs, caps, lengths = next(iter(train_loader))
    print("Images:", imgs.shape, "Captions:", caps.shape, "Lengths:", lengths[:5])
