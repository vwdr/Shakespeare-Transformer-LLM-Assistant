# build_tokenizer.py
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers import decoders

VOCAB_SIZE = 2000  # small for demo; you can bump to 5000 later

def build_tokenizer():
    # Initialize empty BPE model
    tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
    tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=True)
    tokenizer.decoder = decoders.ByteLevel()

    trainer = BpeTrainer(
        vocab_size=VOCAB_SIZE,
        min_frequency=2,
        special_tokens=["[PAD]", "[BOS]", "[EOS]", "[UNK]"]
    )

    # Train on raw text file
    files = ["data_raw.txt"]
    tokenizer.train(files, trainer)

    # Save tokenizer JSON so we can reload later
    tokenizer.save("tokenizer.json")
    print("Tokenizer trained and saved to tokenizer.json")

if __name__ == "__main__":
    build_tokenizer()
