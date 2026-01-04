import urllib.request

def load_text():
    #  tiny Shakespeare dataset
    url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
    print(f"Downloading tiny Shakespeare from:\n  {url}")
    with urllib.request.urlopen(url) as response:
        text = response.read().decode("utf-8")
    return text

if __name__ == "__main__":
    text = load_text()
    with open("data_raw.txt", "w", encoding="utf-8") as f:
        f.write(text)
    print("Saved raw text to data_raw.txt, length:", len(text))
