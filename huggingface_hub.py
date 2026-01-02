# huggingface_hub.py
"""
Shim module to satisfy Gradio's legacy imports.

Gradio expects:
    from huggingface_hub import HfFolder, whoami

We don't actually use Hugging Face Hub features in this project,
so we provide minimal no-op implementations that keep Gradio happy.
"""

class HfFolder:
    @staticmethod
    def get_token():
        # Gradio uses this to fetch an HF token when doing OAuth / Spaces stuff.
        # We don't need that, so just return None.
        return None

def whoami(*args, **kwargs):
    # Normally returns info about the logged-in HF user.
    # We return an empty dict; our app doesn't rely on this.
    return {}
