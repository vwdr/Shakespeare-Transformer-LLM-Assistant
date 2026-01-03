# huggingface_hub.py
"""
Shim module to satisfy Gradio's legacy imports.

If the real `huggingface_hub` package is available (it ships with modern
Gradio), we re-export it. Otherwise we fall back to no-op stubs so the app
still runs without HF auth.
"""

import importlib.util
import site
import sys
from pathlib import Path


def _load_real_module():
    """
    Attempt to load the installed huggingface_hub package directly from
    site-packages, bypassing this shim.
    """
    search_paths = site.getsitepackages()
    user_site = site.getusersitepackages()
    if isinstance(user_site, str):
        search_paths.append(user_site)

    for base in search_paths:
        candidate = Path(base) / "huggingface_hub" / "__init__.py"
        if candidate.exists():
            spec = importlib.util.spec_from_file_location(__name__, candidate)
            if spec and spec.loader:
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                return module
    return None


_real = _load_real_module()

if _real is not None:
    # Re-export everything from the real module so downstream imports work.
    sys.modules[__name__] = _real
    globals().update(_real.__dict__)
else:
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
