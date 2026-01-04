import importlib.util
import site
import sys
from pathlib import Path


def _load_real_module():
 
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
    sys.modules[__name__] = _real
    globals().update(_real.__dict__)
else:
    class HfFolder:
        @staticmethod
        def get_token():
          
            return None

    def whoami(*args, **kwargs):
   
        return {}
