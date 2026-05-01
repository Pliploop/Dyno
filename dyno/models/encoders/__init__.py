from .base import AudioEncoderBase

# Each encoder has optional heavy dependencies; import lazily so the package
# still loads when only a subset of deps is installed.

def _try_import(name, module_path, class_name):
    try:
        import importlib
        mod = importlib.import_module(module_path, package=__package__)
        return getattr(mod, class_name)
    except Exception:
        return None


CLAPEncoder        = _try_import("CLAPEncoder",        ".clap",         "CLAPEncoder")
MuQEncoder         = _try_import("MuQEncoder",         ".muq",          "MuQEncoder")
MERTEncoder        = _try_import("MERTEncoder",        ".mert",         "MERTEncoder")
USADEncoder        = _try_import("USADEncoder",        ".usad",         "USADEncoder")
Music2LatentEncoder = _try_import("Music2LatentEncoder", ".music2latent", "Music2LatentEncoder")
MatPacEncoder      = _try_import("MatPacEncoder",      ".matpac",       "MatPacEncoder")

__all__ = [
    "AudioEncoderBase",
    "CLAPEncoder", "MuQEncoder", "MERTEncoder",
    "USADEncoder", "Music2LatentEncoder", "MatPacEncoder",
]
