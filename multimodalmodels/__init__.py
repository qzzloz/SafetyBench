# from .instructblip import *
# from .llava import *
# from .qwen import *

# MiniGPT-4 pulls in heavy optional deps (omegaconf, etc.). Import it guarded so
# that using any OTHER model (e.g. the Qwen2.5-VL white-box wrapper for JailBound)
# does not require MiniGPT-4's dependency stack. If you actually use MiniGPT-4 and
# it's unavailable, you'll get this warning and a clear error only at use time.
try:
    from .minigpt4.minigpt4_model import MiniGPT4  # noqa: F401
except Exception as _e:  # pragma: no cover
    import logging as _logging
    _logging.getLogger(__name__).warning(
        "multimodalmodels: MiniGPT4 not imported (%s). "
        "Install its deps only if you need MiniGPT-4.", _e
    )