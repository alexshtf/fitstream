from importlib.metadata import version as _version

from .batching import iter_batches as iter_batches
from .events import Event as Event
from .fit import (
    augment as augment,
    ema as ema,
    early_stop as early_stop,
    epoch_stream as epoch_stream,
    pipe as pipe,
    print_keys as print_keys,
    take as take,
    tap as tap,
    tick as tick,
)
from .sinks import collect as collect, collect_jsonl as collect_jsonl, collect_pd as collect_pd
from .augmentations import validation_loss as validation_loss
from .tracking import BestModel as BestModel, best_model as best_model

__version__ = _version("fitstream")
