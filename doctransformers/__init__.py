import sys
from typing import TYPE_CHECKING

from ._lazyImport import _LazyImport

# Base objects, independent of any specific backend
_import_structure = {
    "datasets" : [
        "create_doc_dataset",
        "DocDataset",
    ],
    "transformers" : [
        "DocTrainer",
    ],

}
if TYPE_CHECKING:
    from .datasets import (
        create_doc_dataset,
        DocDataset,
    )
    from .transformers import (
        DocTrainer,
    )
    
# Import
else:
    sys.modules[__name__] = _LazyImport(
        __name__,
        globals()["__file__"],
        _import_structure,
        module_spec=__spec__,
        extra_objects={"__version__": __version__},
    )


