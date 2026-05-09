"""Type stubs for the Rust extension module.

The actual implementation lives in ``rust/lib.rs`` and is compiled by
maturin into ``kaos_nlp_transformers/_rust.abi3.so``. These stubs keep
``ty`` resolving imports like ``from kaos_nlp_transformers._rust import
__version__`` even though the runtime artifact is a binary cdylib.
"""

__version__: str

# Submodules are populated as the migration progresses (P3.1–P3.4).
# At Phase 1 the stubs below are empty placeholders.
from kaos_nlp_transformers._rust import embedding as embedding
from kaos_nlp_transformers._rust import registry as registry
from kaos_nlp_transformers._rust import reranker as reranker
from kaos_nlp_transformers._rust import tokenize as tokenize
