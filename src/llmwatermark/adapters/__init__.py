"""Backend adapters.

Each adapter lives in its own module and imports its backend at module import time, not at
package import time, so ``import llmwatermark`` never pulls torch or an inference stack.
Importing an adapter is how you opt in; if the backend is missing, the adapter raises an
ImportError naming the extra that installs it.
"""

from __future__ import annotations

__all__: list[str] = []
