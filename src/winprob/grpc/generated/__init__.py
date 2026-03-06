"""
Generated gRPC and protobuf code for winprob.v1.

After running scripts/gen_proto.sh, this package contains winprob.v1
subpackage. We alias it so that generated code's internal imports
(from winprob.v1 import ...) resolve correctly.
"""
from __future__ import annotations

import sys

# Alias winprob.grpc.generated.winprob.v1 as winprob.v1 so that
# generated *_pb2.py and *_pb2_grpc.py imports (from winprob.v1 import ...)
# resolve when this package is loaded.
try:
    import winprob.grpc.generated.winprob.v1 as _v1  # noqa: F401
    sys.modules["winprob.v1"] = _v1
except ImportError:
    pass  # Proto code not generated yet; run scripts/gen_proto.sh
