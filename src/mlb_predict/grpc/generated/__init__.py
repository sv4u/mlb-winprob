"""
Generated gRPC and protobuf code for mlb_predict.v1.

After running scripts/gen_proto.sh, this package contains mlb_predict.v1
subpackage. We alias it so that generated code's internal imports
(from mlb_predict.v1 import ...) resolve correctly.
"""
from __future__ import annotations

import sys

# Alias mlb_predict.grpc.generated.mlb_predict.v1 as mlb_predict.v1 so that
# generated *_pb2.py and *_pb2_grpc.py imports (from mlb_predict.v1 import ...)
# resolve when this package is loaded.
try:
    import mlb_predict.grpc.generated.mlb_predict.v1 as _v1  # noqa: F401
    sys.modules["mlb_predict.v1"] = _v1
except ImportError:
    pass  # Proto code not generated yet; run scripts/gen_proto.sh
