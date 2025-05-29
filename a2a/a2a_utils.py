from datetime import datetime, timezone
from uuid import uuid4

def to_envelope(method: str, params: dict):
    """Wrap any payload in a valid JSON-RPC 2.0 envelope."""
    return {
        "jsonrpc": "2.0",
        "id": str(uuid4()),
        "method": method,
        "params": params,
        "timestamp": datetime.now(timezone.utc).isoformat()
    }
