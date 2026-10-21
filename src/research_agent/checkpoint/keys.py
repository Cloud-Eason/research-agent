"""Redis key layout.

The thread id is wrapped in {braces}: a Redis Cluster "hash tag". Only the part
inside the braces is hashed to pick a slot, so every key of one thread lands on
the same node and can be updated together in one MULTI transaction.

    ckpt:{t}:<ns>:<id>         HASH  checkpoint (without channel values), metadata,
                                     parent id, channel versions (JSON), seq
    ckpt_idx:{t}:<ns>          ZSET  checkpoint ids scored by a monotonic sequence
    ckpt_seq:{t}:<ns>          STR   INCR counter feeding the scores above
    ckpt_blobs:{t}:<ns>        HASH  ["channel", version] -> serialized channel value
    ckpt_writes:{t}:<ns>:<id>  HASH  ["task_id", idx] -> pending write
    ckpt_ns:{t}                SET   namespaces used by the thread
    toolcache:{t}:<call>:<h>   STR   cached tool result (JSON)
    toolcache_idx:{t}          SET   cache entries of the thread (for deletion)
"""

from __future__ import annotations

import json
from typing import Any


def _tag(thread_id: str) -> str:
    return "{" + thread_id + "}"


def checkpoint_key(thread_id: str, ns: str, checkpoint_id: str) -> str:
    return f"ckpt:{_tag(thread_id)}:{ns}:{checkpoint_id}"


def index_key(thread_id: str, ns: str) -> str:
    return f"ckpt_idx:{_tag(thread_id)}:{ns}"


def seq_key(thread_id: str, ns: str) -> str:
    return f"ckpt_seq:{_tag(thread_id)}:{ns}"


def blobs_key(thread_id: str, ns: str) -> str:
    return f"ckpt_blobs:{_tag(thread_id)}:{ns}"


def writes_key(thread_id: str, ns: str, checkpoint_id: str) -> str:
    return f"ckpt_writes:{_tag(thread_id)}:{ns}:{checkpoint_id}"


def namespaces_key(thread_id: str) -> str:
    return f"ckpt_ns:{_tag(thread_id)}"


def tool_cache_key(thread_id: str, entry: str) -> str:
    return f"toolcache:{_tag(thread_id)}:{entry}"


def tool_cache_index_key(thread_id: str) -> str:
    return f"toolcache_idx:{_tag(thread_id)}"


def blob_field(channel: str, version: Any) -> str:
    # Channel names can contain ":" (e.g. "branch:to:agent"), so encode as JSON.
    return json.dumps([channel, version])


def thread_from_namespaces_key(key: str) -> str:
    return key[len("ckpt_ns:{") : -1]
