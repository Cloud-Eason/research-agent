"""LangGraph checkpointer backed by Redis, written against langgraph-checkpoint 4.x.

Storage model (mirrors InMemorySaver's semantics, see keys.py for the layout):

* Channel values are stored as versioned blobs, and each put() writes only the
  channels listed in `new_versions`. The message history grows every step, so
  storing full snapshots would make storage quadratic in the number of steps.
* Each checkpoint hash stores the checkpoint minus its channel values, plus the
  channel versions as JSON so compaction can find live blobs without
  deserializing anything.
* Pending writes follow InMemorySaver: regular writes are first-writer-wins
  (HSETNX); special writes (__error__, __interrupt__, ...) overwrite (HSET).
* Only the newest `keep_last` checkpoints are kept; blobs no longer referenced
  by a kept checkpoint are garbage-collected. Trade-off: no time travel past N.
* Every write refreshes the TTL of all of the thread's keys (sliding expiry).

Async methods delegate to the sync ones with asyncio.to_thread. One code path
means sync and async cannot drift apart; the thread hop costs microseconds next
to multi-second LLM steps. A native redis.asyncio port would only pay off at
far higher request rates.

Concurrency assumption: one graph run per thread at a time (LangGraph's own
model). Different threads never share keys.
"""

from __future__ import annotations

import asyncio
import builtins
import json
from collections.abc import AsyncIterator, Iterator, Sequence
from typing import Any

import redis
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.base import (
    WRITES_IDX_MAP,
    BaseCheckpointSaver,
    ChannelVersions,
    Checkpoint,
    CheckpointMetadata,
    CheckpointTuple,
    get_checkpoint_id,
    get_checkpoint_metadata,
)
from langgraph.checkpoint.serde.base import SerializerProtocol

from research_agent.checkpoint import keys
from research_agent.checkpoint.serde import build_serializer


def _pack(typed: tuple[str, bytes]) -> bytes:
    kind, data = typed
    return kind.encode() + b":" + data


def _unpack(raw: bytes) -> tuple[str, bytes]:
    kind, _, data = raw.partition(b":")
    return kind.decode(), data


def _s(value: bytes | str) -> str:
    return value.decode() if isinstance(value, bytes) else value


class RedisCheckpointSaver(BaseCheckpointSaver[int]):
    def __init__(
        self,
        client: redis.Redis,
        *,
        ttl_s: int | None = 86_400,
        keep_last: int | None = 20,
        serde: SerializerProtocol | None = None,
    ) -> None:
        super().__init__(serde=serde or build_serializer())
        # redis-py's return types are unions with Awaitable; Any keeps call sites clean.
        self.client: Any = client
        self.ttl_s = ttl_s
        self.keep_last = keep_last

    # ------------------------------------------------------------------ put
    def put(
        self,
        config: RunnableConfig,
        checkpoint: Checkpoint,
        metadata: CheckpointMetadata,
        new_versions: ChannelVersions,
    ) -> RunnableConfig:
        conf = config["configurable"]
        thread_id = str(conf["thread_id"])
        ns = str(conf.get("checkpoint_ns", ""))
        parent_id = conf.get("checkpoint_id") or ""
        checkpoint_id = checkpoint["id"]

        stored = checkpoint.copy()
        values: dict[str, Any] = stored.pop("channel_values")  # type: ignore[misc]
        blobs = {
            keys.blob_field(channel, version): _pack(
                self.serde.dumps_typed(values[channel]) if channel in values else ("empty", b"")
            )
            for channel, version in new_versions.items()
        }

        seq = int(self.client.incr(keys.seq_key(thread_id, ns)))
        pipe = self.client.pipeline(transaction=True)
        pipe.hset(
            keys.checkpoint_key(thread_id, ns, checkpoint_id),
            mapping={
                "checkpoint": _pack(self.serde.dumps_typed(stored)),
                "metadata": _pack(
                    self.serde.dumps_typed(get_checkpoint_metadata(config, metadata))
                ),
                "parent_id": parent_id,
                "versions": json.dumps(checkpoint["channel_versions"]),
                "seq": seq,
            },
        )
        if blobs:
            pipe.hset(keys.blobs_key(thread_id, ns), mapping=blobs)
        pipe.zadd(keys.index_key(thread_id, ns), {checkpoint_id: seq})
        pipe.sadd(keys.namespaces_key(thread_id), ns)
        pipe.execute()

        self._compact(thread_id, ns)
        self._touch(thread_id, ns)
        return {
            "configurable": {
                "thread_id": thread_id,
                "checkpoint_ns": ns,
                "checkpoint_id": checkpoint_id,
            }
        }

    # ----------------------------------------------------------- put_writes
    def put_writes(
        self,
        config: RunnableConfig,
        writes: Sequence[tuple[str, Any]],
        task_id: str,
        task_path: str = "",
    ) -> None:
        conf = config["configurable"]
        thread_id = str(conf["thread_id"])
        ns = str(conf.get("checkpoint_ns", ""))
        key = keys.writes_key(thread_id, ns, str(conf["checkpoint_id"]))

        pipe = self.client.pipeline(transaction=True)
        for idx, (channel, value) in enumerate(writes):
            write_idx = WRITES_IDX_MAP.get(channel, idx)
            field = json.dumps([task_id, write_idx])
            header = json.dumps(
                {"task_id": task_id, "channel": channel, "task_path": task_path, "idx": write_idx}
            )
            payload = header.encode() + b"\n" + _pack(self.serde.dumps_typed(value))
            if write_idx >= 0:
                pipe.hsetnx(key, field, payload)  # first writer wins, as in InMemorySaver
            else:
                pipe.hset(key, field, payload)  # special writes overwrite
        if self.ttl_s:
            pipe.expire(key, self.ttl_s)
        pipe.execute()

    # ------------------------------------------------------------ get_tuple
    def get_tuple(self, config: RunnableConfig) -> CheckpointTuple | None:
        conf = config["configurable"]
        thread_id = str(conf["thread_id"])
        ns = str(conf.get("checkpoint_ns", ""))
        checkpoint_id = get_checkpoint_id(config)
        if not checkpoint_id:
            latest = self.client.zrevrange(keys.index_key(thread_id, ns), 0, 0)
            if not latest:
                return None
            checkpoint_id = _s(latest[0])
        return self._load(thread_id, ns, checkpoint_id)

    # ----------------------------------------------------------------- list
    def list(
        self,
        config: RunnableConfig | None,
        *,
        filter: dict[str, Any] | None = None,
        before: RunnableConfig | None = None,
        limit: int | None = None,
    ) -> Iterator[CheckpointTuple]:
        only_id: str | None = None
        if config is None:
            targets = [(t, ns) for t in self._threads() for ns in self._namespaces(t)]
        else:
            conf = config["configurable"]
            thread_id = str(conf["thread_id"])
            if "checkpoint_ns" in conf:
                targets = [(thread_id, str(conf["checkpoint_ns"]))]
            else:
                targets = [(thread_id, ns) for ns in self._namespaces(thread_id)]
            only_id = get_checkpoint_id(config)
        before_id = get_checkpoint_id(before) if before else None

        yielded = 0
        for thread_id, ns in targets:
            for raw in self.client.zrevrange(keys.index_key(thread_id, ns), 0, -1):
                checkpoint_id = _s(raw)
                if only_id and checkpoint_id != only_id:
                    continue
                # Checkpoint ids are time-ordered UUIDs, so string order is time order.
                if before_id and not checkpoint_id < before_id:
                    continue
                item = self._load(thread_id, ns, checkpoint_id)
                if item is None:
                    continue
                if filter and not all(item.metadata.get(k) == v for k, v in filter.items()):
                    continue
                yield item
                yielded += 1
                if limit is not None and yielded >= limit:
                    return

    # -------------------------------------------------------- delete_thread
    def delete_thread(self, thread_id: str) -> None:
        thread_id = str(thread_id)
        doomed: builtins.list[str] = []
        for ns in self._namespaces(thread_id):
            for raw in self.client.zrange(keys.index_key(thread_id, ns), 0, -1):
                cid = _s(raw)
                doomed += [
                    keys.checkpoint_key(thread_id, ns, cid),
                    keys.writes_key(thread_id, ns, cid),
                ]
            doomed += [
                keys.index_key(thread_id, ns),
                keys.seq_key(thread_id, ns),
                keys.blobs_key(thread_id, ns),
            ]
        for raw in self.client.smembers(keys.tool_cache_index_key(thread_id)):
            doomed.append(keys.tool_cache_key(thread_id, _s(raw)))
        doomed += [keys.tool_cache_index_key(thread_id), keys.namespaces_key(thread_id)]
        self.client.delete(*doomed)

    # ---------------------------------------------------------------- async
    async def aput(
        self,
        config: RunnableConfig,
        checkpoint: Checkpoint,
        metadata: CheckpointMetadata,
        new_versions: ChannelVersions,
    ) -> RunnableConfig:
        return await asyncio.to_thread(self.put, config, checkpoint, metadata, new_versions)

    async def aput_writes(
        self,
        config: RunnableConfig,
        writes: Sequence[tuple[str, Any]],
        task_id: str,
        task_path: str = "",
    ) -> None:
        await asyncio.to_thread(self.put_writes, config, writes, task_id, task_path)

    async def aget_tuple(self, config: RunnableConfig) -> CheckpointTuple | None:
        return await asyncio.to_thread(self.get_tuple, config)

    async def alist(
        self,
        config: RunnableConfig | None,
        *,
        filter: dict[str, Any] | None = None,
        before: RunnableConfig | None = None,
        limit: int | None = None,
    ) -> AsyncIterator[CheckpointTuple]:
        items = await asyncio.to_thread(
            lambda: builtins.list(self.list(config, filter=filter, before=before, limit=limit))
        )
        for item in items:
            yield item

    async def adelete_thread(self, thread_id: str) -> None:
        await asyncio.to_thread(self.delete_thread, thread_id)

    # -------------------------------------------------------------- helpers
    def _load(self, thread_id: str, ns: str, checkpoint_id: str) -> CheckpointTuple | None:
        data = self.client.hgetall(keys.checkpoint_key(thread_id, ns, checkpoint_id))
        if not data:
            return None
        checkpoint: Checkpoint = self.serde.loads_typed(_unpack(data[b"checkpoint"]))
        versions = checkpoint["channel_versions"]
        values: dict[str, Any] = {}
        if versions:
            channels = builtins.list(versions.items())
            raws = self.client.hmget(
                keys.blobs_key(thread_id, ns), [keys.blob_field(c, v) for c, v in channels]
            )
            for (channel, _), raw in zip(channels, raws, strict=True):
                if raw is None:
                    continue
                typed = _unpack(raw)
                if typed[0] != "empty":
                    values[channel] = self.serde.loads_typed(typed)

        parent_id = _s(data.get(b"parent_id", b""))
        return CheckpointTuple(
            config={
                "configurable": {
                    "thread_id": thread_id,
                    "checkpoint_ns": ns,
                    "checkpoint_id": checkpoint_id,
                }
            },
            checkpoint={**checkpoint, "channel_values": values},
            metadata=self.serde.loads_typed(_unpack(data[b"metadata"])),
            parent_config=(
                {
                    "configurable": {
                        "thread_id": thread_id,
                        "checkpoint_ns": ns,
                        "checkpoint_id": parent_id,
                    }
                }
                if parent_id
                else None
            ),
            pending_writes=self._load_writes(thread_id, ns, checkpoint_id),
        )

    def _load_writes(
        self, thread_id: str, ns: str, checkpoint_id: str
    ) -> builtins.list[tuple[str, str, Any]]:
        raw = self.client.hgetall(keys.writes_key(thread_id, ns, checkpoint_id))
        entries = []
        for payload in raw.values():
            header, _, body = payload.partition(b"\n")
            meta = json.loads(header)
            entries.append((meta["task_path"], meta["task_id"], meta["idx"], meta["channel"], body))
        entries.sort(key=lambda e: (e[0], e[1], e[2]))
        return [
            (task_id, channel, self.serde.loads_typed(_unpack(body)))
            for _, task_id, _, channel, body in entries
        ]

    def _compact(self, thread_id: str, ns: str) -> None:
        if not self.keep_last:
            return
        index = keys.index_key(thread_id, ns)
        total = int(self.client.zcard(index))
        if total <= self.keep_last:
            return
        stale = [_s(x) for x in self.client.zrange(index, 0, total - self.keep_last - 1)]
        pipe = self.client.pipeline(transaction=True)
        for cid in stale:
            pipe.delete(
                keys.checkpoint_key(thread_id, ns, cid), keys.writes_key(thread_id, ns, cid)
            )
        pipe.zrem(index, *stale)
        pipe.execute()
        self._collect_blobs(thread_id, ns)

    def _collect_blobs(self, thread_id: str, ns: str) -> None:
        """Delete blobs that no remaining checkpoint references."""
        kept = [_s(x) for x in self.client.zrange(keys.index_key(thread_id, ns), 0, -1)]
        pipe = self.client.pipeline(transaction=False)
        for cid in kept:
            pipe.hget(keys.checkpoint_key(thread_id, ns, cid), "versions")
        live = {
            keys.blob_field(channel, version)
            for raw in pipe.execute()
            if raw
            for channel, version in json.loads(raw).items()
        }
        blobs = keys.blobs_key(thread_id, ns)
        dead = [f for f in (_s(x) for x in self.client.hkeys(blobs)) if f not in live]
        if dead:
            self.client.hdel(blobs, *dead)

    def _touch(self, thread_id: str, ns: str) -> None:
        if not self.ttl_s:
            return
        kept = [_s(x) for x in self.client.zrange(keys.index_key(thread_id, ns), 0, -1)]
        pipe = self.client.pipeline(transaction=False)
        for key in (
            keys.index_key(thread_id, ns),
            keys.seq_key(thread_id, ns),
            keys.blobs_key(thread_id, ns),
            keys.namespaces_key(thread_id),
        ):
            pipe.expire(key, self.ttl_s)
        for cid in kept:
            pipe.expire(keys.checkpoint_key(thread_id, ns, cid), self.ttl_s)
            pipe.expire(keys.writes_key(thread_id, ns, cid), self.ttl_s)
        pipe.execute()

    def _namespaces(self, thread_id: str) -> builtins.list[str]:
        return sorted(_s(x) for x in self.client.smembers(keys.namespaces_key(thread_id)))

    def _threads(self) -> builtins.list[str]:
        return sorted(
            keys.thread_from_namespaces_key(_s(k)) for k in self.client.scan_iter(match="ckpt_ns:*")
        )
