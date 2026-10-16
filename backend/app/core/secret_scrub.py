"""Bounded, iterative erasure of secret-bearing exception graphs."""
from __future__ import annotations

import types
from enum import Enum
from typing import Any

import httpx

MAX_NODES = 4096
MAX_DEPTH = 64
MAX_CONTAINER_ITEMS = 4096
MAX_ALLOCATIONS = 8192
MAX_DESCRIPTORS = 256

_EXCEPTION_DESCRIPTORS = {
    name: BaseException.__dict__[name]
    for name in ("args", "__cause__", "__context__", "__traceback__", "__dict__")
}


def _exception_get(error: BaseException, name: str) -> object:
    try:
        return _EXCEPTION_DESCRIPTORS[name].__get__(error, type(error))
    except BaseException:
        return None


def _exception_set(error: BaseException, name: str, value: object) -> None:
    try:
        _EXCEPTION_DESCRIPTORS[name].__set__(error, value)
    except BaseException:
        pass


def clear_traceback(traceback: object) -> None:
    remaining = MAX_DEPTH
    while type(traceback) is types.TracebackType and remaining:
        try:
            following = traceback.tb_next
        except BaseException:
            return
        try:
            traceback.tb_frame.clear()
        except BaseException:
            pass
        traceback = following
        remaining -= 1


def _detach_request(request: httpx.Request) -> tuple[object, ...]:
    """Detach credential-bearing request edges without consuming graph budget."""
    detached: list[object] = []
    for name in ("headers", "extensions", "hooks"):
        try:
            component = object.__getattribute__(request, name)
            detached.append(component)
            if name == "headers":
                component.clear()
        except BaseException:
            pass
        try:
            object.__setattr__(request, name, {} if name != "headers" else httpx.Headers())
        except BaseException:
            pass
    try:
        detached.append(object.__getattribute__(request, "stream"))
    except BaseException:
        pass
    for name, replacement in (("_content", b""), ("stream", httpx.ByteStream(b""))):
        try:
            object.__setattr__(request, name, replacement)
        except BaseException:
            pass
    return tuple(detached)


def _detach_response(response: httpx.Response) -> tuple[object, ...]:
    """Sever direct response edges before any budgeted child scheduling."""
    detached: list[object] = []
    for name in ("_request", "history", "next_request", "extensions"):
        try:
            detached.append(object.__getattribute__(response, name))
        except BaseException:
            detached.append(None)
    request, _history, next_request, _extensions = detached
    for name, replacement in (
        ("_request", None), ("next_request", None), ("history", []),
        ("extensions", {}), ("_content", b""), ("stream", httpx.ByteStream(b"")),
    ):
        try:
            object.__setattr__(response, name, replacement)
        except BaseException:
            pass
    try:
        object.__getattribute__(response, "headers").clear()
    except BaseException:
        pass
    if isinstance(request, httpx.Request):
        detached.append(_detach_request(request))
    if isinstance(next_request, httpx.Request):
        detached.append(_detach_request(next_request))
    return tuple(detached)


def scrub_graph(root: object, seen: set[int] | None = None) -> None:
    """Detach first, then traverse within fixed depth/node/allocation budgets."""
    visited = seen if seen is not None else set()
    pending: list[tuple[Any, int]] = [(root, 0)]
    allocations = 1
    nodes = 0

    def schedule(item: object, depth: int) -> None:
        nonlocal allocations
        if depth <= MAX_DEPTH and allocations < MAX_ALLOCATIONS:
            try:
                pending.append((item, depth))
                allocations += 1
            except BaseException:
                pass

    def drain_dict(dictionary: dict[Any, Any], depth: int) -> None:
        count = 0
        while dictionary and count < MAX_CONTAINER_ITEMS:
            try:
                key, item = dict.popitem(dictionary)
            except BaseException:
                break
            schedule(key, depth)
            schedule(item, depth)
            count += 1
        try:
            dict.clear(dictionary)
        except BaseException:
            pass

    try:
        while pending and nodes < MAX_NODES:
            value, depth = pending.pop()
            identity = id(value)
            if value is None or identity in visited:
                continue
            visited.add(identity)
            nodes += 1
            value_type = type(value)
            if value_type in {
                str, bytes, int, float, bool, type, types.ModuleType,
                types.FunctionType, types.CodeType,
            } or isinstance(value, Enum):
                continue
            if value_type is bytearray:
                try:
                    value.clear()  # type: ignore[union-attr]
                except BaseException:
                    pass
                continue
            if value_type is dict:
                drain_dict(value, depth + 1)  # type: ignore[arg-type]
                continue
            if value_type is list:
                count = 0
                while value and count < MAX_CONTAINER_ITEMS:  # type: ignore[truthy-bool]
                    try:
                        schedule(value.pop(), depth + 1)  # type: ignore[union-attr]
                    except BaseException:
                        break
                    count += 1
                try:
                    value.clear()  # type: ignore[union-attr]
                except BaseException:
                    pass
                continue
            if value_type is set:
                count = 0
                while value and count < MAX_CONTAINER_ITEMS:  # type: ignore[truthy-bool]
                    try:
                        schedule(value.pop(), depth + 1)  # type: ignore[union-attr]
                    except BaseException:
                        break
                    count += 1
                try:
                    value.clear()  # type: ignore[union-attr]
                except BaseException:
                    pass
                continue
            if value_type is tuple:
                for index in range(min(len(value), MAX_CONTAINER_ITEMS)):  # type: ignore[arg-type]
                    schedule(value[index], depth + 1)  # type: ignore[index]
                continue
            if value_type is frozenset:
                try:
                    iterator = iter(value)  # type: ignore[arg-type]
                    for _ in range(min(len(value), MAX_CONTAINER_ITEMS)):  # type: ignore[arg-type]
                        schedule(next(iterator), depth + 1)
                except BaseException:
                    pass
                continue

            try:
                if isinstance(value, httpx.Response):
                    for detached in _detach_response(value):
                        schedule(detached, depth + 1)
                    continue
                if isinstance(value, httpx.Request):
                    for detached in _detach_request(value):
                        schedule(detached, depth + 1)
                    continue
            except BaseException:
                pass

            try:
                mro = type.__getattribute__(value_type, "__mro__")
            except BaseException:
                mro = ()
            if BaseException in mro:
                error = value  # type: ignore[assignment]
                for name in ("__cause__", "__context__"):
                    schedule(_exception_get(error, name), depth + 1)
                arguments = _exception_get(error, "args")
                if type(arguments) is tuple:
                    for index in range(min(len(arguments), MAX_CONTAINER_ITEMS)):
                        schedule(arguments[index], depth + 1)
                dictionary = _exception_get(error, "__dict__")
                if type(dictionary) is dict:
                    drain_dict(dictionary, depth + 1)
                traceback = _exception_get(error, "__traceback__")
                _exception_set(error, "args", ())
                _exception_set(error, "__traceback__", None)
                _exception_set(error, "__cause__", None)
                _exception_set(error, "__context__", None)
                clear_traceback(traceback)

            descriptor_count = 0
            for cls in mro[:MAX_DEPTH]:
                try:
                    mapping = type.__getattribute__(cls, "__dict__")
                    descriptors = iter(mapping.values())
                except BaseException:
                    continue
                while descriptor_count < MAX_DESCRIPTORS:
                    try:
                        descriptor = next(descriptors)
                    except StopIteration:
                        break
                    except BaseException:
                        break
                    descriptor_count += 1
                    if isinstance(descriptor, types.GetSetDescriptorType):
                        try:
                            dictionary = descriptor.__get__(value, value_type)
                        except BaseException:
                            dictionary = None
                        if type(dictionary) is dict:
                            drain_dict(dictionary, depth + 1)
                    if not isinstance(descriptor, types.MemberDescriptorType):
                        continue
                    try:
                        item = descriptor.__get__(value, value_type)
                        descriptor.__set__(value, None)
                    except BaseException:
                        continue
                    schedule(item, depth + 1)
                if descriptor_count >= MAX_DESCRIPTORS:
                    break


    except BaseException:
        pass
    finally:
        try:
            pending.clear()
        except BaseException:
            pass
