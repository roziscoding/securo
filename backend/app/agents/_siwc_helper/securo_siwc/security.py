"""Exception-graph scrubbing for helper secret boundaries."""
from __future__ import annotations

from enum import Enum
from functools import wraps
import types
from typing import Any, Callable

import httpx


_SAFE_RUNTIME_MESSAGES = frozenset({
    "Securo is unreachable",
    "Securo redirect rejected",
    "Securo response too large",
    "Securo returned invalid response size",
    "Securo compressed response rejected",
    "Securo request too large",
    "Securo returned malformed JSON",
    "Securo completion could not be confirmed",
    "OpenAI is unreachable",
    "OpenAI redirect rejected",
    "OpenAI response too large",
    "OpenAI response size is invalid",
    "OpenAI compressed response rejected",
    "OpenAI signing keys unavailable",
    "OpenAI token exchange failed",
    "OpenAI token response is malformed",
    "untrusted OpenAI endpoint",
    "unsupported OpenAI request option",
})


def safe_exception_message(error: BaseException, default: str) -> str:
    """Extract only a fixed allowlisted message without calling exception code."""
    try:
        raw_args = BaseException.__dict__["args"].__get__(error, type(error))
    except BaseException:
        return default
    text = (
        raw_args[0]
        if type(raw_args) is tuple
        and len(raw_args) == 1
        and type(raw_args[0]) is str
        else None
    )
    if text in _SAFE_RUNTIME_MESSAGES:
        return text
    if (
        isinstance(text, str)
        and text.startswith("Securo pairing request failed (")
        and text.endswith(")")
        and text.removeprefix("Securo pairing request failed (")[:-1].isdigit()
    ):
        return text
    return default


_MAX_SCRUB_NODES = 4096
_MAX_SCRUB_DEPTH = 64
_MAX_SCRUB_CONTAINER_ITEMS = 4096
_MAX_SCRUB_ALLOCATIONS = 8192
_MAX_SCRUB_DESCRIPTORS = 256


def _clear_traceback(traceback: types.TracebackType | None) -> None:
    remaining = _MAX_SCRUB_DEPTH
    while traceback is not None and remaining:
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


def _exception_get(error: BaseException, name: str) -> object:
    try:
        return BaseException.__dict__[name].__get__(error, type(error))
    except BaseException:
        return None


def _exception_set(error: BaseException, name: str, value: object) -> None:
    try:
        BaseException.__dict__[name].__set__(error, value)
    except BaseException:
        pass


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
    request, _history, next_request, response_extensions = detached
    # httpx exposes its live pooled socket through the private
    # ``network_stream`` response extension. It is transport state, not
    # credential material; traversing it erases the socket and corrupts the
    # owned client before the next request on the keep-alive connection.
    # Detach it without scheduling it for graph erasure. Other extension
    # values remain scrubbed because test-only/unknown metadata may retain
    # request data.
    if type(response_extensions) is dict:
        try:
            dict.pop(response_extensions, "network_stream", None)
        except BaseException:
            pass
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


def _scrub_graph(root: object, seen: set[int] | None = None) -> None:
    """Bounded, iterative, hook-free best-effort graph erasure."""
    visited = seen if seen is not None else set()
    pending: list[tuple[Any, int]] = [(root, 0)]
    allocations = 1
    nodes = 0

    def schedule(item: object, depth: int) -> None:
        nonlocal allocations
        if depth <= _MAX_SCRUB_DEPTH and allocations < _MAX_SCRUB_ALLOCATIONS:
            try:
                pending.append((item, depth))
                allocations += 1
            except BaseException:
                pass

    def drain_dictionary(dictionary: dict[Any, Any], depth: int) -> None:
        count = 0
        while dictionary and count < _MAX_SCRUB_CONTAINER_ITEMS:
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
        while pending and nodes < _MAX_SCRUB_NODES:
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
                    bytearray.clear(value)
                except BaseException:
                    pass
                continue
            if value_type is dict:
                drain_dictionary(value, depth + 1)
                continue
            if value_type is list:
                count = 0
                while value and count < _MAX_SCRUB_CONTAINER_ITEMS:
                    try:
                        schedule(list.pop(value), depth + 1)
                    except BaseException:
                        break
                    count += 1
                try:
                    list.clear(value)
                except BaseException:
                    pass
                continue
            if value_type is set:
                count = 0
                while value and count < _MAX_SCRUB_CONTAINER_ITEMS:
                    try:
                        schedule(set.pop(value), depth + 1)
                    except BaseException:
                        break
                    count += 1
                try:
                    set.clear(value)
                except BaseException:
                    pass
                continue
            if value_type is tuple:
                for index in range(min(tuple.__len__(value), _MAX_SCRUB_CONTAINER_ITEMS)):
                    try:
                        schedule(tuple.__getitem__(value, index), depth + 1)
                    except BaseException:
                        break
                continue
            if value_type is frozenset:
                try:
                    iterator = iter(value)
                    for _ in range(min(len(value), _MAX_SCRUB_CONTAINER_ITEMS)):
                        schedule(next(iterator), depth + 1)
                except (StopIteration, BaseException):
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
            is_exception = BaseException in mro
            if is_exception:
                error: Any = value
                for name in ("__cause__", "__context__"):
                    schedule(_exception_get(error, name), depth + 1)
                arguments = _exception_get(error, "args")
                if type(arguments) is tuple:
                    for index in range(min(tuple.__len__(arguments), _MAX_SCRUB_CONTAINER_ITEMS)):
                        schedule(tuple.__getitem__(arguments, index), depth + 1)
                dictionary = _exception_get(error, "__dict__")
                if type(dictionary) is dict:
                    drain_dictionary(dictionary, depth + 1)
                traceback = _exception_get(error, "__traceback__")
                _exception_set(error, "args", ())
                _exception_set(error, "__traceback__", None)
                _exception_set(error, "__cause__", None)
                _exception_set(error, "__context__", None)
                _clear_traceback(traceback if isinstance(traceback, types.TracebackType) else None)

            descriptors = 0
            for cls in mro[:_MAX_SCRUB_DEPTH]:
                try:
                    mapping = type.__getattribute__(cls, "__dict__")
                    iterator = iter(mapping.values())
                except BaseException:
                    continue
                while descriptors < _MAX_SCRUB_DESCRIPTORS:
                    try:
                        descriptor = next(iterator)
                    except StopIteration:
                        break
                    except BaseException:
                        break
                    descriptors += 1
                    if isinstance(descriptor, types.GetSetDescriptorType):
                        try:
                            dictionary = descriptor.__get__(value, value_type)
                        except BaseException:
                            dictionary = None
                        if type(dictionary) is dict:
                            drain_dictionary(dictionary, depth + 1)
                    if not isinstance(descriptor, types.MemberDescriptorType):
                        continue
                    try:
                        item = descriptor.__get__(value, value_type)
                        descriptor.__set__(value, None)
                    except BaseException:
                        continue
                    schedule(item, depth + 1)
                if descriptors >= _MAX_SCRUB_DESCRIPTORS:
                    break


    except BaseException:
        pass
    finally:
        try:
            pending.clear()
        except BaseException:
            pass


def _scrub_request(request: httpx.Request) -> None:
    try:
        _scrub_graph(request)
    except BaseException:
        pass


def _scrub_response(response: httpx.Response) -> None:
    try:
        _scrub_graph(response)
    except BaseException:
        pass


def _scrub_exception(error: BaseException, seen: set[int] | None = None) -> None:
    try:
        _scrub_graph(error, seen)
    except BaseException:
        # Cleanup must never replace the fresh fixed/fatal signal.
        pass


def secret_boundary(default_message: str) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Raise a fresh fixed error after detaching every internal failure graph."""
    def decorate(function: Callable[..., Any]) -> Callable[..., Any]:
        @wraps(function)
        def guarded(*args: Any, **kwargs: Any) -> Any:
            message: str | None = None
            signal: type[BaseException] | None = None
            internal_traceback = None
            try:
                return function(*args, **kwargs)
            except (KeyboardInterrupt, SystemExit) as exc:
                signal = type(exc)
                traceback = _exception_get(exc, "__traceback__")
                internal_traceback = (
                    traceback if isinstance(traceback, types.TracebackType) else None
                )
                _scrub_exception(exc)
            except BaseException as exc:
                message = safe_exception_message(exc, default_message)
                traceback = _exception_get(exc, "__traceback__")
                internal_traceback = (
                    traceback if isinstance(traceback, types.TracebackType) else None
                )
                _scrub_exception(exc)
            finally:
                kwargs.clear()
                args = ()
            _clear_traceback(internal_traceback)
            internal_traceback = None
            if signal is KeyboardInterrupt:
                raise KeyboardInterrupt() from None
            if signal is SystemExit:
                raise SystemExit(1) from None
            raise RuntimeError(message or default_message) from None

        return guarded
    return decorate
