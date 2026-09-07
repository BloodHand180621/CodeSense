"""共享 LLM 客户端服务。

所有需要调用大模型的功能都应经过这里。除了统一 provider 选择，这一层
还负责有限重试、并发上限、熔断冷却、缓存以及 provider 故障切换。
"""

from dataclasses import dataclass, field
from enum import Enum
import hashlib
import json
import logging
import os
import random
import threading
import time
import uuid
from typing import Any, Dict, List, Optional, Tuple

from services.api_keys import api_keys


logger = logging.getLogger(__name__)


class LLMProvider(Enum):
    """支持的 LLM provider。"""

    ZHIPU = "zhipu"
    OPENAI = "openai"


class LLMServiceError(RuntimeError):
    """不暴露 provider 原始异常内容的流式服务错误。"""

    def __init__(self, code: str):
        self.code = code
        super().__init__(code)


class _EmptyResponseError(RuntimeError):
    """Provider 返回成功状态但没有可用文本。"""


@dataclass
class _ProviderHealth:
    consecutive_failures: int = 0
    opened_until: float = 0.0
    last_error: str = ""


@dataclass
class _ProviderState:
    provider: LLMProvider
    client: Any
    model: str
    health: _ProviderHealth = field(default_factory=_ProviderHealth)


@dataclass
class _InflightCall:
    """同一进程内相同请求的 single-flight 状态。"""

    event: threading.Event = field(default_factory=threading.Event)
    result: Optional[str] = None


@dataclass
class _LLMTrace:
    """脱敏的单次逻辑 LLM 请求事件。

    The event deliberately contains only bounded operational metadata.  It is
    shaped like an OpenTelemetry log/event record without requiring an
    OpenTelemetry runtime dependency in the application.
    """

    request_id: str
    request_kind: str
    stream: bool
    started_at: float = field(default_factory=time.perf_counter, repr=False)
    queue_wait_ms: float = 0.0
    llm_latency_ms: float = 0.0
    attempts: int = 0
    providers_tried: List[str] = field(default_factory=list)
    models_tried: List[str] = field(default_factory=list)
    fallback: bool = False
    cache_hit: bool = False
    provider: Optional[str] = None
    model: Optional[str] = None
    error_class: Optional[str] = None
    stop_reason: Optional[str] = None
    _emitted: bool = field(default=False, init=False, repr=False)

    def record_attempt(
        self,
        state: _ProviderState,
        model: Optional[str],
        queue_wait_seconds: float,
        llm_latency_seconds: float,
    ) -> None:
        self.attempts += 1
        self.queue_wait_ms += max(0.0, queue_wait_seconds * 1000.0)
        self.llm_latency_ms += max(0.0, llm_latency_seconds * 1000.0)
        provider = state.provider.value
        effective_model = str(model or state.model)
        self.provider = provider
        self.model = effective_model
        previous_provider = self.providers_tried[-1] if self.providers_tried else None
        previous_model = self.models_tried[-1] if self.models_tried else None
        if provider not in self.providers_tried:
            self.providers_tried.append(provider)
        if effective_model not in self.models_tried:
            self.models_tried.append(effective_model)
        if self.attempts > 1 and (
            previous_provider != provider
            or previous_model != effective_model
        ):
            self.fallback = True

    def note_error(self, error: Optional[Exception]) -> None:
        if error is not None:
            self.error_class = _failure_code(error)

    def finish(
        self,
        stop_reason: str,
        *,
        provider: Optional[str] = None,
        model: Optional[str] = None,
        error: Optional[Exception] = None,
        error_class: Optional[str] = None,
    ) -> None:
        self.stop_reason = stop_reason
        if provider:
            self.provider = provider
        if model:
            self.model = model
        if error is not None:
            self.note_error(error)
        elif error_class:
            self.error_class = error_class
        elif stop_reason in {"completed", "cache_hit", "singleflight_wait"}:
            self.error_class = None

    def emit(self) -> None:
        if self._emitted:
            return
        payload: Dict[str, Any] = {
            "event_name": "codesense.llm.request",
            "schema_version": 1,
            "request_id": self.request_id,
            "request_kind": self.request_kind,
            "stream": self.stream,
            "cache_hit": self.cache_hit,
            "fallback": self.fallback,
            "attempts": self.attempts,
            "retry_count": max(0, self.attempts - 1),
            "queue_wait_ms": round(self.queue_wait_ms, 2),
            "llm_latency_ms": round(self.llm_latency_ms, 2),
            "duration_ms": round(max(0.0, time.perf_counter() - self.started_at) * 1000.0, 2),
            "stop_reason": self.stop_reason or "unknown",
        }
        if self.provider:
            payload["provider"] = self.provider
        if self.model:
            payload["model"] = self.model
        if self.providers_tried:
            payload["providers_tried"] = self.providers_tried[:4]
        if self.models_tried:
            payload["models_tried"] = self.models_tried[:4]
        if self.error_class:
            payload["error_class"] = self.error_class
        logger.info("llm_trace %s", json.dumps(payload, ensure_ascii=False, sort_keys=True))
        self._emitted = True


_RETRYABLE_STATUS_CODES = frozenset({408, 409, 425, 429, 500, 502, 503, 504})
_DEFAULT_PROVIDER_ORDER = (LLMProvider.ZHIPU, LLMProvider.OPENAI)
_REQUEST_KINDS = frozenset(
    {
        "interactive",
        "background",
        "batch",
        "async",
        "ability_analysis",
        "submission",
        "stage1",
        "stage2",
        "stage3",
        "forum",
        "companion",
        "stt",
        "code_advice",
    }
)


def _env_int(name: str, default: int, *, minimum: int, maximum: int) -> int:
    try:
        value = int(os.environ.get(name, default))
    except (TypeError, ValueError):
        value = default
    return max(minimum, min(maximum, value))


def _env_float(name: str, default: float, *, minimum: float, maximum: float) -> float:
    try:
        value = float(os.environ.get(name, default))
    except (TypeError, ValueError):
        value = default
    return max(minimum, min(maximum, value))


def _normalize_request_kind(value: Any) -> str:
    """Keep trace labels useful without allowing arbitrary high-cardinality data."""

    candidate = str(value or "").strip().lower()
    return candidate if candidate in _REQUEST_KINDS else "interactive"


def _trace_request_id(value: Optional[str]) -> str:
    """Return an opaque, bounded id; never put arbitrary caller text in logs."""

    candidate = str(value or "").strip()
    if not candidate:
        return uuid.uuid4().hex
    try:
        return str(uuid.UUID(candidate))
    except (ValueError, AttributeError, TypeError):
        return hashlib.sha256(candidate.encode("utf-8", "replace")).hexdigest()[:32]


def _status_code(error: Any) -> Optional[int]:
    for candidate in (
        getattr(error, "status_code", None),
        getattr(error, "status", None),
        getattr(getattr(error, "response", None), "status_code", None),
    ):
        try:
            if candidate is not None:
                return int(candidate)
        except (TypeError, ValueError):
            continue
    return None


def _error_label(error: Optional[Exception]) -> str:
    if error is None:
        return "UNKNOWN"
    status = _status_code(error)
    suffix = f" status={status}" if status is not None else ""
    return f"{type(error).__name__}{suffix}"


def _is_rate_limit(error: Optional[Exception]) -> bool:
    if error is None:
        return False
    if _status_code(error) == 429:
        return True
    text = f"{type(error).__name__} {error}".lower()
    return any(
        marker in text
        for marker in (
            "429",
            "1305",
            "rate limit",
            "rate_limit",
            "too many requests",
            "apireachlimiterror",
            "访问量过大",
            "频率",
        )
    )


def _is_retryable_error(error: Optional[Exception]) -> bool:
    if error is None:
        return False
    if isinstance(error, _EmptyResponseError):
        return True
    status = _status_code(error)
    if status in _RETRYABLE_STATUS_CODES:
        return True
    if status is not None and status < 500:
        return False
    if isinstance(error, (TimeoutError, ConnectionError, OSError)):
        return True
    text = f"{type(error).__name__} {error}".lower()
    return any(
        marker in text
        for marker in (
            "timeout",
            "timed out",
            "connection reset",
            "connection aborted",
            "connection refused",
            "broken pipe",
            "temporarily unavailable",
            "service unavailable",
            "winerror 10013",
            "cannot connect",
            "eof",
            "overloaded",
            "too many requests",
            "rate limit",
            "访问量过大",
        )
    )


def _failure_code(error: Optional[Exception]) -> str:
    """Map provider failures to stable, low-cardinality trace values."""

    if error is None:
        return "LLM_UNAVAILABLE"
    status = _status_code(error)
    if status in (401, 403):
        return "AUTH_FAILED"
    if status == 429 or _is_rate_limit(error):
        return "RATE_LIMITED"
    text = " ".join(
        part.lower()
        for part in (type(error).__name__, str(error), str(getattr(error, "__cause__", "")))
    )
    if any(marker in text for marker in ("timeout", "timed out", "readtimeout", "connecttimeout")):
        return "TIMEOUT"
    if isinstance(error, (ConnectionError, OSError)) or any(
        marker in text
        for marker in (
            "api connection error",
            "connection error",
            "connecterror",
            "cannot connect",
            "connection refused",
            "network is unreachable",
            "all connection attempts failed",
        )
    ):
        return "NETWORK_UNAVAILABLE"
    return "UPSTREAM_ERROR"


def _retry_after_seconds(error: Optional[Exception]) -> Optional[float]:
    response = getattr(error, "response", None) if error else None
    headers = getattr(response, "headers", None)
    if not headers:
        return None
    value = headers.get("Retry-After") or headers.get("retry-after")
    try:
        return max(0.0, min(60.0, float(value)))
    except (TypeError, ValueError):
        return None


def _message_content(response: Any) -> str:
    choices = response.get("choices") if isinstance(response, dict) else getattr(response, "choices", None)
    if not choices:
        return ""
    first = choices[0]
    message = first.get("message") if isinstance(first, dict) else getattr(first, "message", None)
    content = message.get("content") if isinstance(message, dict) else getattr(message, "content", None)
    if isinstance(content, str):
        return content.strip()
    return str(content).strip() if content is not None else ""


def _stream_chunk_content(chunk: Any) -> str:
    choices = chunk.get("choices") if isinstance(chunk, dict) else getattr(chunk, "choices", None)
    if not choices:
        return ""
    first = choices[0]
    delta = first.get("delta") if isinstance(first, dict) else getattr(first, "delta", None)
    content = delta.get("content") if isinstance(delta, dict) else getattr(delta, "content", None)
    return content if isinstance(content, str) else ""


class SharedLLMClient:
    """带容错和故障切换的共享 LLM 客户端单例。

    ``_client``、``_provider`` 和 ``_model_name`` 是历史兼容属性，项目旧
    调用点仍可以读取它们；新的请求统一从 ``_provider_states`` 选择健康
    provider，不会因某一次网络抖动永久失效。
    """

    last_user_request_time = 0.0
    _instance: Optional["SharedLLMClient"] = None
    _initialized = False

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if getattr(self, "_initialized", False):
            return
        self._client = None
        self._provider: Optional[LLMProvider] = None
        self._model_name = ""
        self._available = False
        self._redis_cache = None
        self._provider_states: Dict[LLMProvider, _ProviderState] = {}
        self._provider_order: List[LLMProvider] = []
        self._credentials_signature: Tuple[str, str] = ("", "")
        self._state_lock = threading.RLock()
        self._cache_lock = threading.RLock()
        self._local_cache: Dict[str, Tuple[float, str]] = {}
        self._inflight_lock = threading.RLock()
        self._inflight: Dict[str, _InflightCall] = {}
        self._request_semaphore = threading.BoundedSemaphore(
            _env_int("AI_MAX_CONCURRENT_REQUESTS", 3, minimum=1, maximum=32)
        )
        self._retry_attempts = _env_int("AI_RETRY_ATTEMPTS", 3, minimum=1, maximum=6)
        self._retry_base_delay = _env_float(
            "AI_RETRY_BASE_DELAY_SECONDS", 0.8, minimum=0.05, maximum=10.0
        )
        self._retry_max_delay = _env_float(
            "AI_RETRY_MAX_DELAY_SECONDS", 8.0, minimum=0.1, maximum=60.0
        )
        self._circuit_failure_threshold = _env_int(
            "AI_CIRCUIT_FAILURE_THRESHOLD", 2, minimum=1, maximum=10
        )
        self._circuit_cooldown = _env_float(
            "AI_CIRCUIT_COOLDOWN_SECONDS", 30.0, minimum=1.0, maximum=600.0
        )
        self._request_queue_timeout = _env_float(
            "AI_REQUEST_QUEUE_TIMEOUT_SECONDS", 2.0, minimum=0.1, maximum=30.0
        )
        self._background_priority_window = _env_float(
            "AI_BACKGROUND_PRIORITY_WINDOW_SECONDS", 3.0, minimum=0.0, maximum=30.0
        )
        self._background_max_wait = _env_float(
            "AI_BACKGROUND_MAX_WAIT_SECONDS", 4.0, minimum=0.0, maximum=60.0
        )
        self._cache_ttl = _env_int(
            "AI_CACHE_TTL_SECONDS", 86400, minimum=60, maximum=604800
        )
        self._singleflight_wait = _env_float(
            "AI_SINGLEFLIGHT_WAIT_SECONDS", 90.0, minimum=1.0, maximum=300.0
        )
        self._last_error = ""
        self._init_redis()
        self._init_client()
        self._initialized = True

    def _init_redis(self):
        redis_url = os.environ.get("REDIS_URL") or "redis://127.0.0.1:6379/0"
        try:
            import redis

            self._redis_cache = redis.from_url(redis_url, socket_timeout=1)
            self._redis_cache.ping()
            print("[OK] LLM 接口缓存启用")
        except Exception as exc:
            self._redis_cache = None
            logger.info(
                "LLM Redis cache unavailable; using local cache: %s",
                type(exc).__name__,
            )

    def _init_client(self):
        try:
            api_keys.refresh()
        except Exception as exc:
            logger.warning("AI credential refresh failed: %s", type(exc).__name__)
        with self._state_lock:
            self._provider_states = {}
            self._provider_order = []
            self._provider = None
            self._client = None
            self._model_name = ""
            self._available = False
            self._credentials_signature = (api_keys.zhipu_key, api_keys.openai_key)
        if not api_keys.has_any_key:
            print("[WARN] 没有可用的 API 密钥，LLM 客户端不可用")
            return
        for provider in self._configured_provider_order():
            if provider is LLMProvider.ZHIPU and api_keys.has_zhipu:
                self._init_zhipu_client()
            elif provider is LLMProvider.OPENAI and api_keys.has_openai:
                self._init_openai_client()
        with self._state_lock:
            self._available = bool(self._provider_states)
            if self._provider_order:
                self._set_active(self._provider_states[self._provider_order[0]])
        if not self._available:
            print("[WARN] 已配置密钥，但没有成功初始化可用的 LLM provider")

    def _configured_provider_order(self) -> List[LLMProvider]:
        raw_order = os.environ.get("AI_PROVIDER_ORDER", "zhipu,openai")
        result: List[LLMProvider] = []
        for value in raw_order.split(","):
            try:
                provider = LLMProvider(value.strip().lower())
            except ValueError:
                continue
            if provider not in result:
                result.append(provider)
        for provider in _DEFAULT_PROVIDER_ORDER:
            if provider not in result:
                result.append(provider)
        return result or list(_DEFAULT_PROVIDER_ORDER)

    def _register_provider(self, provider: LLMProvider, client: Any, model: str) -> None:
        with self._state_lock:
            self._provider_states[provider] = _ProviderState(provider, client, model)
            if provider not in self._provider_order:
                self._provider_order.append(provider)
        print(f"[OK] 共享 {provider.value} AI 客户端初始化成功")

    def _init_zhipu_client(self):
        try:
            from zhipuai import ZhipuAI

            if not api_keys.zhipu_key:
                return
            kwargs: Dict[str, Any] = {"api_key": api_keys.zhipu_key}
            if os.environ.get("ZHIPU_BASE_URL"):
                kwargs["base_url"] = os.environ["ZHIPU_BASE_URL"]
            self._register_provider(
                LLMProvider.ZHIPU,
                ZhipuAI(**kwargs),
                os.environ.get("ZHIPU_MODEL", "glm-4.5-flash"),
            )
        except ImportError:
            print("[WARN] 未安装 zhipuai 库")
        except Exception as exc:
            print(f"[WARN] 智谱 AI 客户端初始化失败: {type(exc).__name__}")
            logger.warning("Zhipu client initialization failed", exc_info=True)

    def _init_openai_client(self):
        try:
            from openai import OpenAI

            if not api_keys.openai_key:
                return
            kwargs: Dict[str, Any] = {"api_key": api_keys.openai_key}
            if os.environ.get("OPENAI_BASE_URL"):
                kwargs["base_url"] = os.environ["OPENAI_BASE_URL"]
            self._register_provider(
                LLMProvider.OPENAI,
                OpenAI(**kwargs),
                os.environ.get("OPENAI_MODEL", "gpt-4-turbo"),
            )
        except ImportError:
            print("[WARN] 未安装 openai 库")
        except Exception as exc:
            print(f"[WARN] OpenAI 客户端初始化失败: {type(exc).__name__}")
            logger.warning("OpenAI client initialization failed", exc_info=True)

    def _refresh_configuration(self) -> None:
        try:
            api_keys.refresh()
        except Exception:
            return
        signature = (api_keys.zhipu_key, api_keys.openai_key)
        if signature == self._credentials_signature:
            return
        with self._state_lock:
            if signature != self._credentials_signature:
                self._init_client()

    def is_available(self) -> bool:
        self._refresh_configuration()
        with self._state_lock:
            return bool(self._provider_states)

    @property
    def provider(self) -> Optional[str]:
        return self._provider.value if self._provider else None

    @property
    def model_name(self) -> str:
        return self._model_name

    def set_model(self, model: str) -> None:
        value = str(model or "").strip()
        if not value:
            return
        with self._state_lock:
            if self._provider in self._provider_states:
                self._provider_states[self._provider].model = value
            self._model_name = value

    def _set_active(self, state: _ProviderState) -> None:
        self._provider = state.provider
        self._client = state.client
        self._model_name = state.model

    def chat(
        self,
        messages: list,
        temperature: float = 0.7,
        max_tokens: int = 2000,
        *,
        provider: Optional[str] = None,
        model: Optional[str] = None,
        request_kind: str = "interactive",
        request_id: Optional[str] = None,
    ) -> Optional[str]:
        trace = _LLMTrace(
            request_id=_trace_request_id(request_id),
            request_kind=_normalize_request_kind(request_kind),
            stream=False,
        )
        if not self.is_available():
            trace.finish("unavailable", error_class="LLM_UNAVAILABLE")
            trace.emit()
            print("[WARN] LLM 客户端不可用")
            return None
        cache_key = self._cache_key(messages, temperature, max_tokens, provider, model)
        cached = self._cache_get(cache_key)
        if cached:
            trace.cache_hit = True
            trace.finish("cache_hit", provider=self.provider, model=self.model_name)
            trace.emit()
            return cached

        # 缓存未命中时合并同一进程内的并发相同请求，防止页面重复点击、
        # 多个后台触发器同时把同一份 prompt 发给 provider。
        inflight_lock = getattr(self, "_inflight_lock", None)
        if inflight_lock is None:
            # 保持对旧测试/第三方通过 object.__new__ 构造客户端的兼容。
            inflight_lock = threading.RLock()
            self._inflight_lock = inflight_lock
            self._inflight = {}
        with inflight_lock:
            inflight = self._inflight.get(cache_key)
            leader = inflight is None
            if leader:
                inflight = _InflightCall()
                self._inflight[cache_key] = inflight
        if not leader:
            if not inflight.event.wait(getattr(self, "_singleflight_wait", 90.0)):
                trace.finish("singleflight_timeout", error_class="TIMEOUT")
                trace.emit()
                logger.warning("LLM single-flight wait timed out")
                return None
            result = self._cache_get(cache_key) or inflight.result
            trace.finish(
                "singleflight_wait",
                provider=self.provider,
                model=self.model_name,
            )
            trace.emit()
            return result

        self._mark_request_priority()
        result = None
        try:
            for state in self._candidate_states(provider):
                requested_model = self._model_for_state(state, model, provider)
                content = self._chat_with_provider(
                    state,
                    messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    model=requested_model,
                    trace=trace,
                )
                if content:
                    result = content
                    self._cache_set(cache_key, content)
                    trace.finish(
                        "completed",
                        provider=state.provider.value,
                        model=requested_model or state.model,
                    )
                    trace.emit()
                    return content
            trace.finish(
                "provider_error",
                error_class=trace.error_class or "LLM_UNAVAILABLE",
            )
            trace.emit()
            logger.warning("All configured LLM providers failed for one request")
            return None
        except Exception as exc:
            trace.finish("internal_error", error=exc)
            trace.emit()
            raise
        finally:
            with inflight_lock:
                current = self._inflight.get(cache_key)
                if current is inflight:
                    inflight.result = result
                    self._inflight.pop(cache_key, None)
                    inflight.event.set()

    def chat_stream(
        self,
        messages: list,
        temperature: float = 0.7,
        max_tokens: int = 2000,
        *,
        provider: Optional[str] = None,
        model: Optional[str] = None,
        request_kind: str = "interactive",
        request_id: Optional[str] = None,
    ):
        trace = _LLMTrace(
            request_id=_trace_request_id(request_id),
            request_kind=_normalize_request_kind(request_kind),
            stream=True,
        )
        if not self.is_available():
            trace.finish("unavailable", error_class="LLM_UNAVAILABLE")
            trace.emit()
            print("[WARN] LLM 客户端不可用")
            return
        cache_key = self._cache_key(messages, temperature, max_tokens, provider, model)
        cached = self._cache_get(cache_key)
        if cached:
            trace.cache_hit = True
            trace.finish("cache_hit", provider=self.provider, model=self.model_name)
            trace.emit()
            for index in range(0, len(cached), 64):
                yield cached[index:index + 64]
            return
        self._mark_request_priority()
        for state in self._candidate_states(provider):
            emitted = False
            chunks: List[str] = []
            last_error: Optional[Exception] = None
            requested_model = self._model_for_state(state, model, provider)
            for attempt in range(self._retry_attempts):
                acquire_started = time.perf_counter()
                acquired = self._request_semaphore.acquire(timeout=self._request_queue_timeout)
                queue_wait_seconds = time.perf_counter() - acquire_started
                if not acquired:
                    last_error = TimeoutError("AI_REQUEST_QUEUE_TIMEOUT")
                    trace.record_attempt(state, requested_model, queue_wait_seconds, 0.0)
                else:
                    call_started = time.perf_counter()
                    try:
                        response = self._create_completion(
                            state,
                            messages,
                            temperature=temperature,
                            max_tokens=max_tokens,
                            stream=True,
                            model=self._model_for_state(state, model, provider),
                        )
                        for chunk in response:
                            content = _stream_chunk_content(chunk)
                            if content:
                                emitted = True
                                chunks.append(content)
                                yield content
                        if not chunks:
                            raise _EmptyResponseError("empty stream")
                        content = "".join(chunks)
                        self._record_success(state)
                        self._cache_set(cache_key, content)
                        trace.finish(
                            "completed",
                            provider=state.provider.value,
                            model=requested_model or state.model,
                        )
                        trace.emit()
                        return
                    except Exception as exc:
                        last_error = exc
                    finally:
                        trace.record_attempt(
                            state,
                            requested_model,
                            queue_wait_seconds,
                            time.perf_counter() - call_started,
                        )
                        self._request_semaphore.release()
                trace.note_error(last_error)
                if emitted:
                    self._record_failure(state, last_error)
                    trace.finish(
                        "stream_interrupted",
                        provider=state.provider.value,
                        model=requested_model or state.model,
                        error=last_error,
                    )
                    trace.emit()
                    raise LLMServiceError("STREAM_INTERRUPTED") from last_error
                if not last_error or not _is_retryable_error(last_error):
                    break
                if attempt < self._retry_attempts - 1:
                    self._maybe_use_fallback_model(state, last_error)
                    time.sleep(self._retry_delay(attempt, last_error))
            self._record_failure(state, last_error)
        trace.finish(
            "provider_error",
            error_class=trace.error_class or "LLM_UNAVAILABLE",
        )
        trace.emit()
        logger.warning("All configured LLM providers failed before stream output")

    def _chat_with_provider(
        self,
        state: _ProviderState,
        messages: list,
        *,
        temperature: float,
        max_tokens: int,
        model: Optional[str] = None,
        trace: Optional[_LLMTrace] = None,
    ) -> Optional[str]:
        last_error: Optional[Exception] = None
        for attempt in range(self._retry_attempts):
            acquire_started = time.perf_counter()
            acquired = self._request_semaphore.acquire(timeout=self._request_queue_timeout)
            queue_wait_seconds = time.perf_counter() - acquire_started
            if not acquired:
                last_error = TimeoutError("AI_REQUEST_QUEUE_TIMEOUT")
                if trace:
                    trace.record_attempt(state, model, queue_wait_seconds, 0.0)
            else:
                call_started = time.perf_counter()
                try:
                    content = self._chat_once(
                        state,
                        messages,
                        temperature=temperature,
                        max_tokens=max_tokens,
                        model=model,
                    )
                    self._record_success(state)
                    return content
                except Exception as exc:
                    last_error = exc
                finally:
                    if trace:
                        trace.record_attempt(
                            state,
                            model,
                            queue_wait_seconds,
                            time.perf_counter() - call_started,
                        )
                    self._request_semaphore.release()
            if trace:
                trace.note_error(last_error)
            if not last_error or not _is_retryable_error(last_error):
                break
            if attempt < self._retry_attempts - 1:
                self._maybe_use_fallback_model(state, last_error)
                delay = self._retry_delay(attempt, last_error)
                logger.warning(
                    "LLM %s request failed (%s); retrying in %.2fs (%d/%d)",
                    state.provider.value,
                    _error_label(last_error),
                    delay,
                    attempt + 1,
                    self._retry_attempts,
                )
                time.sleep(delay)
        self._record_failure(state, last_error)
        return None

    def _chat_once(self, state, messages, *, temperature, max_tokens, model=None):
        response = self._create_completion(
            state,
            messages,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=False,
            model=model,
        )
        content = _message_content(response)
        if not content:
            raise _EmptyResponseError("empty response")
        return content

    def _model_for_state(
        self,
        state: _ProviderState,
        requested_model: Optional[str],
        preferred_provider: Optional[str],
    ) -> Optional[str]:
        """Keep a provider-specific model from leaking into failover calls."""

        if not requested_model:
            return None
        target_provider = None
        if preferred_provider:
            try:
                target_provider = LLMProvider(str(preferred_provider).lower())
            except ValueError:
                target_provider = None
        if target_provider is None:
            with self._state_lock:
                target_provider = self._provider
        if target_provider is None or state.provider is target_provider:
            return requested_model
        return None

    def _create_completion(self, state, messages, *, temperature, max_tokens, stream, model=None):
        kwargs: Dict[str, Any] = {
            "model": model or state.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if stream:
            kwargs["stream"] = True
        if state.provider is LLMProvider.ZHIPU:
            kwargs["extra_body"] = {"thinking": {"type": "disabled"}}
        return state.client.chat.completions.create(**kwargs)

    def _candidate_states(self, preferred_provider=None) -> List[_ProviderState]:
        now = time.monotonic()
        with self._state_lock:
            order: List[LLMProvider] = []
            preferred = None
            if preferred_provider:
                try:
                    preferred = LLMProvider(str(preferred_provider).lower())
                except ValueError:
                    preferred = None
            if preferred in self._provider_states:
                order.append(preferred)
            if self._provider in self._provider_states and self._provider not in order:
                order.append(self._provider)
            order.extend(provider for provider in self._provider_order if provider not in order)
            return [
                self._provider_states[provider]
                for provider in order
                if self._provider_states[provider].health.opened_until <= now
            ]

    def _record_success(self, state: _ProviderState) -> None:
        with self._state_lock:
            state.health = _ProviderHealth()
            self._available = True
            self._set_active(state)

    def _record_failure(self, state: _ProviderState, error: Optional[Exception]) -> None:
        label = _error_label(error) if error else "UNKNOWN"
        with self._state_lock:
            state.health.consecutive_failures += 1
            state.health.last_error = label
            self._last_error = label
            if state.health.consecutive_failures >= self._circuit_failure_threshold:
                state.health.opened_until = time.monotonic() + self._circuit_cooldown
                logger.warning(
                    "LLM provider %s circuit opened for %.1fs (%s)",
                    state.provider.value,
                    self._circuit_cooldown,
                    label,
                )

    def _maybe_use_fallback_model(self, state: _ProviderState, error: Exception) -> None:
        if state.provider is not LLMProvider.ZHIPU or not _is_rate_limit(error):
            return
        fallback = os.environ.get("ZHIPU_FALLBACK_MODEL", "glm-4.5-flash").strip()
        if fallback and state.model != fallback:
            state.model = fallback
            with self._state_lock:
                if self._provider is state.provider:
                    self._model_name = fallback

    def _retry_delay(self, attempt: int, error: Optional[Exception]) -> float:
        retry_after = _retry_after_seconds(error)
        exponential = min(
            self._retry_max_delay,
            self._retry_base_delay * (2 ** max(0, attempt)),
        )
        delay = max(exponential, retry_after or 0.0)
        jitter = random.uniform(0.0, min(0.25, delay * 0.2))
        return min(self._retry_max_delay, delay + jitter)

    def _mark_request_priority(self) -> None:
        is_worker = threading.current_thread().name.startswith("worker-")
        now = time.time()
        if not is_worker:
            SharedLLMClient.last_user_request_time = now
            return
        deadline = time.monotonic() + self._background_max_wait
        while (
            time.time() - SharedLLMClient.last_user_request_time < self._background_priority_window
            and time.monotonic() < deadline
        ):
            time.sleep(0.25)

    def _cache_key(self, messages, temperature, max_tokens, provider=None, model=None):
        with self._state_lock:
            models = [
                (provider_name.value, self._provider_states[provider_name].model)
                for provider_name in self._provider_order
                if provider_name in self._provider_states
            ]
        serialized = json.dumps(
            {
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "models": models,
                "preferred_provider": provider,
                "preferred_model": model,
            },
            sort_keys=True,
            ensure_ascii=False,
            default=str,
        )
        return f"llm_cache:{hashlib.sha256(serialized.encode('utf-8')).hexdigest()}"

    def _cache_get(self, cache_key: str) -> Optional[str]:
        if self._redis_cache:
            try:
                cached = self._redis_cache.get(cache_key)
                if cached:
                    return cached.decode("utf-8") if isinstance(cached, bytes) else str(cached)
            except Exception as exc:
                logger.debug("Redis LLM cache read failed: %s", type(exc).__name__)
        now = time.time()
        with self._cache_lock:
            entry = self._local_cache.get(cache_key)
            if entry:
                expires_at, value = entry
                if expires_at > now:
                    return value
                self._local_cache.pop(cache_key, None)
        return None

    def _cache_set(self, cache_key: str, content: str) -> None:
        if not content:
            return
        if self._redis_cache:
            try:
                self._redis_cache.setex(cache_key, self._cache_ttl, content)
                return
            except Exception as exc:
                logger.debug("Redis LLM cache write failed: %s", type(exc).__name__)
        with self._cache_lock:
            if len(self._local_cache) >= 128:
                self._local_cache.pop(next(iter(self._local_cache)))
            self._local_cache[cache_key] = (time.time() + self._cache_ttl, content)

    def health_snapshot(self) -> Dict[str, Any]:
        """Return a redacted runtime health snapshot for diagnostics/UI."""
        now = time.monotonic()
        with self._state_lock:
            providers = {}
            for provider, state in self._provider_states.items():
                providers[provider.value] = {
                    "model": state.model,
                    "configured": True,
                    "circuit_open": state.health.opened_until > now,
                    "consecutive_failures": state.health.consecutive_failures,
                    "last_error": state.health.last_error,
                }
            return {
                "available": bool(self._provider_states),
                "active_provider": self.provider,
                "active_model": self.model_name,
                "providers": providers,
            }

    def evaluate_code(self, code: str, assignment_title: str = None) -> Tuple[int, str]:
        """使用 LLM 评估代码。"""
        if not self.is_available():
            return 3, "LLM 服务不可用，使用默认评分"

        system_prompt = """你是一名经验丰富的C++编程教师，评估学生代码质量。
评分标准: 1-5分，5分最高。请先给出分数，再用"分析："分隔详细反馈。

格式: "分数：X\n分析：..."""
        user_prompt = f"题目：{assignment_title or '未指定'}\n\n代码：\n{code}"
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        response = self.chat(
            messages, temperature=0.2, request_kind="submission"
        )
        if not response:
            return 3, "LLM 响应失败，使用默认评分"

        import re

        score_match = re.search(r"分数[：:]\s*(\d+)", response)
        score = max(0, min(5, int(score_match.group(1)))) if score_match else 3
        feedback = response.split("分析：", 1)[1].strip() if "分析：" in response else response
        return score, feedback

    def generate_image(self, prompt: str) -> Optional[str]:
        """调用当前 provider 的图像生成接口。"""
        if not self.is_available():
            print("[WARN] LLM 客户端不可用，无法生成图像")
            return None
        self._mark_request_priority()
        for state in self._candidate_states():
            acquired = self._request_semaphore.acquire(timeout=self._request_queue_timeout)
            if not acquired:
                continue
            try:
                if state.provider is LLMProvider.ZHIPU:
                    response = state.client.images.generations(
                        model=os.environ.get("ZHIPU_IMAGE_MODEL", "cogview-4"),
                        prompt=prompt,
                        size="1024x1024",
                    )
                else:
                    response = state.client.images.generate(
                        model=os.environ.get("OPENAI_IMAGE_MODEL", "dall-e-3"),
                        prompt=prompt,
                        size="1024x1024",
                        n=1,
                    )
                if response and getattr(response, "data", None):
                    self._record_success(state)
                    return response.data[0].url
            except Exception as exc:
                self._record_failure(state, exc)
            finally:
                self._request_semaphore.release()
        return None


llm_client = SharedLLMClient()


_http_session_local = threading.local()


def _thread_http_session():
    """Return one pooled requests session per worker thread."""
    session = getattr(_http_session_local, "session", None)
    if session is not None:
        return session
    import requests
    from requests.adapters import HTTPAdapter

    session = requests.Session()
    adapter = HTTPAdapter(
        pool_connections=_env_int("AI_HTTP_POOL_CONNECTIONS", 8, minimum=1, maximum=64),
        pool_maxsize=_env_int("AI_HTTP_POOL_MAXSIZE", 8, minimum=1, maximum=64),
        max_retries=0,
    )
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    _http_session_local.session = session
    return session


def _response_error_details(response: Any) -> Tuple[Optional[str], str]:
    try:
        payload = response.json()
    except Exception:
        return None, ""
    if not isinstance(payload, dict):
        return None, ""
    error = payload.get("error")
    if isinstance(error, dict):
        code = str(error.get("code")) if error.get("code") is not None else None
        return code, str(error.get("message") or "")
    code = str(payload.get("code")) if payload.get("code") is not None else None
    return code, str(payload.get("message") or "")


def _http_retry_delay(attempt: int, error: Optional[Exception]) -> float:
    retry_after = _retry_after_seconds(error)
    base = _env_float("AI_RETRY_BASE_DELAY_SECONDS", 0.8, minimum=0.05, maximum=10.0)
    maximum = _env_float("AI_RETRY_MAX_DELAY_SECONDS", 8.0, minimum=0.1, maximum=60.0)
    delay = max(min(maximum, base * (2 ** max(0, attempt))), retry_after or 0.0)
    return min(maximum, delay + random.uniform(0.0, min(0.25, delay * 0.2)))


def safe_zhipu_post(url, headers, json_data, timeout=30, stream=False):
    """发送智谱请求并处理瞬时网络错误、限流和 1305。

    非重试型 HTTP 错误返回 response；网络错误在最后一次失败时抛出，
    调用方可以据此返回合适的 SSE/HTTP 错误。
    """
    import copy
    import requests

    try:
        llm_client._mark_request_priority()
    except Exception:
        pass

    payload = copy.deepcopy(json_data or {})
    current_model = str(payload.get("model") or "glm-4.5-flash")
    attempts = _env_int("AI_HTTP_RETRY_ATTEMPTS", 3, minimum=1, maximum=6)
    fallback_model = os.environ.get("ZHIPU_FALLBACK_MODEL", "glm-4.5-flash").strip()
    session = _thread_http_session()
    last_response = None
    last_error: Optional[Exception] = None

    for attempt in range(attempts):
        payload["model"] = current_model
        if "thinking" not in payload:
            payload["thinking"] = {"type": "disabled"}
        response = None
        try:
            response = session.post(
                url,
                headers=headers,
                json=payload,
                timeout=timeout,
                stream=stream,
            )
            last_response = response
            code, message = _response_error_details(response)
            status = getattr(response, "status_code", None)
            is_rate_limit = str(code) == "1305" or _is_rate_limit(response)
            retryable = is_rate_limit or _is_retryable_error(response)
            if status is not None and int(status) < 400 and not is_rate_limit:
                return response
            if not retryable:
                return response
            last_error = requests.exceptions.HTTPError(
                f"Zhipu request failed: {code or status or 'unknown'} {message}".strip(),
                response=response,
            )
        except Exception as exc:
            last_error = exc
            last_response = getattr(exc, "response", None)
            if not _is_retryable_error(exc) and not _is_rate_limit(exc):
                raise

        if attempt >= attempts - 1:
            if last_response is not None:
                return last_response
            if last_error is not None:
                raise last_error
            return None

        if last_error is not None and _is_rate_limit(last_error) and fallback_model:
            if current_model != fallback_model:
                current_model = fallback_model
                try:
                    llm_client.set_model(fallback_model)
                except Exception:
                    pass
        delay = _http_retry_delay(attempt, last_error)
        logger.warning(
            "Zhipu HTTP request failed (%s); retrying in %.2fs (%d/%d)",
            _error_label(last_error),
            delay,
            attempt + 1,
            attempts,
        )
        if response is not None:
            try:
                response.close()
            except Exception:
                pass
        time.sleep(delay)

    return last_response
