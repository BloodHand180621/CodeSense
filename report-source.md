# CodeSense AI 输出流优化研究底稿

研究日期：2026-09-07  
范围：教师端班级学情分析流、学生端能力分析流，以及复用 `consumeSSE` 的代码建议/指导/提问/思维陪练流。  
目标：降低首段内容出现前的空白感、避免流式渲染造成主线程卡顿，并让首块延迟、流中断和完整耗时可被区分观测。

## 结论摘要

“流式”不等于“每收到一个 token 就重绘一次”。成熟实现通常把传输、语义分块和 UI 更新分开：服务端尽早发送生命周期/状态事件；模型文本按可读边界持续到达；前端以约 50ms 的节奏批量更新，而不是在每个小 chunk 上重新解析完整 Markdown。

CodeSense 当前最明显的本地问题是：

1. 教师端在首次可见文本到达后才发送 `start`，模型首 token 等待期间只有加载态；收到每个 chunk 后都对完整缓冲区调用 `marked.parse` 并强制滚动。
2. 学生端对已经存在数据库中的完整分析结果，按约 2 个字符、每次 50ms 播放，并额外模拟打错字和删除。这会把本来可以立即显示的缓存内容人为拖慢。
3. 共用的 fetch SSE 客户端把每个 delta 同步交给页面回调，代码建议、代码指导、问答和思维陪练页面也可能在每个小片段上重排 DOM。
4. 服务端已有 SSE 禁缓冲配置，但缺少统一的首块/首可见文本指标；仅看请求总耗时无法判断慢在排队、模型首 token 还是前端渲染。

## 外部证据

### 传输层

- [Nginx `proxy` 模块官方文档](https://nginx.org/en/docs/http/ngx_http_proxy_module.html)：`proxy_buffering` 默认开启；关闭后响应会在收到时同步转发。`proxy_read_timeout` 是两次读取之间的超时，不是整个响应的总时长。因此，反代必须关闭缓冲，同时长流还要有足够的读取间隔预算。
- [MDN：Using server-sent events](https://developer.mozilla.org/en-US/docs/Web/API/Server-sent_events/Using_server-sent_events)：SSE 支持 `event`、`data`、`id`、`retry` 等生命周期字段；连接断开时浏览器会按重试时间重连。对于不可恢复的一次性生成流，客户端不应无条件重连并再次触发模型调用。
- [Anthropic Messages streaming 官方文档](https://platform.claude.com/docs/en/build-with-claude/streaming)：成熟流协议区分消息/内容块开始、delta、结束、ping 和 error；客户端应能识别错误和未知事件，而不是只把所有内容当成普通文本。

### 分块与感知延迟

- [Vercel AI SDK：`smoothStream`](https://ai-sdk.dev/docs/reference/ai-sdk-core/smooth-stream)：通过变换器按词、行或自定义边界释放文本，并针对中文给出按汉字/非空白词分块的思路；这说明“可读节奏”比机械逐 token 更适合 UI。
- [Vercel AI SDK：Chatbot](https://ai-sdk.dev/docs/ai-sdk-ui/chatbot)：明确提供 `experimental_throttle: 50`，因为每个 chunk 都触发一次更新会让复杂 Markdown UI 过载；同时提供停止生成和错误状态。
- [Google Cloud：流式生成答案](https://cloud.google.com/generative-ai-app-builder/docs/stream-answer?hl=en)：长答案分成按顺序返回的多个部分，通常以句子为单位，目标是降低用户感知到的等待时间。
- [MDN：`requestAnimationFrame`](https://developer.mozilla.org/en-US/docs/Web/API/Window/requestAnimationFrame)：DOM 视觉更新应安排在浏览器下一次重绘前执行，避免在同一批网络事件中反复同步修改界面。
- [Chrome：Long Animation Frames](https://developer.chrome.com/docs/web-platform/long-animation-frames)：长任务和长动画帧可用于识别主线程卡顿；流式 Markdown 解析属于应重点观察的主线程工作。

### 指标与排障

- [OpenTelemetry GenAI 语义约定](https://opentelemetry.io/docs/specs/semconv/registry/attributes/gen-ai/)：定义了 `gen_ai.request.stream` 和 `gen_ai.response.time_to_first_chunk` 等属性。首块时间和完整响应时间必须分开，否则“模型很快开始输出但 UI 卡住”和“模型迟迟没有首 token”会被混为一谈。
- [Vercel AI SDK：Telemetry](https://ai-sdk.dev/docs/ai-sdk-core/telemetry)：把 `ai.stream.firstChunk` 作为流式请求的关键事件，适合作为 CodeSense 日志的命名参考。

## 对 CodeSense 的落地决策

本轮先做低风险、可回滚的优化：

1. `consumeSSE` 默认把 delta 回调节流到 50ms，并在 `done/error` 前强制 flush；解析错误、HTTP 错误和读流异常仍保持原有 Promise 错误语义。
2. 教师流在调用模型前发送 `start`，并改用只识别完整 `===JSON===` 分隔符的增量拆分，避免普通正文中出现“JSON”就停止可见输出；前端以 50ms 批量 Markdown 渲染、只在用户位于底部附近时自动滚动。
3. 学生端缓存分析改为较大的语义块立即发送，移除人为打字、错字和删除模拟；前端的已有 50ms 防抖保留，并在完成事件前 flush。
4. LLM trace 增加首 chunk 延迟、chunk 数和输出字符数；这些字段不含 prompt、学号或模型正文。

暂不做的事情：

- 不引入新的消息队列、WebSocket 或数据库 schema；现有 SSE 已经足够支持本轮目标。
- 不对不可恢复的一次性教师 EventSource 做自动重连；否则断线后可能重复发起一次付费分析。错误时保留已显示正文并提示用户重试。
- 不凭空承诺模型供应商的首 token 会变快；本轮首先消除浏览器端“收到但没及时显示”的额外延迟，并把真正的服务端首块延迟记录出来。

## 验收指标

- 首个 `start` 事件应在服务端进入模型等待阶段前到达浏览器。
- `consumeSSE` 的页面 delta 回调最多约每 50ms 执行一次；最终 `done` 前不得丢失缓冲内容。
- 缓存分析不再因为固定 sleep 被人为限速。
- `llm_trace` 可区分 `time_to_first_chunk_ms`、`duration_ms`、`stream_chunks` 和 `output_chars`。
- 现有 SSE 契约、教师端 JSON 解析和全量测试保持通过。

