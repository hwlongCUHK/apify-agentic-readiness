# Apify 定价制度变化 × AI/Agent 需求 — 项目设计与可行性备忘录

> 本文件是项目的 **canonical reference**：研究设计、平台勘察结论、数据 schema、时间线、identification 策略与 pipeline 架构。
> 最后更新：**2026-08-18**（勘察日，以下结论均基于当日对 Apify 平台与 API 的实测）。

---

## 1. 核心研究问题

当 AI/自动化工具从**固定订阅/租赁定价（rental）**转向 **usage-based / pay-per-event（PPE）** 定价后，需求（adoption）、使用强度（usage）、竞争（competition）与供给行为（seller effort）如何变化？

- **分析单位**：Actor i × Day t（Apify Store 每个 Actor 视作一个软件产品）
- **核心 treatment**：`pricing_regime` 的制度切换（rental → PPE / pay-per-usage）
- **核心 outcome**（优先级降序）：`runs_30d` > `monthly_users` > `total_users` > `reviews`
- **主要 heterogeneity**：AI/Agent-native vs 传统自动化产品（Agent consumer 理论上更偏好 sporadic 的 `$0.002/call` 而非 predictable 的 `$29/month`）

---

## 2. 平台勘察关键结论（2026-08-18 实测）

### 2.1 数据源：无需爬 HTML，直接用匿名 REST API

```
GET https://api.apify.com/v2/store          # 无需 token，匿名可调
```

一次返回每个 Actor 的 metadata + usage stats + pricing + 分类。**没有 GraphQL 端点**。

- 文档：https://docs.apify.com/api/v2/store-get
- OpenAPI：`https://cdn.jsdelivr.net/npm/@apify/openapi@0.0.123/openapi.yaml`（path `/v2/store`）
- 限流：`x-ratelimit-limit: 60`/min；实测宽松（66 连发 0 throttle），但建议 **≤1 req/s**
- robots.txt 全开放（`Allow: /`，明确 `ai-input=yes`）；AUP 允许**内部**收集，禁止**商业转售** catalog

**查询参数**：`limit`（默认/最大 1000）、`offset`、`search`、`sortBy`（`relevance|popularity|newest|lastUpdate`）、`category`、`username`、`pricingModel`（`FREE|FLAT_PRICE_PER_MONTH|PRICE_PER_DATASET_ITEM|PAY_PER_EVENT`）、`allowsAgenticUsers`、`responseFormat`（`full|agent`）、`includeUnrunnableActors`（bool，默认 false）。

### 2.2 规模

| 项目 | 数值 |
|---|---|
| 总 Actor | ~61,150（sitemap）/ 61,301（API `includeUnrunnableActors=true`）|
| 可运行活跃（默认） | ~48,200（`includeUnrunnableActors=false`）|
| 唯一 developer username | **5,334** |
| 全量抓取耗时 | 5,334 req @1/s ≈ **1.5 h** |

### 2.3 关键限制（对 schema 的影响）

1. **`offset` 分页在 ~15k 后静默返回空** → 无法线性翻完 61k，必须按 username 分区枚举（见 §6）。
2. **无公开 `created_at` / `last_modified`** → seller effort 需用代理变量（`totalBuilds`、`lastRunStartedAt`）。
3. **HTML 详情页字段远少于 API**：`runs_30d` 绝对值、`review_count`、分层定价只在 API 里，不在渲染页面 → 务必存 raw **API JSON**（不是只存 HTML）。
4. **无公开历史时序**：只有当前快照 + 7/30/90 天聚合窗口；历史需靠每日自存累积（这是本项目的根本目的）。

---

## 3. 定价模型与时间线（identification 命门）

### 3.1 定价模型 → regime 编码

| regime 码 | 含义 | Apify enum | 说明 |
|---|---|---|---|
| 0 | rental（固定月费）| `FLAT_PRICE_PER_MONTH` | 始终 "月费 + usage"，本身带 hybrid 成分 |
| 1 | pay-per-event（PPE）| `PAY_PER_EVENT` | 现行标准，developer 自定义 event 与单价 |
| 2 | pay-per-result（PPR）| `PRICE_PER_DATASET_ITEM` | legacy，已迁 ~2,000 个到 PPE |
| 3 | pay-per-usage / free | `FREE` | 只付平台 usage（Compute Units）|

> Apify **没有真正的 "hybrid" enum**。你的原设计 0/1/2/3 编码需按上表映射；"pay per usage" 在 enum 里表现为 `FREE`。

相关术语：**Compute Units (CU)** = `1 GB 内存 × 1 小时`；**80/20 分成**（developer/Apify）；PPE 允许用户设**每 run 最大费用上限**。

### 3.2 时间线

```
2025-03-03   PPE 上线（changelog，早鸟 0% 佣金激励）
2025–2026初  已迁移 ~2,000 Actor（PPR → PPE）
2026-04-01   ❄ 冻结：禁止新 rental，存量 rental 定价锁定
2026-04-14   📢 官方博客宣布 rental sunset（"within 6 months"）
2026-08-18   ← 今天（开始收集 = pre-period 尾端）
2026-10-01   💥 rental 彻底退役，剩余 rental Actor 自动迁移到 pay-per-usage
```

### 3.3 对 identification 的含义（重要）

1. **Treatment 是外生平台政策**（硬 deadline），不是 seller 的自愿决定 —— 这是最理想的场景，直接化解 §十四 最大的内生性担忧。
2. **内生性转移**：developer 只能选「目的地」（手动 PPE vs 自动 pay-per-usage fallback），**不能留在 rental**。新的识别挑战是 **「谁选了 PPE vs 谁被自动 fallback」**，它与 developer 特征 / 需求弹性相关。
3. **进场偏晚**：距 10-01 只剩 ~6 周 pre-period；至今仍未迁的 rental Actor 是**负向选择的 laggards**（影响外部有效性）。
4. **`currentPricingInfo.createdAt` 可给每个 Actor 锚定 `T_i`**（定价切换日期）——对已迁移 Actor 也能精确 event time，是 event study 的关键输入。
5. **对策**：立即开抓攒 pre-period；用 `totalUsers90Days vs 30Days` 的差、`totalUsers`/`totalRuns` 的**累积量日间差分**近似重建 pre-period 需求轨迹；对 4–10 月已迁 Actor，用 `pricing_created_at` + web.archive.org 补历史。

---

## 4. Schema（5 张核心表 + 辅助 + raw 归档）

DDL 见 [`schema.sql`](schema.sql)。语义说明：

| 表 | 粒度 | 用途 |
|---|---|---|
| `actor_day` | actor × crawl_date | 需求/usage 面板（users、runs、rating、bookmarks、builds）|
| `pricing_day` | actor × crawl_date | 定价制度细节（regime、fee、price_per_unit、raw JSON）|
| `pricing_event_detail` | actor × crawl_date × event | per-event 计费明细（长表）|
| `actor_static` | actor | 产品/开发者静态特征 + 研究者分类（`ai_native`、`agent_native`、`mcp_compatible`、`vertical`）|
| `pricing_events` | 事件 | **检测到的制度切换**（treatment 事件，含 `switch_reason`）|
| `category_day` | category × crawl_date | 竞争结果（num_products、share_ppe、entry/exit、HHI）|
| `snapshot_manifest` | crawl_date | 每日抓取的 provenance |

**Raw 归档**（§五 的核心，永远保存）：`data/raw/{date}.jsonl.gz` —— 每个 Actor 一行完整 API JSON（含 `stats` + `currentPricingInfo`），是唯一事实来源。规范化表只是它的派生视图。

---

## 5. 字段映射（原设计 → 真实来源）

### `actor_day` 来源（API `stats` 对象）

| 原设计字段 | API 真实来源 | 状态 |
|---|---|---|
| `actor_id` | `id` | ✅ |
| `name` / `developer` / `url` | `title` / `username`+`userFullName` / `url` | ✅ |
| `monthly_active_users` | `stats.totalUsers30Days`（另有 `7Days`/`90Days`）| ✅ 标签实为 "monthly users" |
| `total_users` | `stats.totalUsers`（lifetime）| ✅ |
| `runs_30d` | `stats.publicActorRunStats30Days.TOTAL` | ✅ 仅 API 有 |
| `successful/failed_runs_30d` | `...SUCCEEDED` / `...FAILED`（+`ABORTED`/`TIMED-OUT`）| ✅ |
| `rating` | `stats.actorReviewRating` | ✅ |
| `review_count` | `stats.actorReviewCount` | ✅ 仅 API 有 |
| `bookmarks` | `stats.bookmarkCount` | ✅ |
| `last_modified` | ❌ 无；代理=`stats.totalBuilds`（更新频率）、`stats.lastRunStartedAt` | ⚠️ |
| `created_at` | ⚠️ `currentPricingInfo.createdAt` 是**定价**创建时间，非 Actor 创建时间 | ⚠️ |
| `category` | `categories[]`（17 个官方分类）| ✅ |

### `pricing_day` 来源（API `currentPricingInfo` 对象）

| 原设计字段 | API 真实来源 | 状态 |
|---|---|---|
| `pricing_regime` | `pricingModel` | ✅ |
| `fixed_fee` / `period` | `pricePerUnitUsd`（rental 时=月费，period=month，**待验证**）| ⚠️ |
| `event_name` / `event_price` | `pricingPerEvent`（完整计费表）| ✅ |
| `minimum_charge` | ⚠️ API 只有 `minimalMaxTotalChargeUsd`（是**上限**不是下限）| ⚠️ |
| `included_quantity` / `free_tier` | ❌ 无公开字段 | ❌ |
| `pricing_raw` | 整个 `currentPricingInfo` JSON | ✅ |

### 官方分类（文档列 17 个，实测 raw 有 24 个 distinct 值）

`AI`、`AGENTS`、`AUTOMATION`、`DEVELOPER_TOOLS`、`ECOMMERCE`、`FOR_CREATORS`、`JOBS`、`LEAD_GENERATION`、`NEWS`、`SEO_TOOLS`、`SOCIAL_MEDIA`、`TRAVEL`、`VIDEOS`、`REAL_ESTATE`、`INTEGRATIONS`、`OPEN_SOURCE`、`MCP_SERVERS`

> 实测 raw 数据里还有 7 个文档未列的类目：`BUSINESS`、`EDUCATION`、`GAMES`、`MARKETING`、`OTHER`、`SPORTS`、`DEVELOPER_EXAMPLES`。共 24 个 distinct category 值。

> **原生就有 `AI`、`AGENTS`、`MCP_SERVERS`** 分类。Agent-compatible 的现成特征有两个：**响应字段 `isWhiteListedForAgenticPayments`**（bool，顶层，已实测）与查询参数 `allowsAgenticUsers`/`responseFormat=agent`。你的 A/B/C 三类 Actor 可用官方分类打底，再叠加自己的文本分类（`ai_native`、`agent_native`、`mcp_compatible`、`requires_llm`、`llm_provider`）。

---

## 6. 枚举与抓取策略

**枚举**（`enumerate.py`）：
1. `GET https://apify.com/sitemap.xml` → 找到 `actors1..10.xml` + `users.xml`
2. 解析 actor sitemap 的 `<loc>`，从 `https://apify.com/{username}/{actor-name}` 提取 username（跳过 `/api/...` 等子路径）
3. 合并 `users.xml` 的 `/users/{username}`
4. 去重排序 → `data/usernames.json`（~5,334 个）

**抓取**（`fetch.py`）：
- 对每个 username 用 `limit=1000` + `offset` **分页循环**，直到取满 envelope 的 `total`（避免 >1000 actor 的开发者被截断；如 `parseforge` 实测有 1,798 个）
- 逐 actor 写入 `data/raw/{date}.jsonl.gz`（一行一个完整 JSON；先写 `.tmp` 再原子 `rename`）
- 同时写 `{date}.usernames.json`（当日名单归档 + sha256）与 `{date}.audit.jsonl`（每 username 的 status/items/total）
- 5,334 次请求 @1/s ≈ 1.5–2.5h，全量、无重叠、可证明完备

> **为什么不用全局 offset 翻全量目录**：实测 offset ~15k–16k 后静默返回 0 条，无法线性翻完 61k。但 **username 内部** 的 offset 是安全的（单开发者 actor 数 << 15k），所以分页只在 username 内部做。

---

## 7. 存储架构

```
data/
├── raw/{date}.jsonl.gz          # raw API 归档（唯一事实来源，永不删除）
├── raw/{date}.manifest.json     # 当日抓取 provenance
├── apify_panel.duckdb           # 规范化表（actor_day / pricing_day / ...）
└── parquet/                     # 可选：DuckDB COPY TO 导出（供 pandas/polars/statsmodels）
```

- **raw 归档**：JSONL.gz（忠实、可重新解析、压缩好）——对应你 §五「永远保存完整 raw 页面状态」。
- **规范化层**：DuckDB（嵌入式、分析型 SQL、直接 `read_json_auto` 读 raw、原生 `COPY TO parquet`）。
- **换栈成本低**：DuckDB 可 attach SQLite/Parquet；若你更想用纯 SQLite 或纯 Parquet，只需改 `normalize.py` 的写入目标，raw 归档不变。

---

## 8. Pipeline 架构

```
 sitemap.xml ──► enumerate.py ──► data/usernames.json
                                      │
                                      ▼
        GET /v2/store?username=… (fetch.py, 限流+重试)
                                      │
                              data/raw/{date}.jsonl.gz
                                      │
                    ┌─────────────────┴─────────────────┐
                    ▼                                   ▼
            normalize.py                        diff_alert.py
       (raw → 5 张规范化表)              (今天 vs 昨天 → 事件检测)
                    │                                   │
            actor_day/pricing_day/              pricing_events /
            actor_static/category_day           entry / exit / 需求跳变 alerts
```

**每日 SOP**（`run_pipeline.py` 一键执行）：
1. `enumerate.py`（usernames 缓存，可跳过）
2. `fetch.py --date YYYY-MM-DD`
3. `normalize.py --date YYYY-MM-DD`
4. `diff_alert.py --cur YYYY-MM-DD`

**alerts 四类**（§八）：pricing regime 变化 / 定价数值变化 / monthly_users 大幅跳变 / actor 消失或新增 / description 变化。

---

## 9. 技术栈

| 组件 | 选择 | 理由 |
|---|---|---|
| 语言 | Python 3.11+ | 生态（pandas/statsmodels/linearmodels/pyfixest 做 staggered DiD）|
| HTTP | `requests` | 简单、够用（自带重试+限流封装）|
| 存储 | DuckDB（+ gzip JSONL raw）| 分析型 SQL、嵌入式单文件、原生读 JSON/写 Parquet |
| 导出 | Parquet（可选）| 与 pandas/polars/statsmodels 互通 |

依赖清单见 `requirements.txt`：`requests`、`duckdb`（可选 `pyarrow`）。

---

## 10. 目录结构

```
Apify Project/
├── DESIGN.md              # 本文件（canonical reference）
├── README.md              # 快速上手
├── requirements.txt
├── .gitignore
├── config.py              # 集中配置（路径、API、限流、regime 映射）
├── schema.sql             # DuckDB DDL（5+2 表）
├── enumerate.py           # sitemap → usernames
├── fetch.py               # API 抓取（限流+重试）→ raw JSONL.gz
├── normalize.py           # raw → 规范化表
├── diff_alert.py          # 今天 vs 昨天 → pricing_events / alerts
├── run_pipeline.py        # 一键编排
└── data/                  # gitignored；raw + duckdb + parquet
```

---

## 11. 待验证 / 开放问题（2026-08-18 冒烟测试后更新）

**已校准（真实 payload 实测）：**
1. ✅ **Agent 标志** = 顶层字段 `isWhiteListedForAgenticPayments`（bool，非 `allowsAgenticUsers`，后者只是查询参数）。
2. ✅ **`pricingPerEvent` 结构** = `{"actorChargeEvents": {"<eventKey>": {"eventTitle", "eventDescription", "isOneTimeEvent", "eventPriceUsd", "isPrimaryEvent?"}}}`。已写入 `pricing_event_detail`（event_key/name/price/is_primary/is_one_time/description）。
3. ✅ **rental 语义** = `pricePerUnitUsd` 是月费，`trialMinutes` 是免费试用期（分钟）；`minimalMaxTotalChargeUsd` 是 PPE 的**最大消费上限**（rental 下为 NULL）。
4. ✅ **`totalMetamorphs` 字段不存在**（勘察 agent 臆测，已从 schema 移除）。
5. ✅ **`notice` 字段** = `NONE` | `UNDER_MAINTENANCE`，已入 `actor_day`。

**仍待验证：**
- **`lastUpdate` 字段**：`sortBy=lastUpdate` 存在，响应里是否有对应字段待确认（更精确的 seller-effort 指标）。
- **`badge`**：本样本全 null，但可能对部分 Actor 非空（如 "Recommended for agents"），值得后续检查。
- **`is_runnable`**：list payload 未暴露 per-actor runnable 标志；`includeUnrunnableActors` 参数控制是否返回，但响应无对应字段。

**方法论待办：**
- **pre-period 重建**：评估 web.archive.org 抓历史页面 + `totalUsers90Days` 反推的可行性，以补足 8 月之前的需求轨迹。
- **AUP 边界**：内部研究收集 OK；若未来要**公开/共享 catalog 数据**需 Apify 书面许可。

---

## 12. 研究设计要点（原设计摘要，供对照）

- **Cohort 设计**：Day-0 全市场 census → 建立 longitudinal cohort（N≈2,000–5,000/日，oversample remaining-rental、AI/Agent、MCP、high-usage、竞争密集 category）。
- **Reduced form**：staggered event study `Y_it = α_i + γ_t + Σ β_k·1(t−T_i=k) + ε_it`，outcome = `logMonthlyUsers` / `logRuns` / `SuccessRate`。
- **四个 margin**：extensive（users↑）、intensive（runs/user↑）、quality（failure rate）、seller behavior（update frequency）。
- **Heterogeneity**：`PPE_it × Agent_i`，核心假设 `β_PPE·Agent > β_PPE·Traditional`。
- **竞争 spillover**：`CompetitorPPEShare_{−j,t}` 的 IO 扩展。
- **Structural**：nonlinear pricing 下 consumer sorting（heavy user 偏好 subscription、light/sporadic user 偏好 usage）→ counterfactual（全 rental / 全 PPE / 降 50% / agent share 20%→60%）。
- **Identification**：voluntary switch vs platform-forced 的区分 → 本项目已由时间线确认是 **platform-forced + hard deadline**（见 §3），这是最有力的场景。
