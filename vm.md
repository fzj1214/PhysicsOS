# PhysicsOS 与 foamvm 的账号、邀请码、CLI Runner 重构建议

本文讨论两个目录的重构：

```text
D:\foamvm
D:\javascript\cae-agent\physicsOS
```

目标是把 `https://foamvm.vercel.app/` 升级成 PhysicsOS Cloud / Runner Portal，让可安装的 `physicsOS` CLI agent 复用同一套账号、邀请码、配额和运行记录。

核心结论：

```text
1. PhysicsOS Python 包里不应该保留本地 Node.js runner。
2. CLI 不应该自带 E2B、Docker、OpenFOAM 环境。
3. foamvm 应该成为云端 runner service 和账号/配额中心。
4. CLI 只生成 manifest、提交任务、拉取 artifacts、做 verification/report。
5. 邀请码/run token 应该直接成为 CLI 运行权限和额度系统的一部分。
```

---

## 1. 当前状态

### 1.1 `D:\foamvm`

`D:\foamvm` 已经具备成为 PhysicsOS Cloud 的基础：

```text
Next.js 16 App Router
Supabase Auth
Supabase Postgres
Supabase Storage
E2B sandbox
OpenFOAM E2B template
邀请码 / run token
运行记录 / event stream / output files
```

当前核心表：

```text
profiles
invite_batches
run_tokens
run_consumptions
run_events
run_output_files
admin_audit_logs
```

当前网页流程：

```text
用户登录 foamvm
管理员生成 run token / 邀请码
用户兑换 token
POST /api/cfd 消耗一个 redeemed token
foamvm 启动 E2B sandbox
OpenFOAM 环境里执行任务
输出上传到 Supabase Storage
网页显示日志和 artifacts
```

这套机制很适合扩展成 CLI runner 的权限系统。

### 1.2 `D:\javascript\cae-agent\physicsOS`

当前 PhysicsOS 是 Python agent / CLI 项目雏形：

```text
physicsos/
  agents/
  backends/
  schemas/
  tools/
  workflows/

ARCHITECTURE.md
taps.md
vm.md
examples/
tests/
```

已有能力：

```text
TAPS-first workflow
geometry/mesh agent
taps-agent
verification-agent
postprocess-agent
knowledge-agent
full_solver_runner_manifest 生成
submit_full_solver_job 的 dry_run/local_echo/http scaffold
```

当前已经临时复制了：

```text
runners/foamvm
```

并验证过 OpenFOAM smoke run：

```text
blockMesh
icoFoam
foamToVTK
status=completed
OpenFOAM version=2412
```

但是这个复制目录只应该作为一次验证成果，不应该成为 Python 包的长期组成部分。

---

## 2. 为什么 Python 包里应该去掉本地 runner

如果 PhysicsOS 要成为可安装 CLI agent：

```bash
pip install physicsos
physicsos solve case.yaml
```

那它不应该要求用户同时安装：

```text
Node.js
Next.js
Supabase 配置
E2B template
OpenFOAM
Docker
```

否则 CLI 会变成一个难安装、难维护的混合网站项目。

正确边界应该是：

```text
PhysicsOS CLI:
  本地建模
  agent 编排
  TAPS 求解
  mesh/geometry 工具
  生成 full_solver_runner_manifest
  提交远程 runner
  下载结果
  verification / report

foamvm / PhysicsOS Cloud:
  账号
  邀请码
  配额
  计费/防欠费
  E2B_API_KEY
  OpenFOAM runtime
  artifacts 存储
  web dashboard
```

因此，`D:\javascript\cae-agent\physicsOS` 中长期不应保留：

```text
runners/foamvm
runners/foamvm/node_modules
runners/foamvm/.next
任何 .env.local
任何 E2B_API_KEY
任何 Supabase service role key
```

建议后续处理：

```text
短期：保留本地复制目录作为参考，但不纳入 Python packaging。
中期：把 runners/foamvm 从 physicsOS 仓库移除，改到 D:\foamvm 或单独 physicsos-cloud 仓库。
长期：physicsOS 仓库只保留 docs 中的远程 runner API 文档和客户端代码。
```

---

## 3. 邀请码如何更丝滑地用于 CLI agent

当前 foamvm 的邀请码/run token 逻辑已经很接近可用状态：

```text
管理员生成 run token
用户登录网页
用户 redeem token
token 状态从 unused -> redeemed
运行任务时 consume token
token 状态从 redeemed -> consumed
```

要让 CLI 使用更丝滑，关键是减少“复制粘贴 token”和“浏览器/CLI 割裂感”。

### 3.1 推荐体验：CLI 登录 + 浏览器确认 + 自动绑定邀请码额度

用户流程：

```bash
physicsos auth login
```

CLI 输出：

```text
打开浏览器：
https://foamvm.vercel.app/cli/activate

输入设备码：
AB12-CD34
```

用户在网页端登录并确认。

如果用户还没有额度，网页提示：

```text
你还没有可用运行次数。
请输入邀请码 / run token。
```

用户输入：

```text
SCM-XXXX-YYYY-ZZZZ
```

网页完成：

```text
redeem run token
approve CLI device
issue CLI access token
```

CLI 自动拿到：

```text
foamvm 用户身份
可用运行次数
短期 CLI token
默认 runner URL
```

用户之后直接：

```bash
physicsos solve case.yaml --runner foamvm --approve
```

不用再手动处理邀请码。

这个流程的关键点是：CLI 不直接接触邀请码，也不负责判断邀请码是否有效。邀请码只在 foamvm 网页端兑换，兑换结果进入用户账号的 run quota。CLI 只拿到一个可撤销的用户级访问令牌，然后每次提交 full-solver job 时由 foamvm 服务端扣减额度。

推荐状态机：

```text
CLI:
  physicsos auth login
  -> 请求 /api/cli/device/start
  -> 本地保存 device_code_hash 对应的轮询状态
  -> 打开 /cli/activate?user_code=AB12-CD34
  -> 轮询 /api/cli/device/poll

Web:
  用户登录 Supabase Auth
  -> 输入或确认 user_code
  -> 如果 available_runs = 0，先引导 redeem invite
  -> approve device
  -> 创建短期 CLI token

Runner:
  CLI 提交 /api/physicsos/jobs
  -> verifyCliToken
  -> 检查 redeemed run token / quota
  -> consume quota
  -> 启动 E2B full solver
```

这样邀请码体验会比较顺：

```text
新用户只需要做一次网页登录。
没有额度时在同一个激活页面兑换邀请码。
CLI 不要求用户复制邀请码。
CLI token 泄露后可单独撤销。
管理员仍然只管理 invite/run token，不需要管理每台机器。
```

### 3.2 邀请码与 CLI token 的关系

邀请码/run token 不应该直接等于 CLI token。

推荐关系：

```text
run token / invite code:
  用于增加用户运行额度
  可一次性兑换
  不用于 API 鉴权

CLI token:
  用于证明 CLI 属于某个用户
  可撤销
  可过期
  有 scopes
  不直接代表运行额度
```

这样更安全：

```text
邀请码泄露：最多被抢兑一次。
CLI token 泄露：可以撤销，不影响管理员发码机制。
E2B key：永远不发给 CLI。
```

---

## 4. 数据库建议

### 4.1 保留现有 run token 表

继续使用：

```text
invite_batches
run_tokens
run_consumptions
run_events
run_output_files
```

但建议给 `run_consumptions` 增加结构化字段：

```sql
alter table public.run_consumptions
add column if not exists source text not null default 'web_prompt',
add column if not exists backend text,
add column if not exists solver text,
add column if not exists job_manifest jsonb,
add column if not exists estimated_cost_usd numeric,
add column if not exists runtime_seconds integer,
add column if not exists output_bytes bigint;
```

`source` 可取：

```text
web_prompt
physicsos_cli
api
admin_test
```

这样网页 prompt 任务和 CLI 结构化任务可以共用运行记录。

### 4.2 新增 CLI token 表

```sql
create table if not exists public.cli_tokens (
  id uuid primary key default gen_random_uuid(),
  user_id uuid not null references public.profiles(id) on delete cascade,
  token_hash text not null unique,
  name text,
  scopes text[] not null default array['runner:submit', 'runner:read'],
  expires_at timestamptz,
  revoked_at timestamptz,
  last_used_at timestamptz,
  created_at timestamptz not null default now()
);
```

建议 scopes：

```text
runner:submit
runner:read
runner:cancel
artifacts:read
account:read
```

### 4.3 设备码登录表

CLI 登录只采用 device-code 流程：

```sql
create table if not exists public.cli_device_codes (
  id uuid primary key default gen_random_uuid(),
  user_code text not null unique,
  device_code_hash text not null unique,
  user_id uuid references public.profiles(id) on delete cascade,
  status text not null default 'pending'
    check (status in ('pending', 'approved', 'expired', 'revoked')),
  expires_at timestamptz not null,
  created_at timestamptz not null default now(),
  approved_at timestamptz
);
```

API：

```text
POST /api/cli/device/start
POST /api/cli/device/activate
POST /api/cli/device/poll
POST /api/cli/tokens/revoke
```

### 4.4 配额和防欠费表

第一版可以继续“一次运行消耗一个 redeemed run token”。

后续建议增加：

```sql
create table if not exists public.runner_plans (
  id text primary key,
  max_wall_time_seconds integer not null,
  max_output_mb integer not null,
  max_case_bundle_mb integer not null,
  max_parallel_jobs integer not null,
  allowed_backends text[] not null,
  allowed_solvers text[] not null,
  created_at timestamptz not null default now()
);

alter table public.profiles
add column if not exists runner_plan_id text references public.runner_plans(id);
```

还建议增加 usage ledger：

```sql
create table if not exists public.usage_ledger (
  id uuid primary key default gen_random_uuid(),
  user_id uuid not null references public.profiles(id) on delete cascade,
  consumption_id uuid references public.run_consumptions(id) on delete set null,
  provider text not null,
  backend text not null,
  estimated_cost_usd numeric,
  runtime_seconds integer,
  output_bytes bigint,
  status text not null,
  created_at timestamptz not null default now()
);
```

---

## 5. API 设计建议

### 5.1 CLI 认证 API

只保留设备码登录，不提供手动复制访问令牌的主流程：

```text
POST /api/cli/device/start
POST /api/cli/device/activate
POST /api/cli/device/poll
POST /api/cli/tokens/revoke
```

### 5.2 用户额度 API

CLI 需要知道自己能不能运行：

```text
GET /api/physicsos/me
```

返回：

```json
{
  "user": {
    "id": "...",
    "email": "user@example.com"
  },
  "quota": {
    "available_runs": 3,
    "consumed_runs": 5,
    "max_parallel_jobs": 1
  },
  "allowed_backends": ["openfoam"],
  "allowed_solvers": ["icoFoam", "simpleFoam"]
}
```

### 5.3 结构化 runner API

PhysicsOS CLI 不应调用 prompt-based 的 `/api/cfd`。

应使用：

```text
POST /api/physicsos/jobs
GET  /api/physicsos/jobs/:id
GET  /api/physicsos/jobs/:id/events?after=123
GET  /api/physicsos/jobs/:id/artifacts
POST /api/physicsos/jobs/:id/cancel
```

提交请求：

```json
{
  "schema_version": "physicsos.full_solver_job.v1",
  "problem_id": "problem:cavity",
  "backend": "openfoam",
  "backend_command": "icoFoam",
  "budget": {
    "max_wall_time_seconds": 600,
    "max_memory_gb": 8
  },
  "openfoam": {
    "solver": "icoFoam",
    "case_files": [
      {
        "path": "system/controlDict",
        "content": "..."
      }
    ]
  }
}
```

提交时服务端必须执行：

```text
verifyCliToken
load profile
check available redeemed run token
check max parallel jobs
validate manifest
clamp budget
consumeRunTokenForUser
create run_consumption(source='physicsos_cli')
start E2B job
append run_events
store run_output_files
write usage_ledger
```

---

## 6. 安全边界

### 6.1 密钥放哪里

服务端环境变量：

```text
E2B_API_KEY
E2B_TEMPLATE_ID
SUPABASE_SERVICE_ROLE_KEY
ANTHROPIC_API_KEY
```

CLI 本地只保存：

```text
foamvm runner URL
用户级 CLI token
```

浏览器只保存：

```text
Supabase user session
```

E2B sandbox 默认不应注入长期密钥。

### 6.2 禁止的行为

必须禁止：

```text
CLI 持有 E2B_API_KEY
CLI 持有 SUPABASE_SERVICE_ROLE_KEY
用户 manifest 传任意 shell command
普通用户使用 openfoam.run_commands
case_files 使用绝对路径
case_files 包含 ..
输出任意读取 /workspace 以外路径
日志里打印 Authorization header
日志里打印 E2B_API_KEY
```

### 6.3 OpenFOAM 白名单

第一版只开放：

```text
backend=openfoam
solver=icoFoam
solver=simpleFoam
```

后续再增加：

```text
pimpleFoam
rhoSimpleFoam
interFoam
chtMultiRegionFoam
```

每增加一个 solver，都要配套：

```text
case schema
文件模板
日志 parser
失败诊断
输出 artifact 规则
预算上限
```

---

## 7. 对 `D:\foamvm` 的重构建议

### 7.1 产品定位

把 foamvm 从“CFD 网站”升级为：

```text
PhysicsOS Cloud
```

页面结构建议：

```text
Dashboard
Runs
Artifacts
Invite Codes
CLI Access
Runner Settings
Admin
Apps
  CFD / OpenFOAM
  Surrogate Models
  TAPS Validation
```

当前 CFD 页面可以保留，但应变成 PhysicsOS Cloud 的一个 app。

### 7.2 API 分层

保留：

```text
/api/cfd
```

用于网页 prompt CFD 体验。

新增并重点维护：

```text
/api/physicsos/jobs
/api/physicsos/jobs/:id
/api/physicsos/jobs/:id/events
/api/physicsos/jobs/:id/artifacts
/api/physicsos/me
/api/cli/*
```

### 7.3 邀请码 UX

当前邀请码兑换流程应扩展成：

```text
用户登录
如果没有额度，Dashboard 显示 Redeem invite
兑换后 available runs 增加
Account -> CLI Access 显示 CLI 登录步骤
CLI auth login 时，如果用户无额度，网页 activate 页面直接提示兑换邀请码
```

CLI 侧体验：

```bash
physicsos auth login
physicsos account
```

输出：

```text
已登录：user@example.com
可用运行次数：3
默认 runner：https://foamvm.vercel.app
允许 backend：openfoam
```

### 7.4 E2B 运行逻辑

将 OpenFOAM runner 从 prompt 模式中拆出来：

```text
lib/physicsos/openfoam-runner.ts
lib/physicsos/manifest.ts
lib/physicsos/quota.ts
lib/physicsos/artifacts.ts
lib/physicsos/security.ts
```

`/api/cfd` 可以继续用 Claude Code prompt。

`/api/physicsos/jobs` 必须走结构化 manifest，不走 prompt。

---

## 8. 对 `D:\javascript\cae-agent\physicsOS` 的重构建议

### 8.1 去掉本地 runner

Python 包内不再保留 Node runner。

建议删除或移出：

```text
runners/foamvm
```

如果暂时需要保留，应明确：

```text
仅用于历史验证和参考
不进入 wheel
不进入默认安装
不作为 CLI 运行依赖
不保存任何 .env.local
```

长期更推荐：

```text
D:\foamvm 维护云端网站和 runner
D:\javascript\cae-agent\physicsOS 维护 Python CLI agent
```

两边通过 HTTP API 连接，不通过复制目录连接。

当前 `pyproject.toml` 的 setuptools 配置只包含：

```toml
[tool.setuptools.packages.find]
include = ["physicsos*"]
```

这意味着 `runners/foamvm` 即使暂时存在，也不会进入 Python wheel。后续仍建议显式增加排除项，防止维护者误改打包配置：

```toml
exclude = [
  "runners*",
  "data*",
  "models*",
  "configs*",
  "scratch*",
  "docs*",
  "scripts*",
  "tests*",
  "examples*",
]
```

更重要的是代码层边界：

```text
physicsos 包内不 import runners.*
physicsos 包内不 spawn node/npm/next
physicsos 包内不调用 e2b SDK
physicsos 包内不包含 OpenFOAM case 执行器
physicsos 包内不提供 local_runner mode 作为默认能力
```

如果需要本地调试 full solver，只能作为开发者文档中的外部服务运行：

```text
cd D:\foamvm
npm run dev
physicsos runner submit manifest.json --runner-url http://localhost:3000
```

这仍然是“远程 HTTP runner”，不是 Python 包内 runner。

### 8.2 CLI 配置

新增：

```text
~/.physicsos/config.toml
```

内容：

```toml
[cloud]
runner_url = "https://foamvm.vercel.app"
access_token = "psos_cli_..."
```

### 8.3 CLI 命令

建议命令：

```bash
physicsos auth login
physicsos auth status
physicsos account

physicsos solve case.yaml
physicsos solve case.yaml --runner foamvm --dry-run
physicsos solve case.yaml --runner foamvm --approve

physicsos runner submit manifest.json
physicsos runner status <job_id>
physicsos runner logs <job_id>
physicsos runner collect <job_id>
```

默认策略：

```text
本地 TAPS 优先
full solver 默认 dry-run
远程付费 runner 必须 --approve
没有登录或没有额度则拒绝提交
```

### 8.4 Python 中的 runner client

保留的是“客户端”，不是“本地 runner”：

```text
physicsos/cloud/foamvm_client.py
physicsos/cloud/auth.py
physicsos/cloud/config.py
```

职责：

```text
读取 ~/.physicsos/config.toml
提交 manifest 到 foamvm
轮询 job status
下载 artifacts
转换成 SolverResult
交给 verification-agent
```

不负责：

```text
启动 E2B
启动 Docker
启动 OpenFOAM
保存 E2B key
保存 Supabase service role
```

### 8.5 修改当前 solver tools

当前：

```text
submit_full_solver_job(mode="http")
```

应改为调用：

```text
POST {runner_url}/api/physicsos/jobs
```

而不是通用：

```text
POST {base_url}/jobs
```

响应应转成：

```text
SolverResult(status="partial" or "success")
artifacts=[remote artifact refs]
provenance.source="foamvm"
```

---

## 9. 推荐迁移步骤

### 阶段 1：把本地 runner 从 Python 项目中移除

在 `D:\javascript\cae-agent\physicsOS`：

```text
删除 runners/foamvm 或移动到 docs/archive/foamvm-proof
确认 pyproject.toml 只 package physicsos*
保留 vm.md 中的接口设计
保留 solver_tools 的 manifest/client scaffold
```

### 阶段 2：在 `D:\foamvm` 实现 CLI token

```text
新增 cli_tokens 表
新增 cli_device_codes 表
新增 /cli/activate 页面
新增 Account -> CLI Access 登录说明页面
新增 verifyCliToken helper
新增 GET /api/physicsos/me
```

### 阶段 3：让 `/api/physicsos/jobs` 复用邀请码额度

```text
Authorization: Bearer <cli_token>
verifyCliToken
consumeRunTokenForUser
create run_consumption(source='physicsos_cli')
start E2B OpenFOAM
store artifacts
return job_id
```

### 阶段 4：PhysicsOS CLI 接入 foamvm

```text
physicsos auth login
physicsos account
physicsos solve --runner foamvm --dry-run
physicsos solve --runner foamvm --approve
physicsos runner status/logs/collect
```

---

## 10. 最终决策

最终推荐架构：

```text
PhysicsOS CLI:
  Python package
  no Node.js dependency
  no local foamvm runner
  no E2B key
  no Docker requirement

foamvm / PhysicsOS Cloud:
  shared account
  invite code / run token
  CLI token
  quota and budget enforcement
  E2B/OpenFOAM runner
  artifact storage
  web dashboard
```

邀请码应该成为统一的“运行额度”系统：

```text
网页 prompt CFD 消耗额度
PhysicsOS CLI full solver 消耗额度
未来 SU2 / RunPod / Slurm runner 也消耗额度
```

Python 包里只保留：

```text
manifest generator
remote runner client
artifact parser
verification
report
```

不保留：

```text
本地 Next.js runner
本地 OpenFOAM VM
E2B_API_KEY
Supabase service role key
Docker runner 作为默认依赖
```

---

## 11. Supabase 后端迁移命令

foamvm 代码部署前，Supabase Postgres 需要创建 PhysicsOS CLI / Runner 相关表和字段。最直接的方式是在 Supabase Dashboard 中打开：

```text
Project -> SQL Editor -> New query
```

然后执行下面整段 SQL。

如果使用本地命令行，也可以把 SQL 保存为：

```text
D:\foamvm\supabase\physicsos_cloud_migration.sql
```

然后用任一方式执行：

```bash
supabase db push
```

或：

```bash
psql "$SUPABASE_DB_URL" -f supabase/physicsos_cloud_migration.sql
```

实际 SQL：

```sql
create extension if not exists pgcrypto;

alter table public.run_consumptions
add column if not exists source text not null default 'web_prompt',
add column if not exists backend text,
add column if not exists solver text,
add column if not exists job_manifest jsonb,
add column if not exists estimated_cost_usd numeric,
add column if not exists runtime_seconds integer,
add column if not exists output_bytes bigint;

create table if not exists public.cli_tokens (
  id uuid primary key default gen_random_uuid(),
  user_id uuid not null references public.profiles (id) on delete cascade,
  token_hash text not null unique,
  name text,
  scopes text[] not null default array[
    'runner:submit',
    'runner:read',
    'runner:cancel',
    'artifacts:read',
    'account:read'
  ],
  expires_at timestamptz,
  revoked_at timestamptz,
  last_used_at timestamptz,
  created_at timestamptz not null default now()
);

create table if not exists public.cli_device_codes (
  id uuid primary key default gen_random_uuid(),
  user_code text not null unique,
  device_code_hash text not null unique,
  user_id uuid references public.profiles(id) on delete cascade,
  status text not null default 'pending'
    check (status in ('pending', 'approved', 'expired', 'revoked')),
  expires_at timestamptz not null,
  created_at timestamptz not null default now(),
  approved_at timestamptz
);

create table if not exists public.runner_plans (
  id text primary key,
  max_wall_time_seconds integer not null,
  max_output_mb integer not null,
  max_case_bundle_mb integer not null,
  max_parallel_jobs integer not null,
  allowed_backends text[] not null,
  allowed_solvers text[] not null,
  created_at timestamptz not null default now()
);

alter table public.profiles
add column if not exists runner_plan_id text references public.runner_plans(id);

insert into public.runner_plans (
  id,
  max_wall_time_seconds,
  max_output_mb,
  max_case_bundle_mb,
  max_parallel_jobs,
  allowed_backends,
  allowed_solvers
)
values (
  'private_beta',
  1800,
  512,
  32,
  1,
  array['openfoam'],
  array['icoFoam', 'simpleFoam']
)
on conflict (id) do nothing;

create table if not exists public.usage_ledger (
  id uuid primary key default gen_random_uuid(),
  user_id uuid not null references public.profiles(id) on delete cascade,
  consumption_id uuid references public.run_consumptions(id) on delete set null,
  provider text not null,
  backend text not null,
  estimated_cost_usd numeric,
  runtime_seconds integer,
  output_bytes bigint,
  status text not null,
  created_at timestamptz not null default now()
);

alter table public.cli_tokens enable row level security;
alter table public.cli_device_codes enable row level security;
alter table public.runner_plans enable row level security;
alter table public.usage_ledger enable row level security;

drop policy if exists "cli_tokens_select_own" on public.cli_tokens;
create policy "cli_tokens_select_own"
on public.cli_tokens
for select
to authenticated
using (user_id = auth.uid());

drop policy if exists "runner_plans_select_all" on public.runner_plans;
create policy "runner_plans_select_all"
on public.runner_plans
for select
to authenticated
using (true);

drop policy if exists "usage_ledger_select_own" on public.usage_ledger;
create policy "usage_ledger_select_own"
on public.usage_ledger
for select
to authenticated
using (user_id = auth.uid());
```

执行完成后，需要确认以下服务端环境变量仍只存在于 Vercel / 服务端，不要放进 CLI：

```text
NEXT_PUBLIC_SUPABASE_URL
NEXT_PUBLIC_SUPABASE_ANON_KEY
SUPABASE_SERVICE_ROLE_KEY
E2B_API_KEY
E2B_TEMPLATE_ID
ANTHROPIC_API_KEY
ANTHROPIC_BASE_URL
```

还需要重新部署 foamvm：

```bash
git pull origin main
npm install
npm run build
```

Vercel 部署时通常只需要 push 后自动部署；如果数据库没迁移，`/api/cli/device/start`、`/api/physicsos/me`、`/api/physicsos/jobs` 会因为缺表而失败。
