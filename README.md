<div align="center">

# snowmarten
🛡️ The missing security layer for AI agents. Prompt injection protection, tool authorization, sandboxed execution — like OAuth, but for LLM agents.


[![PyPI version](https://badge.fury.io/py/snowmarten.svg)](https://badge.fury.io/py/snowmarten)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://python.org)
[![Discord](https://img.shields.io/discord/XXXXXX?color=7289DA&label=Discord)](https://discord.gg/snowmarten)
[![GitHub Stars](https://img.shields.io/github/stars/snowmarten/snowmarten)](https://github.com/snowmarten/snowmarten)

**Snowmarten is an open-source security framework for LLM agents.**  
It prevents prompt injection, tool abuse, jailbreaks, and data exfiltration  
across LangChain, LangGraph, OpenAI, and LlamaIndex agents.

[Documentation](https://docs.snowmarten.dev) · 
[Quickstart](#quickstart) · 
[Attack Demos](examples/) · 
[Discord Community](https://discord.gg/snowmarten)

</div>

---

## ⚠️ The Problem

You've built an AI agent. It has access to your database, filesystem, 
APIs, and maybe other agents. Now ask yourself:

- What happens if a user submits a prompt injection in a support ticket?
- What if your agent scrapes a webpage containing hidden instructions?  
- What if a tool call is manipulated to exfiltrate customer PII?
- What if code execution is used to escape your application sandbox?

**None of the major agent frameworks (LangChain, OpenAI, LlamaIndex) 
have a security model.** They are orchestration frameworks, not security 
frameworks. Snowmarten fills that gap.

> Snowmarten is to AI agents what OAuth 2.0 is to APIs.

---

## ✅ The Solution

Snowmarten wraps your existing agent with a multi-layer security stack:

```
Input → [Prompt Firewall] → [Policy Engine] → [Tool Permission Layer] → [Sandbox Runtime] → Agent → [Output Filter] → Response
```

Every input, tool call, and output is validated against security 
policies before execution.

---

## 🏗️ Architecture

```
User Input
    │
    ▼
┌──────────────────┐
│  Prompt Firewall │  ← Injection detection, jailbreak patterns, 
└────────┬─────────┘    semantic similarity scoring
         │
         ▼
┌──────────────────┐
│  Policy Engine   │  ← YAML policies, risk scoring, allow/deny/audit
└────────┬─────────┘
         │
    ┌────┴────┐
    ▼         ▼
┌───────┐  ┌──────────────────────┐
│  LLM  │  │ Tool Permission Layer│  ← RBAC, param validation,
└───┬───┘  └──────────┬───────────┘    rate limits, allowlists
    │                 │
    └────────┬────────┘
             ▼
┌─────────────────────────┐
│  Sandbox Runtime        │  ← Isolated execution, resource limits,
└────────────┬────────────┘    network egress control
             │
             ▼
┌─────────────────────────┐
│  Telemetry & Audit Log  │  ← Full trace, anomaly detection,
└─────────────────────────┘    SIEM integration

┌─────────────────────────────────────────────────────────────┐ │ Your Application │ └──────────────────────────┬──────────────────────────────────┘ │ ┌──────────────────────────▼──────────────────────────────────┐ │ Snowmarten Layer │ │ │ │ ┌─────────────┐ ┌─────────────┐ ┌──────────────────┐ │ │ │ Prompt │ │ Policy │ │ Tool │ │ │ │ Firewall │──▶│ Engine │──▶│ Permission │ │ │ │ │ │ │ │ Layer │ │ │ └─────────────┘ └─────────────┘ └──────────────────┘ │ │ │ │ ┌─────────────────────────────┐ ┌──────────────────────┐ │ │ │ Sandbox Runtime │ │ Telemetry & Audit │ │ │ │ (gVisor / Docker / rPython)│ │ Logs │ │ │ └─────────────────────────────┘ └──────────────────────┘ │ └──────────────────────────┬──────────────────────────────────┘ │ ┌──────────────────────────▼──────────────────────────────────┐ │ LangChain / LangGraph / OpenAI / LlamaIndex │ └─────────────────────────────────────────────────────────────┘
```


---

## ✨ Features

- 🔥 **Prompt Firewall** — Multi-stage injection detection (syntactic + semantic + contextual)
- 🔐 **Tool Permission Layer** — OAuth-style scopes for every tool
- 📋 **Policy Engine** — Declarative YAML/Python security policies
- 📦 **Sandbox Runtime** — Isolated code execution (Docker, gVisor, rPython)
- 📊 **Telemetry & Audit** — Full audit trail, SIEM integration, real-time alerts
- 🔗 **Framework Agnostic** — LangChain, LangGraph, OpenAI, LlamaIndex adapters
- ⚡ **Low Overhead** — <5ms median latency on firewall checks
- 🧪 **Attack Test Suite** — 200+ attack scenarios for CI/CD testing

---

## 🚀 Quickstart

### Installation

```bash
pip install snowmarten
# With extras:
pip install snowmarten[langchain]     # LangChain integration
pip install snowmarten[openai]        # OpenAI integration
pip install snowmarten[sandbox-docker] # Docker sandbox support
```

### Basic Usage

```python
from snowmarten import SecureAgent
from snowmarten.policies import NoExfiltrationPolicy, PIIProtectionPolicy
from snowmarten.tools import ToolPermission, PermissionScope
from langchain.tools import DuckDuckGoSearchRun, ReadFileTool

agent = SecureAgent(
    model="gpt-4",
    policies=[
        NoExfiltrationPolicy(),
        PIIProtectionPolicy(action="redact"),
    ],
    tools=[
        ToolPermission(
            tool=ReadFileTool(),
            scope=PermissionScope(allowed_paths=["/app/public/*"])
        ),
        ToolPermission(
            tool=DuckDuckGoSearchRun(),
            scope=PermissionScope(rate_limit="20/hour")
        ),
    ],
    sandbox_enabled=True,
)

response = agent.run("Summarize the public documentation")
```

### Zero-Config Security (Sensible Defaults)


```python
from snowmarten import secure_langchain_agent

# Wrap any existing LangChain agent with one line
secure_agent = secure_langchain_agent(
    your_existing_agent,
    preset="production"  # Applies recommended security profile
)
```

## 🔒 Security Model
Snowmarten implements a defense-in-depth security model with five independent layers. An attack must bypass ALL layers to succeed.

```
Layer 1: Prompt Firewall    — stops known attacks before LLM sees them
Layer 2: Policy Engine      — stops policy-violating actions at decision time
Layer 3: Tool Permissions   — stops unauthorized tool calls at execution time
Layer 4: Sandbox Runtime    — limits blast radius if execution proceeds
Layer 5: Audit & Anomaly    — detects attacks that bypassed all layers
```

## Threat Coverage

| Threat | Detection Method | Action |
| --- | --- | --- |
| Prompt Injection | Pattern + Semantic detection | Block / Sanitize |
| Indirect Injection | Document scanning | Strip / Alert |
| Tool Abuse | Permission scope validation | Block |
| Data Exfiltration | Domain/content policy | Block + Alert |
| Code Escape | Sandbox isolation | Contain |
| Jailbreak | Policy engine + context tracking | Block |
| Multi-hop Injection | Inter-agent message scanning | Quarantine |
| PII Leakage | Output filtering | Redact |

## Policy Precedence

```
DENY (explicit) > REQUIRE_APPROVAL > SANDBOX > RATE_LIMIT > ALLOW
```

## 🔗 Integrations

| Framework | Status | Install |
| --- | --- | --- |
| LangChain | ✅ Stable | pip install snowmarten[langchain] |
| LangGraph | ✅ Stable | pip install snowmarten[langgraph] |
| OpenAI Assistants | ✅ Stable | pip install snowmarten[openai] |
| LlamaIndex | ✅ Stable | pip install snowmarten[llamaindex] |
| AutoGPT | 🚧 Beta | pip install snowmarten[autogpt] |
| CrewAI | 🚧 Beta | pip install snowmarten[crewai] |
| Semantic Kernel | 📋 Planned | — |

```python
# LangChain integration
from snowmarten.integrations.langchain import SecureLangChainAgent
from langchain.agents import create_openai_tools_agent

base_agent = create_openai_tools_agent(llm, tools, prompt)
secure_agent = SecureLangChainAgent(base_agent, policies=[...])

# LlamaIndex integration
from snowmarten.integrations.llamaindex import SecureQueryEngine
from llama_index.core import VectorStoreIndex

index = VectorStoreIndex.from_documents(docs)
secure_engine = SecureQueryEngine(
    index.as_query_engine(),
    scan_retrieved_nodes=True,  # Scan for injection before context injection
    policies=[...]
)

```

## 🗺️ Roadmap
v0.1 — Foundation (Current)
- [x] Prompt Firewall (syntactic)
- [x] Tool Permission Layer
- [x] Policy Engine (YAML)
- [x] LangChain integration
- [x] Basic audit logging

v0.2 — Intelligence
- [ ] Semantic injection detector (fine-tuned model)
- [ ] LangGraph + multi-agent support
- [ ] OpenAI Assistants integration
- [ ] Policy-as-code (Python DSL)

v0.3 — Production
- [ ] Docker/gVisor sandbox
- [ ] SIEM integration (Splunk, Datadog, PagerDuty)
- [ ] Policy server (centralized management)
- [ ] Real-time dashboard

v1.0 — Standard
- [ ] AgentSec Policy Specification v1.0
- [ ] Certification program
- [ ] Cloud-hosted policy management


## 🤝 Contributing
We welcome contributions in these areas:

Attack Research — Submit new attack vectors via research/attacks/
Detector Development — Improve injection detection accuracy
Framework Integrations — Add support for new agent frameworks
Policy Library — Contribute reusable policy templates
Documentation — Attack walkthroughs, deployment guides
See CONTRIBUTING.md [blocked] for guidelines.

Good first issues: labeled good-first-issue

## 📄 License
MIT License — free for commercial and open-source use.

Built by the AI security community. Star ⭐ if you think AI agents need security.

