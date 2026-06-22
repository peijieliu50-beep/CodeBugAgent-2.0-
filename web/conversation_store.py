# -*- coding: utf-8 -*-
"""
web/conversation_store.py
=========================
对话持久化：把每次会话（用户提问、Agent 推理步骤、最终回答、以及供模型接续的
完整消息历史）保存为本地 JSON 文件，支持列出 / 加载 / 新建 / 删除，并能自动恢复
最近一次对话——这样"关掉再打开"时 Agent 仍记得之前聊了什么。

存储位置：logs/conversations/<会话ID>.json
每个会话结构：
{
  "id": "conv_20260619_153000",
  "title": "首问的前 20 字",
  "created": "2026-06-19T15:30:00",
  "updated": "2026-06-19T15:45:00",
  "turns": [                      # 用于前端回放展示
    {"user": "...", "steps": [ReActStep.to_dict(), ...], "final": "..."}
  ],
  "llm_messages": [ ... ]         # 不含 system 的对话消息，传给引擎做多轮上下文
}
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from config.config import CONFIG

_STORE_DIR = CONFIG.paths.logs / "conversations"
_STORE_DIR.mkdir(parents=True, exist_ok=True)


def _now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def new_conversation_id() -> str:
    return "conv_" + datetime.now().strftime("%Y%m%d_%H%M%S")


def empty_conversation(conv_id: Optional[str] = None) -> Dict[str, Any]:
    """构造一个空会话对象。"""
    cid = conv_id or new_conversation_id()
    return {
        "id": cid,
        "title": "新对话",
        "created": _now_iso(),
        "updated": _now_iso(),
        "turns": [],
        "llm_messages": [],
    }


def _path(conv_id: str) -> Path:
    return _STORE_DIR / f"{conv_id}.json"


def save_conversation(conv: Dict[str, Any]) -> None:
    """保存（覆盖写）一个会话。"""
    conv["updated"] = _now_iso()
    # 标题取首个用户问题前 20 字
    if conv.get("turns"):
        first = conv["turns"][0].get("user", "").strip().replace("\n", " ")
        conv["title"] = (first[:20] + "…") if len(first) > 20 else (first or "新对话")
    try:
        _path(conv["id"]).write_text(
            json.dumps(conv, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    except Exception:  # noqa: BLE001
        pass


def load_conversation(conv_id: str) -> Optional[Dict[str, Any]]:
    """按 ID 加载会话。"""
    fp = _path(conv_id)
    if not fp.exists():
        return None
    try:
        return json.loads(fp.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return None


def list_conversations() -> List[Dict[str, str]]:
    """列出全部会话（按更新时间倒序），返回 [{id, title, updated}]。"""
    items = []
    for fp in _STORE_DIR.glob("conv_*.json"):
        try:
            data = json.loads(fp.read_text(encoding="utf-8"))
            items.append({
                "id": data.get("id", fp.stem),
                "title": data.get("title", "未命名"),
                "updated": data.get("updated", ""),
                "turns": str(len(data.get("turns", []))),
            })
        except Exception:  # noqa: BLE001
            continue
    items.sort(key=lambda x: x["updated"], reverse=True)
    return items


def delete_conversation(conv_id: str) -> None:
    fp = _path(conv_id)
    if fp.exists():
        try:
            fp.unlink()
        except Exception:  # noqa: BLE001
            pass


def latest_conversation_id() -> Optional[str]:
    """返回最近更新的会话 ID（用于自动恢复）。"""
    items = list_conversations()
    return items[0]["id"] if items else None


def append_turn(
    conv: Dict[str, Any],
    user_text: str,
    steps: List[Dict[str, Any]],
    final_answer: str,
    llm_messages: List[Dict[str, Any]],
) -> None:
    """把一轮完整问答追加进会话，并更新供模型接续的消息历史。

    Args:
        conv: 会话对象（原地修改）
        user_text: 本轮用户输入
        steps: 本轮 ReActStep.to_dict() 列表
        final_answer: 本轮最终回答
        llm_messages: 引擎返回的完整 messages（含 system），这里去掉 system 后存储
    """
    conv["turns"].append({
        "user": user_text,
        "steps": steps,
        "final": final_answer,
    })
    # 存储不含 system 的消息，供下轮作为 history（build_messages 会重新加 system）
    conv["llm_messages"] = [m for m in (llm_messages or []) if m.get("role") != "system"]


def history_for_engine(conv: Dict[str, Any]) -> List[Dict[str, Any]]:
    """取出可直接作为 ReActEngine.run(history=...) 的历史消息。

    注意：只保留 role 为 user/assistant 的纯文本消息，剥离 tool_calls 与 tool 结果，
    避免把上一轮残留的"工具调用但未配对结果"带进新一轮导致接口报错；
    这样既保留对话记忆，又保证消息结构干净。
    """
    clean: List[Dict[str, Any]] = []
    for m in conv.get("llm_messages", []):
        role = m.get("role")
        if role == "user":
            clean.append({"role": "user", "content": m.get("content", "")})
        elif role == "assistant":
            content = m.get("content", "")
            if content:  # 跳过只含 tool_calls、无文本的 assistant 消息
                clean.append({"role": "assistant", "content": content})
    return clean
