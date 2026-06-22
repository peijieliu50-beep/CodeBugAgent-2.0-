# -*- coding: utf-8 -*-
"""
web/app.py
==========
ReAct PyTorch 代码助手 —— Streamlit 交互界面。

界面要点：
    - 全宽聊天区，输入框固定在页面底部（不随内容生成而移动）；文件随输入框一起上传。
    - 每轮的推理过程（思考 + 工具调用）收进可折叠面板，默认折叠、点箭头展开，避免刷屏；
      最终结论始终展开显示。
    - 侧边栏：配置（密钥/参数/知识库）+ 对话管理（新建/切换/删除历史对话）。
    - 多轮记忆 + 本地持久化：关闭再打开自动恢复最近一次对话。

启动：
    streamlit run web/app.py
"""

import os
import sys
from datetime import datetime
from pathlib import Path

# 确保可导入项目根模块
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import streamlit as st

from config.config import CONFIG
from web.deploy_guard import (
    inject_api_key, check_password, consume_quota, remaining_quota,
    get_daily_limit, get_access_password,
)
from web import conversation_store as cs

# ---- 页面基础配置 ----
st.set_page_config(page_title="ReAct PyTorch 代码助手", layout="wide")

inject_api_key()
check_password()


# ============================================================
# 辅助
# ============================================================
def _esc(text: str) -> str:
    return str(text).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def _preview(text: str, n: int = 40) -> str:
    """取一行预览（折叠面板标题用）。"""
    t = " ".join(str(text).split())
    return (t[:n] + "…") if len(t) > n else t


def render_steps(steps):
    """把一轮的推理步骤渲染进可折叠面板。steps 为 dict 列表。"""
    if not steps:
        return
    # 取最后一步思考做预览，让用户大致看到在想什么
    last_thought = ""
    for s in steps:
        if s.get("thought"):
            last_thought = s["thought"]
    title = f"推理过程（{len(steps)} 步）" + (f" · {_preview(last_thought)}" if last_thought else "")
    with st.expander(title, expanded=False):
        for s in steps:
            idx = s.get("index", "?")
            if s.get("thought") and not s.get("is_final"):
                st.markdown(f"**第 {idx} 轮 · 思考**")
                st.markdown(
                    f"<div style='white-space:pre-wrap;color:#444;font-size:14px;"
                    f"border-left:3px solid #b9a7d6;padding-left:8px;margin:2px 0 8px'>"
                    f"{_esc(s['thought'])}</div>", unsafe_allow_html=True)
            if s.get("action"):
                import json
                args = json.dumps(s.get("action_input", {}), ensure_ascii=False)
                obs = s.get("observation", "")
                obs = obs if len(obs) < 1000 else obs[:1000] + " …(截断)"
                st.markdown(
                    f"<div style='font-size:13px;border-left:3px solid #9cc79e;"
                    f"padding-left:8px;margin:2px 0 8px'>"
                    f"<b>调用工具</b> <code>{_esc(s['action'])}</code> "
                    f"<span style='color:#888'>{_esc(args)}</span><br>"
                    f"<b>结果</b><br><span style='white-space:pre-wrap;color:#555'>"
                    f"{_esc(obs)}</span></div>", unsafe_allow_html=True)


# ============================================================
# 会话状态：当前对话对象
# ============================================================
if "conv" not in st.session_state:
    last = cs.latest_conversation_id()
    st.session_state.conv = cs.load_conversation(last) if last else cs.empty_conversation()
    if st.session_state.conv is None:
        st.session_state.conv = cs.empty_conversation()

conv = st.session_state.conv


# ============================================================
# 侧边栏：配置 + 对话管理
# ============================================================
with st.sidebar:
    st.title("ReAct 代码助手")
    st.caption("人工智能导论 · 课程设计 | 科研工具与代码开发类 Agent")
    st.divider()

    # ---- 对话管理 ----
    st.subheader("对话")
    if st.button("＋ 新建对话", use_container_width=True):
        cs.save_conversation(st.session_state.conv)
        st.session_state.conv = cs.empty_conversation()
        st.rerun()

    convs = cs.list_conversations()
    if convs:
        labels = {c["id"]: f"{c['title']}  ({c['turns']}轮)" for c in convs}
        ids = [c["id"] for c in convs]
        cur_id = conv.get("id")
        idx = ids.index(cur_id) if cur_id in ids else 0
        chosen = st.selectbox("历史对话", ids, index=idx,
                              format_func=lambda i: labels.get(i, i))
        c1, c2 = st.columns(2)
        with c1:
            if st.button("加载", use_container_width=True) and chosen != cur_id:
                cs.save_conversation(st.session_state.conv)
                loaded = cs.load_conversation(chosen)
                if loaded:
                    st.session_state.conv = loaded
                    st.rerun()
        with c2:
            if st.button("删除", use_container_width=True):
                cs.delete_conversation(chosen)
                if chosen == cur_id:
                    st.session_state.conv = cs.empty_conversation()
                st.rerun()

    st.divider()

    # ---- 配置 ----
    st.subheader("配置")
    _deployed = bool(get_access_password())
    if _deployed:
        st.success("在线演示模式")
        if get_daily_limit() > 0:
            st.caption(f"今日剩余调用次数：{remaining_quota()} / {get_daily_limit()}")
        if not CONFIG.validate_api():
            st.error("服务端未配置 API 密钥，请联系作者。")
    else:
        api_key = st.text_input("DeepSeek API 密钥", value=os.getenv("DEEPSEEK_API_KEY", ""),
                                type="password", help="留空则读取环境变量 DEEPSEEK_API_KEY")
        if api_key:
            os.environ["DEEPSEEK_API_KEY"] = api_key
            CONFIG.api_key = api_key

    temperature = st.slider("采样温度", 0.0, 1.0, CONFIG.temperature, 0.1)
    max_iter = st.slider("最大迭代轮数", 3, 25, CONFIG.max_iterations, 1)
    CONFIG.temperature = temperature
    CONFIG.max_iterations = max_iter

    st.divider()

    # ---- 知识库 ----
    st.subheader("知识库")
    try:
        from tools.knowledge_tools import get_engine
        st.caption(get_engine().stats())
    except Exception as e:  # noqa: BLE001
        st.caption(f"知识库未就绪: {e}")

    kb_files = st.file_uploader("上传资料入库（PDF/DOCX/MD/TXT）",
                                type=["pdf", "docx", "md", "txt"],
                                accept_multiple_files=True, key="kb_upload")
    if kb_files and st.button("导入知识库", use_container_width=True):
        from tools.file_parser import parse_file
        from tools.knowledge_tools import get_engine
        added = []
        for f in kb_files:
            dst = CONFIG.paths.knowledge_docs / f.name
            dst.write_bytes(f.getbuffer())
            text = parse_file(str(dst))
            if not text.startswith("["):
                added.append((f.name, text))
        if added:
            st.success(get_engine().add_documents(added))
        else:
            st.warning("未能解析上传的文件")

    st.divider()
    # ---- 报告导出 ----
    if conv.get("turns"):
        from web.report_export import build_markdown_report
        from core.react_engine import ReActResult, ReActStep
        last_turn = conv["turns"][-1]
        steps_obj = [ReActStep(**s) for s in last_turn.get("steps", [])]
        res = ReActResult(True, last_turn.get("final", ""), steps_obj,
                          len(steps_obj), "final_answer")
        md = build_markdown_report(res, last_turn.get("user", ""),
                                   timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        st.download_button("导出调试报告 (Markdown)", md,
                           file_name="调试报告_" + datetime.now().strftime("%Y%m%d_%H%M%S") + ".md",
                           mime="text/markdown", use_container_width=True)


# ============================================================
# 主区：标题 + 对话
# ============================================================
st.markdown("### 基于 ReAct 范式的 PyTorch 深度学习代码助手")

# ---- 回放历史对话 ----
for turn in conv.get("turns", []):
    with st.chat_message("user"):
        st.markdown(_esc(turn.get("user", "")))
    with st.chat_message("assistant"):
        render_steps(turn.get("steps", []))
        if turn.get("final"):
            st.markdown(turn["final"])


# ============================================================
# 输入区（chat_input 自动固定在页面底部；文件随输入一起上传）
# ============================================================
_chat = st.chat_input(
    "描述你的 PyTorch 代码问题，或要求按需修改代码…",
    accept_file="multiple",
    file_type=["py", "txt", "json", "log", "md", "csv", "png", "jpg", "jpeg"],
)

# 兼容不同 Streamlit 版本：accept_file 返回对象(.text/.files) 或纯字符串
user_text = ""
up_files = []
if _chat:
    if isinstance(_chat, str):
        user_text = _chat
    else:
        user_text = getattr(_chat, "text", "") or ""
        up_files = getattr(_chat, "files", []) or []


if user_text:
    # 解析上传文件为上下文
    extra_context = ""
    if up_files:
        from tools.file_parser import parse_file
        parts = []
        for f in up_files:
            dst = CONFIG.paths.workspace / f.name
            dst.write_bytes(f.getbuffer())
            text = parse_file(str(dst))
            parts.append(f"【文件 {f.name}】\n{text[:3000]}")
        extra_context = "\n\n".join(parts)

    with st.chat_message("user"):
        st.markdown(_esc(user_text))
        if up_files:
            st.caption("已上传：" + "、".join(f.name for f in up_files))

    if not CONFIG.validate_api():
        st.error("未配置 DeepSeek API 密钥，请在侧边栏填写或设置环境变量 DEEPSEEK_API_KEY。")
    elif not consume_quota():
        st.error(f"今日演示调用次数已达上限（{get_daily_limit()} 次/天），请明天再来。")
    else:
        with st.chat_message("assistant"):
            status = st.status("正在思考…", expanded=True)
            live_steps = []

            def _on_step(step):
                live_steps.append(step)
                with status:
                    if step.thought and not step.is_final:
                        status.update(label=f"第 {step.index} 轮：{_preview(step.thought, 30)}")
                        st.markdown(f"**第 {step.index} 轮 · 思考**")
                        st.markdown(
                            f"<div style='white-space:pre-wrap;color:#444;font-size:14px;"
                            f"border-left:3px solid #b9a7d6;padding-left:8px;margin:2px 0 8px'>"
                            f"{_esc(step.thought)}</div>", unsafe_allow_html=True)
                    if step.action:
                        import json
                        args = json.dumps(step.action_input, ensure_ascii=False)
                        obs = step.observation if len(step.observation) < 1000 else step.observation[:1000] + " …(截断)"
                        st.markdown(
                            f"<div style='font-size:13px;border-left:3px solid #9cc79e;"
                            f"padding-left:8px;margin:2px 0 8px'>"
                            f"<b>调用工具</b> <code>{_esc(step.action)}</code> "
                            f"<span style='color:#888'>{_esc(args)}</span><br>"
                            f"<b>结果</b><br><span style='white-space:pre-wrap;color:#555'>"
                            f"{_esc(obs)}</span></div>", unsafe_allow_html=True)

            # 仅注册"代码助手"相关工具（不含训练）
            import tools.file_tools, tools.log_tools, tools.knowledge_tools  # noqa
            from core.react_engine import ReActEngine

            try:
                engine = ReActEngine(max_iterations=max_iter, on_step=_on_step)
                history = cs.history_for_engine(conv)
                result = engine.run(user_text, history=history, extra_context=extra_context)
                status.update(label=f"完成（共 {result.iterations} 轮）", state="complete", expanded=False)
                st.markdown(result.final_answer)

                # 落盘记忆
                cs.append_turn(conv, user_text,
                               [s.to_dict() for s in result.steps],
                               result.final_answer, result.messages)
                cs.save_conversation(conv)
                st.rerun()
            except Exception as e:  # noqa: BLE001
                status.update(label="执行出错", state="error")
                st.error(f"执行出错：{e}")
