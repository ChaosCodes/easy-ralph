"""
Interactive terminal selector for Ralph SDK.

Provides @clack/prompts style interactive selection UI:
- Arrow key navigation with highlight bar
- Direct typing to switch to free text input
- Multi-select with space toggle
- Collapsed confirmation view

Adapted from claude_ask_user_demo.py.
Unix only (termios/tty). Falls back to simple input() on unsupported platforms.
"""

import os
import sys
import unicodedata

try:
    import termios
    import tty
    from select import select as io_select

    HAS_TERMIOS = True
except ImportError:
    HAS_TERMIOS = False


# ── ANSI style constants ────────────────────────────────────────────────

CYAN = "\x1b[36m"
GREEN = "\x1b[32m"
YELLOW = "\x1b[33m"
BOLD = "\x1b[1m"
DIM = "\x1b[2m"
GRAY = "\x1b[90m"
RESET = "\x1b[0m"
HIDE_CUR = "\x1b[?25l"
SHOW_CUR = "\x1b[?25h"
BG_ACTIVE = "\x1b[48;5;236m"

BAR = f"  {CYAN}│{RESET}"


# ── Terminal raw input ───────────────────────────────────────────────────


def _read_key(fd: int) -> str | None:
    """Read a single keypress, handling arrow key escape sequences and UTF-8."""
    b = os.read(fd, 1)

    if b == b"\x1b":
        if io_select([fd], [], [], 0.05)[0]:
            b2 = os.read(fd, 1)
            if b2 == b"[" and io_select([fd], [], [], 0.05)[0]:
                b3 = os.read(fd, 1)
                return {b"A": "up", b"B": "down", b"C": "right", b"D": "left"}.get(b3)
        return "escape"

    if b in (b"\r", b"\n"):
        return "enter"
    if b in (b"\x7f", b"\x08"):
        return "backspace"
    if b == b"\x03":
        raise KeyboardInterrupt
    if b == b" ":
        return "space"

    first = b[0]
    if first & 0x80 == 0:
        ch = b.decode()
        return ch if ch.isprintable() else None
    elif first & 0xE0 == 0xC0:
        b += os.read(fd, 1)
    elif first & 0xF0 == 0xE0:
        b += os.read(fd, 2)
    elif first & 0xF8 == 0xF0:
        b += os.read(fd, 3)
    else:
        return None
    try:
        return b.decode()
    except UnicodeDecodeError:
        return None


# ── Helpers ──────────────────────────────────────────────────────────────


def _visual_width(s: str) -> int:
    """Terminal display width (CJK characters count as 2)."""
    return sum(2 if unicodedata.east_asian_width(c) in ("F", "W") else 1 for c in s)


def _pad_to_width(s: str, width: int) -> str:
    """Pad string to target visual width."""
    return s + " " * max(0, width - _visual_width(s))


# ── Normalize options ────────────────────────────────────────────────────


def _normalize_options(options: list) -> list[dict]:
    """Normalize options to list of {"label": ..., "description": ...} dicts."""
    result = []
    for opt in options:
        if isinstance(opt, str):
            result.append({"label": opt})
        elif isinstance(opt, dict):
            label = opt.get("label", opt.get("name", str(opt)))
            desc = opt.get("description", "")
            result.append({"label": label, "description": desc})
        else:
            result.append({"label": str(opt)})
    return result


# ── Interactive selector ─────────────────────────────────────────────────


def display_question(question: dict) -> str:
    """
    @clack/prompts style interactive selector.

    Features:
    - Cyan vertical bar on left
    - Arrow key navigation with background highlight
    - Direct typing switches to text input
    - Multi-select with space toggle (◉/○)
    - Collapsed confirmation view

    Args:
        question: Dict with keys:
            - question (str): The question text
            - header (str, optional): Short label
            - options (list): Options as strings or {"label", "description"} dicts
            - multiSelect (bool, optional): Enable multi-select

    Returns:
        The selected answer as a string.
    """
    if not HAS_TERMIOS:
        return _display_question_fallback(question)

    header = question.get("header", "")
    text = question.get("question", "")
    raw_options = question.get("options", [])
    multi = question.get("multiSelect", False)
    options = _normalize_options(raw_options)

    if header:
        title_styled = f"{YELLOW}{header}{RESET}  {BOLD}{text}{RESET}"
    else:
        title_styled = f"{BOLD}{text}{RESET}"

    type_idx = len(options)  # "Type something." index

    cursor = 0
    typed_text = ""
    checked: set[int] = set()
    header_lines = 2  # ╭ title + │ blank
    prev_lines = 0
    prev_cursor_offset = 0

    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)

    # Highlight bar width
    num_w = len(str(len(options)))
    max_label_w = max((_visual_width(o["label"]) for o in options), default=0)
    bar_w = max(max_label_w + num_w + 7, 22)

    # Keyboard hint
    if multi:
        hint = f"{DIM}↑↓ move  space select  enter confirm{RESET}"
    else:
        hint = f"{DIM}↑↓ move  enter confirm{RESET}"

    def render():
        nonlocal prev_lines, prev_cursor_offset
        out = sys.stdout
        out.write(HIDE_CUR)

        if prev_lines > 0:
            up = prev_lines - 1 - prev_cursor_offset
            out.write("\r")
            if up > 0:
                out.write(f"\x1b[{up}A")
            out.write("\x1b[J")

        lines = 0
        for i, opt in enumerate(options):
            active = cursor == i
            label = opt["label"]
            desc = opt.get("description", "")
            num = f"{i + 1}."

            if multi:
                chk_char = "◉" if i in checked else "○"
                if active:
                    content = f" ❯ {chk_char} {num} {label}"
                    padded = _pad_to_width(content, bar_w)
                    out.write(f"{BAR} {BG_ACTIVE}{CYAN}{BOLD}{padded}{RESET}\n")
                else:
                    chk = f"{CYAN}◉{RESET}" if i in checked else f"{DIM}○{RESET}"
                    out.write(f"{BAR}    {chk} {GRAY}{num}{RESET} {label}\n")
            else:
                if active:
                    content = f" ❯ {num} {label}"
                    padded = _pad_to_width(content, bar_w)
                    out.write(f"{BAR} {BG_ACTIVE}{CYAN}{BOLD}{padded}{RESET}\n")
                else:
                    out.write(f"{BAR}    {GRAY}{num}{RESET} {label}\n")
            lines += 1

            if desc:
                if active:
                    out.write(f"{BAR} {BG_ACTIVE}{GRAY}      {desc} {RESET}\n")
                else:
                    out.write(f"{BAR}       {DIM}{desc}{RESET}\n")
                lines += 1

            if i < len(options) - 1 and not desc:
                out.write(f"{BAR}\n")
                lines += 1

        # Separator
        out.write(f"  {CYAN}├{'─' * (bar_w + 1)}{RESET}\n")
        lines += 1

        # Text input area
        is_typing = cursor == type_idx
        type_line = lines
        if typed_text:
            if is_typing:
                out.write(f"{BAR}  {CYAN}❯{RESET} {typed_text}")
            else:
                out.write(f"{BAR}    {typed_text}")
        else:
            if is_typing:
                content = " ❯ Type something."
                padded = _pad_to_width(content, bar_w)
                out.write(f"{BAR} {BG_ACTIVE}{GRAY}{padded}{RESET}")
            else:
                out.write(f"{BAR}    {GRAY}Type something.{RESET}")
        lines += 1

        # Bottom line
        out.write(f"\n  {CYAN}╰{'─' * (bar_w + 1)}{RESET}")
        lines += 1

        # Hint
        out.write(f"\n    {hint}")
        lines += 1

        # Move cursor back to input line when typing
        new_offset = 0
        if is_typing and typed_text:
            lines_below = lines - 1 - type_line
            out.write(f"\x1b[{lines_below}A")
            col = 7 + _visual_width(typed_text)
            out.write(f"\r\x1b[{col}C")
            out.write(SHOW_CUR)
            new_offset = lines_below

        out.flush()
        prev_lines = lines
        prev_cursor_offset = new_offset

    def render_collapsed(answer: str):
        out = sys.stdout
        out.write(HIDE_CUR)

        up = header_lines + prev_lines - 1 - prev_cursor_offset
        out.write("\r")
        if up > 0:
            out.write(f"\x1b[{up}A")
        out.write("\x1b[J")

        out.write(f"  {CYAN}╭{RESET} {title_styled}\n")
        out.write(f"{BAR}  {GREEN}{answer}{RESET}\n")
        out.write(f"  {CYAN}╰{RESET}\n")

        out.write(SHOW_CUR)
        out.flush()

    try:
        out = sys.stdout
        out.write(f"\n  {CYAN}╭{RESET} {title_styled}\n")
        out.write(f"{BAR}\n")
        out.flush()

        tty.setcbreak(fd)
        render()

        while True:
            key = _read_key(fd)

            if key == "up" and cursor > 0:
                cursor -= 1
                render()
            elif key == "down" and cursor < type_idx:
                cursor += 1
                render()
            elif key == "enter":
                answer = None
                if cursor == type_idx and typed_text.strip():
                    if multi:
                        parts = [options[i]["label"] for i in sorted(checked)]
                        parts.append(typed_text.strip())
                        answer = ", ".join(parts)
                    else:
                        answer = typed_text.strip()
                elif cursor < type_idx:
                    if multi:
                        parts = [options[i]["label"] for i in sorted(checked)]
                        answer = ", ".join(parts) if parts else options[cursor]["label"]
                    else:
                        answer = options[cursor]["label"]
                if answer is not None:
                    render_collapsed(answer)
                    return answer
            elif key == "space":
                if multi and cursor < type_idx:
                    checked.symmetric_difference_update({cursor})
                    render()
                elif cursor == type_idx:
                    typed_text += " "
                    render()
            elif key == "backspace":
                if typed_text:
                    typed_text = typed_text[:-1]
                    render()
            elif key and len(key) == 1:
                cursor = type_idx
                typed_text += key
                render()
    finally:
        sys.stdout.write(SHOW_CUR)
        sys.stdout.flush()
        termios.tcsetattr(fd, termios.TCSADRAIN, old)


# ── Fallback for non-Unix ────────────────────────────────────────────────


def _display_question_fallback(question: dict) -> str:
    """Simple input()-based fallback when termios is unavailable."""
    text = question.get("question", "")
    raw_options = question.get("options", [])
    options = _normalize_options(raw_options)

    print(f"\n  {text}")
    for i, opt in enumerate(options, 1):
        desc = f" - {opt['description']}" if opt.get("description") else ""
        print(f"  {i}. {opt['label']}{desc}")
    print(f"  {len(options) + 1}. Other (free text)")

    while True:
        choice = input(f"  Choose [1-{len(options) + 1}]: ").strip()
        try:
            idx = int(choice) - 1
            if 0 <= idx < len(options):
                return options[idx]["label"]
            elif idx == len(options):
                return input("  Enter: ").strip()
        except ValueError:
            if choice:
                return choice


# ── High-level: ask multiple questions ───────────────────────────────────


def ask_user_interactive(questions: list[dict]) -> dict[str, str]:
    """Present multiple questions interactively and collect answers.

    Args:
        questions: List of question dicts. Each can have:
            - question (str): The question text
            - header (str, optional): Short label
            - options (list): Strings or {"label", "description"} dicts
            - multiSelect (bool, optional): Enable multi-select

    Returns:
        Dict mapping question text to user's answer.
    """
    answers = {}
    for i, q in enumerate(questions, 1):
        question_text = q.get("question", q.get("q", ""))
        # Inject "Q1", "Q2" ... as header prefix
        numbered = dict(q)
        existing_header = numbered.get("header", "")
        numbered["header"] = f"Q{i} {existing_header}".strip()
        answer = display_question(numbered)
        answers[question_text] = answer
    return answers
