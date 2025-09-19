#!/usr/bin/env python3
"""
Multi-page PDF generator from your Bash comparison output.

Top area on the FIRST table page:
- Title (centered)
- node version: <first non-empty line> (left-aligned)
- git info: (left-aligned), lines taken from the block between:
      START GIT INFO
      ... (your lines)
      END GIT INFO
  (If the git block is missing, it's skipped.)

Then:
- Colored results table (PASS=green, FAIL=red, MISSING=gray), paginated as needed.
- "Artifact changes details" section, paginated, format:
    NEW_RUN vs OLD_RUN   (bold, slightly larger)
    test1                (bold)
    no changes
    test2
    ADDED:
      path...
    REMOVED:
      path...
    CHANGED:
      path...
"""

import argparse, sys, textwrap, re
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

TOKENS = {"PASS", "FAIL", "MISSING"}

# ---------------- Models ----------------

@dataclass
class TestDiff:
    name: str
    new_status: str = "MISSING"
    old_status: str = "MISSING"
    added: List[str] = field(default_factory=list)
    removed: List[str] = field(default_factory=list)
    changed: List[str] = field(default_factory=list)

@dataclass
class RunComparison:
    new_run: str
    old_run: str
    tests: List[TestDiff] = field(default_factory=list)

# ---------------- Parse helpers ----------------

def is_status_line(s: str) -> bool:
    if " vs " not in s: return False
    left, right = s.split(" vs ", 1)
    return left.strip().upper() in TOKENS and right.strip().upper() in TOKENS

def is_header_line(s: str) -> bool:
    return " vs " in s and not is_status_line(s)

def extract_node_and_git(raw: str) -> tuple[str, List[str], str]:
    """
    Returns (node_info, git_lines, remainder_text_without_node_or_git)
    - node_info: first non-empty line (or "")
    - git_lines: lines between START GIT INFO and END GIT INFO (trimmed), or []
    - remainder: raw text with node line and git block removed
    """
    lines = raw.splitlines()
    # Node: first non-empty
    node_info = ""
    idx = 0
    while idx < len(lines) and not lines[idx].strip():
        idx += 1
    if idx < len(lines):
        node_info = lines[idx].strip()
        # remove node line
        del lines[idx]
    # Git block
    text_wo_node = "\n".join(lines)
    m = re.search(r"(?ms)^START GIT INFO\s*\n(.*?)\nEND GIT INFO\s*$", text_wo_node)
    git_lines: List[str] = []
    if m:
        git_block = m.group(1)
        git_lines = [ln.rstrip() for ln in git_block.splitlines()]
        # remove the whole block
        start = m.start()
        end = m.end()
        text_wo_git = text_wo_node[:start] + text_wo_node[end:]
    else:
        text_wo_git = text_wo_node
    return node_info, git_lines, text_wo_git

def parse_shell_output(text: str) -> List[RunComparison]:
    lines = [ln.rstrip("\n") for ln in text.splitlines()]
    comps: List[RunComparison] = []
    i = 0
    current: Optional[RunComparison] = None

    def peek(idx: int) -> str:
        return lines[idx].strip() if 0 <= idx < len(lines) else ""

    while i < len(lines):
        line = lines[i].strip()
        if not line:
            i += 1; continue

        if is_header_line(line):
            new_run, old_run = [x.strip() for x in line.split(" vs ", 1)]
            current = RunComparison(new_run=new_run, old_run=old_run)
            comps.append(current); i += 1; continue

        if current is None:
            i += 1; continue

        test_name = line; i += 1
        new_status = "MISSING"; old_status = "MISSING"
        if is_status_line(peek(i)):
            ns, os = [x.strip() for x in lines[i].strip().split(" vs ", 1)]
            new_status, old_status = ns, os; i += 1

        td = TestDiff(name=test_name, new_status=new_status, old_status=old_status)

        while i < len(lines):
            s = lines[i].strip()
            if not s: i += 1; break
            if is_header_line(s): break
            if s.lower().startswith("no changes"): i += 1; continue
            if s.startswith("ADDED  "):   td.added.append(s[7:]);  i += 1; continue
            if s.startswith("REMOVED  "): td.removed.append(s[9:]); i += 1; continue
            if s.startswith("CHANGED  "): td.changed.append(s[9:]); i += 1; continue
            break

        current.tests.append(td)

    return comps

# ---------------- Consolidation ----------------

def consolidate(comps: List[RunComparison]):
    if not comps: return "", [], [], {}, {}

    new_run = comps[0].new_run
    old_runs: List[str] = []
    tests_set = set()
    seen_old = set()

    for c in comps:
        if c.old_run not in seen_old:
            old_runs.append(c.old_run); seen_old.add(c.old_run)
        for td in c.tests:
            tests_set.add(td.name)

    tests = sorted(tests_set)
    status_table: Dict[str, Dict[str, str]] = {t: {"NEW": "MISSING"} for t in tests}
    change_map: Dict[str, Dict[str, Tuple[List[str], List[str], List[str]]]] = {t: {} for t in tests}

    for c in comps:
        for td in c.tests:
            if status_table[td.name]["NEW"] == "MISSING":
                status_table[td.name]["NEW"] = td.new_status
            status_table[td.name][c.old_run] = td.old_status
            change_map[td.name][c.old_run] = (td.added, td.removed, td.changed)

    return new_run, old_runs, tests, status_table, change_map

# ---------------- Rendering: table (paginated) ----------------

def render_table_pages(pdf: PdfPages, new_run: str, old_runs: List[str], tests: List[str],
                       status_table: Dict[str, Dict[str, str]],
                       node_info: Optional[str],
                       git_lines: List[str],
                       rows_per_page: int = 32):
    headers = ["Test", "NEW"] + old_runs
    color_map = {"PASS": "#d8f5d0", "FAIL": "#ffd6d6", "MISSING": "#e9e9e9"}

    # Pre-wrap git lines (keep them as-is per line; they’re already “lines”)
    note_lines = []
    if node_info:
        note_lines.append(f"node version: {node_info}")
    if git_lines:
        note_lines.append("git info:")
        # keep lines as-is; optionally wrap very long ones:
        for g in git_lines:
            # wrap very long lines so they don’t run off the page
            wrapped = textwrap.wrap(g, width=110, replace_whitespace=False, drop_whitespace=False)
            note_lines.extend(wrapped if wrapped else [""])
    # vertical spacing per note line (empirical)
    per_note = 0.02

    # Split into pages
    for page_idx in range(0, len(tests) or 1, rows_per_page):
        chunk = tests[page_idx:page_idx + rows_per_page] if tests else []
        data = []
        for t in chunk:
            row = [t, status_table[t].get("NEW", "MISSING")]
            for o in old_runs:
                row.append(status_table[t].get(o, "MISSING"))
            data.append(row)

        fig = plt.figure(figsize=(8.27, 11.69))  # A4
        left, right = 0.04, 0.04
        # Base top margin for title; add space for note lines on the FIRST table page only
        top_margin_base = 0.12
        extra_top = per_note * len(note_lines) if (page_idx == 0 and note_lines) else 0.0
        top_margin, bottom_margin = top_margin_base + extra_top, 0.06

        ax = fig.add_axes([left, bottom_margin, 1.0 - left - right, 1.0 - top_margin - bottom_margin])
        ax.axis("off")

        # Title
        title = f"{new_run} vs " + " vs ".join(old_runs) if old_runs else new_run
        fig.text(0.5, 1.0 - (top_margin_base * 0.75 + extra_top), title,
                 ha="center", va="center", fontsize=12, fontweight="bold")

        # Node & Git notes (first table page only)
        if page_idx == 0 and note_lines:
            y = 1.0 - (top_margin_base * 0.75 + extra_top) - 0.03
            for ln in note_lines:
                fig.text(left, y, ln, ha="left", va="center", fontsize=10, family="monospace")
                y -= per_note

        table = ax.table(cellText=data, colLabels=headers, loc="upper left",
                         cellLoc="left", colLoc="left")
        table.auto_set_font_size(False)
        # Base font: adjust for #cols
        ncols = len(headers)
        base_font = 9 if (ncols <= 6) else 8
        table.set_fontsize(base_font)

        # Row heights
        for (r, c), cell in table.get_celld().items():
            cell.set_linewidth(0.3)
            if r == 0:
                cell.set_fontsize(base_font + 1)
                cell.set_height(0.035)
            else:
                cell.set_height(0.032)

        # Color status cells
        for (r, c), cell in table.get_celld().items():
            if r == 0 or c == 0:
                continue
            text = str(cell.get_text().get_text()).strip().upper()
            if text in color_map:
                cell.set_facecolor(color_map[text])

        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

# ---------------- Rendering: details (paginated) ----------------

def wrap_lines(items: List[str], width: int = 120) -> List[str]:
    out: List[str] = []
    for s in items:
        out.extend(textwrap.wrap(s, width=width, replace_whitespace=False, drop_whitespace=False))
    return out

def build_detail_lines(new_run: str, old_runs: List[str], tests: List[str],
                       change_map: Dict[str, Dict[str, Tuple[List[str], List[str], List[str]]]]
                       ) -> List[Tuple[str, str]]:
    """
    Produce a list of (text, style) tuples for details.
    Styles: 'header', 'subheader', 'test', 'normal'
    """
    lines: List[Tuple[str, str]] = []
    lines.append(("Artifact changes details", "header"))
    lines.append(("", "normal"))

    for o in old_runs:
        lines.append((f"{new_run} vs {o}", "subheader"))
        for t in tests:
            lines.append((t, "test"))
            added, removed, changed = change_map[t].get(o, ([], [], []))
            if not (added or removed or changed):
                lines.append(("no changes", "normal"))
            else:
                if added:
                    lines.append(("ADDED:", "normal"))
                    for w in wrap_lines([f"  {p}" for p in added], width=120):
                        lines.append((w, "normal"))
                if removed:
                    lines.append(("REMOVED:", "normal"))
                    for w in wrap_lines([f"  {p}" for p in removed], width=120):
                        lines.append((w, "normal"))
                if changed:
                    lines.append(("CHANGED:", "normal"))
                    for w in wrap_lines([f"  {p}" for p in changed], width=120):
                        lines.append((w, "normal"))
        lines.append(("", "normal"))
    return lines

def render_detail_pages(pdf: PdfPages, detail_lines: List[Tuple[str, str]],
                        title: str = "Artifact changes details"):
    # Pagination settings
    max_lines_per_page = 55
    base_font = 9
    header_boost = 3
    subheader_boost = 2

    # Paginate
    idx = 0
    page_num = 0
    while idx < len(detail_lines):
        fig = plt.figure(figsize=(8.27, 11.69))  # A4
        left, right = 0.05, 0.05
        top_margin, bottom_margin = 0.06, 0.06
        ax = fig.add_axes([left, bottom_margin, 1.0 - left - right, 1.0 - top_margin - bottom_margin])
        ax.axis("off")

        # Continuation header from page 2 onward
        if page_num >= 1:
            ax.text(0.0, 1.02, f"{title} (cont.)", ha="left", va="bottom",
                    fontsize=base_font + subheader_boost, fontweight="bold")

        # Render lines
        y = 0.98
        lines_used = 0
        while idx < len(detail_lines) and lines_used < max_lines_per_page:
            text, style = detail_lines[idx]
            if style == "header":
                fs = base_font + header_boost; fw = "bold"; fam = None
            elif style == "subheader":
                fs = base_font + subheader_boost; fw = "bold"; fam = None
            elif style == "test":
                fs = base_font; fw = "bold"; fam = "monospace"
            else:
                fs = base_font; fw = None; fam = "monospace"

            ax.text(0.0, y, text, ha="left", va="top", fontsize=fs,
                    fontweight=fw, family=fam)
            y -= 0.018
            idx += 1
            lines_used += 1

        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)
        page_num += 1

# ---------------- Main ----------------

def main():
    ap = argparse.ArgumentParser(description="Multi-page test comparison PDF (colored table + artifact changes details + node/git info).")
    ap.add_argument("-i", "--input", default="-", help="Input file (default: stdin). Use '-' for stdin.")
    ap.add_argument("-o", "--output", default="run_comparison_report.pdf", help="Output PDF filename.")
    args = ap.parse_args()

    # Read raw input and extract node & git info
    if args.input == "-" or args.input == "/dev/stdin":
        raw = sys.stdin.read()
    else:
        with open(args.input, "r", encoding="utf-8", errors="ignore") as f:
            raw = f.read()

    node_info, git_lines, rest_text = extract_node_and_git(raw)

    comps = parse_shell_output(rest_text)
    if not comps:
        print("No comparisons found in input.", file=sys.stderr)
        sys.exit(1)

    new_run, old_runs, tests, status_table, change_map = consolidate(comps)

    with PdfPages(args.output) as pdf:
        # Table pages first
        render_table_pages(pdf, new_run, old_runs, tests, status_table,
                           node_info=node_info, git_lines=git_lines, rows_per_page=32)
        # Details pages
        detail_lines = build_detail_lines(new_run, old_runs, tests, change_map)
        render_detail_pages(pdf, detail_lines, title="Artifact changes details")

    print(f"Wrote PDF: {args.output}")

if __name__ == "__main__":
    main()
