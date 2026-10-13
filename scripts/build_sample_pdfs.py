"""Build the sample policy PDFs in data/policies/ from the Markdown sources in data/source/.

Brightwave Analytics is a fictional company. These documents exist only so the
assistant has realistic, multi-page PDFs to search and cite.

A line containing only ---PAGE--- starts a new page, so each fact sits on a known
page. The evaluation questions in eval/questions.jsonl depend on these page numbers.

Usage:
    python scripts/build_sample_pdfs.py
"""

from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import mm
from reportlab.platypus import PageBreak, Paragraph, SimpleDocTemplate, Spacer

ROOT = Path(__file__).resolve().parents[1]
SOURCE_DIR = ROOT / "data" / "source"
OUTPUT_DIR = ROOT / "data" / "policies"

TITLE = ParagraphStyle("title", fontName="Helvetica-Bold", fontSize=18, leading=22, spaceAfter=4)
META = ParagraphStyle("meta", fontName="Helvetica", fontSize=9, leading=12,
                      textColor=colors.HexColor("#555555"), spaceAfter=14)
HEADING = ParagraphStyle("heading", fontName="Helvetica-Bold", fontSize=12, leading=15,
                         spaceBefore=10, spaceAfter=4)
BODY = ParagraphStyle("body", fontName="Helvetica", fontSize=10.5, leading=14.5, spaceAfter=7)
BULLET = ParagraphStyle("bullet", parent=BODY, leftIndent=14, bulletIndent=4, spaceAfter=3)


def markdown_to_story(text: str) -> tuple[str, list]:
    """Turn the small Markdown subset used in data/source/ into reportlab flowables."""
    title = ""
    story = []
    paragraph: list[str] = []

    def flush():
        if paragraph:
            story.append(Paragraph(" ".join(paragraph), BODY))
            paragraph.clear()

    lines = text.splitlines()
    for i, raw in enumerate(lines):
        line = raw.strip()
        if not line:
            flush()
        elif line == "---PAGE---":
            flush()
            story.append(PageBreak())
        elif line.startswith("# "):
            flush()
            title = line[2:]
            story.append(Paragraph(title, TITLE))
            # The line after the title holds the document number, version and date.
            if i + 1 < len(lines) and lines[i + 1].strip():
                story.append(Paragraph(lines[i + 1].strip(), META))
        elif line.startswith("## "):
            flush()
            story.append(Paragraph(line[3:], HEADING))
        elif line.startswith("- "):
            flush()
            story.append(Paragraph(line[2:], BULLET, bulletText="•"))
        elif i > 0 and lines[i - 1].startswith("# "):
            continue  # metadata line, already added under the title
        else:
            paragraph.append(line)
    flush()
    return title, story


def build(source: Path) -> Path:
    title, story = markdown_to_story(source.read_text(encoding="utf-8"))
    output = OUTPUT_DIR / f"{source.stem}.pdf"

    def footer(canvas, doc):
        canvas.saveState()
        canvas.setFont("Helvetica", 8)
        canvas.setFillColor(colors.HexColor("#777777"))
        canvas.drawString(20 * mm, 12 * mm,
                          f"Brightwave Analytics (fictional sample company) | {title}")
        canvas.drawRightString(A4[0] - 20 * mm, 12 * mm, f"Page {doc.page}")
        canvas.restoreState()

    doc = SimpleDocTemplate(str(output), pagesize=A4, title=title,
                            author="Brightwave Analytics (fictional)",
                            leftMargin=20 * mm, rightMargin=20 * mm,
                            topMargin=20 * mm, bottomMargin=22 * mm)
    doc.build(story + [Spacer(1, 1)], onFirstPage=footer, onLaterPages=footer)
    return output


if __name__ == "__main__":
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(SOURCE_DIR.glob("*.md")):
        print("built", build(path).relative_to(ROOT))
