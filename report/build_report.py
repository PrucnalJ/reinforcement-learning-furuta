"""Build the submission PDF from a Markdown export of the report.

Usage:  python report/build_report.py report/report.md

Export the report as Markdown, save it as report/report.md, then run this.
It swaps the image placeholders for the files in figures/, sets the page in
11 pt type, and prints it to report/report.pdf with Edge or Chrome
(headless).  Edit AUTHORS and DATE below for the title block.
"""

import pathlib
import re
import subprocess
import sys

import markdown

AUTHORS = "Jacek Prucnal and Rocco DeLuca"
DATE = "October 12, 2026"

HERE = pathlib.Path(__file__).resolve().parent
FIGS = HERE.parent / "figures"
# Placeholders appear in this order in the report.
FIGURES = ["fig_training.png", "fig_eval_success.png", "fig_eval_traces.png",
           "fig_comparison.png", "fig_visits.png"]
WIDTH = {"fig_training.png": "52%", "fig_eval_success.png": "46%",
         "fig_eval_traces.png": "44%", "fig_comparison.png": "56%", "fig_visits.png": "50%"}
EQUATIONS = {
    "SARSA": "SARSA: &nbsp; Q(S<sub>t</sub>,A<sub>t</sub>) &larr; Q(S<sub>t</sub>,A<sub>t</sub>) + "
             "&alpha;[R<sub>t+1</sub> + &gamma; Q(S<sub>t+1</sub>,A<sub>t+1</sub>) &minus; Q(S<sub>t</sub>,A<sub>t</sub>)]",
    "Q-learning": "Q-learning: &nbsp; Q(S<sub>t</sub>,A<sub>t</sub>) &larr; Q(S<sub>t</sub>,A<sub>t</sub>) + "
                  "&alpha;[R<sub>t+1</sub> + &gamma; max<sub>a</sub> Q(S<sub>t+1</sub>,a) &minus; Q(S<sub>t</sub>,A<sub>t</sub>)]",
}
BROWSERS = [r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
            r"C:\Program Files\Google\Chrome\Application\chrome.exe"]

CSS = """
@page { size: letter; margin: 0.7in 0.75in; }
body { font-family: 'Times New Roman', Times, serif; font-size: 10.5pt; line-height: 1.18;
       color: #000; margin: 0; }
h1 { font-size: 15.5pt; text-align: center; margin: 0 0 4pt; line-height: 1.2; }
.byline { text-align: center; margin: 0 0 10pt; }
h2 { font-size: 12pt; margin: 11pt 0 3pt; break-after: avoid; }
p { margin: 0 0 4pt; text-align: justify; }
ul, ol { margin: 0 0 5pt; padding-left: 18pt; } li { margin: 0 0 2pt; }
table { border-collapse: collapse; margin: 4pt auto 6pt; font-size: 9pt; }
tr { break-inside: avoid; }
th, td { border-top: 0.5pt solid #444; border-bottom: 0.5pt solid #444; padding: 1.5pt 5pt;
         text-align: left; vertical-align: top; }
th { font-weight: bold; }
code { font-family: Consolas, monospace; font-size: 9pt; }
pre { font-size: 8.3pt; background: #f4f4f4; padding: 4pt 6pt; margin: 3pt 0 6pt;
      white-space: pre-wrap; break-inside: avoid; }
.eq { text-align: center; margin: 3pt 0; }
figure { margin: 4pt 0 6pt; text-align: center; break-inside: avoid; }
figure img { display: block; margin: 0 auto 2pt; }
.pair { display: flex; gap: 14pt; align-items: flex-start; break-inside: avoid; }
.pair figure { flex: 1; margin: 2pt 0 4pt; }
p.caption { font-size: 9.5pt; text-align: center; margin: 0 0 6pt; }
a { color: #000; text-decoration: none; }
"""


def convert(md_text):
    lines = md_text.splitlines()
    # Title and byline: replace the Docs chips line with a plain byline.
    title = lines[0].lstrip("# ").strip()
    body = "\n".join(lines[1:])
    body = re.sub(r"\A\s*\n[^\n]*·[^\n]*\n", "\n", body, count=1)
    # Equations written as fenced LaTeX blocks.
    def eq(m):
        key = "Q-learning" if "max" in m.group(1) else "SARSA"
        return f'\n<div class="eq">{EQUATIONS[key]}</div>\n'
    body = re.sub(r"```latex\n(.*?)\n```", eq, body, flags=re.S)
    # Image placeholders -> figures, in order.
    figs = iter(FIGURES)
    def img(_):
        f = next(figs)
        return (f'\n<figure><img src="{(FIGS / f).as_uri()}" style="width:{WIDTH[f]}"></figure>\n')
    body = re.sub(r"&#91;image:[^\n]*?\\\]", img, body)
    html = markdown.markdown(body, extensions=["tables", "fenced_code", "sane_lists"])
    # Figure captions: paragraphs that are only an italic "Figure N." sentence.
    html = re.sub(r"<p><em>(Figure \d+\..*?)</em>(.*?)</p>",
                  r'<p class="caption"><em>\1</em></p><p>\2</p>', html, flags=re.S)
    html = html.replace("<p></p>", "")
    # Figures 2 and 3 side by side.
    pair = re.compile(r'<figure><img (src="[^"]*fig_eval_success.png")[^>]*></figure>\s*(<p class="caption">.*?</p>)\s*'
                      r'<figure><img (src="[^"]*fig_eval_traces.png")[^>]*></figure>\s*(<p class="caption">.*?</p>)', re.S)
    html = pair.sub(r'<div class="pair"><figure><img \1 style="width:100%">\2</figure>'
                    r'<figure><img \3 style="width:100%">\4</figure></div>', html)
    # Keep every other caption on the same page as its figure.
    html = re.sub(r'</figure>\s*(<p class="caption">.*?</p>)', r'\1</figure>', html, flags=re.S)
    return (f"<!doctype html><html><head><meta charset='utf-8'><style>{CSS}</style></head><body>"
            f"<h1>{title}</h1><div class='byline'>{AUTHORS} &middot; COSC 527 &middot; {DATE}</div>"
            f"{html}</body></html>")


def main():
    src = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else HERE / "report.md")
    html_path = HERE / "report.html"
    pdf_path = HERE / "report.pdf"
    html_path.write_text(convert(src.read_text(encoding="utf-8")), encoding="utf-8")
    browser = next(b for b in BROWSERS if pathlib.Path(b).exists())
    subprocess.run([browser, "--headless", "--disable-gpu", "--no-pdf-header-footer",
                    f"--print-to-pdf={pdf_path}", html_path.as_uri()], check=True,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    print(f"wrote {pdf_path}")


if __name__ == "__main__":
    main()
