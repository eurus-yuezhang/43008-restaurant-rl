"""Step 2: combine the page parts and the recorded traces into one self-contained HTML file.

Run:  python demo_web/build_page.py
Output: demo_web/restaurant_policy_lab.html  (open it in any browser, no internet needed except fonts)
"""
from pathlib import Path

HERE = Path(__file__).resolve().parent
src = HERE / "src"
# Read the page sources and existing trajectories; building the page runs no simulation.
head = (src / "page_head.html").read_text(encoding="utf-8")
body = (src / "page_body.html").read_text(encoding="utf-8")
script = (src / "page_script.js").read_text(encoding="utf-8")
traces = (HERE / "traces.json").read_text(encoding="utf-8")

# Embed the recorded JSON and replay script so the page can open directly from a file.
page = ("<!doctype html>\n<html lang=\"en\">\n<head>\n<meta charset=\"utf-8\">\n"
        "<meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">\n"
        + head + "\n</head>\n<body>\n" + body
        + "\n<script type=\"application/json\" id=\"traces\">" + traces + "</script>\n"
        + "<script>\n" + script + "\n</script>\n</body>\n</html>\n")
out = HERE / "restaurant_policy_lab.html"
out.write_text(page, encoding="utf-8")
print("wrote", out, f"({len(page) // 1024} KB)")
