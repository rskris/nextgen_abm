"""Generate publication-quality HTML and PDF documentation from technical_features.md.

Leverages:
- marked.js for Markdown parsing
- KaTeX for mathematical vector typography
- Mermaid.js for vector SVG flowcharts, state diagrams, and sequence diagrams
- Prism.js for code syntax highlighting
- Google Chrome headless for PDF generation with @page print rules
"""

import os
import subprocess
import sys
import time

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
MD_PATH = os.path.join(PROJECT_ROOT, "technical_features.md")
HTML_PATH = os.path.join(PROJECT_ROOT, "technical_features.html")
PDF_PATH = os.path.join(PROJECT_ROOT, "technical_features.pdf")

HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>NextGen ABM: Technical Architecture & Feature Specification</title>
  
  <!-- KaTeX CSS & JS -->
  <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/katex@0.16.8/dist/katex.min.css">
  <script src="https://cdn.jsdelivr.net/npm/katex@0.16.8/dist/katex.min.js"></script>
  <script src="https://cdn.jsdelivr.net/npm/katex@0.16.8/dist/contrib/auto-render.min.js"></script>
  
  <!-- Mermaid.js -->
  <script src="https://cdn.jsdelivr.net/npm/mermaid@10.6.1/dist/mermaid.min.js"></script>
  
  <!-- Marked.js -->
  <script src="https://cdn.jsdelivr.net/npm/marked@9.1.6/marked.min.js"></script>

  <style>
    :root {
      --primary: #1e3a8a;
      --secondary: #0284c7;
      --accent: #0f766e;
      --text: #1f2937;
      --bg: #ffffff;
      --code-bg: #f8fafc;
      --border: #e2e8f0;
      --header-bg: #f1f5f9;
    }

    * {
      box-sizing: border-box;
    }

    body {
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
      color: var(--text);
      background-color: var(--bg);
      line-height: 1.65;
      font-size: 14px;
      padding: 0;
      margin: 0 auto;
      max-width: 900px;
    }

    .container {
      padding: 40px 40px;
    }

    h1 {
      font-size: 26px;
      color: var(--primary);
      border-bottom: 2.5px solid var(--primary);
      padding-bottom: 8px;
      margin-top: 0;
      margin-bottom: 16px;
    }

    h2 {
      font-size: 20px;
      color: var(--primary);
      border-bottom: 1px solid var(--border);
      padding-bottom: 6px;
      margin-top: 32px;
      margin-bottom: 14px;
      page-break-after: avoid;
    }

    h3 {
      font-size: 16px;
      color: #1e40af;
      margin-top: 24px;
      margin-bottom: 10px;
      page-break-after: avoid;
    }

    h4 {
      font-size: 14.5px;
      color: #334155;
      margin-top: 18px;
      margin-bottom: 8px;
      page-break-after: avoid;
    }

    h5 {
      font-size: 13.5px;
      color: #475569;
      margin-top: 14px;
      margin-bottom: 6px;
    }

    p {
      margin-top: 0;
      margin-bottom: 12px;
      text-align: justify;
    }

    blockquote {
      margin: 16px 0;
      padding: 12px 18px;
      background-color: #f8fafc;
      border-left: 4px solid var(--secondary);
      color: #334155;
      font-size: 13.5px;
      border-radius: 0 4px 4px 0;
    }

    blockquote p:last-child {
      margin-bottom: 0;
    }

    /* Math display */
    .katex-display {
      margin: 14px 0 !important;
      padding: 10px 14px !important;
      background: #f8fafc !important;
      border-radius: 6px !important;
      border: 1px solid #e2e8f0 !important;
      overflow-x: auto !important;
      page-break-inside: avoid !important;
    }

    .katex {
      font-size: 1.05em !important;
    }

    /* Mermaid diagrams */
    .mermaid {
      margin: 20px 0;
      padding: 16px;
      background: #ffffff;
      border: 1px solid var(--border);
      border-radius: 8px;
      text-align: center;
      page-break-inside: avoid;
      box-shadow: 0 1px 3px rgba(0, 0, 0, 0.05);
    }

    .mermaid svg {
      max-width: 100% !important;
      height: auto !important;
    }

    /* Tables */
    table {
      width: 100%;
      border-collapse: collapse;
      margin: 16px 0;
      font-size: 12.5px;
      page-break-inside: avoid;
    }

    th, td {
      border: 1px solid var(--border);
      padding: 7px 10px;
      text-align: left;
    }

    th {
      background-color: var(--header-bg);
      color: #0f172a;
      font-weight: 600;
    }

    tr:nth-child(even) td {
      background-color: #fcfcfd;
    }

    /* Code blocks */
    pre {
      background-color: var(--code-bg);
      border: 1px solid var(--border);
      border-radius: 6px;
      padding: 12px 14px;
      overflow-x: auto;
      font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
      font-size: 12px;
      line-height: 1.5;
      page-break-inside: avoid;
    }

    code {
      font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
      font-size: 12px;
      background-color: #f1f5f9;
      padding: 2px 4px;
      border-radius: 3px;
    }

    pre code {
      background-color: transparent;
      padding: 0;
    }

    ul, ol {
      margin-top: 0;
      margin-bottom: 12px;
      padding-left: 24px;
    }

    li {
      margin-bottom: 4px;
    }

    hr {
      border: 0;
      height: 1px;
      background: var(--border);
      margin: 28px 0;
    }

    /* Print styles */
    @media print {
      body {
        max-width: 100%;
        font-size: 12px;
        line-height: 1.5;
      }

      .container {
        padding: 0;
      }

      @page {
        size: letter;
        margin: 16mm 14mm 16mm 14mm;
      }

      h1 { font-size: 22px; }
      h2 { font-size: 17px; margin-top: 24px; }
      h3 { font-size: 14.5px; margin-top: 18px; }
      h4 { font-size: 13px; }

      .mermaid {
        box-shadow: none;
        border: 1px solid #cbd5e1;
        padding: 10px;
        margin: 14px 0;
      }

      .katex-display {
        background: #f8fafc !important;
        border: 1px solid #cbd5e1 !important;
        padding: 8px !important;
        margin: 10px 0 !important;
      }

      table {
        font-size: 11px;
      }

      th, td {
        padding: 5px 8px;
      }

      pre {
        font-size: 10.5px;
        padding: 8px 10px;
      }
    }
  </style>
</head>
<body>
  <div class="container" id="content">
    <!-- Rendered Markdown Content Will Appear Here -->
  </div>

  <!-- Raw Markdown Data Island -->
  <script type="text/plain" id="raw-markdown">
__RAW_MARKDOWN_CONTENT__
  </script>

  <script>
    document.addEventListener("DOMContentLoaded", async function() {
      const rawMd = document.getElementById("raw-markdown").textContent;
      
      // Step 1: Protect math blocks and inline math from Marked parser
      const mathBlocks = [];
      const mathInlines = [];
      
      // Extract $$ ... $$ blocks
      let protectedMd = rawMd.replace(/\$\$([\s\S]*?)\$\$/g, function(match, mathContent) {
        const id = mathBlocks.length;
        mathBlocks.push(mathContent.trim());
        return "\n\nZZMATHBLOCK" + id + "ZZ\n\n";
      });
      
      // Extract $ ... $ inline math (excluding escaped \$)
      protectedMd = protectedMd.replace(/(^|[^\\])\$([^$\n]+)\$/g, function(match, prefix, mathContent) {
        const id = mathInlines.length;
        mathInlines.push(mathContent.trim());
        return prefix + "ZZMATHINLINE" + id + "ZZ";
      });

      // Step 2: Custom Marked Renderer for Mermaid blocks
      const renderer = new marked.Renderer();
      const defaultCodeRenderer = renderer.code.bind(renderer);
      
      renderer.code = function(code, lang) {
        if (lang === "mermaid") {
          return '<div class="mermaid">' + code + '</div>';
        }
        return defaultCodeRenderer(code, lang);
      };

      marked.setOptions({
        renderer: renderer,
        gfm: true,
        breaks: false
      });

      // Step 3: Parse Markdown to HTML
      let html = marked.parse(protectedMd);

      // Step 4: Re-insert Math elements with direct indices
      html = html.replace(/ZZMATHBLOCK(\d+)ZZ/g, function(match, id) {
        return '<div class="katex-display-target" data-math-id="' + id + '"></div>';
      });

      html = html.replace(/ZZMATHINLINE(\d+)ZZ/g, function(match, id) {
        return '<span class="katex-inline-target" data-math-id="' + id + '"></span>';
      });

      document.getElementById("content").innerHTML = html;

      // Step 5: Render KaTeX Math synchronously directly into targets
      document.querySelectorAll(".katex-display-target").forEach(function(el) {
        const id = parseInt(el.getAttribute("data-math-id"), 10);
        const math = mathBlocks[id];
        if (math) {
          try {
            katex.render(math, el, { displayMode: true, throwOnError: false });
          } catch(e) {
            el.textContent = "$$" + math + "$$";
          }
        }
      });

      document.querySelectorAll(".katex-inline-target").forEach(function(el) {
        const id = parseInt(el.getAttribute("data-math-id"), 10);
        const math = mathInlines[id];
        if (math) {
          try {
            katex.render(math, el, { displayMode: false, throwOnError: false });
          } catch(e) {
            el.textContent = "$" + math + "$";
          }
        }
      });

      // Step 6: Render Mermaid SVGs
      try {
        mermaid.initialize({
          startOnLoad: false,
          theme: "neutral",
          flowchart: { curve: "basis" },
          securityLevel: "loose"
        });
        await mermaid.run();
      } catch (err) {
        console.error("Mermaid render error:", err);
      }

      // Mark page as ready for headless print snapshot
      document.body.setAttribute("data-render-complete", "true");
      console.log("PDF generation prep completed successfully!");
    });
  </script>
</body>
</html>
"""


def build():
    if not os.path.exists(MD_PATH):
        print(f"Error: {MD_PATH} not found.")
        sys.exit(1)

    with open(MD_PATH, "r", encoding="utf-8") as f:
        md_content = f.read()

    # Escape any closing script tag inside markdown if any
    safe_md = md_content.replace("</script>", "<\\/script>")

    full_html = HTML_TEMPLATE.replace("__RAW_MARKDOWN_CONTENT__", safe_md)

    with open(HTML_PATH, "w", encoding="utf-8") as f:
        f.write(full_html)
    print(f"✓ Standalone interactive HTML generated: {HTML_PATH}")

    # Check for Chrome
    chrome_app = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
    if not os.path.exists(chrome_app):
        print(f"Warning: Chrome not found at {chrome_app}. PDF generation requires Chrome.")
        return

    print("▶ Rendering PDF via Google Chrome headless...")
    file_url = f"file://{HTML_PATH}"

    cmd = [
        chrome_app,
        "--headless",
        "--disable-gpu",
        "--no-pdf-header-footer",
        "--virtual-time-budget=10000",
        f"--print-to-pdf={PDF_PATH}",
        file_url,
    ]

    res = subprocess.run(cmd, capture_output=True, text=True)
    if os.path.exists(PDF_PATH) and os.path.getsize(PDF_PATH) > 0:
        size_kb = os.path.getsize(PDF_PATH) / 1024
        print(f"✓ High-quality PDF generated: {PDF_PATH} ({size_kb:.1f} KB)")
    else:
        print(f"Error during Chrome PDF generation:\n{res.stderr}")


if __name__ == "__main__":
    build()
