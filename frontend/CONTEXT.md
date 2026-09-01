# frontend/

## Intent
Single-file browser UI for the EvoCoCo backend (`index.html`). It connects to the
backend over WebSocket, displays the pipeline's execution trace and runtime logs,
renders the three output views (Python code, blueprint, IR analysis), and submits
the MATLAB source code entered in the editor for processing.

## API Surface

### WebSocket endpoint
- Derives the URL from the page origin: `ws(s)://<host>/ws`; falls back to
  `ws://localhost:8000/ws` when opened via `file://`.
- Reconnects automatically with exponential backoff (1s -> 2s -> ... capped at 30s).

### Client -> server
- `{ "code": "<matlab source>" }` — sent once when the user clicks Run.

### Server -> client (field names are load-bearing, do NOT rename)
- Every message carries `{ type, title, message, step_id, extra_data, is_success, icon }`.
- Handled `type` values:
  - `log` — append a line to the runtime terminal (`title`/`message` text, styling inferred from content).
  - `step_start` — add a step card to the execution trace (`step_id`, `title`, `message`, `icon`).
  - `step_done` — mark a step card success/failure (`step_id`, `is_success`, `message`, `extra_data`); `step_id === 'finish'` (or `title` Success/Failed) ends the run.
  - `result_ir` / `result_blueprint` — markdown rendered into the Analyst/Blueprint views (auto-switches tab).
  - `result_code` — Python source, syntax-highlighted in the code view (auto-switches tab).
- Unknown `type` values are logged, not fatal.

## Constraints
- Vanilla HTML/CSS/JS only. **No CDNs, no external dependencies, no build step** —
  must work when opened via `file://` or served statically.
- All UI must be self-contained in `index.html` (inline SVG icon sprite, regex
  tokenizers, minimal markdown renderer).
- WebSocket protocol field names above are load-bearing — changing them breaks the backend.
- Backend-supplied strings must only be rendered via `textContent` / `escapeHtml()`
  (XSS). The markdown renderer escapes all input before applying its own markup.
- Preserve the element IDs the backend/browser automation may rely on:
  `source-input`, `highlight-overlay`, `line-numbers`, `run-btn`, `stop-btn`,
  `flow-container`, `empty-state`, `terminal-content`, `code-content`,
  `view-code`, `view-blueprint`, `view-ir`, `tab-code`, `tab-blueprint`, `tab-ir`,
  `copy-btn`, `connection-dot`, `connection-text`, `theme-icon`,
  `resizer-left`, `resizer-middle`, `resizer-right`, `panel-left`,
  `panel-middle`, `panel-middle-top`, `panel-middle-bottom`, `main-container`, `step-template`.

## Known Issues
- MathJax was removed (was a CDN dependency): backend `$...$` / `$$...$$` math now
  renders as literal text inside markdown.
- Syntax highlighting is regex-based (best-effort), not a real lexer; deeply nested
  or exotic MATLAB/Python may mis-tokenize (cosmetic only — the underlying text is intact).
- `navigator.clipboard` may be unavailable under `file://`; an `execCommand('copy')`
  textarea fallback covers most browsers, but some hardened environments block both.
- The seed editor content (`initialScript`) is preserved verbatim from the original UI.