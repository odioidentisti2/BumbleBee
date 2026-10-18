# Third-party notices

## htmx 2.0.11

- Project: https://github.com/bigskysoftware/htmx
- License: 0BSD (Zero-Clause BSD)
- License text: [static/vendor/LICENSE.htmx.txt](static/vendor/LICENSE.htmx.txt)
- Vendored file: `static/vendor/htmx.min.js`
- SHA-256: `d6fdc75f204e6bdefa99b69bf1e6d4ac69b8a364f77929f45c13476b4000f717`

## htmx-ext-sse 2.2.4

- Project: https://github.com/bigskysoftware/htmx-extensions
- License: 0BSD (BSD Zero Clause), Copyright (c) 2023 Alexander Petros
- License text: [static/vendor/LICENSE.htmx-sse.txt](static/vendor/LICENSE.htmx-sse.txt)
- Vendored file: `static/vendor/sse.min.js`
- SHA-256: `98a46496de0c3605fbffdce9167ba427bdd9553184f83f149c261891a92c0136`

Upgrade policy: keep htmx and htmx-ext-sse on the same major. The SSE
extension declares a dependency on `htmx.org ^2.0.2`; do not mix it with
htmx 4.x.
