# CollegeSaathi AI — frontend prototype

A single-file, dependency-free prototype of the CollegeSaathi AI frontend (landing, AI counsellor, college explorer,
college details, cutoff checker, comparison). No build step, no `node_modules`, no environment variables or secrets.

## Run
Open `index.html` in any modern browser, or serve the folder:

    python3 -m http.server 8080    # then visit http://localhost:8080

## Theme behaviour
The theme changes **only** when the Dark/Light toggle button in the navbar (`data-theme-toggle`) is clicked.
The choice is stored in `localStorage` (`cs_theme`) and applied before first paint to avoid a flash. With no saved
choice, the system preference is used. All theme logic lives in the single `Theme` module in `index.html`.

## Data
All colleges, fees, packages and cutoffs are **demo placeholders, not verified** and are labelled as such in the UI.
Replace the bodies of `Svc.colleges()`, `Svc.ask()` and `Svc.cutoffs()` with real API calls.

## Note
This is a standalone prototype, not a Next.js/TypeScript project; there is no lint, type-check or production build
configured. Port the markup and the `Theme` module into your app's framework as needed.
