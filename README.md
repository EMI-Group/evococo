# EvoCoder

EvoCoder is a lightweight local application with a **FastAPI backend** and a **browser-based frontend**.
The backend provides API services, while the frontend interacts with it through a simple web interface.

---

## Running the Backend

Start the backend server from the project root directory:

```
python -m uvicorn backend.main:app --reload --reload-dir backend --port 8000
```

### Arguments

* `backend.main:app` — FastAPI application entry point
* `--reload` — Enables auto-reload when code changes
* `--reload-dir backend` — Watches the `backend` directory for updates
* `--port 8000` — Runs the server on port `8000`

After starting successfully, the backend API will be available at:

```
http://localhost:8000
```

---

## Opening the Frontend

Once the backend is running, open the frontend page in your browser:

```
frontend/index.html
```

You can open it by:

* Double-clicking `index.html`
* Dragging the file into a browser
* Opening the file path directly in a browser

Example:

```
file:///path/to/project/frontend/index.html
```

The frontend will automatically communicate with the backend running at:

```
http://localhost:8000
```

---


## Notes

* The backend **must be running before opening the frontend**.
* If the backend is not started, frontend API requests will fail.
