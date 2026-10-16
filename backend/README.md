# Backend

`app.py` exposes the FastAPI `POST /api/process` upload endpoint and mounts
`../frontend/` at the root path for both local use and Vercel static promotion.
The route delegates filtering and plot generation to
[`signals_systems/processing.py`](../signals_systems/processing.py).

The API caps uploads at 4 MB and recordings at 40 seconds to remain within
serverless request/response payload limits.
