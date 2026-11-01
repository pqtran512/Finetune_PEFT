# AJAC Frontend (React)

Console UI theo phong cách Secure Agentic BI: split-pane, NDJSON live trace, X-Ray, Run panel.

## Dev (hot reload)

1. Terminal 1 — Flask API (mock khuyến nghị khi UI-only):

```bat
..\run_prototype_mock.bat
```

hoặc `python prototype/app.py --mock` từ root repo.

2. Terminal 2 — Vite:

```bat
npm install
npm run dev
```

Mở http://127.0.0.1:5173 (proxy `/api` → Flask `:5000`).

## Production / demo qua Flask

```bat
npm run build
```

Sau đó `run_prototype.bat` / `run_prototype_mock.bat` phục vụ SPA tại http://127.0.0.1:5000  
UI cũ: http://127.0.0.1:5000/legacy
