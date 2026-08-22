Place the original GIST regulation source documents in this directory, then run:

```bash
docker compose exec backend python -m app.admin_cli upload-regulations \
  /imports/gist-regulations
```

The command scans supported files recursively. The legacy `index.faiss` and `index.pkl` files are not consumed by the service.
