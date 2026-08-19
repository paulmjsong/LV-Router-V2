SaeGyeol low-latency routing and streaming patch

Apply from PowerShell:

  Expand-Archive .\saegyeol_streaming_patch_bundle.zip -DestinationPath .\saegyeol_streaming_patch_bundle
  python .\saegyeol_streaming_patch_bundle\apply_patch.py .\saegyeol_lab_ai_local_router

Then, from the project directory:

  docker compose --profile ollama down --remove-orphans
  docker compose --profile ollama up -d --build --force-recreate
  docker compose logs -f backend litellm ollama open-webui

The updater creates a timestamped .patch-backup-* directory inside the project before replacing files.
