.PHONY: up down setup catalog download transcribe chunk index summaries serve status eval pipeline
up:        ; docker compose up -d
down:      ; docker compose down
setup:     ; pip install -r requirements.txt && ollama pull nomic-embed-text && ollama pull qwen2.5:14b
catalog:   ; python -m ragio catalog
download:  ; python -m ragio download $(ARGS)
transcribe:; python -m ragio transcribe $(ARGS)
chunk:     ; python -m ragio chunk
index:     ; python -m ragio index $(ARGS)
summaries: ; python -m ragio index --summaries
status:    ; python -m ragio status
eval:      ; python -m ragio eval
serve:     ; uvicorn app:app --reload --port 8000
# First taste: 3 episodes end-to-end
pipeline:  ; $(MAKE) catalog && $(MAKE) download ARGS="--limit 3" && $(MAKE) transcribe ARGS="--limit 3" && $(MAKE) chunk && $(MAKE) index
