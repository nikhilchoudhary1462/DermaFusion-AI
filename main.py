import os
import sys
import webbrowser
import threading
import time
import uvicorn

def open_browser():
    time.sleep(1.5)
    url = "http://127.0.0.1:8000"
    print(f"\n[DermaFusion AI] Opening website at {url} ...")
    webbrowser.open(url)

if __name__ == "__main__":
    print("=====================================================")
    print("  DermaFusion AI — Multimodal Skin Lesion Web App   ")
    print("=====================================================")
    print("Starting server on http://127.0.0.1:8000")
    print("Press Ctrl+C to stop the server.\n")

    threading.Thread(target=open_browser, daemon=True).start()
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=False)
