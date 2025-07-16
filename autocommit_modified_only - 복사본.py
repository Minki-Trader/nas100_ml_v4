import time
import os
import subprocess
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

REPO_PATH = r"C:\Users\awdse\OneDrive\Desktop\nas100_ml_v4"
INTERVAL_SECONDS = 5

class SmartCommitHandler(FileSystemEventHandler):
    def __init__(self):
        self.modified_files = set()
        self.last_commit_time = time.time()

    def on_modified(self, event):
        if not event.is_directory:
            rel_path = os.path.relpath(event.src_path, REPO_PATH)
            self.modified_files.add(rel_path)

    def on_created(self, event):
        if not event.is_directory:
            rel_path = os.path.relpath(event.src_path, REPO_PATH)
            self.modified_files.add(rel_path)

    def try_commit(self):
        now = time.time()
        if self.modified_files and now - self.last_commit_time > INTERVAL_SECONDS:
            os.chdir(REPO_PATH)
            try:
                for file in self.modified_files:
                    subprocess.run(["git", "add", file], check=True)
                subprocess.run(["git", "commit", "-m", f"🔄 자동 커밋: {len(self.modified_files)}개 파일 변경"], check=True)
                subprocess.run(["git", "push"], check=True)
                print(f"✅ {len(self.modified_files)}개 파일 자동 커밋 & 푸시됨")
            except subprocess.CalledProcessError as e:
                print("❌ Git 오류:", e)
            self.modified_files.clear()
            self.last_commit_time = now

if __name__ == "__main__":
    os.chdir(REPO_PATH)
    event_handler = SmartCommitHandler()
    observer = Observer()
    observer.schedule(event_handler, path=REPO_PATH, recursive=True)
    observer.start()
    print(f"👀 {REPO_PATH}에서 생성/수정 파일 감시 중…")

    try:
        while True:
            time.sleep(1)
            event_handler.try_commit()
    except KeyboardInterrupt:
        observer.stop()
    observer.join()
