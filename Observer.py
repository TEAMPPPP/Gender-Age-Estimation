from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import time
import os
import subprocess
from rich.console import Console  # Import the Console class from rich.

console = Console()  # Create a console object.

class Watcher:
    DIRECTORY_TO_WATCH = "./input"

    def __init__(self):
        self.observer = Observer()

    def run(self):
        console.print("[bold magenta]Script started. Watching directory '{}'...[/bold magenta]".format(self.DIRECTORY_TO_WATCH), style="bold blue")
        event_handler = Handler()
        self.observer.schedule(event_handler, self.DIRECTORY_TO_WATCH, recursive=True)
        self.observer.start()
        try:
            while True:
                time.sleep(5)
        except KeyboardInterrupt:
            self.observer.stop()
            console.print("[bold red]Stopping watcher.[/bold red]", style="bold red")
        self.observer.join()

class Handler(FileSystemEventHandler):
    @staticmethod
    def on_created(event):
        if not event.is_directory:
            console.print("[blue]****************************************************************************************************[/blue]", style="bold blue")
            console.print(f"[green]New image detected: {event.src_path}[/green]", style="bold green")
            subprocess.run(['python', './tools/demo.py', '--image_path', event.src_path], check=True)
            console.print("[blue]****************************************************************************************************[/blue]", style="bold blue")
if __name__ == "__main__":
    w = Watcher()
    w.run()
