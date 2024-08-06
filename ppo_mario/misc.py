from pathlib import Path
import shutil


def get_device() -> str:
    import torch

    if torch.backends.mps.is_available():
        return "mps"
    elif torch.cuda.is_available():
        return "cuda"
    else:
        return "cpu"


def copy_preivous_logs(model_file: str | Path, work_dir: Path):
    """Copy the logs from the previous training if available"""
    model_file = Path(model_file)
    # ignore if alredy in the work directory
    if str(model_file.parent.absolute()).startswith(str(work_dir.absolute())):
        print("The previous model is already in the work directory.")
        return

    # find the log dir
    previous_log_dir = model_file.parent / "logs"
    if not (previous_log_dir.exists or previous_log_dir.is_dir()):
        print("No previous logs found")
        return

    # enumerate the log files
    log_files = list(previous_log_dir.glob("events.*"))
    print(f"Found {len(log_files)} log files.")

    # copy the log files
    target = work_dir / "logs"
    target.mkdir(parents=True, exist_ok=True)
    for log_file in log_files:
        shutil.copy2(log_file, target)

    print(f"Copied {len(log_files)} log files to {target}")


def launch_tensorboard(log_dir: Path):
    """Start a tensorboard instance with the given log directory"""
    from tensorboard.program import TensorBoard

    tensorboard = TensorBoard()
    tensorboard.configure(argv=[None, "--logdir", str(log_dir), "--port", "7007"])
    tensorboard.launch()


def get_sub_process_start_method() -> str:
    import platform

    if platform.platform() == "Darwin":
        return "spawn"
    else:
        return "fork"


class WorkDir:

    def __init__(self, work_dir: str | Path):
        self.root = Path(work_dir)

    def child(self, name: str | Path) -> Path:
        return self.root / name

    @property
    def config(self) -> Path:
        return self.child("config.json")

    @property
    def logs(self) -> Path:
        return self.child("logs")

    @property
    def checkpoints(self) -> Path:
        return self.child("checkpoints")

    @property
    def saved_model(self) -> Path:
        return self.child("model.zip")

    @property
    def base_model(self) -> Path:
        return self.child("base_model.zip")

    def mkdirs(self):
        """Create the directories"""
        self.logs.mkdir(parents=True, exist_ok=False)
        self.checkpoints.mkdir(parents=False, exist_ok=False)
