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
    # find the log dir
    previous_log_dir = Path(model_file).parent / "logs"
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
