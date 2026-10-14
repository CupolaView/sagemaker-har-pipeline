"""Make src/ importable in tests, mirroring how SageMaker runs the scripts:
from inside src/, with flat imports like `from model import HARConvNet`."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
