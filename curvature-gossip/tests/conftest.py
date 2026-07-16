"""测试入口：将 src 布局加入 Python 路径，避免依赖预先安装包。"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

