# download.py 仅用于下载模型，不要写到chainst.py里
import os
# 全局环境变量，放在最顶部
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HF_HOME"] = r"E:\school\2025-2026-3\shixi\huggingface_cache"
os.environ["HF_HUB_DISABLE_SYMLINKS"] = "1"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] = "600"

from huggingface_hub import snapshot_download

# 这里才可以传local_dir_use_symlinks、resume_download
snapshot_download(
    repo_id="BAAI/bge-m3",
    local_dir_use_symlinks=False,
    resume_download=True
)