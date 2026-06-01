import argparse
import os
import sys
from modelscope.hub.snapshot_download import snapshot_download

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"


def download_from_huggingface(model_name: str, cache_dir: str = None):
    try:
        from huggingface_hub import snapshot_download as hf_snapshot_download
    except ImportError:
        print("huggingface_hub not found. Installing...")
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "huggingface-hub"])
        from huggingface_hub import snapshot_download as hf_snapshot_download

    print(f"Downloading from HuggingFace: {model_name}")

    model_dir = hf_snapshot_download(
        repo_id=model_name,
        cache_dir=cache_dir,
        revision="main"
    )

    print(f"Download success! Model saved to: {model_dir}")
    return model_dir


def download_from_modelscope(model_name: str, cache_dir: str = None):
    print(f"download model start...: {model_name}")

    model_dir = snapshot_download(
        model_id=model_name,
        cache_dir=cache_dir,
        revision="master"
    )

    print(f"download model success!: {model_dir}")
    return model_dir


def download_model(model_name: str, cache_dir: str = None, source: str = "modelscope"):
    print(f"Download model start...: {model_name} from {source}")

    if source == "modelscope":
        return download_from_modelscope(model_name, cache_dir)
    elif source == "huggingface":
        return download_from_huggingface(model_name, cache_dir)
    else:
        raise ValueError(f"Unsupported source: {source}. Use 'modelscope' or 'huggingface'")


def print_example():
    example_text = """
========== 使用示例 ==========

1. 从 ModelScope 下载模型（默认）：
   python download_model.py --name "Qwen/Qwen-7B" --cache_dir ./my_models

2. 从 HuggingFace 下载模型：
   python download_model.py --name "meta-llama/Llama-2-7b-hf" --source huggingface --cache_dir ./hf_models

3. 单独显示本示例信息：
   python download_model.py --example

=============================
    """
    print(example_text)
    sys.exit(0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="download model from modelscope")
    parser.add_argument(
        "--name",
        type=str,
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default="./",
    )

    parser.add_argument(
        "--source",
        type=str,
        default="modelscope",
        choices=["modelscope", "huggingface"],
        help="Download source: modelscope or huggingface"
    )
    parser.add_argument(
        "--example",
        action="store_true",
        help="显示详细的使用示例并退出"
    )

    args = parser.parse_args()
    if args.example:
        print_example()
    download_model(args.name, args.cache_dir, args.source)