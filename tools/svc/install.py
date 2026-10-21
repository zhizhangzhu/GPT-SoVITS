"""Reproducible runtime installer: python -m tools.svc.install.

Run with the same Python/CUDA environment as the main workbench.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import shutil
import subprocess
import sys
import venv

from .storage import ROOT, digest, write

COMMIT = "730930d337d171479eadf305f96cbed4bb393e77"
REPO = "https://github.com/svc-develop-team/so-vits-svc.git"
RUNTIME = ROOT / "runtime/so-vits-svc"
ENV = ROOT / "runtime/svc-env"
ASSETS = ROOT / "runtime/svc-assets"
ENCODER_REPO = "NaruseMioShirakana/MoeSS-SUBModel"
PRETRAIN_REPO = "innnky/sovits_pretrained"
ENCODER_REVISION = "3ff2e37e161ebcb3c4711e7ec870472342260842"
PRETRAIN_REVISION = "ca0b06142829e73873324e52fbc30e6a78e28262"
ENCODER_FILE = "vec-256-layer-9.onnx"
ENCODER_NAME = "vec256l9-onnx"
ENCODER_768_FILE = "vec-768-layer-12.onnx"
ENCODER_768_NAME = "vec768l12-onnx"
ENCODER_VARIANTS = {
    ENCODER_NAME: (ENCODER_FILE, 256),
    ENCODER_768_NAME: (ENCODER_768_FILE, 768),
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-pip", action="store_true")
    parser.add_argument("--skip-models", action="store_true")
    parser.add_argument("--with-vec768", action="store_true", help="安装外部 768 维模型所需的 ContentVec 编码器")
    parser.add_argument("--with-vocal-split", action="store_true", help="安装主唱 / 和声分离模型（约 1.7 GB）")
    parser.add_argument("--with-separation", action="store_true", help="安装新版 Deux 与 BS-RoFormer Karaoke 分离模型（合计约 640 MB）")
    parser.add_argument("--with-dereverb", action="store_true", help="安装可选的 BS-RoFormer De-Reverb 模型（约 204 MB）")
    parser.add_argument("--with-pitch", action="store_true", help="安装 RMVPE 神经音高模型；FCPE 随依赖安装")
    parser.add_argument("--with-pitch-correction", action="store_true", help="安装可选 PC-NSF-HiFiGAN 修音声码器")
    args = parser.parse_args()
    if not RUNTIME.exists():
        subprocess.run(["git", "clone", "--depth", "1", "--branch", "4.1-Stable", REPO, str(RUNTIME)], check=True)
    revision = subprocess.check_output(["git", "-C", str(RUNTIME), "rev-parse", "HEAD"], text=True).strip()
    if revision != COMMIT:
        raise RuntimeError(f"SVC 运行目录版本不匹配：{revision}，要求 {COMMIT}；请保存现有目录后重新安装")
    if not (ENV / "bin/python").exists():
        venv.EnvBuilder(with_pip=True, system_site_packages=True).create(ENV)
    if not args.skip_pip:
        subprocess.run([str(ENV / "bin/python"), "-m", "pip", "install", "-r", str(ROOT / "requirements-svc.txt")], check=True)
    ASSETS.mkdir(parents=True, exist_ok=True)
    if not args.skip_models:
        from .download import download
        import os
        endpoint = os.environ.get("HF_ENDPOINT", "https://huggingface.co")
        assets = [
            (ENCODER_REPO, ENCODER_FILE, ENCODER_FILE, "GPL-3.0"),
            (PRETRAIN_REPO, "sovits4/G_0.pth", "G_0.pth", "CC-BY-NC-4.0"),
            (PRETRAIN_REPO, "sovits4/D_0.pth", "D_0.pth", "CC-BY-NC-4.0"),
        ]
        if args.with_vec768:
            assets.append((ENCODER_REPO, ENCODER_768_FILE, ENCODER_768_FILE, "GPL-3.0"))
        def fetch(asset):
            repo, filename, target, license_name = asset
            print(f"下载 {repo}/{filename}", flush=True)
            revision = ENCODER_REVISION if repo == ENCODER_REPO else PRETRAIN_REVISION
            verified = download(repo, filename, ASSETS / target, endpoint=endpoint, workers=8, revision=revision)
            return {"repo": repo, "file": filename, "local": target, "license": license_name, **verified}
        with ThreadPoolExecutor(max_workers=3) as pool:
            manifest = list(pool.map(fetch, assets))
        write(ASSETS / "manifest.json", manifest)
    (RUNTIME / "pretrain").mkdir(exist_ok=True)
    target = RUNTIME / "pretrain" / ENCODER_FILE
    if not target.exists() and (ASSETS / target.name).exists():
        target.symlink_to(ASSETS / target.name)
    if not args.skip_models:
        subprocess.run([str(ENV / "bin/python"), "-m", "tools.svc.check_backend"], cwd=ROOT, check=True)
    if args.with_vocal_split:
        from .vocal_split import install_model
        install_model()
    if args.with_separation:
        from .separation import install_model
        for model_id in ("mel_deux", "bs_karaoke"):
            install_model(model_id)
    if args.with_dereverb:
        from .separation import install_model
        install_model("dereverb")
    if args.with_pitch:
        from .pitch import install_models
        install_models()
    if args.with_pitch_correction:
        from .pitch_correction import install_model
        install_model()
    write(ROOT / "runtime/svc-runtime.json", {"commit": COMMIT, "repo": REPO, "license": "AGPL-3.0",
          "python": str(ENV / "bin/python"), "assets": str(ASSETS), "encoder": ENCODER_NAME})
    print("SVC 后端已安装。默认预训练权重限非商业使用，来源记录在 runtime/svc-assets/manifest.json。")


if __name__ == "__main__":
    main()
