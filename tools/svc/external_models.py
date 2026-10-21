"""Import user-provided So-VITS-SVC inference weights as immutable model snapshots."""
import hashlib
import json
from pathlib import Path
import shutil

from . import storage as db
from .shallow_diffusion import FILES as DIFFUSION_FILES, source_files as diffusion_files, validate_config as validate_diffusion
from .install import ASSETS, ENCODER_NAME, ENCODER_768_NAME, ENCODER_VARIANTS
from tools.workspace_resources import require_space

INBOX = db.ROOT / "svc_imports"
LIBRARY = db.DATA / "imported_models"


def candidates():
    """Only direct child folders and ordinary files may be imported."""
    INBOX.mkdir(exist_ok=True)
    result, problems = [], []
    for folder in sorted(INBOX.iterdir()):
        if folder.is_symlink() or not folder.is_dir():
            continue
        config = folder / "config.json"
        weights = sorted(path for path in folder.glob("*.pth")
                         if path.is_file() and not path.is_symlink()
                         and not path.name.lower().startswith(("d_", "state_", "optimizer")))
        if not config.is_file() or config.is_symlink():
            problems.append(f"{folder.name}：缺少 config.json")
            continue
        if not weights:
            problems.append(f"{folder.name}：缺少生成器 .pth 权重")
        for weight in weights:
            result.append({"key": folder.name + "/" + weight.name,
                           "name": folder.name, "weight": weight.name})
    return result, problems


def source_for(key):
    parts = str(key or "").split("/")
    if len(parts) != 2 or any(part in {"", ".", ".."} for part in parts):
        raise ValueError("请选择本地模型目录中的权重")
    folder = INBOX / parts[0]
    weight, config = folder / parts[1], folder / "config.json"
    if (folder.is_symlink() or not folder.is_dir() or weight.is_symlink()
            or config.is_symlink() or not weight.is_file() or not config.is_file()
            or weight.suffix.lower() != ".pth" or weight.name.lower().startswith(("d_", "state_", "optimizer"))):
        raise ValueError("模型文件不完整或已更改，请重新扫描目录")
    return folder, weight, config


def normalize_config(path):
    try:
        config = json.loads(Path(path).read_text(encoding="utf-8"))
        model, data = config["model"], config["data"]
        encoder = model.get("speech_encoder") or ("vec256l9" if model.get("ssl_dim") == 256 else "")
        dimension = int(model.get("ssl_dim", 0))
        if encoder in {"vec256l9", ENCODER_NAME} and dimension == 256:
            normalized_encoder = ENCODER_NAME
        elif encoder in {"vec768l12", ENCODER_768_NAME} and dimension == 768:
            normalized_encoder = ENCODER_768_NAME
        else:
            raise ValueError("仅支持 ContentVec 第 9 层 / 256 维或第 12 层 / 768 维模型")
        if int(data.get("sampling_rate", 0)) != 44100:
            raise ValueError("当前转换器仅支持 44.1 kHz SVC 模型")
        speakers = config.get("spk")
        if not speakers and int(model.get("n_speakers", 0)) == 1:
            speakers = {"默认音色": 0}
        if not isinstance(speakers, dict) or not speakers:
            raise ValueError("config.json 缺少可选的说话人映射 spk")
        speakers = {str(name): int(index) for name, index in speakers.items()}
        if any(not name or index < 0 or index >= int(model["n_speakers"])
               for name, index in speakers.items()):
            raise ValueError("config.json 中的说话人编号与模型配置不符")
        model["speech_encoder"] = normalized_encoder
        model["sampling_rate"] = data["sampling_rate"]
        config["spk"] = speakers
        return config
    except (KeyError, TypeError, json.JSONDecodeError, UnicodeError) as error:
        raise ValueError("config.json 不是完整的 So-VITS-SVC 配置") from error


def records():
    result = []
    for path in LIBRARY.glob("*/model.json"):
        try:
            item = db.read(path)
            db.identifier(item["id"])
            if path.parent.name == item["id"] and (path.parent / "G.pth").is_file() and (path.parent / "config.json").is_file():
                result.append(item)
        except (OSError, KeyError, TypeError, ValueError):
            continue
    return sorted(result, key=lambda item: item["created_at"], reverse=True)


def record(model_id):
    model_id = db.identifier(model_id)
    path = LIBRARY / model_id / "model.json"
    item = db.read(path)
    if item.get("id") != model_id or not (path.parent / "G.pth").is_file() or not (path.parent / "config.json").is_file():
        raise ValueError("导入模型不完整，请重新导入")
    return item


def import_candidate(worker):
    folder, weight, config_path = source_for(worker.options["candidate"])
    diffusion = diffusion_files(folder)
    required = weight.stat().st_size + sum(path.stat().st_size for path in diffusion.values()) if diffusion else weight.stat().st_size
    require_space(LIBRARY, required * 2 + 64 * 1024 ** 2, "导入 SVC 模型")
    model_id = db.new_id()
    temporary = LIBRARY / (model_id + ".part")
    temporary.mkdir(parents=True)
    try:
        config = normalize_config(config_path)
        encoder_name = config["model"]["speech_encoder"]
        encoder_file = ENCODER_VARIANTS[encoder_name][0]
        if not (ASSETS / encoder_file).is_file():
            raise ValueError(f"缺少内容编码器 {encoder_file}，请先下载后重试导入")
        worker.update("复制并校验模型权重", 3)
        source_stat = weight.stat()
        digest = hashlib.sha256()
        copied = 0
        progress = worker.hash_progress("复制模型权重", 3, 65)
        with weight.open("rb") as source, (temporary / "G.pth").open("wb") as target:
            while block := source.read(1024 * 1024):
                worker.check()
                target.write(block)
                digest.update(block)
                copied += len(block)
                progress(copied, source_stat.st_size, weight.name)
        after = weight.stat()
        if copied != source_stat.st_size or (source_stat.st_dev, source_stat.st_ino, source_stat.st_mtime_ns) != (after.st_dev, after.st_ino, after.st_mtime_ns):
            raise ValueError("复制过程中源模型发生变化，请完成下载后重新导入")
        (temporary / "config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
        worker.check()
        worker.update("核对模型参数结构", 70)
        import torch
        from .engine import upstream, model_for, compatible_load
        upstream()
        torch.set_num_threads(4)
        model = model_for(config, "cpu")
        compatible_load(temporary / "G.pth", model, inference_only=True)
        del model
        worker.check()
        diffusion_meta = None
        diffusion_max_steps = None
        if diffusion:
            validate_diffusion(folder, config)
            diffusion_meta = {}
            total = sum(path.stat().st_size for path in diffusion.values())
            copied_diffusion = 0
            progress = worker.hash_progress("复制浅扩散与声码器", 75, 98)
            for key, source_path in diffusion.items():
                target = temporary / DIFFUSION_FILES[key]
                target.parent.mkdir(parents=True, exist_ok=True)
                before = source_path.stat()
                digest = hashlib.sha256()
                with source_path.open("rb") as source, target.open("wb") as output:
                    while block := source.read(1024 * 1024):
                        worker.check()
                        output.write(block)
                        digest.update(block)
                        copied_diffusion += len(block)
                        progress(copied_diffusion, total, source_path.name)
                after = source_path.stat()
                if (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns) != (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns):
                    raise ValueError("复制过程中浅扩散文件发生变化，请重新导入")
                diffusion_meta[key] = {"sha256": digest.hexdigest(), "bytes": target.stat().st_size}
            worker.update("核对浅扩散与声码器参数结构", 98)
            from .shallow_diffusion import load as load_diffusion
            diffusion_model, vocoder, _ = load_diffusion(temporary, config, "cpu")
            diffusion_max_steps = diffusion_model.k_step_max
            del diffusion_model, vocoder
            worker.check()
        item = {"id": model_id, "name": folder.name, "source": str(weight),
                "weight_name": weight.name, "sha256": digest.hexdigest(),
                "bytes": copied, "config_sha256": db.digest(temporary / "config.json"),
                "speakers": config["spk"], "encoder": encoder_name,
                "sampling_rate": 44100, "inference_mode": "base+shallow" if diffusion_meta else "base",
                "diffusion": diffusion_meta, "diffusion_max_steps": diffusion_max_steps,
                "created_at": db.now()}
        db.write(temporary / "model.json", item)
        temporary.rename(LIBRARY / model_id)
        worker.update("现成歌手模型已导入，可在完整音频转换中选择", 100,
                      outputs={"model_id": model_id})
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def attach_diffusion(model_id, folder):
    """Complete an existing base-only import with a matching diffusion bundle."""
    item = record(model_id)
    directory = LIBRARY / item["id"]
    folder = Path(folder).resolve()
    paths = diffusion_files(folder)
    if not paths:
        raise ValueError("目录中没有完整的浅扩散文件")
    if item.get("diffusion") or (directory / "diffusion").exists():
        raise ValueError("该模型已关联浅扩散文件")
    base = db.read(directory / "config.json")
    validate_diffusion(folder, base)
    from .engine import upstream
    from .shallow_diffusion import load as load_diffusion
    upstream()
    diffusion_model, vocoder, _ = load_diffusion(folder, base, "cpu")
    diffusion_max_steps = diffusion_model.k_step_max
    del diffusion_model, vocoder
    require_space(directory, sum(path.stat().st_size for path in paths.values()) * 2 + 1024 ** 2,
                  "关联浅扩散文件")
    staging = directory / "diffusion.part"
    staging.mkdir()
    metadata = {}
    try:
        for key, source_path in paths.items():
            relative = Path(DIFFUSION_FILES[key]).relative_to("diffusion")
            target = staging / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            before = source_path.stat()
            digest = hashlib.sha256()
            with source_path.open("rb") as source, target.open("wb") as output:
                while block := source.read(1024 * 1024):
                    output.write(block)
                    digest.update(block)
            after = source_path.stat()
            if (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns) != (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns):
                raise ValueError("浅扩散源文件在复制时发生变化")
            metadata[key] = {"sha256": digest.hexdigest(), "bytes": target.stat().st_size}
        staging.rename(directory / "diffusion")
        item["diffusion"] = metadata
        item["inference_mode"] = "base+shallow"
        item["diffusion_max_steps"] = diffusion_max_steps
        db.write(directory / "model.json", item)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return item
