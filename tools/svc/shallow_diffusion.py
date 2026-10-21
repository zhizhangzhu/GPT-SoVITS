"""Optional 4.1 shallow diffusion for imported SVC inference models."""
import json
from pathlib import Path

import torch
import yaml


FILES = {
    "model": "diffusion/model_30000.pt",
    "config": "diffusion/config.yaml",
    "vocoder": "diffusion/vocoder/model",
    "vocoder_config": "diffusion/vocoder/config.json",
}


def source_files(folder):
    folder = Path(folder)
    if (folder / "diffusion").is_symlink() or (folder / "diffusion/vocoder").is_symlink():
        raise ValueError("浅扩散目录不能是符号链接")
    paths = {key: folder / relative for key, relative in FILES.items()}
    present = [path.is_file() and not path.is_symlink() for path in paths.values()]
    if any(present) and not all(present):
        missing = [key for (key, _), exists in zip(paths.items(), present) if not exists]
        raise ValueError("浅扩散文件不完整，缺少：" + "、".join(missing))
    return paths if all(present) else None


def validate_config(folder, base):
    paths = source_files(folder)
    if not paths:
        return None
    config = yaml.safe_load(paths["config"].read_text(encoding="utf-8"))
    vocoder = json.loads(paths["vocoder_config"].read_text(encoding="utf-8"))
    data, model = config["data"], config["model"]
    if (int(data["sampling_rate"]) != int(base["data"]["sampling_rate"])
            or int(data["block_size"]) != int(base["data"]["hop_length"])
            or int(data["encoder_out_channels"]) != int(base["model"]["ssl_dim"])
            or data["encoder"].removesuffix("-onnx") != base["model"]["speech_encoder"].removesuffix("-onnx")
            or config["vocoder"]["type"] not in {"nsf-hifigan", "nsf-hifigan-log10"}
            or int(vocoder["sampling_rate"]) != int(data["sampling_rate"])
            or int(vocoder["hop_size"]) != int(data["block_size"]) or int(vocoder["num_mels"]) != 128):
        raise ValueError("浅扩散配置与基础 SVC 模型或声码器不匹配")
    if int(model["n_spk"]) < max(base["spk"].values()) + 1:
        raise ValueError("浅扩散模型的说话人数量与基础模型不匹配")
    if config["infer"]["method"] not in {"dpm-solver", "dpm-solver++", "pndm", "ddim", "unipc"}:
        raise ValueError("浅扩散采样方式不受支持")
    if int(config["infer"]["speedup"]) < 1:
        raise ValueError("浅扩散采样加速参数无效")
    return config


def load(folder, base, device):
    """Use weights_only for both externally supplied checkpoints."""
    config = validate_config(folder, base)
    if config is None:
        raise ValueError("当前导入模型没有完整的浅扩散文件")
    from diffusion.unit2mel import Unit2Mel
    from diffusion.vocoder import Vocoder
    from vdecoder.nsf_hifigan.models import Generator

    paths = source_files(folder)
    vocoder = Vocoder(config["vocoder"]["type"], str(paths["vocoder"]), device=device)
    vocoder_model = Generator(vocoder.vocoder.h).to(device)
    try:
        state = torch.load(paths["model"], map_location="cpu", weights_only=True)
        schedule_steps = int(state["model"]["decoder.betas"].numel())
        if not 1 <= schedule_steps <= int(config["model"]["timesteps"]):
            raise ValueError("浅扩散权重中的噪声调度长度无效")
        vocoder_state = torch.load(paths["vocoder"], map_location="cpu", weights_only=True)
        vocoder_model.load_state_dict(vocoder_state["generator"], strict=True)
        vocoder_model.eval()
        vocoder_model.remove_weight_norm()
        vocoder.vocoder.model = vocoder_model
        settings = config["model"]
        diffusion = Unit2Mel(config["data"]["encoder_out_channels"], settings["n_spk"],
                             settings["use_pitch_aug"], vocoder.dimension,
                             settings["n_layers"], settings["n_chans"], settings["n_hidden"],
                             schedule_steps, schedule_steps).to(device)
        diffusion.load_state_dict(state["model"], strict=True)
    except (KeyError, RuntimeError, TypeError) as error:
        raise ValueError("浅扩散权重或声码器与配置不兼容") from error
    diffusion.eval()
    return diffusion, vocoder, config
