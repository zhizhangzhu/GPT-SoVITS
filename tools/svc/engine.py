# SPDX-License-Identifier: AGPL-3.0-only
# Training adapter derived from the objectives and interfaces of:
# svc-develop-team/so-vits-svc, commit 730930d337d171479eadf305f96cbed4bb393e77.
# Upstream license is preserved by the installer at runtime/so-vits-svc/LICENSE.
"""Single GPU training and timed conversion through pinned upstream models."""
import os
from pathlib import Path
import random
import sys
import tempfile

import numpy as np
import soundfile as sf
import torch
from torch.nn import functional as F

from . import storage as db
from .install import RUNTIME, ASSETS, ENCODER_NAME, ENCODER_768_NAME, ENCODER_VARIANTS
from .training_order import resumed_order


def upstream():
    sys.path.insert(0, str(RUNTIME))
    # The upstream encoder's relative paths are intentionally confined to its runtime.
    os.chdir(RUNTIME)
    import utils
    return utils


def model_for(config, device):
    from models import SynthesizerTrn
    return SynthesizerTrn(config["data"]["filter_length"] // 2 + 1,
                          config["train"]["segment_size"] // config["data"]["hop_length"], **config["model"]).to(device)


def compatible_load(path, model, initial=False, inference_only=False):
    # Locally generated or user-selected checkpoints only; reject TTS and incompatible architectures.
    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    except Exception as error:
        raise ValueError("无法安全读取此权重，请选择兼容 SVC 的 G 文件；不接受 GPT-SoVITS-TTS 或包含自定义对象的检查点") from error
    saved = checkpoint.get("model") if isinstance(checkpoint, dict) else None
    if not isinstance(saved, dict) or "pre.weight" not in saved or "emb_uv.weight" not in saved:
        raise ValueError("这不是兼容的 So-VITS-SVC 权重；GPT-SoVITS-TTS 权重不能用于 SVC")
    expected = model.state_dict()
    # A new single-singer experiment owns its speaker table. Pretraining may
    # omit that table, or contain multiple speakers: reuse row zero only.
    if initial and "emb_g.weight" in saved and saved["emb_g.weight"].shape[1:] == expected["emb_g.weight"].shape[1:]:
        saved = dict(saved)
        saved["emb_g.weight"] = saved["emb_g.weight"][:expected["emb_g.weight"].shape[0]]
    wrong = [key for key in expected if key in saved and expected[key].shape != saved[key].shape]
    missing = [key for key in expected if key not in saved]
    permitted = lambda key: (initial and (key.startswith("f0_decoder.") or key == "emb_g.weight")) or (inference_only and key.startswith("enc_q."))
    if wrong or any(not permitted(key) for key in missing):
        raise ValueError("SVC 权重结构与模型配置不符：" + ", ".join((wrong + missing)[:8]))
    model.load_state_dict({key: value for key, value in saved.items() if key in expected}, strict=not (initial or inference_only))
    if missing:
        if inference_only:
            print(f"推理权重省略了 {len(missing)} 个训练专用后验编码器张量；推理参数已完整加载", flush=True)
        else:
            print(f"兼容 4.0 底模：目标歌手嵌入及新增 F0 预测器中 {len(missing)} 个张量将参与微调；推理保留源 F0", flush=True)
    return checkpoint


def encoder_for(device, encoder_name=ENCODER_NAME):
    if encoder_name not in ENCODER_VARIANTS:
        raise ValueError("模型使用的内容编码器尚未支持")
    filename, dimension = ENCODER_VARIANTS[encoder_name]
    asset = ASSETS / filename
    if not asset.is_file():
        raise ValueError(f"缺少内容编码器 {filename}，请先安装对应模型文件")
    if encoder_name == ENCODER_768_NAME:
        from vencoder.ContentVec768L12_Onnx import ContentVec768L12_Onnx as Encoder
    else:
        from vencoder.ContentVec256L9_Onnx import ContentVec256L9_Onnx as Encoder
    # CPU ONNX keeps installation independent of CUDA/cuDNN provider versions.
    import onnxruntime as ort
    encoder = Encoder.__new__(Encoder)
    encoder.dev = torch.device(device)
    encoder.hidden_dim = dimension
    settings = ort.SessionOptions()
    settings.intra_op_num_threads = 4
    settings.inter_op_num_threads = 1
    encoder.model = ort.InferenceSession(str(asset), sess_options=settings, providers=["CPUExecutionProvider"])
    return encoder


def predictor_for(config, name):
    from .pitch import METHODS, Predictor
    if name not in METHODS:
        raise ValueError("请选择有效的 F0 提取方式")
    if name in {"rmvpe", "fcpe"}:
        return Predictor(name, config["data"]["sampling_rate"], config["data"]["hop_length"])
    import utils
    return utils.get_f0_predictor(name, hop_length=config["data"]["hop_length"], sampling_rate=config["data"]["sampling_rate"])


def features(worker, directory, config, device):
    import librosa
    from modules.mel_processing import spectrogram_torch
    encoder = encoder_for(device)
    predictor = predictor_for(config, worker.options["predictor"])
    snapshot = db.read(directory / "dataset_snapshot.json")
    key = dict(snapshot["feature_key"], encoder_sha256=db.digest(ASSETS / ENCODER_FILE))
    previous = db.read(directory / "features/status.json", {"key": None, "files": {}})
    valid = previous["files"] if previous["key"] == key else {}
    for index, row in enumerate(snapshot["clips"]):
        worker.check()
        path = Path(row["snapshot_file"])
        generated = [Path(str(path) + ".soft.pt"), Path(str(path) + ".f0.npy"), path.with_suffix(".spec.pt")]
        if valid.get(row["id"]) == row["sha256"] and all(p.exists() for p in generated):
            continue
        samples, rate = sf.read(path, dtype="float32")
        if rate != 44100 or samples.ndim != 1 or db.digest(path) != row["sha256"]:
            raise ValueError("训练快照被修改或不是 44.1 kHz 单声道人声，请重新创建实验")
        tensor = torch.from_numpy(samples).to(device)
        wave16 = librosa.resample(samples, orig_sr=rate, target_sr=16000)
        with torch.no_grad():
            content = encoder.encoder(torch.from_numpy(wave16).to(device)).cpu()
            spec = spectrogram_torch(tensor.unsqueeze(0), config["data"]["filter_length"], rate,
                                     config["data"]["hop_length"], config["data"]["win_length"], center=False).squeeze(0).cpu()
        f0, uv = predictor.compute_f0_uv(samples)
        torch.save(content, generated[0])
        np.save(generated[1], np.array([f0, uv], dtype=object))
        torch.save(spec, generated[2])
        valid[row["id"]] = row["sha256"]
        db.write(directory / "features/status.json", {"key": key, "files": valid})
        worker.update(f"内容特征与 F0 {index + 1}/{len(snapshot['clips'])}", 5 + (index + 1) / len(snapshot["clips"]) * 20)
    del encoder


def validation_previews(directory, snapshot):
    """Choose stable validation clips across source recordings and median F0."""
    paths = set((directory / "filelists/val.txt").read_text(encoding="utf-8").splitlines())
    target = directory / "previews/selection.json"
    if target.exists():
        saved = db.read(target)
        if saved and all(row["file"] in paths and Path(row["file"]).is_file() for row in saved):
            return saved
    candidates = []
    for row in snapshot["clips"]:
        path = row["snapshot_file"]
        if path not in paths:
            continue
        f0, uv = np.load(path + ".f0.npy", allow_pickle=True)
        f0, uv = np.asarray(f0, dtype=float), np.asarray(uv, dtype=float)
        voiced = f0[(uv > 0) & np.isfinite(f0) & (f0 > 0)]
        note = float(69 + 12 * np.log2(np.median(voiced) / 440)) if len(voiced) else None
        candidates.append({"id": row["id"], "file": path, "source_id": row["source_id"],
                           "source_name": row.get("source_name", "源素材"), "median_note": note,
                           "duration": row.get("duration", 0)})
    if not candidates:
        raise ValueError("验证集没有可用的试听片段")
    voiced = sorted((row for row in candidates if row["median_note"] is not None), key=lambda row: (row["median_note"], row["id"]))
    targets = [voiced[round(index * (len(voiced) - 1) / max(1, min(3, len(candidates)) - 1))]["median_note"]
               for index in range(min(3, len(candidates)))] if voiced else [None] * min(3, len(candidates))
    selected, used_sources = [], set()
    for target_note in targets:
        remaining = [row for row in candidates if row["id"] not in {item["id"] for item in selected}]
        chosen = min(remaining, key=lambda row: (
            row["source_id"] in used_sources,
            row["median_note"] is None,
            abs(row["median_note"] - target_note) if row["median_note"] is not None and target_note is not None else 0,
            row["id"]))
        selected.append(chosen)
        used_sources.add(chosen["source_id"])
    target.parent.mkdir(exist_ok=True)
    db.write(target, selected)
    return selected


def train(worker):
    utils = upstream()
    if not torch.cuda.is_available():
        raise ValueError("SVC 训练需要可用的 CUDA GPU")
    device = torch.device("cuda:0")
    torch.set_num_threads(4)
    directory = db.experiment_dir(worker.options["experiment_id"])
    config = db.read(directory / "config.json")
    config["model"]["sampling_rate"] = config["data"]["sampling_rate"]
    db.write(directory / "config.json", config)
    identity = db.read(directory / "experiment.json")
    identity["status"] = "training"
    db.write(directory / "experiment.json", identity)
    random.seed(config["train"]["seed"])
    torch.manual_seed(config["train"]["seed"])
    np.random.seed(config["train"]["seed"])
    # Reject incompatible user weights before spending time on feature extraction.
    preflight = db.read(directory / "checkpoints/latest.json", {"step": 0})
    initial = not preflight["step"]
    candidate = directory / "initial/G_0.pth" if initial else directory / "checkpoints" / preflight["generator"]
    temporary_model = model_for(config, "cpu")
    compatible_load(candidate, temporary_model, initial=initial)
    del temporary_model
    features(worker, directory, config, device)
    from data_utils import TextAudioCollate, TextAudioSpeakerLoader
    from models import MultiPeriodDiscriminator
    from modules import commons
    from modules.losses import discriminator_loss, generator_loss, feature_loss, kl_loss
    from modules.mel_processing import spec_to_mel_torch, mel_spectrogram_torch
    from torch.utils.data import DataLoader, Subset
    from torch.utils.tensorboard import SummaryWriter
    hps = utils.get_hparams_from_file(str(directory / "config.json"))
    dataset = TextAudioSpeakerLoader(hps.data.training_files, hps, all_in_mem=False, vol_aug=False)
    batch_count = (len(dataset) + config["train"]["batch_size"] - 1) // config["train"]["batch_size"]
    validation = TextAudioSpeakerLoader(hps.data.validation_files, hps, vol_aug=False)
    preview_rows = validation_previews(directory, db.read(directory / "dataset_snapshot.json"))
    generator = model_for(config, device)
    discriminator = MultiPeriodDiscriminator(config["model"]["use_spectral_norm"]).to(device)
    options = config["train"]
    optim_g = torch.optim.AdamW(generator.parameters(), lr=options["learning_rate"], betas=tuple(options["betas"]), eps=options["eps"])
    optim_d = torch.optim.AdamW(discriminator.parameters(), lr=options["learning_rate"], betas=tuple(options["betas"]), eps=options["eps"])
    checkpoints = directory / "checkpoints"
    checkpoints.mkdir(exist_ok=True)
    latest = db.read(checkpoints / "latest.json", {"step": 0})
    step, epoch, skip = int(latest["step"]), 1, 0
    resume_rng = None
    sample_order = None
    training_list_digest = db.digest(directory / "filelists/train.txt")
    if step:
        compatible_load(checkpoints / latest["generator"], generator)
        discriminator.load_state_dict(torch.load(checkpoints / latest["discriminator"], map_location="cpu", weights_only=True)["model"], strict=True)
        state = torch.load(checkpoints / latest["state"], map_location="cpu", weights_only=True)
        if state.get("training_list_digest") not in (None, training_list_digest):
            raise ValueError("训练清单与检查点不一致，已停止续训")
        optim_g.load_state_dict(state["optimizer_g"])
        optim_d.load_state_dict(state["optimizer_d"])
        epoch, skip = state["epoch"], state["next_batch"]
        resume_rng = state
        worker.update(f"从第 {step} 步续训，目标 {options['max_steps']} 步", 25)
    else:
        compatible_load(directory / "initial/G_0.pth", generator, initial=True)
        saved_d = torch.load(directory / "initial/D_0.pth", map_location="cpu", weights_only=True)
        try:
            discriminator.load_state_dict(saved_d["model"], strict=True)
        except (KeyError, RuntimeError) as error:
            raise ValueError("预训练判别器与当前 SVC 配置不兼容") from error
    writer = SummaryWriter(str(directory / "logs"))
    next_batch = skip
    last_evaluated_step = None
    if step:
        sample_order = resumed_order(len(dataset), options["seed"], epoch, resume_rng)

    def save():
        if step == 0:
            return
        names = {"generator": f"G_{step}.pth", "discriminator": f"D_{step}.pth", "state": f"state_{step}.pth"}
        payloads = [
            (names["generator"], {"model": generator.state_dict(), "iteration": epoch, "learning_rate": optim_g.param_groups[0]["lr"], "format": db.VERSION, "step": step}),
            (names["discriminator"], {"model": discriminator.state_dict(), "iteration": epoch, "learning_rate": optim_d.param_groups[0]["lr"], "step": step}),
            (names["state"], {"optimizer_g": optim_g.state_dict(), "optimizer_d": optim_d.state_dict(), "epoch": epoch,
                              "next_batch": next_batch, "sample_order": sample_order, "training_list_digest": training_list_digest,
                              "random": random.getstate(), "torch_rng": torch.get_rng_state(),
                              "cuda_rng": torch.cuda.get_rng_state_all(), "step": step}),
        ]
        from tools.workspace_resources import require_space
        def tensor_bytes(value):
            if isinstance(value, torch.Tensor):
                return value.numel() * value.element_size()
            if isinstance(value, dict):
                return sum(tensor_bytes(item) for item in value.values())
            if isinstance(value, (tuple, list)):
                return sum(tensor_bytes(item) for item in value)
            return 0
        require_space(checkpoints, sum(tensor_bytes(payload) for _, payload in payloads)
                      + 16 * 1024 ** 2, "保存完整训练检查点")
        for name, payload in payloads:
            temp = checkpoints / (name + ".tmp")
            torch.save(payload, temp)
            os.replace(temp, checkpoints / name)
        index_path = checkpoints / "checkpoint-index.json"
        index = db.read(index_path, {})
        index[str(step)] = {key: {"name": name, "bytes": (checkpoints / name).stat().st_size,
                                  "sha256": db.digest(checkpoints / name)} for key, name in names.items()}
        db.write(index_path, index)
        db.write(checkpoints / "latest.json", dict(names, step=step))
        worker.update(f"已保存检查点：第 {step} 步", checkpoint=str(checkpoints / names["generator"]))

    def evaluate():
        python_rng = random.getstate()
        cpu_rng = torch.get_rng_state()
        cuda_rng = torch.cuda.get_rng_state_all()
        generator.eval()
        with torch.no_grad():
            target = directory / "previews"
            target.mkdir(exist_ok=True)
            losses = []
            for index, row in enumerate(preview_rows):
                c, f0, spec, wave, speaker, uv, volume = validation.get_audio(row["file"])
                converted, _ = generator.infer(c.unsqueeze(0).to(device), f0.unsqueeze(0).to(device), uv.unsqueeze(0).to(device),
                                              g=speaker.unsqueeze(0).to(device), predict_f0=False)
                samples = converted[0, 0].float().cpu().numpy()
                reference = target / f"reference-{row['id']}.wav"
                if not reference.exists():
                    sf.write(reference, wave[0].numpy(), 44100, subtype="PCM_16")
                sf.write(target / f"step-{step}-{row['id']}.wav", samples, 44100, subtype="PCM_16")
                preview_mel = mel_spectrogram_torch(converted.squeeze(1), hps.data.filter_length, hps.data.n_mel_channels,
                    hps.data.sampling_rate, hps.data.hop_length, hps.data.win_length, hps.data.mel_fmin, hps.data.mel_fmax)
                reference_mel = spec_to_mel_torch(spec.unsqueeze(0).to(device), hps.data.filter_length, hps.data.n_mel_channels,
                    hps.data.sampling_rate, hps.data.mel_fmin, hps.data.mel_fmax)
                count = min(preview_mel.shape[-1], reference_mel.shape[-1])
                loss = F.l1_loss(preview_mel[..., :count], reference_mel[..., :count]).item()
                losses.append(loss)
                writer.add_scalar(f"validation/clip_{index + 1}_mel_l1", loss, step)
            mean_loss = sum(losses) / len(losses)
            writer.add_scalar("validation/mel_l1", mean_loss, step)
            worker.update(f"{len(losses)} 段固定验证素材平均 mel L1 {mean_loss:.4f}（非音色相似度评分）")
        generator.train()
        random.setstate(python_rng)
        torch.set_rng_state(cpu_rng)
        torch.cuda.set_rng_state_all(cuda_rng)

    try:
        while step < options["max_steps"]:
            saved_epoch_state = resume_rng
            if resume_rng is not None:
                random.setstate(resume_rng["random"])
                torch.set_rng_state(resume_rng["torch_rng"])
                torch.cuda.set_rng_state_all(resume_rng["cuda_rng"])
                resume_rng = None
            # Persist the actual permutation. Old checkpoints complete the current
            # sequential epoch before later epochs adopt shuffled sampling.
            sample_order = resumed_order(len(dataset), options["seed"], epoch, saved_epoch_state)
            remaining = Subset(dataset, sample_order[min(skip * options["batch_size"], len(dataset)):])
            iterator_rng = torch.Generator().manual_seed(options["seed"] + epoch)
            remaining_loader = DataLoader(remaining, batch_size=options["batch_size"], num_workers=0,
                                          shuffle=False, collate_fn=TextAudioCollate(), generator=iterator_rng)
            for relative_index, items in enumerate(remaining_loader):
                batch_index = skip + relative_index
                worker.check()
                c, f0, spec, wave, speaker, lengths, uv, volume = items
                c, f0, spec, wave, speaker, lengths, uv = [x.to(device) for x in (c, f0, spec, wave, speaker, lengths, uv)]
                generator.train()
                output, starts, mask, latent, pred_f0, _, target_f0 = generator(c, f0, uv, spec, g=speaker, c_lengths=lengths, spec_lengths=lengths)
                _, z_p, m_p, logs_p, _, logs_q = latent
                reference = commons.slice_segments(wave, starts * hps.data.hop_length, hps.train.segment_size)
                real, fake, _, _ = discriminator(reference, output.detach())
                loss_d, _, _ = discriminator_loss(real, fake)
                if not torch.isfinite(loss_d):
                    raise ValueError("判别器损失非有限值，已停止训练")
                optim_d.zero_grad(set_to_none=True)
                loss_d.backward()
                torch.nn.utils.clip_grad_norm_(discriminator.parameters(), 100)
                optim_d.step()
                real, fake, real_maps, fake_maps = discriminator(reference, output)
                source_mel = spec_to_mel_torch(spec, hps.data.filter_length, hps.data.n_mel_channels, hps.data.sampling_rate, hps.data.mel_fmin, hps.data.mel_fmax)
                source_mel = commons.slice_segments(source_mel, starts, hps.train.segment_size // hps.data.hop_length)
                output_mel = mel_spectrogram_torch(output.squeeze(1), hps.data.filter_length, hps.data.n_mel_channels, hps.data.sampling_rate, hps.data.hop_length, hps.data.win_length, hps.data.mel_fmin, hps.data.mel_fmax)
                adversarial, _ = generator_loss(fake)
                loss_g = adversarial + feature_loss(real_maps, fake_maps) + F.l1_loss(source_mel, output_mel) * options["c_mel"] + kl_loss(z_p, logs_q, m_p, logs_p, mask) * options["c_kl"]
                if generator.use_automatic_f0_prediction:
                    loss_g = loss_g + F.mse_loss(pred_f0, target_f0)
                if not torch.isfinite(loss_g):
                    raise ValueError("生成器损失非有限值，已停止训练，请检查数据音量与质量")
                optim_g.zero_grad(set_to_none=True)
                loss_g.backward()
                torch.nn.utils.clip_grad_norm_(generator.parameters(), 100)
                optim_g.step()
                step += 1
                next_batch = batch_index + 1
                writer.add_scalar("loss/generator", loss_g.item(), step)
                writer.add_scalar("loss/discriminator", loss_d.item(), step)
                if step % 10 == 0 or step == 1 or step == options["max_steps"]:
                    worker.update(f"训练 {step}/{options['max_steps']} 步 · G {loss_g.item():.4f} · D {loss_d.item():.4f}", 25 + 75 * step / options["max_steps"], step=step)
                if step % options["eval_interval"] == 0:
                    save()
                    evaluate()
                    last_evaluated_step = step
                if step >= options["max_steps"]:
                    break
            if next_batch >= batch_count:
                skip, next_batch = 0, 0
                epoch += 1
                for optimizer in (optim_g, optim_d):
                    for group in optimizer.param_groups:
                        group["lr"] *= options["lr_decay"]
        if last_evaluated_step != step:
            evaluate()
            save()
        identity["status"] = "completed"
    except Exception:
        # A cancellation occurs only between complete steps. Save that resumable state.
        from .worker import Cancelled
        if isinstance(sys.exc_info()[1], Cancelled):
            save()
            identity["status"] = "cancelled"
        else:
            identity["status"] = "failed"
        raise
    finally:
        writer.close()
        db.write(directory / "experiment.json", identity)


def convert(worker, vocals, harmony=None):
    utils = upstream()
    import librosa
    from scipy.ndimage import gaussian_filter1d
    from .audio import envelope, OverlapWriter, write_audio, mix_tracks
    if worker.options.get("model_source") == "imported":
        from .external_models import LIBRARY
        config = db.read(LIBRARY / worker.options["model_id"] / "config.json")
        speaker_id = int(worker.options["speaker_id"])
    else:
        config = db.read(db.experiment_dir(worker.options["experiment_id"]) / "config.json")
        speaker_id = int(config["spk"][worker.job["singer_id"]])
    encoder_name = config["model"]["speech_encoder"]
    if encoder_name not in ENCODER_VARIANTS or config["data"]["sampling_rate"] != 44100:
        raise ValueError("当前工作区只支持 44.1 kHz / ContentVec256L9 或 ContentVec768L12 ONNX 模型")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model_for(config, device)
    compatible_load(worker.options["checkpoint"], model,
                    inference_only=worker.options.get("model_source") == "imported")
    model.eval()
    encoder = encoder_for(device, encoder_name)
    diffusion = vocoder = diffusion_config = None
    if worker.options.get("shallow_diffusion"):
        if worker.options.get("model_source") != "imported":
            raise ValueError("浅扩散仅适用于配套文件完整的导入模型")
        from .shallow_diffusion import load as load_diffusion
        worker.update("正在加载浅扩散模型和声码器", 29 if harmony else 7)
        diffusion, vocoder, diffusion_config = load_diffusion(LIBRARY / worker.options["model_id"], config, device)
        worker.check()
        if int(worker.options["diffusion_steps"]) > diffusion.k_step_max:
            raise ValueError(f"浅扩散步数超过模型上限 {diffusion.k_step_max}")
    rate = 44100
    size = max(1, round(worker.options["chunk"] * rate))
    overlap = round(.1 * rate)
    context = round(.5 * rate)
    hop = size - overlap
    if hop <= 0:
        raise ValueError("转换分段长度必须大于交叠长度")
    speaker = torch.LongTensor([[speaker_id]]).to(device)
    mode = worker.options.get("vocal_mode", "off")

    def render(input_path, floating, semitones, label, begin_progress, finish_progress):
        info = sf.info(input_path)
        total = info.frames
        if info.samplerate != rate or total == 0:
            raise ValueError("推理输入需要非空 44.1 kHz 人声音轨")
        from .pitch import analyze, mono, continuous, sample_curve
        curves, channel_info = analyze(worker, input_path, worker.options["predictor"], config["data"]["hop_length"])
        diagnostics = db.read(worker.path / "channel-analysis.json", {})
        diagnostics[label] = channel_info
        db.write(worker.path / "channel-analysis.json", diagnostics)
        if channel_info["suspected"]:
            worker.update(f"{label}检测到单声道抵消风险，分析使用{channel_info['selection']}声道；原音轨保留")
        with sf.SoundFile(input_path) as source, sf.SoundFile(floating, "w", samplerate=rate, channels=1, subtype="FLOAT") as destination:
            writer = OverlapWriter(destination, hop)
            for start in range(0, total, hop):
                worker.check()
                end = min(total, start + size)
                left, right = max(0, start - context), min(total, end + context)
                source.seek(left)
                chunk = mono(source.read(right - left, dtype="float32", always_2d=True), channel_info["selection"])
                if len(chunk) != right - left:
                    raise ValueError("输入音轨读取长度异常，已阻止时间轴错位")
                output = np.zeros(end - start, dtype=np.float32)
                if np.max(np.abs(chunk)) > 1e-5:
                    grid = left / rate + np.arange(max(1, len(chunk) // config["data"]["hop_length"])) * config["data"]["hop_length"] / rate
                    f0, uv = sample_curve(curves["times"], curves["f0"], grid)
                    f0 = continuous(f0, uv)
                    uv = uv.astype(np.float32)
                    if not len(f0):
                        raise ValueError("无法从片段提取 F0，请尝试其他音高提取方式")
                    wave16 = librosa.resample(chunk, orig_sr=rate, target_sr=16000)
                    with torch.no_grad():
                        content = encoder.encoder(torch.from_numpy(wave16).to(device))
                        content = utils.repeat_expand_2d(content.squeeze(0), len(f0), mode=config["data"]["unit_interpolate_mode"]).unsqueeze(0)
                        volume = None
                        if config["model"].get("vol_embedding"):
                            volume = utils.Volume_Extractor(config["data"]["hop_length"]).extract(
                                torch.from_numpy(chunk).float().unsqueeze(0))
                            volume = F.interpolate(volume[None, None, :], size=len(f0), mode="nearest").squeeze(1).to(device)
                        f0_tensor = torch.from_numpy(f0 * (2 ** (semitones / 12))).float().unsqueeze(0).to(device)
                        generated, _ = model.infer(content, f0_tensor,
                            torch.from_numpy(uv).float().unsqueeze(0).to(device), g=speaker, predict_f0=False, noice_scale=.35, vol=volume)
                        audio_tensor = generated[0, 0].float()
                        if diffusion is not None:
                            mel = vocoder.extract(audio_tensor.unsqueeze(0), rate)
                            frames = mel.shape[1]
                            units = F.interpolate(content.float(), size=frames, mode="nearest").transpose(1, 2)
                            pitch = F.interpolate(f0_tensor.unsqueeze(1), size=frames, mode="nearest").transpose(1, 2)
                            if volume is None:
                                loudness = utils.Volume_Extractor(config["data"]["hop_length"]).extract(audio_tensor.unsqueeze(0))
                                loudness = loudness[None, None, :].to(device)
                            else:
                                loudness = volume.unsqueeze(1)
                            loudness = F.interpolate(loudness.float(), size=frames, mode="nearest").transpose(1, 2)
                            refined_mel = diffusion(units, pitch, loudness, spk_id=speaker, gt_spec=mel,
                                                    infer=True, infer_speedup=int(diffusion_config["infer"]["speedup"]),
                                                    method=diffusion_config["infer"]["method"],
                                                    k_step=int(worker.options["diffusion_steps"]), use_tqdm=False)
                            audio_tensor = vocoder.infer(refined_mel, pitch).squeeze().float()
                        converted = audio_tensor.cpu().numpy()
                    converted = np.pad(converted[:len(chunk)], (0, max(0, len(chunk) - len(converted))))
                    output = converted[start - left:end - left]
                # The gate uses this chunk's source samples at the original positions.
                rms, frame = envelope(chunk[start - left:end - left], rate)
                gate = gaussian_filter1d((rms > 10 ** (-55 / 20)).astype(np.float32), sigma=2)
                gate = np.interp(np.arange(end - start) / rate, np.arange(len(gate)) * frame, gate)
                output *= gate
                window = np.ones(end - start, dtype=np.float32)
                fade = min(overlap, len(window))
                if start:
                    window[:fade] *= np.linspace(0, 1, fade)
                if end < total:
                    window[-fade:] *= np.linspace(1, 0, fade)
                writer.append(start, output, window)
                progress = begin_progress + (finish_progress - begin_progress) * end / total
                worker.update(f"{label} {end / rate:.1f}/{total / rate:.1f} 秒", progress*.9 if worker.options.get("output_reverb",{}).get("enabled") else progress)
            if writer.frames != total or len(writer.tail):
                raise ValueError("转换输出样本数异常，已阻止时间轴错位")
        return writer.peak

    from .audio import sum_tracks
    info = sf.info(vocals)
    total = info.frames
    target = worker.path / "converted_vocals.wav"
    outputs = dict(db.read(worker.path / "job.json").get("outputs", {}))
    with tempfile.TemporaryDirectory(prefix=".svc-convert-", dir=worker.path) as directory:
        directory = Path(directory)
        lead_float = directory / "lead.wav"
        lead_peak = render(vocals, lead_float, worker.options["transpose"],
                           "转换主唱" if harmony else "转换人声",
                           30 if harmony else 10, 58 if harmony and mode == "convert" else 88)
        combined = lead_float
        common_peak = lead_peak
        harmony_float = None
        if harmony is not None and mode == "convert":
            harmony_float = directory / "harmony.wav"
            harmony_peak = render(harmony, harmony_float,
                                  worker.options["transpose"] + worker.options.get("harmony_transpose", 0),
                                  "转换和声", 58, 88)
            common_peak = max(common_peak, harmony_peak)
        if harmony is not None and mode in {"keep", "convert"}:
            combined = directory / "combined.wav"
            common_peak = max(common_peak, sum_tracks(
                [(lead_float, 1.), (harmony_float if harmony_float else harmony,
                                   worker.options.get("harmony_gain", 1.))], combined, worker.check))
        post_reverb = worker.options.get("output_reverb",{}).get("enabled",False)
        worker.update("合并人声并写入完整音轨", 82.8 if post_reverb else 92)
        normalization = write_audio(combined, target, worker.check)
        outputs.update(vocals=str(target), duration=total / rate, samples=total, sampling_rate=rate,
                       vocal_normalization_gain=normalization)
        if harmony is not None:
            lead_target = worker.path / "converted_lead.wav"
            write_audio(lead_float, lead_target, worker.check)
            outputs["converted_lead"] = str(lead_target)
        if harmony_float:
            harmony_target = worker.path / "converted_backing.wav"
            write_audio(harmony_float, harmony_target, worker.check)
            outputs["converted_backing"] = str(harmony_target)
        if worker.options["mixed"]:
            worker.update("按设定增益分块混音（不自动衰减）", 86.4 if post_reverb else 96)
            accompaniment = worker.path / "accompaniment.wav"
            mix_path = worker.path / "converted_mix.wav"
            gain = mix_tracks(combined, accompaniment, mix_path, worker.options["vocal_gain"],
                              worker.options["instrumental_gain"], normalization, worker.check)
            outputs.update(mix=str(mix_path), mix_normalization_gain=gain, accompaniment=str(accompaniment))
    db.write(worker.path / "outputs.json", outputs)
    worker.update("转换人声已生成，原始时间轴与静音已保留", 90 if post_reverb else 100, outputs=outputs)
