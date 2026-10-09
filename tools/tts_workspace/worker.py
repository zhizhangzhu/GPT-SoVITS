"""Model worker using stdin/stdout messages; this process serves no HTTP UI."""

import json
import os
from pathlib import Path
import queue
import runpy
import sys
import threading
import traceback
from types import SimpleNamespace


def main():
    channel = sys.stdout
    sys.stdout = sys.stderr  # Model diagnostic output must not enter the protocol.
    commands = queue.Queue()
    stopped = threading.Event()
    engine = {}

    def send(kind, **values):
        channel.write(json.dumps({"kind": kind, **values}, ensure_ascii=False) + "\n")
        channel.flush()

    def read_commands():
        for line in sys.stdin:
            try:
                message = json.loads(line)
                if message.get("command") == "cancel":
                    stopped.set()
                    pipeline = engine.get("tts_pipeline")
                    if pipeline is not None:
                        pipeline.stop()
                else:
                    commands.put(message)
            except Exception:
                traceback.print_exc()
        # Closing the owner's stdin also releases weights if it exits mid-generation.
        os._exit(0)

    threading.Thread(target=read_commands, daemon=True).start()
    try:
        config = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
        fast = config["parallel"]
        filename = "inference_webui_fast.py" if fast else "inference_webui.py"
        sys.argv = [filename, config["language"]]
        engine.update(
            runpy.run_path(
                str(Path("GPT_SoVITS") / filename), run_name="tts_workspace_engine"
            )
        )
        version = engine["model_version"]
        if fast:
            path = config["sovits_path"]
            path = engine["name2sovits_path"].get(path, path)
            version = engine["get_sovits_version_from_path_fast"](path)[1]
        send(
            "loaded",
            model_version=version,
            parallel=fast,
            languages=list(engine["dict_language"]),
            graph_available=bool(engine["graph_available"]),
            flash_available=bool(engine["flash_attn_supported"]),
        )
    except Exception as error:
        traceback.print_exc()
        send("error", message=str(error))
        return

    while True:
        message = commands.get()
        if message.get("command") != "generate":
            continue
        stopped.clear()
        try:
            data = message["inputs"]
            refs = [
                SimpleNamespace(name=path) for path in data.get("aux_refs", [])
            ] or None
            common = dict(
                prompt_text=data["prompt_text"],
                top_k=data["top_k"],
                top_p=data["top_p"],
                temperature=data["temperature"],
                sample_steps=data["sample_steps"],
                cfg_rate=data["cfg_rate"],
            )
            if fast:
                stream = engine["inference"](
                    text=data["text"],
                    text_lang=data["text_language"],
                    ref_audio_path=data["reference"],
                    aux_ref_audio_paths=refs,
                    prompt_lang=data["prompt_language"],
                    text_split_method=data["cut"],
                    batch_size=data["batch_size"],
                    speed_factor=data["speed"],
                    ref_text_free=data["ref_free"],
                    split_bucket=data["split_bucket"],
                    fragment_interval=data["pause"],
                    seed=data["seed"],
                    keep_random=data["random_seed"],
                    use_cuda_graph=data["cuda_graph"],
                    use_flash_attention=data["flash_attention"],
                    repetition_penalty=data["repetition_penalty"],
                    super_sampling=data["super_sampling"],
                    **common,
                )
            else:
                stream = engine["get_tts_wav"](
                    ref_wav_path=data["reference"],
                    prompt_language=data["prompt_language"],
                    text=data["text"],
                    text_language=data["text_language"],
                    how_to_cut=data["cut"],
                    ref_free=data["ref_free"],
                    speed=data["speed"],
                    if_freeze=data["freeze"],
                    inp_refs=refs,
                    if_sr=data["super_sampling"],
                    pause_second=data["pause"],
                    use_cuda_graph=data["cuda_graph"],
                    use_flash_attention=data["flash_attention"],
                    **common,
                )
            produced = False
            for index, item in enumerate(stream):
                if stopped.is_set():
                    break
                audio, seed = item if fast else (item, data["seed"])
                import soundfile as sf

                target = Path(message["output_dir"]) / f"{message['id']}-{index}.wav"
                sf.write(target, audio[1], int(audio[0]), subtype="PCM_16")
                send("audio", id=message["id"], path=str(target), seed=int(seed))
                produced = True
            if not produced and not stopped.is_set():
                raise ValueError("没有生成音频，请检查参考音频、标注和模型设置")
            send("done", id=message["id"], cancelled=stopped.is_set())
        except Exception as error:
            traceback.print_exc()
            send("error", id=message["id"], message=str(error))


if __name__ == "__main__":
    main()
