"""Installation readiness check, including exact pretrained tensor shapes."""
import numpy as np
import torch

from . import storage as db
from .install import RUNTIME, ASSETS, ENCODER_NAME
from .engine import upstream, model_for, compatible_load, encoder_for, predictor_for


def main():
    upstream()
    torch.set_num_threads(4)
    config = db.read(RUNTIME / "configs_template/config_template.json")
    config["model"].update(speech_encoder=ENCODER_NAME, ssl_dim=256, gin_channels=256, n_speakers=1)
    model = model_for(config, "cpu")
    compatible_load(ASSETS / "G_0.pth", model, initial=True)
    del model
    from models import MultiPeriodDiscriminator
    discriminator = MultiPeriodDiscriminator(config["model"]["use_spectral_norm"])
    discriminator.load_state_dict(torch.load(ASSETS / "D_0.pth", map_location="cpu", weights_only=True)["model"], strict=True)
    del discriminator
    encoder = encoder_for("cpu")
    features = encoder.encoder(torch.from_numpy(np.zeros(16000, dtype=np.float32)))
    if features.ndim != 3 or features.shape[1] != 256:
        raise ValueError("编码器输出不是 256 维，与底模配置不符")
    for name in ["dio", "harvest", "pm"]:
        predictor_for(config, name)
    print("SVC 运行环境、内容编码器、生成器和判别器配置匹配。")


if __name__ == "__main__":
    main()
