"""話速推定CNN（docs/PLAN.md 第5段階 5-1、docs/spec.md「モデル構造」）。

docs/spec.md が定める構造は次のとおりである。

    周波数方向を圧縮する2次元畳み込み層 → 時間方向の膨張畳み込み層
    → フレームごとのsoftplus出力 → 時間方向の総和

入力は正規化済みの対数メルスペクトログラム ``(バッチ, フレーム, n_mels)`` の float32
テンソルである（``spkrate.features.melspec`` の出力と同じ並び。時間が第1軸）。
出力は ``(バッチ,)`` の非負スカラー列で、**その区間に含まれるモーラ数**である。
毎秒モーラ数への換算（モーラ数 ÷ 区間長）は評価側で行う。

## 可変長と詰め物（docs/decisions/005-window-strategy.md 4.2節）

第4段階4-3の判断により、学習は方式A（クリップ全体を入力、クリップ全体のモーラ数を正解）
で行う。クリップ長は1.1〜13.3秒と幅があるため、バッチには詰め物のフレームが生じる。
softplus の出力は常に正なので、**詰め物のフレームをそのまま総和に入れるとモーラ数が
水増しされる**。損失が下がりながら静かに壊れる種類の誤りなので、本実装では次を守る。

1. ``lengths`` で有効フレーム数を受け取り、入力の時点で詰め物のフレームを0にする
2. 各層の出力でも詰め物のフレームを0に戻す
3. softplus の**後**にマスクしてから時間方向に総和をとる

畳み込みの時間方向の詰め方は零詰め（``padding_mode="zeros"``）である。有効領域の外側が
常に厳密に0であるため、「1件だけを詰め物なしで流した場合」と「長い系列と同じバッチに
入れて詰め物付きで流した場合」で、有効フレームの値が一致する（数値誤差を除く）。
これは ``tests/test_cnn.py::test_output_does_not_depend_on_padding`` で固定してある。

同じ理由から、バッチ内の他の標本や時間方向の他の位置を混ぜる正規化層（バッチ正規化・
グループ正規化・インスタンス正規化）は使えない。``norm`` の選択肢はフレームごとに閉じた
``"layer"``（チャネル方向の正規化）と ``"none"`` の2つだけである。

## 受容野

時間方向の刻みは常に1（フレーム数を保つ）なので、受容野は

    1 + Σ (カーネル長 - 1) × 膨張率

で求まる（``receptive_field_frames``）。docs/decisions/005-window-strategy.md 0節の制約に
より **201フレーム（2.0秒）以下**でなければならず、推奨は51〜101フレームである。
``CnnConfig`` は201フレームを超える設定を作れない（``ValueError`` になる）。

## 使い方

>>> from spkrate.models.cnn import SpeechRateCNN, load_config
>>> config = load_config("configs/model/cnn_base.yaml")
>>> model = SpeechRateCNN(config)
>>> mora = model(features, lengths)   # (バッチ,) モーラ数
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields, replace
from pathlib import Path

import torch
from torch import Tensor, nn

__all__ = [
    "ACTIVATIONS",
    "FRAME_RATE_HZ",
    "FREQ_POOLS",
    "MAX_RECEPTIVE_FIELD_FRAMES",
    "NORMS",
    "ChannelLayerNorm",
    "CnnConfig",
    "SpeechRateCNN",
    "build_valid_mask",
    "count_parameters",
    "estimate_macs",
    "load_config",
    "model_summary",
    "receptive_field_frames",
    "receptive_field_seconds",
]

# 1秒あたりのフレーム数。hop_length=160 / 16000Hz = 10ミリ秒（docs/spec.md「特徴量」）。
FRAME_RATE_HZ: float = 100.0

# 受容野の上限。docs/spec.md「モデルの入力単位: 固定長窓2.0秒」＝201フレーム。
# docs/decisions/005-window-strategy.md 0節が「必ず収める」と定めた値。
MAX_RECEPTIVE_FIELD_FRAMES: int = 201

ACTIVATIONS: tuple[str, ...] = ("relu", "gelu", "silu")
# バッチや時間方向を混ぜる正規化は詰め物に影響されるため選択肢に入れない（モジュール冒頭）。
NORMS: tuple[str, ...] = ("none", "layer")
FREQ_POOLS: tuple[str, ...] = ("flatten", "mean")


# --------------------------------------------------------------------------------------
# 設定


@dataclass(frozen=True)
class CnnConfig:
    """モデル構造の設定。層数・チャネル数・膨張率をここから指定する。

    層数は列の長さで決まる。``freq_channels`` と ``freq_strides`` の長さが周波数段の
    層数、``temporal_channels`` と ``dilations`` の長さが時間段の層数である。

    Attributes:
        n_mels: 入力のメル次元数。``spkrate.features.melspec`` の ``n_mels``。
        freq_channels: 周波数段の各層の出力チャネル数。
        freq_strides: 周波数段の各層の周波数方向の刻み。時間方向の刻みは常に1。
        freq_kernel: 周波数段のカーネル ``(周波数方向, 時間方向)``。ともに奇数。
        freq_pool: 周波数段の後に残った周波数軸の畳み方。``flatten`` か ``mean``。
        temporal_channels: 時間段の各層の出力チャネル数。
        dilations: 時間段の各層の膨張率。
        temporal_kernel: 時間段のカーネル長（奇数）。
        activation: 活性化関数。``relu`` / ``gelu`` / ``silu``。
        norm: 正規化層。``none`` かフレームごとの ``layer``。
        dropout: 各層の後のドロップアウト率。
        max_receptive_field_frames: 受容野の上限（フレーム）。既定は201（2.0秒）。
    """

    n_mels: int = 80
    freq_channels: tuple[int, ...] = (32, 64, 64)
    freq_strides: tuple[int, ...] = (2, 2, 2)
    freq_kernel: tuple[int, int] = (3, 3)
    freq_pool: str = "flatten"
    temporal_channels: tuple[int, ...] = (128, 128, 128, 128, 128)
    dilations: tuple[int, ...] = (1, 2, 4, 8, 16)
    temporal_kernel: int = 3
    activation: str = "relu"
    norm: str = "none"
    dropout: float = 0.0
    max_receptive_field_frames: int = MAX_RECEPTIVE_FIELD_FRAMES

    def __post_init__(self) -> None:
        # yaml から読むと列は list で来るので tuple に正規化する（frozen なので直に書く）。
        object.__setattr__(self, "freq_channels", tuple(int(v) for v in self.freq_channels))
        object.__setattr__(self, "freq_strides", tuple(int(v) for v in self.freq_strides))
        object.__setattr__(self, "freq_kernel", tuple(int(v) for v in self.freq_kernel))
        object.__setattr__(
            self, "temporal_channels", tuple(int(v) for v in self.temporal_channels)
        )
        object.__setattr__(self, "dilations", tuple(int(v) for v in self.dilations))
        object.__setattr__(self, "n_mels", int(self.n_mels))
        object.__setattr__(self, "temporal_kernel", int(self.temporal_kernel))
        object.__setattr__(self, "dropout", float(self.dropout))
        object.__setattr__(
            self, "max_receptive_field_frames", int(self.max_receptive_field_frames)
        )
        self._validate()

    def _validate(self) -> None:
        if self.n_mels < 1:
            raise ValueError(f"n_mels は1以上である必要がある: {self.n_mels}")
        if not self.freq_channels:
            raise ValueError("freq_channels が空である。周波数段は1層以上必要")
        if len(self.freq_channels) != len(self.freq_strides):
            raise ValueError(
                "freq_channels と freq_strides の長さが違う: "
                f"{len(self.freq_channels)} != {len(self.freq_strides)}"
            )
        if not self.temporal_channels:
            raise ValueError("temporal_channels が空である。時間段は1層以上必要")
        if len(self.temporal_channels) != len(self.dilations):
            raise ValueError(
                "temporal_channels と dilations の長さが違う: "
                f"{len(self.temporal_channels)} != {len(self.dilations)}"
            )
        if any(c < 1 for c in self.freq_channels + self.temporal_channels):
            raise ValueError("チャネル数は1以上である必要がある")
        if any(s < 1 for s in self.freq_strides):
            raise ValueError("freq_strides は1以上である必要がある")
        if any(d < 1 for d in self.dilations):
            raise ValueError("dilations は1以上である必要がある")
        if len(self.freq_kernel) != 2:
            raise ValueError(f"freq_kernel は (周波数, 時間) の2要素: {self.freq_kernel}")
        if any(k < 1 or k % 2 == 0 for k in self.freq_kernel):
            raise ValueError(f"freq_kernel は正の奇数である必要がある: {self.freq_kernel}")
        if self.temporal_kernel < 1 or self.temporal_kernel % 2 == 0:
            raise ValueError(
                f"temporal_kernel は正の奇数である必要がある: {self.temporal_kernel}"
            )
        if self.activation not in ACTIVATIONS:
            raise ValueError(f"activation は {ACTIVATIONS} のいずれか: {self.activation}")
        if self.norm not in NORMS:
            raise ValueError(
                f"norm は {NORMS} のいずれか: {self.norm}。"
                "バッチ・時間方向を混ぜる正規化は詰め物に影響されるため使えない"
            )
        if self.freq_pool not in FREQ_POOLS:
            raise ValueError(f"freq_pool は {FREQ_POOLS} のいずれか: {self.freq_pool}")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError(f"dropout は0以上1未満: {self.dropout}")
        if self.max_receptive_field_frames < 1:
            raise ValueError(
                f"max_receptive_field_frames は1以上: {self.max_receptive_field_frames}"
            )
        field = receptive_field_frames(self)
        if field > self.max_receptive_field_frames:
            raise ValueError(
                f"時間方向の受容野が上限を超えている: {field} > "
                f"{self.max_receptive_field_frames} フレーム。"
                "docs/decisions/005-window-strategy.md により201フレーム（2.0秒）以下に"
                "収める必要がある（推奨51〜101）"
            )

    # -- 構造から決まる値 ---------------------------------------------------------------

    @property
    def freq_dim_out(self) -> int:
        """周波数段を通したあとに残る周波数方向の次元数。"""
        dim = self.n_mels
        pad = self.freq_kernel[0] // 2
        for stride in self.freq_strides:
            dim = (dim + 2 * pad - self.freq_kernel[0]) // stride + 1
            if dim < 1:
                return 0
        return dim

    @property
    def temporal_in_channels(self) -> int:
        """時間段に入る時点のチャネル数（周波数段の出力を畳んだあと）。"""
        if self.freq_pool == "flatten":
            return self.freq_channels[-1] * self.freq_dim_out
        return self.freq_channels[-1]

    @property
    def num_layers(self) -> int:
        """畳み込み層の総数（1×1の出力層を除く）。"""
        return len(self.freq_channels) + len(self.temporal_channels)

    @property
    def receptive_field_frames(self) -> int:
        """時間方向の受容野（フレーム）。"""
        return receptive_field_frames(self)

    @property
    def receptive_field_seconds(self) -> float:
        """時間方向の受容野（秒）。"""
        return receptive_field_seconds(self)

    # -- 入出力 -------------------------------------------------------------------------

    def replace(self, **changes: object) -> "CnnConfig":
        return replace(self, **changes)

    def as_dict(self) -> dict[str, object]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict) -> "CnnConfig":
        known = {field.name for field in fields(cls)}
        unknown = set(payload) - known
        if unknown:
            raise ValueError(f"未知のパラメータ: {sorted(unknown)}")
        return cls(**payload)


def receptive_field_frames(config: CnnConfig) -> int:
    """設定から時間方向の受容野（フレーム）を求める。

    時間方向の刻みは全層で1なので、受容野は
    ``1 + Σ (カーネル長 - 1) × 膨張率`` である。周波数段は膨張率1で
    ``freq_kernel[1]`` を時間方向のカーネル長として持つ。
    """
    field = 1
    field += len(config.freq_channels) * (config.freq_kernel[1] - 1)
    field += sum((config.temporal_kernel - 1) * int(d) for d in config.dilations)
    return field


def receptive_field_seconds(config: CnnConfig) -> float:
    """設定から時間方向の受容野（秒）を求める。1フレーム=10ミリ秒。"""
    return receptive_field_frames(config) / FRAME_RATE_HZ


def load_config(path: str | Path) -> CnnConfig:
    """yaml からモデル構造の設定を読む（configs/model/cnn_base.yaml）。

    最上位に ``model:`` があればその中身を、無ければ全体をパラメータとみなす。
    """
    import yaml

    payload = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"設定ファイルの最上位が辞書でない: {path}")
    return CnnConfig.from_dict(payload.get("model", payload))


# --------------------------------------------------------------------------------------
# 部品


class ChannelLayerNorm(nn.Module):
    """チャネル方向だけを正規化する層（位置ごとに閉じている）。

    ``(バッチ, チャネル, ...)`` の第1軸で平均と分散をとる。時間方向・バッチ方向を
    混ぜないので、詰め物のフレームが有効フレームの値に影響しない。
    """

    def __init__(self, num_channels: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.num_channels = int(num_channels)
        self.eps = float(eps)
        self.weight = nn.Parameter(torch.ones(self.num_channels))
        self.bias = nn.Parameter(torch.zeros(self.num_channels))

    def forward(self, x: Tensor) -> Tensor:
        mean = x.mean(dim=1, keepdim=True)
        var = x.var(dim=1, unbiased=False, keepdim=True)
        normalized = (x - mean) * torch.rsqrt(var + self.eps)
        shape = (1, -1) + (1,) * (x.ndim - 2)
        return normalized * self.weight.view(shape) + self.bias.view(shape)

    def extra_repr(self) -> str:
        return f"{self.num_channels}, eps={self.eps}"


def _activation(name: str) -> nn.Module:
    if name == "relu":
        return nn.ReLU()
    if name == "gelu":
        return nn.GELU()
    if name == "silu":
        return nn.SiLU()
    raise ValueError(f"未知の活性化関数: {name}")


def _norm(name: str, num_channels: int) -> nn.Module | None:
    if name == "none":
        return None
    if name == "layer":
        return ChannelLayerNorm(num_channels)
    raise ValueError(f"未知の正規化: {name}")


def build_valid_mask(
    lengths: Tensor | None,
    num_frames: int,
    *,
    device: torch.device,
    dtype: torch.dtype,
    batch_size: int,
) -> Tensor:
    """有効フレームを1、詰め物のフレームを0にした ``(バッチ, フレーム)`` を作る。

    ``lengths`` が ``None`` なら全フレームが有効である。
    """
    if lengths is None:
        return torch.ones((batch_size, num_frames), device=device, dtype=dtype)
    if lengths.ndim != 1 or lengths.shape[0] != batch_size:
        raise ValueError(
            f"lengths の形が不正: {tuple(lengths.shape)}（(バッチ,)={batch_size} が必要）"
        )
    lengths_long = lengths.to(device=device, dtype=torch.long)
    if int(lengths_long.min()) < 1:
        raise ValueError("lengths は1以上である必要がある（空の入力は扱えない）")
    if int(lengths_long.max()) > num_frames:
        raise ValueError(
            f"lengths がフレーム数を超えている: {int(lengths_long.max())} > {num_frames}"
        )
    index = torch.arange(num_frames, device=device).unsqueeze(0)
    return (index < lengths_long.unsqueeze(1)).to(dtype)


# --------------------------------------------------------------------------------------
# モデル


class SpeechRateCNN(nn.Module):
    """話速推定CNN（docs/spec.md「モデル構造」）。

    ``forward`` の戻り値はモーラ数（非負）である。毎秒モーラ数ではない。
    """

    def __init__(self, config: CnnConfig | None = None) -> None:
        super().__init__()
        self.config = config if config is not None else CnnConfig()
        cfg = self.config

        freq_kernel, time_kernel = cfg.freq_kernel
        freq_pad = freq_kernel // 2
        time_pad = time_kernel // 2

        blocks: list[nn.Module] = []
        in_channels = 1
        for out_channels, stride in zip(cfg.freq_channels, cfg.freq_strides, strict=True):
            layers: list[nn.Module] = [
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=(freq_kernel, time_kernel),
                    stride=(stride, 1),  # 時間方向の刻みは常に1（フレーム数を保つ）
                    padding=(freq_pad, time_pad),
                    padding_mode="zeros",
                )
            ]
            norm = _norm(cfg.norm, out_channels)
            if norm is not None:
                layers.append(norm)
            layers.append(_activation(cfg.activation))
            if cfg.dropout > 0.0:
                layers.append(nn.Dropout(cfg.dropout))
            blocks.append(nn.Sequential(*layers))
            in_channels = out_channels
        self.freq_blocks = nn.ModuleList(blocks)

        temporal_blocks: list[nn.Module] = []
        in_channels = cfg.temporal_in_channels
        for out_channels, dilation in zip(
            cfg.temporal_channels, cfg.dilations, strict=True
        ):
            layers = [
                nn.Conv1d(
                    in_channels,
                    out_channels,
                    kernel_size=cfg.temporal_kernel,
                    dilation=int(dilation),
                    padding=(cfg.temporal_kernel // 2) * int(dilation),
                    padding_mode="zeros",
                )
            ]
            norm = _norm(cfg.norm, out_channels)
            if norm is not None:
                layers.append(norm)
            layers.append(_activation(cfg.activation))
            if cfg.dropout > 0.0:
                layers.append(nn.Dropout(cfg.dropout))
            temporal_blocks.append(nn.Sequential(*layers))
            in_channels = out_channels
        self.temporal_blocks = nn.ModuleList(temporal_blocks)

        # フレームごとの出力（1×1畳み込み）。この後に softplus をかける。
        self.head = nn.Conv1d(cfg.temporal_channels[-1], 1, kernel_size=1)

    # -- 前向き計算 ---------------------------------------------------------------------

    def frame_counts(self, features: Tensor, lengths: Tensor | None = None) -> Tensor:
        """フレームごとのモーラ数 ``(バッチ, フレーム)`` を返す。

        すべて非負（softplus）で、詰め物のフレームは厳密に0である。
        """
        if features.ndim != 3:
            raise ValueError(
                f"入力は (バッチ, フレーム, n_mels) の3次元: {tuple(features.shape)}"
            )
        if features.dtype == torch.float64:
            raise ValueError("float64 は使わない（CLAUDE.md「実行環境」）")
        batch, num_frames, n_mels = features.shape
        if n_mels != self.config.n_mels:
            raise ValueError(
                f"メル次元数が設定と異なる: {n_mels} != {self.config.n_mels}"
            )
        if num_frames < 1:
            raise ValueError("フレーム数が0の入力は扱えない")

        features = features.to(self.head.weight.dtype)
        mask = build_valid_mask(
            lengths,
            num_frames,
            device=features.device,
            dtype=features.dtype,
            batch_size=batch,
        )
        mask4 = mask[:, None, None, :]
        mask3 = mask[:, None, :]

        # (バッチ, 1, n_mels, フレーム)。詰め物のフレームは入口で0にする。
        x = features.transpose(1, 2).unsqueeze(1) * mask4

        for block in self.freq_blocks:
            # 各層の後で詰め物のフレームを0に戻す。畳み込みの零詰めと同じ値になるので、
            # 詰め物の有無によらず有効フレームの値が変わらない。
            x = block(x) * mask4

        if self.config.freq_pool == "flatten":
            x = x.reshape(batch, -1, num_frames)
        else:
            x = x.mean(dim=2)

        for block in self.temporal_blocks:
            x = block(x) * mask3

        # フレームごとの softplus。softplus は常に正なので、必ずマスクしてから総和する。
        frames = torch.nn.functional.softplus(self.head(x)).squeeze(1) * mask
        return frames

    def forward(self, features: Tensor, lengths: Tensor | None = None) -> Tensor:
        """モーラ数 ``(バッチ,)`` を返す（フレームごとの出力の時間方向の総和）。

        Args:
            features: ``(バッチ, フレーム, n_mels)`` の正規化済み対数メル。
            lengths: ``(バッチ,)`` の有効フレーム数。``None`` なら全フレームが有効。
        """
        return self.frame_counts(features, lengths).sum(dim=1)

    # -- 付随情報 -----------------------------------------------------------------------

    @property
    def receptive_field_frames(self) -> int:
        return receptive_field_frames(self.config)

    @property
    def receptive_field_seconds(self) -> float:
        return receptive_field_seconds(self.config)

    def extra_repr(self) -> str:
        return (
            f"受容野={self.receptive_field_frames}フレーム "
            f"({self.receptive_field_seconds:.2f}秒)"
        )


def count_parameters(model: nn.Module, *, trainable_only: bool = False) -> int:
    """パラメータ数を数える。"""
    parameters = model.parameters()
    if trainable_only:
        parameters = (p for p in model.parameters() if p.requires_grad)
    return sum(int(p.numel()) for p in parameters)


DEFAULT_MACS_FRAMES: int = 201  # 2.0秒窓（docs/spec.md「モデルの入力単位」）


def estimate_macs(config: CnnConfig, num_frames: int = DEFAULT_MACS_FRAMES) -> dict[str, int]:
    """1回の前向き計算の積和の回数を設定から数える（docs/experiments/011-search-plan.md 1.2節・4.1節）。

    畳み込みごとに「入力チャネル × 出力チャネル × カーネルの要素数 × 出力の要素数」を足す。
    偏りの加算・活性化・正規化・マスク・softplus・総和は含まない。出力層（1×1）は含める。
    時間方向の刻みは1なので、出力の時間方向の要素数は ``num_frames`` である。
    ``freq_pool: mean`` の平均は積和に数えない。

    Returns:
        ``total``（合計）、``freq``（周波数段）、``temporal``（時間段）、``head``（出力層）。
    """
    frames = int(num_frames)
    if frames < 1:
        raise ValueError(f"num_frames は1以上: {num_frames}")
    freq_kernel, time_kernel = config.freq_kernel
    pad = freq_kernel // 2
    dim = config.n_mels
    in_channels = 1
    freq = 0
    for out_channels, stride in zip(config.freq_channels, config.freq_strides, strict=True):
        dim = (dim + 2 * pad - freq_kernel) // stride + 1
        freq += in_channels * out_channels * freq_kernel * time_kernel * dim * frames
        in_channels = out_channels
    temporal = 0
    in_channels = config.temporal_in_channels
    for out_channels in config.temporal_channels:
        temporal += in_channels * out_channels * config.temporal_kernel * frames
        in_channels = out_channels
    head = in_channels * 1 * frames
    return {"total": freq + temporal + head, "freq": freq, "temporal": temporal, "head": head}


def model_summary(model: SpeechRateCNN) -> dict[str, object]:
    """パラメータ数・概算サイズ・受容野・積和の回数をまとめる（runs/ の config_snapshot 用）。

    ``macs_201_frames`` は2.0秒窓（201フレーム）1回の積和の回数（``estimate_macs``）。

    ``int8_bytes`` は第3段階の目標「int8量子化後5MB以下」に対する概算で、
    重み1つあたり1バイトとした値である（実際の onnx には量子化の刻み等が加わる）。
    """
    num_parameters = count_parameters(model)
    macs = estimate_macs(model.config, DEFAULT_MACS_FRAMES)
    return {
        "num_parameters": num_parameters,
        "num_trainable_parameters": count_parameters(model, trainable_only=True),
        "float32_bytes": num_parameters * 4,
        "float32_mib": num_parameters * 4 / 1024**2,
        "int8_bytes": num_parameters,
        "int8_mib": num_parameters / 1024**2,
        "receptive_field_frames": model.receptive_field_frames,
        "receptive_field_seconds": model.receptive_field_seconds,
        "num_layers": model.config.num_layers,
        "freq_dim_out": model.config.freq_dim_out,
        "temporal_in_channels": model.config.temporal_in_channels,
        "macs_201_frames": macs["total"],
        "macs_201_frames_freq": macs["freq"],
    }
