"""話速推定CNNの単体テスト（docs/PLAN.md 第5段階 5-1）。

実データには依存しない。入力はすべてこの場で作る乱数である。
学習は行わず、ランダム初期化のモデルの前向き計算だけを見る。

完了条件（PLAN.md 5-1）「任意長の入力で前向き計算が通り、出力が非負のスカラーになる」と、
docs/decisions/005-window-strategy.md 4.2節が課した制約（受容野201フレーム以下、
詰め物を総和の前にマスクする、詰め物の量に出力が依存しない）を固定する。

## 許容誤差

詰め物の有無で出力が一致することを見るテストでは ``rtol=1e-4, atol=1e-4`` を使う。
本実装は数学的には厳密一致するが、畳み込みの実装がバッチの大きさや系列長によって
別の経路（並列の足し合わせ順序）を選ぶため、float32 の丸めの差が残りうる。
モーラ数は1クリップあたり高々70程度（13.3秒 × 5.03 mora/s）なので、この許容誤差は
モーラ数にして0.01未満であり、目標のMAE 0.50に対して十分小さい。
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from spkrate.models.cnn import (
    MAX_RECEPTIVE_FIELD_FRAMES,
    ChannelLayerNorm,
    CnnConfig,
    SpeechRateCNN,
    count_parameters,
    load_config,
    model_summary,
    receptive_field_frames,
    receptive_field_seconds,
)

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs" / "model" / "cnn_base.yaml"

# 詰め物の有無を比べるときの許容誤差（モジュール docstring 参照）。
RTOL = 1e-4
ATOL = 1e-4

N_MELS = 80


def make_model(config: CnnConfig | None = None, seed: int = 20260921) -> SpeechRateCNN:
    """再現可能な初期値のモデルを作る。評価モードで返す。"""
    torch.manual_seed(seed)
    model = SpeechRateCNN(config)
    model.eval()
    return model


def make_features(batch: int, frames: int, seed: int = 0) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    return torch.randn(batch, frames, N_MELS, generator=generator, dtype=torch.float32)


# --------------------------------------------------------------------------------------
# 完了条件: 任意長の入力で前向き計算が通り、出力が非負のスカラーになる


@pytest.mark.parametrize("frames", [1, 2, 5, 50, 201, 463, 1400])
def test_forward_accepts_any_length(frames: int) -> None:
    """短いものから長いものまで、可変長の入力で前向き計算が通る。"""
    model = make_model()
    with torch.no_grad():
        output = model(make_features(2, frames, seed=frames))
    assert output.shape == (2,)
    assert torch.isfinite(output).all()
    assert (output >= 0).all()


def test_output_is_scalar_per_batch_item() -> None:
    """出力の形は (バッチ,) のスカラー列である。"""
    model = make_model()
    for batch in (1, 3, 8):
        with torch.no_grad():
            output = model(make_features(batch, 120, seed=batch))
        assert output.ndim == 1
        assert output.shape == (batch,)


def test_output_is_non_negative_for_extreme_inputs() -> None:
    """極端な入力でも出力は非負である（softplus の総和の取り違えを検出する）。"""
    model = make_model()
    frames = 150
    cases = {
        "大きな負": torch.full((2, frames, N_MELS), -50.0),
        "大きな正": torch.full((2, frames, N_MELS), 50.0),
        "零": torch.zeros(2, frames, N_MELS),
        "無音相当": torch.full((2, frames, N_MELS), -13.8),  # log(1e-6)
    }
    for name, features in cases.items():
        with torch.no_grad():
            output = model(features)
        assert torch.isfinite(output).all(), name
        assert (output >= 0).all(), f"{name}: {output}"


def test_mixed_lengths_in_one_batch() -> None:
    """バッチ内で長さが混在しても前向き計算が通る。"""
    model = make_model()
    lengths = torch.tensor([1, 37, 201, 600])
    features = make_features(4, int(lengths.max()), seed=7)
    with torch.no_grad():
        output = model(features, lengths)
    assert output.shape == (4,)
    assert (output >= 0).all()


def test_frame_counts_sum_to_forward() -> None:
    """forward はフレームごとの出力の総和である。"""
    model = make_model()
    lengths = torch.tensor([300, 90])
    features = make_features(2, 300, seed=11)
    with torch.no_grad():
        frames = model.frame_counts(features, lengths)
        total = model(features, lengths)
    assert frames.shape == (2, 300)
    assert (frames >= 0).all()
    torch.testing.assert_close(frames.sum(dim=1), total, rtol=RTOL, atol=ATOL)


# --------------------------------------------------------------------------------------
# 制約2・3: 詰め物のフレームをマスクする / 詰め物の量に出力が依存しない
# （docs/decisions/005-window-strategy.md 4.2節）


def test_padded_frames_are_exactly_zero() -> None:
    """詰め物のフレームのモーラ数は厳密に0である（総和に混ざらない）。"""
    model = make_model()
    lengths = torch.tensor([200, 64, 5])
    features = make_features(3, 200, seed=13) * 10.0  # 詰め物の領域に大きな値を置く
    with torch.no_grad():
        frames = model.frame_counts(features, lengths)
    for index, length in enumerate(lengths.tolist()):
        assert torch.all(frames[index, length:] == 0.0), index
        # 有効フレーム側は softplus なので正の値を持つ（全部0なら総和も0で無意味になる）
        assert float(frames[index, :length].sum()) > 0.0


def test_output_does_not_depend_on_padding() -> None:
    """同じクリップを、詰め物なしで単独に流した場合と、長いクリップと同じバッチに
    入れて詰め物付きで流した場合とで、出力が一致する（許容誤差 rtol=1e-4, atol=1e-4）。

    詰め物の領域には無音ではなく大きな乱数を置く。マスクを外すと必ず落ちる。
    """
    model = make_model()
    lengths = torch.tensor([640, 401, 120, 11])
    max_frames = int(lengths.max())
    features = make_features(4, max_frames, seed=17) * 8.0

    with torch.no_grad():
        batched = model(features, lengths)
        batched_frames = model.frame_counts(features, lengths)

    for index, length in enumerate(lengths.tolist()):
        single_input = features[index : index + 1, :length]
        with torch.no_grad():
            single = model(single_input)
            single_frames = model.frame_counts(single_input)
        torch.testing.assert_close(batched[index : index + 1], single, rtol=RTOL, atol=ATOL)
        torch.testing.assert_close(
            batched_frames[index : index + 1, :length], single_frames, rtol=RTOL, atol=ATOL
        )


@pytest.mark.parametrize("extra", [0, 1, 100, 500])
def test_output_is_invariant_to_amount_of_padding(extra: int) -> None:
    """詰め物0・1・100・500フレームで出力が一致する（4.2節の要求そのもの）。"""
    model = make_model()
    length = 137
    clip = make_features(1, length, seed=19)
    reference_input = clip
    with torch.no_grad():
        reference = model(reference_input, torch.tensor([length]))

    generator = torch.Generator().manual_seed(1000 + extra)
    garbage = torch.randn(1, extra, N_MELS, generator=generator) * 20.0
    padded = torch.cat([clip, garbage], dim=1)
    with torch.no_grad():
        padded_output = model(padded, torch.tensor([length]))
    torch.testing.assert_close(padded_output, reference, rtol=RTOL, atol=ATOL)


def test_lengths_none_means_all_frames_valid() -> None:
    """lengths を省略した場合は全フレームが有効である。"""
    model = make_model()
    features = make_features(2, 88, seed=23)
    with torch.no_grad():
        implicit = model(features)
        explicit = model(features, torch.tensor([88, 88]))
    torch.testing.assert_close(implicit, explicit, rtol=RTOL, atol=ATOL)


def test_invalid_lengths_are_rejected() -> None:
    model = make_model()
    features = make_features(2, 50, seed=29)
    with pytest.raises(ValueError, match="フレーム数を超えている"):
        model(features, torch.tensor([50, 51]))
    with pytest.raises(ValueError, match="1以上"):
        model(features, torch.tensor([50, 0]))
    with pytest.raises(ValueError, match="lengths の形"):
        model(features, torch.tensor([50, 50, 50]))


def test_input_shape_and_dtype_are_checked() -> None:
    model = make_model()
    with pytest.raises(ValueError, match="3次元"):
        model(torch.randn(50, N_MELS))
    with pytest.raises(ValueError, match="メル次元数"):
        model(torch.randn(1, 50, 40))
    with pytest.raises(ValueError, match="float64"):
        model(torch.randn(1, 50, N_MELS, dtype=torch.float64))


def test_layer_norm_is_also_padding_independent() -> None:
    """norm=layer（フレームごとの正規化）でも詰め物に依存しない。"""
    model = make_model(CnnConfig(norm="layer"))
    lengths = torch.tensor([300, 64])
    features = make_features(2, 300, seed=31) * 5.0
    with torch.no_grad():
        batched = model(features, lengths)
        single = model(features[1:2, :64])
    torch.testing.assert_close(batched[1:2], single, rtol=RTOL, atol=ATOL)


def test_channel_layer_norm_normalizes_channels_only() -> None:
    """ChannelLayerNorm はチャネル方向だけを正規化する（位置ごとに閉じている）。"""
    norm = ChannelLayerNorm(4)
    x = torch.randn(2, 4, 3, 5)
    with torch.no_grad():
        y = norm(x)
    torch.testing.assert_close(y.mean(dim=1), torch.zeros(2, 3, 5), rtol=0, atol=1e-5)
    torch.testing.assert_close(
        y.std(dim=1, unbiased=False), torch.ones(2, 3, 5), rtol=1e-3, atol=1e-3
    )


# --------------------------------------------------------------------------------------
# 制約1: 受容野は201フレーム（2.0秒）以下


def test_default_receptive_field_within_limit() -> None:
    """既定の設定の受容野は201フレーム以下で、推奨の51〜101フレームに入っている。"""
    config = CnnConfig()
    field = receptive_field_frames(config)
    assert field == 69
    assert field <= MAX_RECEPTIVE_FIELD_FRAMES
    assert 51 <= field <= 101, "推奨範囲（docs/decisions/005-window-strategy.md 0節）"
    assert receptive_field_seconds(config) == pytest.approx(0.69)
    assert config.receptive_field_frames == field


def test_config_file_receptive_field_within_limit() -> None:
    """configs/model/cnn_base.yaml の受容野も201フレーム以下である。"""
    config = load_config(CONFIG_PATH)
    assert config.receptive_field_frames <= MAX_RECEPTIVE_FIELD_FRAMES
    assert 51 <= config.receptive_field_frames <= 101


def test_receptive_field_formula() -> None:
    """受容野 = 1 + Σ (カーネル長-1)×膨張率。周波数段も時間方向のカーネルを持つ。"""
    config = CnnConfig(
        freq_channels=(8, 8),
        freq_strides=(2, 2),
        freq_kernel=(3, 5),
        temporal_channels=(16, 16),
        dilations=(1, 3),
        temporal_kernel=3,
    )
    # 1 + 2層×(5-1) + (3-1)×1 + (3-1)×3 = 1 + 8 + 2 + 6 = 17
    assert receptive_field_frames(config) == 17


def test_receptive_field_matches_actual_dependency() -> None:
    """算出した受容野が実際の依存範囲と一致する（外側を変えても出力が変わらない）。"""
    config = CnnConfig(
        freq_channels=(4,),
        freq_strides=(4,),
        temporal_channels=(8, 8),
        dilations=(1, 2),
    )
    field = receptive_field_frames(config)  # 1 + 2 + 2 + 4 = 9
    assert field == 9
    half = field // 2
    model = make_model(config)
    frames = 61
    center = 30
    features = make_features(1, frames, seed=37)
    with torch.no_grad():
        base = model.frame_counts(features)[0, center]

    # 受容野の内側を変えれば出力は変わる
    inside = features.clone()
    inside[0, center] += 5.0
    with torch.no_grad():
        changed = model.frame_counts(inside)[0, center]
    assert abs(float(changed) - float(base)) > 1e-6

    # 受容野の外側を変えても出力は変わらない
    outside = features.clone()
    outside[0, : center - half] += 5.0
    outside[0, center + half + 1 :] += 5.0
    with torch.no_grad():
        unchanged = model.frame_counts(outside)[0, center]
    torch.testing.assert_close(unchanged, base, rtol=RTOL, atol=ATOL)


def test_config_rejects_receptive_field_over_limit() -> None:
    """201フレームを超える設定は作れない（4.2節「201を超える設定を拒否する」）。"""
    with pytest.raises(ValueError, match="受容野"):
        CnnConfig(
            temporal_channels=(32, 32, 32, 32, 32, 32, 32),
            dilations=(1, 2, 4, 8, 16, 32, 64),
        )
    # 上限を明示的に上げれば作れる（実験で意図的に外す場合のみ）
    config = CnnConfig(
        temporal_channels=(32,) * 7,
        dilations=(1, 2, 4, 8, 16, 32, 64),
        max_receptive_field_frames=1000,
    )
    assert config.receptive_field_frames == 1 + 6 + 2 * 127


# --------------------------------------------------------------------------------------
# configs から層数・チャネル数・膨張率を変えられる


def test_load_config_matches_defaults() -> None:
    """configs/model/cnn_base.yaml が既定の設定と一致する。"""
    assert CONFIG_PATH.exists(), f"設定ファイルが無い: {CONFIG_PATH}"
    assert load_config(CONFIG_PATH) == CnnConfig()


def test_load_config_rejects_unknown_keys(tmp_path: Path) -> None:
    path = tmp_path / "bad.yaml"
    path.write_text("model:\n  n_mels: 80\n  nonexistent: 1\n", encoding="utf-8")
    with pytest.raises(ValueError, match="未知のパラメータ"):
        load_config(path)


def test_config_lists_become_tuples(tmp_path: Path) -> None:
    path = tmp_path / "c.yaml"
    path.write_text(
        "model:\n"
        "  freq_channels: [8, 16]\n"
        "  freq_strides: [2, 2]\n"
        "  temporal_channels: [32, 32]\n"
        "  dilations: [1, 2]\n",
        encoding="utf-8",
    )
    config = load_config(path)
    assert config.freq_channels == (8, 16)
    assert config.dilations == (1, 2)


def test_number_of_layers_follows_config() -> None:
    """層数を変えると実際の層数が変わる。"""
    small = CnnConfig(
        freq_channels=(8, 8), freq_strides=(2, 2), temporal_channels=(16,), dilations=(1,)
    )
    large = CnnConfig(
        freq_channels=(8, 8, 8),
        freq_strides=(2, 2, 2),
        temporal_channels=(16, 16, 16),
        dilations=(1, 2, 4),
    )
    small_model = make_model(small)
    large_model = make_model(large)
    assert len(small_model.freq_blocks) == 2
    assert len(small_model.temporal_blocks) == 1
    assert len(large_model.freq_blocks) == 3
    assert len(large_model.temporal_blocks) == 3
    assert small.num_layers == 3
    assert large.num_layers == 6
    # 同じ幅のまま時間段を1層増やせばパラメータ数は増える
    deeper = make_model(large.replace(temporal_channels=(16, 16, 16, 16), dilations=(1, 2, 4, 8)))
    assert count_parameters(deeper) > count_parameters(large_model)
    with torch.no_grad():
        assert small_model(make_features(1, 60, seed=41)).shape == (1,)
        assert large_model(make_features(1, 60, seed=41)).shape == (1,)


def test_channel_counts_follow_config() -> None:
    """チャネル数を変えると実際のチャネル数とパラメータ数が変わる。"""
    config = CnnConfig(
        freq_channels=(4, 12),
        freq_strides=(2, 2),
        temporal_channels=(24, 48),
        dilations=(1, 2),
    )
    model = make_model(config)
    assert model.freq_blocks[0][0].out_channels == 4
    assert model.freq_blocks[1][0].out_channels == 12
    assert model.temporal_blocks[0][0].out_channels == 24
    assert model.temporal_blocks[1][0].out_channels == 48
    assert model.head.in_channels == 48
    # 周波数 80 -> 40 -> 20、flatten で 12*20 = 240 チャネルが時間段に入る
    assert config.freq_dim_out == 20
    assert config.temporal_in_channels == 240
    assert model.temporal_blocks[0][0].in_channels == 240

    wider = make_model(config.replace(temporal_channels=(48, 96)))
    assert count_parameters(wider) > count_parameters(model)


def test_dilations_follow_config() -> None:
    """膨張率を変えると実際の膨張率と受容野が変わる。"""
    config = CnnConfig(
        freq_channels=(8,),
        freq_strides=(4,),
        temporal_channels=(16, 16, 16),
        dilations=(1, 4, 9),
    )
    model = make_model(config)
    assert [block[0].dilation[0] for block in model.temporal_blocks] == [1, 4, 9]
    # 詰め物の量は膨張率に比例し、フレーム数は保たれる
    with torch.no_grad():
        frames = model.frame_counts(make_features(1, 70, seed=43))
    assert frames.shape == (1, 70)
    assert receptive_field_frames(config) == 1 + 2 + 2 * (1 + 4 + 9)
    assert receptive_field_frames(config) != receptive_field_frames(
        config.replace(dilations=(1, 2, 4))
    )


def test_freq_pool_mean_option() -> None:
    """freq_pool=mean でも前向き計算が通り、時間段の入力チャネル数が変わる。"""
    config = CnnConfig(freq_pool="mean")
    model = make_model(config)
    assert config.temporal_in_channels == config.freq_channels[-1]
    assert model.temporal_blocks[0][0].in_channels == config.freq_channels[-1]
    with torch.no_grad():
        output = model(make_features(2, 100, seed=47), torch.tensor([100, 30]))
    assert output.shape == (2,)
    assert (output >= 0).all()


def test_invalid_configs_are_rejected() -> None:
    with pytest.raises(ValueError, match="freq_channels と freq_strides"):
        CnnConfig(freq_channels=(8, 8), freq_strides=(2,))
    with pytest.raises(ValueError, match="temporal_channels と dilations"):
        CnnConfig(temporal_channels=(8, 8), dilations=(1,))
    with pytest.raises(ValueError, match="奇数"):
        CnnConfig(temporal_kernel=4)
    with pytest.raises(ValueError, match="norm"):
        CnnConfig(norm="batch")
    with pytest.raises(ValueError, match="activation"):
        CnnConfig(activation="tanh")
    with pytest.raises(ValueError, match="freq_pool"):
        CnnConfig(freq_pool="max")
    with pytest.raises(ValueError, match="1以上"):
        CnnConfig(freq_channels=(8, 0), freq_strides=(2, 2))


# --------------------------------------------------------------------------------------
# 大きさ（第3段階の目標「int8量子化後5MB以下」）


def test_model_size_within_target() -> None:
    model = make_model(load_config(CONFIG_PATH))
    summary = model_summary(model)
    assert summary["num_parameters"] == count_parameters(model)
    assert summary["float32_bytes"] == summary["num_parameters"] * 4
    # int8 に量子化して5MB以下という目標に対する概算
    assert summary["int8_mib"] < 5.0
    assert summary["receptive_field_frames"] == 69


# --------------------------------------------------------------------------------------
# デバイス（CLAUDE.md「学習デバイスはmps」）


def test_forward_runs_on_mps() -> None:
    """mps でも前向き計算が通る。mps が無い環境では飛ばす。"""
    if not (torch.backends.mps.is_available() and torch.backends.mps.is_built()):
        pytest.skip("mps が使えない環境")
    device = torch.device("mps")
    model = make_model().to(device)
    lengths = torch.tensor([300, 77], device=device)
    features = make_features(2, 300, seed=53).to(device)
    with torch.no_grad():
        output = model(features, lengths)
    assert output.shape == (2,)
    assert torch.isfinite(output).all()
    assert (output >= 0).all()
    # cpu と大きく違わないこと（mps の精度差があるので許容を緩める）
    with torch.no_grad():
        cpu_output = make_model()(features.cpu(), lengths.cpu())
    torch.testing.assert_close(output.cpu(), cpu_output, rtol=1e-3, atol=1e-2)
