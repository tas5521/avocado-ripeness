"""
摂動（決定的な明るさ・回転など）を加えたテスト用の前処理。

通常のテスト精度とは別に、分布シフトをシミュレートして頑健性を比較する用途向け。
"""

from torchvision import transforms
from torchvision.transforms import functional as F


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def apply_pil_perturbation(
    pil_img,
    brightness=1.0,
    contrast=1.0,
    saturation=1.0,
    hue=0.0,
    angle=0.0,
    hflip=False,
):
    """
    PIL 画像に決定的な摂動を適用する（ランダム性なし）。

    Args:
        pil_img: RGB の PIL.Image
        brightness: 1.0 が無変換（adjust_brightness の係数）
        contrast: 1.0 が無変換
        saturation: 1.0 が無変換
        hue: 0.0 が無変換（[-0.5, 0.5] 程度が torchvision の仕様）
        angle: 回転角（度）。0 で無変換
        hflip: True で左右反転
    """
    img = pil_img
    if hflip:
        img = F.hflip(img)
    if angle != 0.0:
        img = F.rotate(
            img,
            angle,
            interpolation=F.InterpolationMode.BILINEAR,
            expand=False,
        )
    if brightness != 1.0:
        img = F.adjust_brightness(img, brightness)
    if contrast != 1.0:
        img = F.adjust_contrast(img, contrast)
    if saturation != 1.0:
        img = F.adjust_saturation(img, saturation)
    if hue != 0.0:
        img = F.adjust_hue(img, hue)
    return img


class _PerturbationTransform:
    """Compose 用：コンストラクタで摂動パラメータを固定する。"""

    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def __call__(self, pil_img):
        return apply_pil_perturbation(pil_img, **self.kwargs)


def get_perturbation_eval_transform(preset: str):
    """
    検証用と同じ Resize→(摂動)→ToTensor→Normalize のパイプラインを返す。

    Args:
        preset: :func:`list_perturbation_presets` のキー

    Returns:
        torchvision.transforms.Compose
    """
    presets = _perturbation_presets_dict()
    if preset not in presets:
        keys = ", ".join(sorted(presets.keys()))
        raise ValueError(f"不明な preset: {preset!r}。次のいずれか: {keys}")

    kwargs = presets[preset]
    return transforms.Compose(
        [
            transforms.Resize((224, 224)),
            _PerturbationTransform(**kwargs),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


def _perturbation_presets_dict():
    """preset 名 -> apply_pil_perturbation に渡す kwargs"""
    return {
        "clean": {},
        "brightness_0.75": {"brightness": 0.75},
        "brightness_1.25": {"brightness": 1.25},
        "contrast_0.85": {"contrast": 0.85},
        "contrast_1.15": {"contrast": 1.15},
        "rotate_-10": {"angle": -10.0},
        "rotate_10": {"angle": 10.0},
        "horizontal_flip": {"hflip": True},
        "brightness_0.85_rotate_8": {"brightness": 0.85, "angle": 8.0},
    }


def list_perturbation_presets():
    """利用可能な preset 名のリスト（表示順固定）。"""
    order = [
        "clean",
        "brightness_0.75",
        "brightness_1.25",
        "contrast_0.85",
        "contrast_1.15",
        "rotate_-10",
        "rotate_10",
        "horizontal_flip",
        "brightness_0.85_rotate_8",
    ]
    d = _perturbation_presets_dict()
    return [k for k in order if k in d]


def describe_preset(name: str) -> str:
    """表用の短い説明（日本語）。"""
    desc = {
        "clean": "摂動なし（通常テストと同条件）",
        "brightness_0.75": "明るさ 0.75×",
        "brightness_1.25": "明るさ 1.25×",
        "contrast_0.85": "コントラスト 0.85×",
        "contrast_1.15": "コントラスト 1.15×",
        "rotate_-10": "回転 -10°",
        "rotate_10": "回転 +10°",
        "horizontal_flip": "左右反転",
        "brightness_0.85_rotate_8": "明るさ 0.85× + 回転 8°",
    }
    return desc.get(name, name)
