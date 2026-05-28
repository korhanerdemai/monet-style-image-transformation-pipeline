from __future__ import annotations

from pathlib import Path
from typing import Any, Optional
import fire
from hydra import compose, initialize
from monet_pipeline.models.baseline_nst import run_nst


def baseline(
    content: str,
    style: str,
    output: str = "stylized.jpg",
    steps: Optional[int] = None,
    image_size: Optional[int] = None,
    style_weight: Optional[float] = None,
    device: Optional[str] = None,
) -> None:
    """Run baseline Neural Style Transfer using default parameters from Hydra.

    Parameters
    ----------
    content : str
        Path to the content image.
    style : str
        Path to the style image.
    output : str
        Path where the stylized output image will be saved. Default: 'stylized.jpg'.
    steps : int, optional
        Number of style transfer steps. Overrides Hydra config.
    image_size : int, optional
        Resize dimension. Overrides Hydra config.
    style_weight : float, optional
        Weight for style loss. Overrides Hydra config.
    device : str, optional
        Device to run on ('cuda' or 'cpu'). Overrides Hydra config.
    """
    with initialize(version_base=None, config_path="conf"):
        cfg = compose(config_name="config")

    nst_cfg = cfg.baseline_nst

    # Override defaults with CLI flags if provided
    final_steps = steps if steps is not None else int(nst_cfg.steps)
    final_image_size = image_size if image_size is not None else int(nst_cfg.image_size)
    final_style_weight = style_weight if style_weight is not None else float(nst_cfg.style_weight)
    final_device = device if device is not None else nst_cfg.get("device", None)

    print(f"Executing Baseline NST Transfer:")
    print(f"  Content: {content}")
    print(f"  Style:   {style}")
    print(f"  Output:  {output}")
    print(f"  Steps:   {final_steps}")
    print(f"  Size:    {final_image_size}")
    print(f"  Weight:  {final_style_weight}")
    print(f"  Device:  {final_device}")

    run_nst(
        content_path=content,
        style_path=style,
        output_path=output,
        steps=final_steps,
        image_size=final_image_size,
        device=final_device,
        style_weight=final_style_weight,
    )


if __name__ == "__main__":
    fire.Fire({
        "baseline": baseline
    })
