#!/usr/bin/env python3
"""
character_to_spritesheet.py

Turn a single character image into a sprite sheet using OpenAI's image model
(ChatGPT Images 2.0 / gpt-image-1) via the openai SDK.

You give it:
  * a reference image of the character, and
  * a prompt describing the action (e.g. "make this character walk").

The reference image is sent to the image-EDIT endpoint, so the generated sheet
is based on your actual character rather than a fresh invention.

DEFAULT (one-shot) MODE
-----------------------
The whole sprite sheet is generated in a SINGLE model call, asking for the full
strip in one image.

Use --per-frame to instead generate each frame in its own call (anchored to the
original reference) and lay them out locally with PIL. That keeps the character
more consistent but is slower and gives less smooth motion.

SETUP
-----
  pip install openai pillow python-dotenv

Put your key in a .env file next to this script (or export it):
  OPENAI_API_KEY=your-key-from-https://platform.openai.com/api-keys

USAGE
-----
  python character_to_spritesheet.py captain.png "make this character walk"
  python character_to_spritesheet.py hero.png "jumping animation" \
      --frames 8 --cols 4 --out hero_jump.png
  python character_to_spritesheet.py hero.png "walk" --per-frame
"""

import argparse
import base64
import io
import os
import sys

from PIL import Image, ImageChops

try:
    from dotenv import load_dotenv
except ImportError:
    sys.exit("python-dotenv is not installed.  pip install python-dotenv")

try:
    from openai import OpenAI
except ImportError:
    sys.exit("openai is not installed.  pip install openai")

# Load OPENAI_API_KEY (and friends) from a .env file next to this script, if present.
load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"))


# OpenAI image model. "ChatGPT Images 2.0" is exposed through the API as
# gpt-image-2; override with --model if your account uses a different id.
MODEL = "gpt-image-2"

# Plain background the model is told to use and that we composite/trim against.
BG = (255, 255, 255)

# Rules shared by every prompt: keep the character identical, don't restyle,
# face the viewer, keep anatomy constant.
_CONSISTENCY_RULES = (
    "USE THE REFERENCE CHARACTER AS-IS. The reference image is the ground "
    "truth. Do not redesign, reinterpret, or 'improve' the character.\n"
    "- IDENTICAL character to the reference: exact same face, hair, hat, "
    "outfit, accessories, colors, and body proportions. It must look like the "
    "same character, only the pose changes.\n"
    "- DO NOT CHANGE THE ART STYLE. Match the reference's exact rendering, "
    "line work, shading, level of detail, and color palette. Do not convert "
    "it to pixel art, cartoon, anime, or any other style.\n"
    "- The character FACES THE VIEWER (front-facing, toward the camera) -- a "
    "front view, not a side or profile view.\n"
    "- Keep anatomy correct and constant: exactly the same number of limbs, "
    "hands, feet, and digits as the reference. Never add extra arms, legs, or "
    "feet, and never drop or merge limbs -- only their position changes.\n"
    "- Plain solid white background, no scenery, no cast shadow on the ground."
)


def build_frame_prompt(action, i, n):
    """Prompt for ONE frame: the same character, one pose in the action cycle."""
    return (
        f"Redraw the character from the provided reference image as a SINGLE "
        f"full-body pose: one frame of a smooth, looping '{action}' animation.\n"
        f"This is frame {i + 1} of {n} in the cycle (the cycle loops back to "
        f"frame 1), so show the pose at about {round(100 * i / n)}% through the "
        f"motion -- distinct from the other frames but clearly the same action.\n"
        "Show ONE character only, full body head-to-toe, centered, at the same "
        "scale as the reference.\n"
        + _CONSISTENCY_RULES
    )


def build_sheet_prompt(action, frames, cols):
    """Prompt for the whole sheet in one image (default one-shot mode)."""
    rows = -(-frames // cols)  # ceil
    if rows == 1:
        layout = (
            f"Lay the animation out as a SINGLE HORIZONTAL ROW (a sprite strip) "
            f"of {frames} frames, read left-to-right, showing one continuous, "
            f"smoothly looping animation cycle."
        )
    else:
        layout = (
            f"Lay the animation out as a clean {cols}x{rows} grid ({frames} "
            f"frames total), read left-to-right, top-to-bottom, showing one "
            f"continuous, smoothly looping animation cycle."
        )
    return (
        f"Make a 2D animation sprite sheet from the provided reference image. "
        f"The action is: {action}.\n\n"
        f"{layout}\n\n"
        + _CONSISTENCY_RULES + "\n"
        "- Same scale and ground level across all frames.\n"
        "- GENEROUS, EVEN EMPTY SPACE between every frame: each frame fully "
        "separated by a clear empty margin (roughly half a character-width of "
        "gap); frames never touch or overlap. Uniform spacing.\n"
        "- No labels, numbers, or grid lines drawn on top."
    )


def _generate(client, prompt, image_path, size, model):
    """Call the image-edit endpoint with the reference image; return a PIL Image."""
    with open(image_path, "rb") as f:
        result = client.images.edit(
            model=model,
            image=f,
            prompt=prompt,
            size=size,
        )
    b64 = result.data[0].b64_json
    return Image.open(io.BytesIO(base64.b64decode(b64))).convert("RGB")


def _trim(img, bg=BG, tol=16):
    """Crop away the uniform background border around the character."""
    diff = ImageChops.difference(img, Image.new("RGB", img.size, bg))
    mask = diff.convert("L").point(lambda p: 255 if p > tol else 0)
    bbox = mask.getbbox()
    return img.crop(bbox) if bbox else img


def pack_frames(frames, cols, gap_frac=0.45, bg=BG):
    """Lay trimmed frames into an evenly-spaced grid/strip on a clean canvas.

    Each frame is trimmed to its character, placed in a uniform cell sized to
    the largest frame, centered horizontally and bottom-aligned (so the ground
    line is consistent). Spacing is fixed here, not left to the model, so the
    gaps are always even -- which is what makes the strip read cleanly.
    """
    trimmed = [_trim(f) for f in frames]
    cw = max(t.width for t in trimmed)
    ch = max(t.height for t in trimmed)
    gap = max(1, int(cw * gap_frac))
    rows = -(-len(trimmed) // cols)
    cell_w, cell_h = cw + gap, ch + gap
    width = cols * cell_w + gap
    height = rows * cell_h + gap
    canvas = Image.new("RGB", (width, height), bg)
    for idx, t in enumerate(trimmed):
        r, c = divmod(idx, cols)
        x = gap + c * cell_w + (cw - t.width) // 2
        y = gap + r * cell_h + (ch - t.height)          # bottom-align
        canvas.paste(t, (x, y))
    return canvas


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("image", help="path to the character reference image")
    ap.add_argument("prompt",
                    help='action to animate, e.g. "make this character walk"')
    ap.add_argument("--frames", type=int, default=8,
                    help="number of animation frames (default: 8)")
    ap.add_argument("--cols", type=int, default=8,
                    help="frames per row (default: 8 = a single horizontal strip)")
    ap.add_argument("--per-frame", action="store_true",
                    help="generate each frame in its own call and lay them out "
                         "locally (more consistent character, but slower and "
                         "the motion is less smooth). Default is one-shot: the "
                         "whole sheet in a single model call.")
    ap.add_argument("--size", default="1536x512",
                    help="image size sent to the model (default: 1536x512, a 3:1 "
                         "strip -- the widest aspect ratio gpt-image-2 allows; "
                         "4:1 like 2048x512 is rejected by the API).")
    ap.add_argument("--model", default=MODEL,
                    help=f"OpenAI image model id (default: {MODEL})")
    ap.add_argument("--out", default=None,
                    help="output PNG path (default: <image>_spritesheet.png)")
    ap.add_argument("--api-key", default=os.environ.get("OPENAI_API_KEY"),
                    help="OpenAI API key (default: $OPENAI_API_KEY)")
    args = ap.parse_args()

    if not args.api_key:
        sys.exit("No API key. Set OPENAI_API_KEY or pass --api-key. "
                 "Get one at https://platform.openai.com/api-keys")

    if not os.path.exists(args.image):
        sys.exit(f"Could not find image '{args.image}'")

    out_path = args.out or (
        os.path.splitext(args.image)[0] + "_spritesheet.png")

    client = OpenAI(api_key=args.api_key)
    print(f"character : {args.image}")
    print(f"action    : {args.prompt}")
    print(f"layout    : {args.frames} frames, {args.cols} per row")
    print(f"model     : {args.model}  (size {args.size})")
    print(f"mode      : {'per-frame' if args.per_frame else 'one-shot'}")

    if args.per_frame:
        frames = []
        for i in range(args.frames):
            print(f"  frame {i + 1}/{args.frames}...")
            frames.append(_generate(
                client, build_frame_prompt(args.prompt, i, args.frames),
                args.image, args.size, args.model))
        sheet = pack_frames(frames, args.cols)
    else:
        print("generating...")
        sheet = _generate(
            client, build_sheet_prompt(args.prompt, args.frames, args.cols),
            args.image, args.size, args.model)

    sheet.save(out_path)
    print(f"saved     : {os.path.abspath(out_path)}  ({sheet.width}x{sheet.height})")


if __name__ == "__main__":
    main()
