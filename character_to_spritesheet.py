#!/usr/bin/env python3
"""
character_to_spritesheet.py

Turn a single character image into a sprite sheet using Google's "Nano Banana"
image model (gemini-2.5-flash-image) via the google-genai SDK.

You give it:
  * a reference image of the character, and
  * a prompt describing the action (e.g. "make this character walk").

It asks Nano Banana to redraw that exact character as a grid of animation
frames -- preserving the character's design, colors, and proportions -- and
saves the resulting sprite sheet PNG.

SETUP
-----
  pip install google-genai pillow python-dotenv

Put your key in a .env file next to this script (or export it):
  GEMINI_API_KEY=your-key-from-https://aistudio.google.com/apikey

USAGE
-----
  python character_to_spritesheet.py captain.png "make this character walk"
  python character_to_spritesheet.py hero.png "jumping animation" \
      --frames 8 --cols 4 --out hero_jump.png
"""

import argparse
import io
import os
import sys

from PIL import Image

try:
    from dotenv import load_dotenv
except ImportError:
    sys.exit("python-dotenv is not installed.  pip install python-dotenv")

try:
    from google import genai
    from google.genai import types
except ImportError:
    sys.exit("google-genai is not installed.  pip install google-genai")

# Load GEMINI_API_KEY (and friends) from a .env file next to this script, if present.
load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"))


# Nano Banana Pro: Gemini 3 Pro Image -- advanced reasoning ("Thinking") for
# following complex instructions and rendering high-fidelity text/assets.
MODEL = "gemini-3-pro-image"


def build_prompt(action, frames, cols):
    """Wrap the user's action description in explicit sprite-sheet instructions.

    Nano Banana follows the reference image strongly, so the framing prompt
    leans hard on three things the model otherwise drifts on: keeping the
    EXACT same character in every frame, NOT restyling the art, and a fixed
    front-facing (toward the viewer) orientation.
    """
    rows = -(-frames // cols)  # ceil
    return (
        f"Make a 2D animation sprite sheet from the provided reference image. "
        f"The action is: {action}.\n\n"
        f"Lay the animation out as a clean {cols}x{rows} grid ({frames} frames "
        f"total), read left-to-right, top-to-bottom, showing one continuous, "
        f"smoothly looping animation cycle.\n\n"
        "USE THE REFERENCE CHARACTER AS-IS. The reference image is the ground "
        "truth. Do not redesign, reinterpret, or 'improve' the character.\n"
        "Requirements:\n"
        "- IDENTICAL character in every single frame: exact same face, hair, "
        "outfit, accessories, colors, and body proportions as the reference. "
        "It must look like the same drawing copied frame to frame, with only "
        "the pose changing for the animation. No variation in design between "
        "frames.\n"
        "- DO NOT CHANGE THE ART STYLE. Match the reference's exact rendering, "
        "line work, shading, level of detail, and color palette. Do not "
        "convert it to pixel art, cartoon, anime, 3D, or any other style. If "
        "the reference is realistic, keep it realistic.\n"
        "- The character FACES THE VIEWER (front-facing, looking toward the "
        "camera) in every frame -- a front view, not a side or profile view.\n"
        "- Full body visible and centered in every frame, with consistent "
        "scale and ground level across all frames.\n"
        "- Keep anatomy correct and constant: the character has exactly the "
        "same number of limbs, hands, feet, and digits as the reference in "
        "EVERY frame. Never add extra arms, legs, or feet, and never drop or "
        "merge limbs -- only their position changes with the pose.\n"
        "- Plain flat solid-color background (no scenery, no shadows on the "
        "ground), even spacing between frames, no labels, numbers, or grid "
        "lines drawn on top."
    )


def generate_sprite_sheet(client, character_img, action, frames, cols):
    """Call Nano Banana and return the first image it produces as a PIL Image."""
    prompt = build_prompt(action, frames, cols)
    resp = client.models.generate_content(
        model=MODEL,
        contents=[prompt, character_img],
        config=types.GenerateContentConfig(
            response_modalities=["IMAGE"],
            # Let Nano Banana Pro "think" before drawing -- it reasons about the
            # layout/consistency before committing to the image. -1 = let the
            # model decide its own thinking budget.
            thinking_config=types.ThinkingConfig(thinking_budget=-1),
        ),
    )

    candidates = resp.candidates or []
    for cand in candidates:
        for part in (cand.content.parts if cand.content else []):
            inline = getattr(part, "inline_data", None)
            if inline and inline.data:
                return Image.open(io.BytesIO(inline.data))

    # No image came back -- surface any text the model returned instead.
    text = getattr(resp, "text", None)
    sys.exit("Nano Banana returned no image."
             + (f" Model said: {text}" if text else ""))


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("image", help="path to the character reference image")
    ap.add_argument("prompt",
                    help='action to animate, e.g. "make this character walk"')
    ap.add_argument("--frames", type=int, default=8,
                    help="number of animation frames (default: 8)")
    ap.add_argument("--cols", type=int, default=4,
                    help="frames per row in the grid (default: 4)")
    ap.add_argument("--out", default=None,
                    help="output PNG path (default: <image>_spritesheet.png)")
    ap.add_argument("--api-key", default=os.environ.get("GEMINI_API_KEY"),
                    help="Gemini API key (default: $GEMINI_API_KEY)")
    args = ap.parse_args()

    if not args.api_key:
        sys.exit("No API key. Set GEMINI_API_KEY or pass --api-key. "
                 "Get one at https://aistudio.google.com/apikey")

    try:
        character_img = Image.open(args.image)
    except (FileNotFoundError, OSError) as e:
        sys.exit(f"Could not open image '{args.image}': {e}")

    out_path = args.out or (
        os.path.splitext(args.image)[0] + "_spritesheet.png")

    client = genai.Client(api_key=args.api_key)
    print(f"character : {args.image}")
    print(f"action    : {args.prompt}")
    print(f"layout    : {args.frames} frames, {args.cols} per row")
    print(f"model     : {MODEL}")
    print("generating...")

    sheet = generate_sprite_sheet(
        client, character_img, args.prompt, args.frames, args.cols)
    sheet.save(out_path)
    print(f"saved     : {os.path.abspath(out_path)}  ({sheet.width}x{sheet.height})")


if __name__ == "__main__":
    main()
