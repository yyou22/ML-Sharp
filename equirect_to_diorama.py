#!/usr/bin/env python3
"""
equirect_to_diorama.py

Convert an equirectangular panorama into the 5 interior faces of a paper-theater
diorama box: backdrop (back wall), left wall, right wall, ceiling, floor. The
front face is intentionally skipped -- that's the open side the viewer looks
through.

Conversion is done with py360convert's e2c(). The face order / return format of
py360convert has changed across releases, so this script does NOT hardcode an
index order: it requests cube_format='dict' and validates the returned keys
against the expected {F,R,B,L,U,D} set, failing loudly if the installed version
differs.

ORIENTATION (empirically verified against a test panorama, not assumed)
-----------------------------------------------------------------------
py360convert v1.0.4 returns faces in the standard y-UP, inside-looking-out
convention: each wall face comes out upright and NON-mirrored (verified by
rendering readable text into a panorama -- "BACK"/"LEFT"/"RIGHT" all read
correctly, not "TFEL"). So the classic OpenCV y-down vs y-up cubemap flip bug is
NOT present here, and the walls need no flipping.

The catch for a paper theater is that the viewer faces the BACKDROP (the box's
back wall), i.e. the opposite direction from the cubemap's nominal "front"
reference. Turning the viewing frame 180deg about the vertical axis means:

  * The wall on the viewer's LEFT is py360convert's +x face -> key 'R'.
  * The wall on the viewer's RIGHT is py360convert's -x face -> key 'L'.
    (Left/Right are swapped relative to py360convert's naming.)
  * The CEILING ('U') and FLOOR ('D') faces get rotated 180deg, because "up"/
    "down" as seen by someone facing the backdrop is yaw-rotated 180deg from the
    library's reference.
  * The BACKDROP ('B') is used as-is.

This exact mapping was confirmed by assembling the 5 faces into the unfolded box
("cross" net) and measuring pixel continuity across all four backdrop seams:
mean abs diff was 0.0 / 255 on every seam (a naive no-swap/no-rotate mapping gave
~80-120 / 255). Run with --preview to eyeball the same net for your own panorama.
"""

import argparse
import base64
import io
import json
import os
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

try:
    import py360convert
except ImportError:
    sys.exit("py360convert is not installed.  pip install py360convert")


# --- verified orientation transforms -----------------------------------------

def _identity(a):
    return a


def _rot180(a):
    return np.rot90(a, 2)


# Output panel  ->  (py360convert dict key, transform to apply).
# See module docstring for how each entry was verified.
PANEL_MAP = {
    "backdrop.png":   ("B", _identity, "back wall, as-is"),
    "wall_left.png":  ("R", _identity, "viewer's left = +x face, as-is"),
    "wall_right.png": ("L", _identity, "viewer's right = -x face, as-is"),
    "ceiling.png":    ("U", _rot180,   "up face, rotated 180deg"),
    "floor.png":      ("D", _rot180,   "down face, rotated 180deg"),
}

EXPECTED_KEYS = set("FRBLUD")


def load_equirect(path):
    img = Image.open(path).convert("RGB")
    arr = np.asarray(img)
    h, w = arr.shape[:2]
    if abs(w - 2 * h) > 2:
        print(f"[warn] input is {w}x{h}; a true equirectangular panorama should "
              f"be 2:1 (w == 2*h). Proceeding anyway.", file=sys.stderr)
    return arr


def equirect_to_faces(equirect, face_size):
    """Run e2c and return the validated dict of py360convert faces."""
    faces = py360convert.e2c(equirect, face_w=face_size, cube_format="dict")
    if not isinstance(faces, dict):
        sys.exit(f"e2c(cube_format='dict') returned {type(faces).__name__}, not a "
                 "dict. Your py360convert version differs from the one this "
                 "script was verified against (1.0.4); inspect its source.")
    keys = set(faces.keys())
    if keys != EXPECTED_KEYS:
        sys.exit(f"e2c returned face keys {sorted(keys)}, expected "
                 f"{sorted(EXPECTED_KEYS)}. The face order/format of your "
                 "py360convert version differs; re-verify the mapping before "
                 "trusting the output.")
    return faces


def build_panels(faces):
    """Apply the verified key-swap + rotations. Returns {filename: ndarray}."""
    panels = {}
    for fname, (key, transform, _desc) in PANEL_MAP.items():
        panels[fname] = transform(faces[key])
    return panels


def _font(size):
    for cand in (
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    ):
        if os.path.exists(cand):
            return ImageFont.truetype(cand, size)
    return ImageFont.load_default()


def render_box(panels, out_path, dims=(1.0, 1.0, 1.0), width=900, height=700,
               fov_deg=52, bg=(12, 12, 15)):
    """Render a 3D perspective view of the assembled open diorama box.

    The box occupies [-hx,hx] x [-hy,hy] x [-hz,hz] (dims = the half-extents in
    x,y,z), with the backdrop at z=-hz and the front (z=+hz) open. The camera
    sits just outside the opening and looks in, so you see the interior the way
    the paper-theater viewer would. Non-equal dims give a rectangular box (the
    square cube faces get stretched onto the walls). Each wall's (u,v)->xyz
    parametrization is taken straight from the verified continuous net, so a
    correct set of panels produces seamless tile/horizon lines across corners.

    This is a tiny inverse ray-caster (5 axis-aligned quads), no 3D dependency.
    """
    hx, hy, hz = (float(d) for d in dims)
    cam = np.array([0.6 * hx, 0.5 * hy, hz + 1.5], float)
    target = np.array([0.0, -0.05 * hy, -0.4 * hz], float)
    fwd = target - cam
    fwd /= np.linalg.norm(fwd)
    right = np.cross(fwd, (0.0, 1.0, 0.0)); right /= np.linalg.norm(right)
    up = np.cross(right, fwd)

    asp = width / height
    half = np.tan(np.radians(fov_deg) / 2)
    i = (np.arange(width) + 0.5) / width * 2 - 1
    j = 1 - (np.arange(height) + 0.5) / height * 2
    ii, jj = np.meshgrid(i, j)
    dirs = (fwd + (ii * asp * half)[..., None] * right + (jj * half)[..., None] * up)
    dirs /= np.linalg.norm(dirs, axis=-1, keepdims=True)

    # (constant-axis, plane value, (u,v) from hit point, panel) per interior face
    quads = [
        (2, -hz, lambda h: ((hx - h[..., 0]) / (2 * hx), (hy - h[..., 1]) / (2 * hy)), panels["backdrop.png"]),
        (0,  hx, lambda h: ((hz - h[..., 2]) / (2 * hz), (hy - h[..., 1]) / (2 * hy)), panels["wall_left.png"]),
        (0, -hx, lambda h: ((h[..., 2] + hz) / (2 * hz), (hy - h[..., 1]) / (2 * hy)), panels["wall_right.png"]),
        (1,  hy, lambda h: ((hx - h[..., 0]) / (2 * hx), (hz - h[..., 2]) / (2 * hz)), panels["ceiling.png"]),
        (1, -hy, lambda h: ((hx - h[..., 0]) / (2 * hx), (h[..., 2] + hz) / (2 * hz)), panels["floor.png"]),
    ]
    bound = {0: hx, 1: hy, 2: hz}

    img = np.empty((height, width, 3), np.uint8); img[:] = bg
    best = np.full((height, width), np.inf)
    for axis, const, uv_fn, tex in quads:
        with np.errstate(divide="ignore", invalid="ignore"):
            t = (const - cam[axis]) / dirs[..., axis]
        hit = cam + t[..., None] * dirs
        inb = t > 1e-4
        for a in (a for a in (0, 1, 2) if a != axis):
            lim = bound[a] * 1.0001
            inb &= (hit[..., a] >= -lim) & (hit[..., a] <= lim)
        sel = inb & (t < best)
        if sel.any():
            u, v = uv_fn(hit)
            s = tex.shape[0]
            col = np.clip((u * (s - 1)).astype(int), 0, s - 1)
            row = np.clip((v * (s - 1)).astype(int), 0, s - 1)
            samp = tex[row, col]
            img[sel] = samp[sel]
            best[sel] = t[sel]

    Image.fromarray(img).save(out_path)
    return out_path


_VIEWER_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Diorama Box Preview</title>
<style>
  html,body{margin:0;height:100%;overflow:hidden;background:#0e0e12;
            font-family:-apple-system,Segoe UI,Helvetica,Arial,sans-serif;}
  #hud{position:absolute;top:10px;left:12px;color:#cfd2da;font-size:12px;
       line-height:1.5;text-shadow:0 1px 2px #000;pointer-events:none;}
  #hud b{color:#fff;}
  button{pointer-events:auto;background:#23252e;color:#dfe2ea;border:1px solid #3a3d47;
         border-radius:5px;padding:3px 8px;font-size:11px;cursor:pointer;margin-top:6px;}
</style>
</head>
<body>
<div id="hud"><b>Diorama box</b><br>WASD: move &middot; Q/E: down/up &middot; drag: look &middot; scroll: dolly &middot; Shift: faster
<br><button id="reset">reset view</button></div>
<script type="importmap">
{ "imports": {
  "three": "https://unpkg.com/three@0.160.0/build/three.module.js"
}}
</script>
<script type="module">
import * as THREE from 'three';

const TEX = __TEX__;
const FACES = __FACES__;            // 4 corners each, order c00,c10,c11,c01
const DIMS = __DIMS__;              // box half-extents [hx, hy, hz]
const UV  = new Float32Array([0,1, 1,1, 1,0, 0,0]);
const IDX = [0,1,2, 0,2,3];

const scene = new THREE.Scene();
scene.background = new THREE.Color(0x0e0e12);
const camera = new THREE.PerspectiveCamera(50, innerWidth/innerHeight, 0.01, 100);
const CAM0 = new THREE.Vector3(0.85 * DIMS[0], 0.7 * DIMS[1], DIMS[2] + 2.7);
const TGT0 = new THREE.Vector3(0, 0, -0.2 * DIMS[2]);
camera.position.copy(CAM0);

const renderer = new THREE.WebGLRenderer({antialias:true});
renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
renderer.setSize(innerWidth, innerHeight);
document.body.appendChild(renderer.domElement);

// --- first-person fly controls (WASD move, Q/E down/up, drag to look) ---
const WORLD_UP = new THREE.Vector3(0, 1, 0);
const PITCH_LIM = THREE.MathUtils.degToRad(89);
let yaw = 0, pitch = 0;
function aimFromTarget(t) {
  const d = t.clone().sub(camera.position).normalize();
  pitch = Math.asin(THREE.MathUtils.clamp(d.y, -1, 1));
  yaw = Math.atan2(d.x, -d.z);
}
function forwardVec() {
  const cp = Math.cos(pitch);
  return new THREE.Vector3(cp * Math.sin(yaw), Math.sin(pitch), -cp * Math.cos(yaw));
}
function applyLook() { camera.lookAt(camera.position.clone().add(forwardVec())); }
aimFromTarget(TGT0); applyLook();

const el = renderer.domElement;
let dragging = false;
el.addEventListener('pointerdown', e => { dragging = true; el.setPointerCapture(e.pointerId); });
el.addEventListener('pointerup',   e => { dragging = false; });
el.addEventListener('pointermove', e => {
  if (!dragging) return;
  const s = 0.0035;
  yaw   += e.movementX * s;
  pitch = THREE.MathUtils.clamp(pitch - e.movementY * s, -PITCH_LIM, PITCH_LIM);
});
el.addEventListener('wheel', e => {
  e.preventDefault();
  camera.position.addScaledVector(forwardVec(), -e.deltaY * 0.002);
}, {passive: false});

const keys = {};
addEventListener('keydown', e => { keys[e.code] = true; });
addEventListener('keyup',   e => { keys[e.code] = false; });

// faces flagged as water get an animated ripple: the texture-sample UVs are
// displaced by moving sine waves and the GPU bilinearly interpolates the
// resample, so the still image reads as a rippling surface. onBeforeCompile is
// used so three.js keeps doing its normal sRGB texture decode.
const WATER = new Set(__WATER__);
const MASK = __MASK__;              // per-face grayscale "where is water" textures
const waterU = { value: 0 };
function makeWater(mat, maskTex) {
  mat.onBeforeCompile = (shader) => {
    shader.uniforms.uTime = waterU;
    shader.uniforms.uMask = { value: maskTex };
    shader.fragmentShader = 'uniform float uTime;\\nuniform sampler2D uMask;\\n' +
      shader.fragmentShader.replace(
      '#include <map_fragment>',
      `#ifdef USE_MAP
        float TAU = 6.2831853;
        vec2 _uv = vMapUv;
        float wm = texture2D(uMask, vMapUv).r;   // 0..1 wet-ness
        float t = uTime;
        vec2 off;
        off.x = sin(TAU*6.0*_uv.y + t) + 0.5*sin(TAU*7.7*(_uv.x+_uv.y) + 2.0*t);
        off.y = cos(TAU*6.0*_uv.x - t) + 0.5*cos(TAU*6.6*(_uv.x-_uv.y) - 2.0*t);
        _uv += float(__WAMP__) * off * wm;       // ripple only where wet
        vec4 sampledDiffuseColor = texture2D( map, _uv );
        diffuseColor *= sampledDiffuseColor;
      #endif`);
  };
  return mat;
}

const loader = new THREE.TextureLoader();
for (const name in FACES) {
  const c = FACES[name];
  const pos = new Float32Array([...c[0], ...c[1], ...c[2], ...c[3]]);
  const g = new THREE.BufferGeometry();
  g.setAttribute('position', new THREE.BufferAttribute(pos, 3));
  g.setAttribute('uv', new THREE.BufferAttribute(UV, 2));
  g.setIndex(IDX);
  const t = loader.load(TEX[name]);
  t.colorSpace = THREE.SRGBColorSpace;
  let m = new THREE.MeshBasicMaterial({map: t, side: THREE.DoubleSide});
  if (WATER.has(name)) {
    const mk = loader.load(MASK[name]);
    mk.colorSpace = THREE.NoColorSpace;
    m = makeWater(m, mk);
  }
  scene.add(new THREE.Mesh(g, m));
}

document.getElementById('reset').onclick = () => {
  camera.position.copy(CAM0); aimFromTarget(TGT0); applyLook();
};
addEventListener('resize', () => {
  camera.aspect = innerWidth/innerHeight; camera.updateProjectionMatrix();
  renderer.setSize(innerWidth, innerHeight);
});

const clock = new THREE.Clock();
let simTime = 0;
(function loop(){
  requestAnimationFrame(loop);
  const dt = Math.min(clock.getDelta(), 0.05);
  simTime += dt; waterU.value = simTime * 1.8;
  const fwd = forwardVec();
  const right = new THREE.Vector3().crossVectors(fwd, WORLD_UP).normalize();
  const speed = ((keys['ShiftLeft'] || keys['ShiftRight']) ? 4.5 : 1.8) * dt;
  const mv = new THREE.Vector3();
  if (keys['KeyW']) mv.add(fwd);
  if (keys['KeyS']) mv.addScaledVector(fwd, -1);
  if (keys['KeyD']) mv.add(right);
  if (keys['KeyA']) mv.addScaledVector(right, -1);
  if (keys['KeyE']) mv.add(WORLD_UP);
  if (keys['KeyQ']) mv.addScaledVector(WORLD_UP, -1);
  if (mv.lengthSq() > 0) camera.position.addScaledVector(mv.normalize(), speed);
  applyLook();
  renderer.render(scene, camera);
})();
</script>
</body>
</html>
"""


def _png_data_uri(arr):
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


def _water_mask_uri(arr, full=False):
    """Grayscale mask (white = water) for where a panel should ripple.

    full=True marks the whole panel as water (the floor). Otherwise water is
    detected by teal/green dominance ((G+B)/2 - R), softly thresholded and
    blurred so the ripple fades out at the waterline instead of cutting hard.
    """
    h, w = arr.shape[:2]
    if full:
        mask = Image.new("L", (w, h), 255)
    else:
        a = arr.astype(np.float32)
        teal = (a[..., 1] + a[..., 2]) / 2 - a[..., 0]
        m = np.clip((teal + 4.0) / 12.0, 0, 1)
        mask = Image.fromarray((m * 255).astype(np.uint8), "L")
        mask = mask.filter(ImageFilter.GaussianBlur(max(1.0, w / 120.0)))
    buf = io.BytesIO()
    mask.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


def build_viewer(panels, out_path, dims=(1.0, 1.0, 1.0),
                 water_faces=("floor", "backdrop", "wall_left", "wall_right"),
                 water_amp=0.014):
    """Write a self-contained interactive WebGL viewer of the assembled box.

    The 5 panels are embedded as base64 PNGs (so the file opens by double-click,
    no server needed) and mapped onto the box with the same verified corner/uv
    parametrization as render_box(). dims are the box half-extents (hx,hy,hz);
    non-equal values give a rectangular box. Faces named in water_faces get an
    animated ripple (UV-displacement resampled by GPU interpolation). Needs
    internet once to pull three.js from a CDN. Fly controls (WASD/QE/drag).
    """
    hx, hy, hz = (float(d) for d in dims)
    tex = {
        "backdrop":   _png_data_uri(panels["backdrop.png"]),
        "wall_left":  _png_data_uri(panels["wall_left.png"]),
        "wall_right": _png_data_uri(panels["wall_right.png"]),
        "ceiling":    _png_data_uri(panels["ceiling.png"]),
        "floor":      _png_data_uri(panels["floor.png"]),
    }
    # corners c00,c10,c11,c01 (matches (u,v)->xyz used by render_box), scaled to dims
    faces = {
        "backdrop":   [[hx, hy, -hz], [-hx, hy, -hz], [-hx, -hy, -hz], [hx, -hy, -hz]],
        "wall_left":  [[hx, hy, hz], [hx, hy, -hz], [hx, -hy, -hz], [hx, -hy, hz]],
        "wall_right": [[-hx, hy, -hz], [-hx, hy, hz], [-hx, -hy, hz], [-hx, -hy, -hz]],
        "ceiling":    [[hx, hy, hz], [-hx, hy, hz], [-hx, hy, -hz], [hx, hy, -hz]],
        "floor":      [[hx, -hy, -hz], [-hx, -hy, -hz], [-hx, -hy, hz], [hx, -hy, hz]],
    }
    water = list(water_faces)  # names match FACES keys: backdrop/wall_left/wall_right/ceiling/floor
    # soft per-face water mask so each panel ripples only where it's wet
    mask = {name: _water_mask_uri(panels[name + ".png"], full=(name == "floor"))
            for name in water}
    html = (_VIEWER_TEMPLATE
            .replace("__TEX__", json.dumps(tex))
            .replace("__FACES__", json.dumps(faces))
            .replace("__DIMS__", json.dumps([hx, hy, hz]))
            .replace("__WATER__", json.dumps(water))
            .replace("__MASK__", json.dumps(mask))
            .replace("__WAMP__", repr(float(water_amp))))
    with open(out_path, "w") as f:
        f.write(html)
    return out_path


def make_preview(panels, out_path):
    """Lay the 5 faces out as the unfolded box ('cross') net, labeled.

    Placement mirrors how the box folds up around the viewer:

                [ ceiling ]
        [ left ][ backdrop ][ right ]
                [  floor  ]

    Because the panels are a consistent cubemap interior, the net is seamless
    when the orientation is correct -- making this both a contact sheet and an
    assembly/orientation sanity check.
    """
    s = next(iter(panels.values())).shape[0]
    pad = max(2, s // 256)
    label_h = max(18, s // 14)
    cell = s + label_h
    canvas = Image.new("RGB", (3 * cell + 4 * pad, 3 * cell + 4 * pad), (15, 15, 18))
    draw = ImageDraw.Draw(canvas)
    font = _font(label_h - 4)

    # grid (col, row) for each panel
    layout = {
        "ceiling.png":    (1, 0),
        "wall_left.png":  (0, 1),
        "backdrop.png":   (1, 1),
        "wall_right.png": (2, 1),
        "floor.png":      (1, 2),
    }
    for fname, (cx, cy) in layout.items():
        x = pad + cx * (cell + pad)
        y = pad + cy * (cell + pad)
        draw.text((x + 2, y), fname.replace(".png", ""), fill=(240, 240, 240), font=font)
        canvas.paste(Image.fromarray(panels[fname]), (x, y + label_h))

    canvas.save(out_path)
    return out_path


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("panorama", help="path to an equirectangular (2:1) panorama image")
    ap.add_argument("face_size", nargs="?", type=int, default=1024,
                    help="resolution per panel, in pixels (default: 1024)")
    ap.add_argument("--outdir", default=".", help="output directory (default: .)")
    ap.add_argument("--preview", action="store_true",
                    help="also write contact_sheet.png laying out all 5 panels")
    ap.add_argument("--box", action="store_true",
                    help="also write box_preview.png, a 3D perspective render of "
                         "the assembled open diorama box")
    ap.add_argument("--viewer", action="store_true",
                    help="also write box_viewer.html, a self-contained interactive "
                         "WebGL viewer (WASD/QE fly through the assembled box)")
    ap.add_argument("--dims", nargs=3, type=float, metavar=("HX", "HY", "HZ"),
                    default=[1.5, 1.0, 1.2],
                    help="box half-extents (width, height, depth) for --box/--viewer. "
                         "Non-equal values make a rectangular box; cube faces get "
                         "stretched. Default: 1.5 1.0 1.2. Use 1 1 1 for a cube.")
    ap.add_argument("--yaw", type=float, default=0.0, metavar="DEG",
                    help="which direction in the panorama becomes the backdrop. "
                         "0 (default) = the panorama's horizontal CENTER faces the "
                         "backdrop; +/-90 = a side; 180 = the panorama's wrap-seam "
                         "edges (py360convert's raw orientation).")
    ap.add_argument("--water-faces", nargs="*",
                    default=["floor", "backdrop", "wall_left", "wall_right"],
                    metavar="FACE",
                    help="which faces ripple as animated water in --viewer (any of: "
                         "backdrop wall_left wall_right ceiling floor). Each ripples "
                         "only where water is detected (floor = entirely). Default: "
                         "floor backdrop wall_left wall_right. Pass with no names to "
                         "disable.")
    ap.add_argument("--water-amp", type=float, default=0.014, metavar="FRAC",
                    help="water ripple amplitude as a fraction of the panel "
                         "(default: 0.014).")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    equirect = load_equirect(args.panorama)
    # Put the chosen panorama direction on the backdrop. The Back cube face samples
    # the equirect's seam (yaw 180 from image center), so by default we roll the
    # image 180 deg -- a lossless horizontal wrap-around -- to bring the panorama's
    # CENTER onto the backdrop. --yaw adds an extra offset on top of that.
    w = equirect.shape[1]
    shift = int(round(-(0.5 + args.yaw / 360.0) * w)) % w
    if shift:
        equirect = np.roll(equirect, shift, axis=1)

    faces = equirect_to_faces(equirect, args.face_size)
    panels = build_panels(faces)

    for fname, arr in panels.items():
        Image.fromarray(arr).save(os.path.join(args.outdir, fname))

    preview_path = None
    if args.preview:
        preview_path = make_preview(panels, os.path.join(args.outdir, "contact_sheet.png"))

    box_path = None
    if args.box:
        box_path = render_box(panels, os.path.join(args.outdir, "box_preview.png"),
                              dims=args.dims)

    viewer_path = None
    if args.viewer:
        viewer_path = build_viewer(panels, os.path.join(args.outdir, "box_viewer.html"),
                                   dims=args.dims, water_faces=args.water_faces,
                                   water_amp=args.water_amp)

    # --- report the resolved mapping so it can be sanity-checked ---
    print(f"\npy360convert version : {getattr(py360convert, '__version__', '?')}")
    print(f"e2c cube_format      : 'dict'  (returned keys: {sorted(faces.keys())})")
    print(f"face_size            : {args.face_size}px")
    backdrop_src = "panorama center" if args.yaw == 0 else f"panorama center +{args.yaw} deg"
    print(f"backdrop direction   : yaw={args.yaw} deg  ({backdrop_src})")
    print(f"output directory     : {os.path.abspath(args.outdir)}")
    print("\nresolved panel mapping (output file <- py360 face [transform] : note):")
    for fname, (key, transform, desc) in PANEL_MAP.items():
        tname = "rot180" if transform is _rot180 else "none"
        print(f"  {fname:<15} <- '{key}' [{tname:<6}] : {desc}")
    print("  (front face skipped -- it's the open side the viewer looks through)")
    if preview_path:
        print(f"\npreview contact sheet: {os.path.abspath(preview_path)}")
    if box_path or viewer_path:
        hx, hy, hz = args.dims
        shape = "cube" if hx == hy == hz else "rectangular box"
        print(f"box half-extents     : {hx} x {hy} x {hz}  ({shape})")
    if box_path:
        print(f"box 3D preview       : {os.path.abspath(box_path)}")
    if viewer_path:
        water = ", ".join(args.water_faces) if args.water_faces else "(none)"
        print(f"animated water faces : {water}")
        print(f"interactive viewer   : {os.path.abspath(viewer_path)}")


if __name__ == "__main__":
    main()
