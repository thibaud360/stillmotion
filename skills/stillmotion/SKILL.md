---
name: stillmotion
description: Turn ONE still photo into a seamless looping cinemagraph (MP4 + GIF) where only part moves: drifting clouds, flowing or rippling water, steam, light shimmer, falling snow. Use when the user wants a photo to come alive, a cinemagraph, a living photo or a looping animation.
---

# stillmotion

Makes a looping cinemagraph from one photo or drawing with `scripts/stillmotion.py` (Python 3, NumPy, OpenCV, Pillow, ffmpeg). No video model: a mask picks the moving area, a noise displacement field moves it, and two phases are blended so the last frame equals the first.

## Effects

| `--effect` | Moves | Auto mask |
|---|---|---|
| `drift` | clouds / sky slide sideways (`--dir`) | sky, found from the top edge down to the horizon |
| `flow` | water or a river flows along `--dir` | water in the lower part of the frame |
| `ripple` | lake surface shimmers in place | water |
| `shimmer` | heat haze + light flicker | bright / glowing areas |
| `steam` | wisps rise from a cup | none: needs `--region` |
| `snow` | looping falling snow overlay | whole frame |

## Steps

1. Look at the photo first (Read tool). Pick the effect from what is in it: sky or clouds `drift`, lake or river `flow`/`ripple`, hot drink `steam`, glowing lights `shimmer`, winter or "make it Christmas" `snow`. Nothing obvious: `--effect auto`.
2. Install if needed: `pip install pillow numpy opencv-python` (add `--break-system-packages` if pip asks). `ffmpeg` must be on the PATH.
3. Run: `python scripts/stillmotion.py PHOTO --effect drift --save-mask -o out`. Always add `--save-mask` on the first run and look at `out/NAME_maskview.jpg` (green tint = the part that moves). If the mask is wrong, set the area yourself:
   - `--region "x1,y1,x2,y2"` in percent of the image, e.g. `--region "0,55,100,100"` for the lower water
   - a polygon: `--region "0,60;100,55;100,100;0,100"`
   - `--mask mask.png` (white = moves), `--invert` to flip it
4. Steam: look where the cup's liquid is and pass a tall box that starts at the liquid surface and goes up, about as wide as the cup opening: `--region "32,0,64,38"`.
5. Extract a frame or two from the MP4 and look at them before sending. Too much warping: lower `--strength`. Too calm: raise it. The script prints a loop check; it must say `seamless`.
6. Send the GIF (previews everywhere) and the MP4 (sharp). Offer one tweak in a line: speed, strength, direction, another region.

## Options

- `--speed 1.0` how fast things move (clouds drift further, steam rises faster, snow falls faster)
- `--strength 1.0` how strong the distortion / how opaque the steam
- `--dir right|left|up|down|DEGREES` for `drift`, `flow`, `snow` (0 = right, 90 = down)
- `--density 1.0` amount of snow
- `--feather PCT` softness of the mask edge, in percent of the image size
- `--seconds 4 --fps 24` loop length and frame rate
- `--live` also writes a 3 s, 30 fps iPhone-friendly clip (`*_live.mp4`)
- `--gif-width 480`, `--no-gif`, `--no-mp4`, `--width 1080` max output width, `--seed N` another noise pattern

Outputs go to `-o DIR` as `NAME_EFFECT.mp4`, `NAME_EFFECT.gif` (kept under about 3.5 MB), `NAME_EFFECT_live.mp4`.

## Notes

- Works best when the moving part has some texture (clouds, ripples, a lit surface). A perfectly smooth gradient sky has nothing visible to move.
- Present it as a tool for the user's own photo. Do not call the result AI art.
- If the script is missing: `git clone --depth 1 https://github.com/thibaud360/stillmotion /tmp/stillmotion` and use `/tmp/stillmotion/skills/stillmotion/scripts/stillmotion.py`.
