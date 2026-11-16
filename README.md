# Showdown vision v6 (det + seg)

Two models that read a landscape 16:9 game frame resized to **960 x 540**. They were trained on frames rendered by libg 68
under qemu (offline battle server), with labels from the game's own state. Both run on onnxruntime-directml, so they work
on NVIDIA and AMD GPUs.

**Model files are in the [v6 release](https://github.com/mlgdoggynoob/showdown-vision-v6/releases/tag/v6)** (too big for the repo itself).

| file | what |
| --- | --- |
| `det.onnx` / `det.pt` | entity detector, YOLO26s (Ultralytics 8.4), from COCO weights, 29 classes |
| `seg_argmax.onnx` | tile / wall segmentation, outputs class ids (ArgMax in the graph) |
| `seg.onnx` / `seg.pt` | the same seg model with logits out; `seg.pt` = state_dict of `smp.FPN('tu-mobilenetv3_large_100', encoder_weights=None, classes=19)` |
| `seg_iou.txt` | per-class val IoU |
| `reference_inference_live_run.py` | our live loop; `detect()` and `segment()` show the exact pre/post-processing (it imports our adapter, so it won't run on its own) |

## Detector (`det.onnx`)
- Input `images` 1 x 3 x 544 x 960, RGB, /255. Letterbox = the 960 x 540 frame with 2 rows of grey (114) above and below.
- Output `output0` 1 x 33 x 10710 = cx, cy, w, h + 29 class scores (the one-to-many head, **no NMS in the graph**).
  Decode: take the best class per column, conf >= 0.3, then NMS per class at IoU 0.6, and subtract 2 px from y.
- Classes: 0 player, 1 teammate, 2 enemy, 3 teammate_summon, 4 enemy_summon, 5 power_box, 6 boss, 7 objective,
  8 teammate_projectile, 9 enemy_projectile, 10 teammate_area_effect, 11 enemy_area_effect, 12 gem, 13 power_cube,
  14 item, 15 player_health, 16 teammate_health, 17 enemy_health, 18 player_ammo, 19 destructible,
  20 movement_joystick, 21 attack_joystick, 22 super_ready, 23 super_empty, 24 gadget_ready, 25 gadget_empty,
  26 hypercharge_ready, 27 hypercharge_empty, 28 teammate_marker
- Val (18.8k frames): mAP50 .914, mAP50-95 .814. On 60 hand-labelled real frames: precision .833, recall .907.
- Known weak spots:
  - Small projectiles: recall by box long side is < 12 px .03, 12-20 px .32, 20-30 px .52, > 30 px .56. We are working on this.
  - It misses the player inside a shield bubble.
  - It boxes the grey chat button as `destructible`.
  - The brawlers SoulCollector, JetpackGirl, Knight, Undertaker and Baseball are not in the training data.

## Segmentation (`seg_argmax.onnx`)
- Input `images` 1 x 3 x 288 x 480: the frame resized to 480 x 272 (bilinear, RGB, /255), with 16 zero rows below.
- Output: 288 x 480 class ids (uint8). Drop the last 16 rows, then resize nearest to 960 x 540.
- Classes: 0 floor, 1 indestructible, 2 destructible_wall, 3 fragile_wall, 4 bush, 5 faded_bush, 6 water,
  7 rope_fence, 8 bouncer, 9 gravity, 10 slow_tile, 11 fast_tile, 12 ice, 13 damage_tile, 14 power_box, 15 poison,
  16 out_of_bounds, 17 destructible, 18 ui
- Val mIoU .849 (per class in `seg_iou.txt`). It ignores gas and destroyed tiles.

## Speed
On an RX 7900 XT with DirectML: det about 10 ms, seg_argmax about 4 ms. Our whole live chain (grab + det + seg + our
adapter) runs at 10 frames/s with p95 53 ms.

## Validation set (same frames our numbers above come from)
On the [v6 release](https://github.com/mlgdoggynoob/showdown-vision-v6/releases/tag/v6): `showdown-v6-val-part*.zip`
and `showdown-v6-val_mk-part*.zip` (stored zips, ~6.9 GB in all). Unzip every part into one folder:
- `images/val/*.png` (16,969 frames, 960 x 540) + `labels/val/*.txt` (YOLO boxes, the 29 classes above) +
  `seg/val/*.png` (uint8 masks at 960 x 540, pixel = class id, names in `seg_names.txt`)
- `images/val_mk` + `labels/val_mk` (1,800 frames with teammate markers drawn in, for class 28; no seg masks)

Detector: `yolo val model=det.pt data=data.yaml imgsz=960` with `data.yaml` from this repo. Seg: mIoU over
`seg/val`, using the input and output steps above.
