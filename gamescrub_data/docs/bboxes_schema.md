# `bboxes.json` schema

Per-frame HUD bounding boxes for one `clip.mp4`, in normalized `[0, 1]`
coordinates (so they're resolution-independent and survive any resize):

```json
{
  "schema": "per_frame_bbox_v1",
  "frames": [
    {
      "frame_index": 0,
      "bboxes_xyxy_norm": [
        [0.02, 0.03, 0.18, 0.09],
        [0.80, 0.85, 0.97, 0.98]
      ]
    },
    {
      "frame_index": 1,
      "bboxes_xyxy_norm": []
    }
  ]
}
```

- `frame_index`: 0-based, matching `clip.mp4`'s own frame order/fps. Not
  every frame needs an entry -- a missing frame is treated as "no HUD boxes
  this frame" (all-black mask).
- `bboxes_xyxy_norm`: zero or more `[x1, y1, x2, y2]` boxes, each a
  separate detected HUD element (minimap, health bar, crosshair, subtitle,
  ...), normalized to the clip's own width/height. Multiple boxes per frame
  are fine and expected -- most game HUDs have several.

Boxes can come from any detector or from manual annotation -- the pipeline
only reads them. The `schema` field is informational and not checked.

Boxes are used exactly as given (no padding), so any HUD pixels outside a
box are left untouched in the output: make the boxes cover each element
fully.
