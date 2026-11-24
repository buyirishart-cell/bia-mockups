# Buy Irish Art mockup tool (v3: rendered frames)

Rooms are photographs of **bare walls**. The tool builds the frame itself at the
painting's exact proportions, in Black / Wood / White (the Gelato range), puts the
**real artwork file** inside untouched, lays that patch of wall's own light and
window shadow over frame and glass, and casts the frame's shadow on the wall.

Nothing is stretched, nothing in the room is covered, the art is never cropped
or redrawn, and any painting shape works.

## Run

```
pip install opencv-python-headless numpy
python3 tool/bia_mockup.py ARTWORK --templates tool/templates/templates.json \
    --out OUTDIR --name <shopify-product-handle> [--finish auto|wood|black|white]
```

Output: six 2048x2048 JPGs: 01 lean (wood), 02 bedroom (wood), 03 living (black),
04 kitchen (black), 05 georgian (white), 06 brushwork (true crop of the master file).
`--finish black` etc. renders every room in that finish (e.g. variant images).

## Templates

`templates/templates.json`, per room (coordinates in the 1024 px source):
- `area` - the box the whole frame must stay inside (clear wall, away from objects)
- `anchor` - `bottom` = resting on floor / ledge, `center` = hung
- `fill`, `moulding` (frame width as a fraction of size), `finish`, `light` (side the sun comes from), `shadow`

To add a room: generate a bare-wall room (no frames), approve it, set its `area`
clear of every object, and check renders for a tall, a square and a wide painting.

Always look at every image before it goes on a product.
