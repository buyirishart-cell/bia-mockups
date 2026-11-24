# Buy Irish Art mockup tool (template method)

Places the **real artwork file** into fixed room templates. The painting is never
redrawn or cropped; the room's own window light and shadow is laid over the glass.

## Run

```
pip install opencv-python-headless numpy
python3 tool/bia_mockup.py ARTWORK.jpg --templates tool/templates/templates.json \
    --out OUTDIR --name <shopify-product-handle>
```

Output: six 2048×2048 JPGs, in this order:

| # | Image | Note |
|---|---|---|
| 01 | lean | oak frame leaning on a lime-plaster wall |
| 02 | bedroom | oak frame on a window ledge, pampas grass |
| 03 | living | walnut frame above a cream sofa |
| 04 | kitchen | black frame, plaster wall, no plants |
| 05 | georgian | oak frame above a marble chimneypiece (Georgian Dublin) |
| 06 | brushwork | a true crop of the master file, no AI |

## How it fits each painting

Every room has portrait, landscape and (some) square templates. For each room
the script picks the template whose blank panel is closest to the painting's
proportions, then resizes the **frame** (never the art) to match exactly.
If that would mean stretching the frame more than the template allows
(`max_grow`, default 1.2×), the room is skipped and reported, so a bad fit
never ships.

## Rules

- Use the largest master file available (Artvee Pro / NGI). Flat files with a
  white border are trimmed automatically.
- Look at every image before it goes on a product.
- Rooms: no people, no text, no plants in the kitchen.

## Templates

`templates/templates.json` holds each template's blank-panel and frame-edge
coordinates (in its 1024 px source). The coordinates were checked corner by corner.
To add a room: generate a blank-frame room (flat light-grey panel, no mat),
add it to the JSON with `panel`, `outer`, `anchor` (bottom = leaning,
center = hung) and `order`, and check the corners magnified before use.
