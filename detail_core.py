"""Core logic for ComfyUI Detail Fixer (no ComfyUI imports, testable on its own).

Boxes are pixel coordinates in the original image: {"label", "x0", "y0", "x1", "y1"}.
"""

import base64
import json
import os
import re
import time
import urllib.error
import urllib.request

GEMINI_MODEL = "gemini-flash-lite-latest"
RETRY_CODES = {429, 500, 502, 503, 529}


class DetailFixerError(Exception):
    pass


def _post_json(url, payload, headers, timeout=180, retries=2, backoff=4):
    data = json.dumps(payload).encode("utf-8")
    for attempt in range(retries + 1):
        req = urllib.request.Request(url, data=data, method="POST",
                                     headers={"Content-Type": "application/json", **headers})
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            body = e.read().decode("utf-8", "replace")[:400]
            if attempt < retries and e.code in RETRY_CODES:
                time.sleep(backoff * (attempt + 1))
                continue
            raise DetailFixerError(f"API error {e.code}: {body}") from e
        except urllib.error.URLError as e:
            raise DetailFixerError(f"Could not reach the API: {e.reason}") from e
        except TimeoutError as e:
            raise DetailFixerError(f"The API did not answer within {timeout} seconds.") from e


DETECT_PROMPT = """Find every {target} in this image.
Return JSON only: a list of objects like {{"label": "short description", "box_2d": [ymin, xmin, ymax, xmax]}}
with coordinates normalized to 0-1000. Include small, partial and distorted ones (for example a garbled or
misspelled logo printed on clothing or a bag). Return [] if there are none. At most {max_items} items."""


def detect_gemini(png_bytes, width, height, target="logo", max_items=5, api_key="", model=""):
    key = (api_key or "").strip() or os.environ.get("GEMINI_API_KEY", "")
    if not key:
        raise DetailFixerError("No Gemini API key. Set GEMINI_API_KEY (free at aistudio.google.com) or use manual boxes.")
    model = (model or "").strip() or GEMINI_MODEL
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
    payload = {
        "contents": [{"role": "user", "parts": [
            {"inline_data": {"mime_type": "image/png", "data": base64.b64encode(png_bytes).decode()}},
            {"text": DETECT_PROMPT.format(target=target.strip() or "logo", max_items=max_items)},
        ]}],
        "generationConfig": {"responseMimeType": "application/json", "temperature": 0},
    }
    r = _post_json(url, payload, {"x-goog-api-key": key})
    try:
        text = "".join(p.get("text", "") for p in r["candidates"][0]["content"]["parts"])
    except (KeyError, IndexError) as e:
        raise DetailFixerError(f"Unexpected Gemini reply: {str(r)[:300]}") from e
    return parse_gemini_boxes(text, width, height, max_items)


def parse_gemini_boxes(text, width, height, max_items=5):
    text = text.strip()
    m = re.search(r"\[.*\]", text, re.S)
    if not m:
        return []
    try:
        items = json.loads(m.group(0))
    except ValueError as e:
        raise DetailFixerError("Gemini returned invalid JSON.") from e
    boxes = []
    for it in items[:max_items]:
        b = it.get("box_2d") if isinstance(it, dict) else None
        if not b or len(b) != 4:
            continue
        ymin, xmin, ymax, xmax = [float(v) for v in b]
        box = {"label": str(it.get("label", "logo")),
               "x0": int(round(min(xmin, xmax) / 1000 * width)), "y0": int(round(min(ymin, ymax) / 1000 * height)),
               "x1": int(round(max(xmin, xmax) / 1000 * width)), "y1": int(round(max(ymin, ymax) / 1000 * height))}
        box = clamp_box(box, width, height)
        if box["x1"] - box["x0"] >= 4 and box["y1"] - box["y0"] >= 4:
            boxes.append(box)
    return boxes


def parse_manual_boxes(text, width, height):
    """One box per line: x,y,w,h in pixels (or x%,y%,w%,h% with a % sign)."""
    boxes = []
    for line in (text or "").splitlines():
        nums = re.findall(r"-?\d+(?:\.\d+)?%?", line)
        if len(nums) < 4:
            continue
        vals = []
        for n, dim in zip(nums[:4], (width, height, width, height)):
            vals.append(float(n[:-1]) / 100 * dim if n.endswith("%") else float(n))
        x, y, w, h = vals
        boxes.append(clamp_box({"label": "manual", "x0": int(x), "y0": int(y),
                                "x1": int(x + w), "y1": int(y + h)}, width, height))
    return boxes


def clamp_box(b, width, height):
    b = dict(b)
    b["x0"], b["x1"] = max(0, min(b["x0"], width)), max(0, min(b["x1"], width))
    b["y0"], b["y1"] = max(0, min(b["y0"], height)), max(0, min(b["y1"], height))
    return b


def crop_region(box, width, height, padding=0.6, target=1024, multiple=16):
    """Square-ish context crop around a box. Returns crop rect + output size for the model."""
    bw, bh = box["x1"] - box["x0"], box["y1"] - box["y0"]
    side = max(bw, bh) * (1 + 2 * padding)
    side = max(side, 64)
    cx, cy = (box["x0"] + box["x1"]) / 2, (box["y0"] + box["y1"]) / 2
    cw, ch = min(side, width), min(side, height)
    x0 = int(round(min(max(cx - cw / 2, 0), width - cw)))
    y0 = int(round(min(max(cy - ch / 2, 0), height - ch)))
    x1, y1 = x0 + int(round(cw)), y0 + int(round(ch))
    scale = target / max(x1 - x0, y1 - y0)
    out_w = max(multiple, int(round((x1 - x0) * scale / multiple)) * multiple)
    out_h = max(multiple, int(round((y1 - y0) * scale / multiple)) * multiple)
    return {"crop": [x0, y0, x1, y1], "box": [box["x0"], box["y0"], box["x1"], box["y1"]],
            "size": [out_w, out_h], "label": box.get("label", "logo")}


def _iou(a, b):
    ix = max(0, min(a["x1"], b["x1"]) - max(a["x0"], b["x0"]))
    iy = max(0, min(a["y1"], b["y1"]) - max(a["y0"], b["y0"]))
    inter = ix * iy
    ua = (a["x1"] - a["x0"]) * (a["y1"] - a["y0"]) + (b["x1"] - b["x0"]) * (b["y1"] - b["y0"]) - inter
    return inter / ua if ua > 0 else 0.0


def parse_bounding_boxes(sources, width, height, max_items=20, iou_merge=0.5):
    """Merge ComfyUI BOUNDING_BOX outputs (e.g. from SAM3 Detect) into Detail Fixer boxes.
    Each source is a list (per image) of lists of {x, y, width, height, score}, or a flat
    list/dict. Overlapping boxes from different prompts are de-duplicated."""
    flat = []

    def walk(v):
        if isinstance(v, dict) and "x" in v and "width" in v:
            flat.append(v)
        elif isinstance(v, (list, tuple)):
            # only first image of a batch
            if v and isinstance(v[0], (list, tuple)):
                walk(v[0])
            else:
                for item in v:
                    walk(item)

    for s in sources:
        if s is not None:
            walk(s)
    flat.sort(key=lambda d: -float(d.get("score", 1.0)))
    boxes = []
    for d in flat:
        b = clamp_box({"label": f"sam3 {float(d.get('score', 1.0)):.2f}",
                       "x0": int(d["x"]), "y0": int(d["y"]),
                       "x1": int(d["x"] + d["width"]), "y1": int(d["y"] + d["height"])}, width, height)
        if b["x1"] - b["x0"] < 2 or b["y1"] - b["y0"] < 2:
            continue
        if any(_iou(b, k) > iou_merge for k in boxes):
            continue
        boxes.append(b)
        if len(boxes) >= max_items:
            break
    return boxes


def merge_touching_boxes(boxes, gap=0.15):
    """Merge boxes that overlap or nearly touch (e.g. a wordmark and its tagline found
    separately), so one logo is regenerated as one piece. gap is relative to the
    smaller box's height."""
    boxes = [dict(b) for b in boxes]
    changed = True
    while changed:
        changed = False
        for i in range(len(boxes)):
            for j in range(i + 1, len(boxes)):
                a, b = boxes[i], boxes[j]
                g = gap * min(a["y1"] - a["y0"], b["y1"] - b["y0"])
                if (a["x0"] - g <= b["x1"] and b["x0"] - g <= a["x1"]
                        and a["y0"] - g <= b["y1"] and b["y0"] - g <= a["y1"]):
                    boxes[i] = {"label": a["label"], "x0": min(a["x0"], b["x0"]), "y0": min(a["y0"], b["y0"]),
                                "x1": max(a["x1"], b["x1"]), "y1": max(a["y1"], b["y1"])}
                    del boxes[j]
                    changed = True
                    break
            if changed:
                break
    return boxes
