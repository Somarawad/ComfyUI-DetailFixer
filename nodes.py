import io
import json

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw

from .detail_core import (DetailFixerError, crop_region, detect_gemini, merge_touching_boxes,
                        parse_bounding_boxes, parse_manual_boxes)


def _to_pil(img):  # img: [H,W,C] float 0-1
    return Image.fromarray((img.clamp(0, 1).cpu().numpy() * 255).astype(np.uint8))


def _png_bytes(img, max_side=1536):
    pil = _to_pil(img)
    if max(pil.size) > max_side:
        pil.thumbnail((max_side, max_side))
    buf = io.BytesIO()
    pil.save(buf, format="PNG")
    return buf.getvalue()


def _resize(img, w, h):  # [H,W,C] -> [h,w,C]
    x = img.permute(2, 0, 1).unsqueeze(0)
    x = F.interpolate(x, size=(h, w), mode="bicubic", align_corners=False)
    return x[0].permute(1, 2, 0).clamp(0, 1)


def _feather(mask, radius):  # mask [H,W]
    if radius <= 0:
        return mask
    k = radius * 2 + 1
    m = mask.unsqueeze(0).unsqueeze(0)
    for _ in range(2):
        m = F.avg_pool2d(F.pad(m, (radius,) * 4, mode="replicate"), k, stride=1)
    return m[0, 0].clamp(0, 1)


def _first(v):
    return v[0] if isinstance(v, list) else v


class DetailDetect:
    """Find what to fix (a logo, a face, hands...) and return its boxes."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "mode": (["sam3", "gemini", "manual"], {"default": "sam3",
                         "tooltip": "sam3: local, connect the bboxes output of one or two SAM3 Detect nodes. gemini: free Gemini API, describe the target below. manual: type boxes yourself."}),
                "target": ("STRING", {"default": "logo or printed text",
                                      "tooltip": "gemini mode only: what to look for (plain English)."}),
                "max_items": ("INT", {"default": 5, "min": 1, "max": 20}),
                "manual_boxes": ("STRING", {"multiline": True, "default": "",
                                            "tooltip": "One box per line: x, y, width, height (pixels, or add % for percent)."}),
            },
            "optional": {
                "api_key": ("STRING", {"default": "", "tooltip": "Prefer the GEMINI_API_KEY environment variable."}),
                "model": ("STRING", {"default": ""}),
                "bboxes": ("BOUNDING_BOX", {"forceInput": True, "tooltip": "sam3 mode: bboxes output of SAM3 Detect."}),
                "bboxes_2": ("BOUNDING_BOX", {"forceInput": True,
                             "tooltip": "sam3 mode: a second SAM3 Detect (e.g. 'logo' + 'text'); overlaps are merged."}),
                "merge_touching": ("BOOLEAN", {"default": True,
                                   "tooltip": "Join boxes that overlap or touch (e.g. a brand name and its tagline found separately) so each logo is fixed in one piece."}),
            },
        }

    RETURN_TYPES = ("STRING", "MASK", "IMAGE", "INT")
    RETURN_NAMES = ("boxes_json", "mask", "preview", "count")
    FUNCTION = "detect"
    CATEGORY = "Detail Fixer"
    DESCRIPTION = "Detects logos and returns their boxes, a mask and a preview with the boxes drawn."

    def detect(self, image, mode, target, max_items, manual_boxes, api_key="", model="",
               bboxes=None, bboxes_2=None, merge_touching=True):
        img = image[0]
        h, w = img.shape[0], img.shape[1]
        try:
            if mode == "manual":
                boxes = parse_manual_boxes(manual_boxes, w, h)
            elif mode == "sam3":
                if bboxes is None and bboxes_2 is None:
                    raise DetailFixerError("sam3 mode needs the bboxes output of a SAM3 Detect node connected.")
                boxes = parse_bounding_boxes([bboxes, bboxes_2], w, h, max_items)
            else:
                boxes = detect_gemini(_png_bytes(img), w, h, target, max_items, api_key, model)
        except DetailFixerError as e:
            raise RuntimeError(f"Detail Detect: {e}") from e
        if merge_touching:
            boxes = merge_touching_boxes(boxes)
        mask = torch.zeros((1, h, w))
        pil = _to_pil(img)
        draw = ImageDraw.Draw(pil)
        lw = max(2, w // 300)
        for i, b in enumerate(boxes, 1):
            mask[0, b["y0"]:b["y1"], b["x0"]:b["x1"]] = 1.0
            draw.rectangle([b["x0"], b["y0"], b["x1"], b["y1"]], outline=(255, 40, 40), width=lw)
            draw.text((b["x0"] + lw + 2, max(0, b["y0"] - 14)), f"{i}: {b['label'][:30]}", fill=(255, 40, 40))
        preview = torch.from_numpy(np.array(pil).astype(np.float32) / 255.0).unsqueeze(0)
        return (json.dumps(boxes), mask, preview, len(boxes))


class DetailCrop:
    """Cut a high-resolution patch around each detected area, ready for an edit model."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "image": ("IMAGE",),
            "boxes_json": ("STRING", {"forceInput": True}),
            "which": ("INT", {"default": 0, "min": 0, "max": 20, "tooltip": "0 = every detection, 1 = first, 2 = second..."}),
            "context": ("FLOAT", {"default": 0.6, "min": 0.1, "max": 3.0, "step": 0.05,
                                  "tooltip": "Extra area around each detection so the model sees the surroundings and lighting."}),
            "size": ("INT", {"default": 1024, "min": 256, "max": 2048, "step": 64,
                             "tooltip": "Longest side of each crop sent to the edit model."}),
        }, "optional": {
            "mask": ("MASK", {"tooltip": "Optional segmentation mask (e.g. SAM3 Detect masks). Gives shape_mask the "
                              "object's real outline instead of its box. Batches are merged."}),
        }}

    RETURN_TYPES = ("IMAGE", "STRING", "MASK", "INT", "MASK")
    RETURN_NAMES = ("crops", "crop_info", "box_mask", "count", "shape_mask")
    OUTPUT_IS_LIST = (True, True, True, False, True)
    FUNCTION = "crop"
    CATEGORY = "Detail Fixer"
    DESCRIPTION = "Outputs one upscaled crop per logo (a list, so the edit model runs once per logo)."

    def crop(self, image, boxes_json, which, context, size, mask=None):
        img = image[0]
        h, w = img.shape[0], img.shape[1]
        boxes = json.loads(boxes_json or "[]")
        if which > 0:
            boxes = boxes[which - 1:which]
        if not boxes:
            # Nothing found: pass a tiny placeholder through so the rest of the slot still runs,
            # and Paste Back leaves the photo untouched.
            ph = torch.zeros((1, 64, 64, 3))
            info = {"skip": True, "crop": [0, 0, 64, 64], "box": [0, 0, 64, 64], "size": [64, 64], "label": "none"}
            z = torch.zeros((1, 64, 64))
            return ([ph], [json.dumps(info)], [z], 0, [z])
        union = None
        if mask is not None:
            mm = mask if mask.dim() == 3 else mask.unsqueeze(0)
            union = mm.float().amax(dim=0).cpu()
            if union.shape != (h, w):
                union = F.interpolate(union[None, None], size=(h, w), mode="bilinear", align_corners=False)[0, 0]
        crops, infos, masks, shapes = [], [], [], []
        for b in boxes:
            info = crop_region(b, w, h, context, size)
            x0, y0, x1, y1 = info["crop"]
            ow, oh = info["size"]
            crops.append(_resize(img[y0:y1, x0:x1], ow, oh).unsqueeze(0))
            m = torch.zeros((oh, ow))
            sx, sy = ow / (x1 - x0), oh / (y1 - y0)
            bx0, by0, bx1, by1 = info["box"]
            m[int((by0 - y0) * sy):int((by1 - y0) * sy), int((bx0 - x0) * sx):int((bx1 - x0) * sx)] = 1
            masks.append(m.unsqueeze(0))
            if union is not None:
                # object outline, limited to a margin around this box so other objects don't leak in
                gx, gy = (bx1 - bx0) * 0.15, (by1 - by0) * 0.15
                lim = torch.zeros((h, w))
                lim[max(0, int(by0 - gy)):int(by1 + gy), max(0, int(bx0 - gx)):int(bx1 + gx)] = 1
                sm = (union * lim)[y0:y1, x0:x1]
                sm = F.interpolate(sm[None, None], size=(oh, ow), mode="bilinear", align_corners=False)[0, 0]
                shapes.append((sm > 0.5).float().unsqueeze(0) if sm.max() > 0.5 else m.unsqueeze(0))
            else:
                shapes.append(m.unsqueeze(0))
            infos.append(json.dumps(info))
        return (crops, infos, masks, len(crops), shapes)


def _box_blur(x, k):  # x [H,W,C] or [H,W]; mean filter with edge padding
    squeeze = x.dim() == 2
    t = (x.unsqueeze(-1) if squeeze else x).permute(2, 0, 1).unsqueeze(0)
    k = max(1, int(k) | 1)
    r = k // 2
    t = F.avg_pool2d(F.pad(t, (r, r, r, r), mode="replicate"), k, stride=1)
    t = t[0].permute(1, 2, 0)
    return t[..., 0] if squeeze else t


def _erase_box(img, x0, y0, x1, y1, ink_threshold=0.12):
    """Remove the printed logo inside a box but keep the fabric.
    1. Find the ink: pixels that differ clearly from the fabric color at the box edges.
    2. Fill only those pixels from the nearby fabric (growing fill), so folds and
       lighting between and around the letters survive."""
    h, w = img.shape[0], img.shape[1]
    x0, y0 = max(int(x0), 1), max(int(y0), 1)
    x1, y1 = min(int(x1), w - 2), min(int(y1), h - 2)
    H, W = y1 - y0, x1 - x0
    if H < 2 or W < 2:
        return img
    b = 3
    if W >= H:  # wide logo: interpolate top -> bottom (sides may touch arms or background)
        a = img[max(y0 - b, 0):y0, x0:x1].median(dim=0).values
        c = img[y1:min(y1 + b, h), x0:x1].median(dim=0).values
        t = torch.linspace(0, 1, H).view(H, 1, 1)
        est = a.unsqueeze(0) * (1 - t) + c.unsqueeze(0) * t
    else:
        a = img[y0:y1, max(x0 - b, 0):x0].median(dim=1).values
        c = img[y0:y1, x1:min(x1 + b, w)].median(dim=1).values
        t = torch.linspace(0, 1, W).view(1, W, 1)
        est = a.unsqueeze(1) * (1 - t) + c.unsqueeze(1) * t
    region = img[y0:y1, x0:x1]
    ink = ((region - est).abs().amax(dim=-1) > ink_threshold).float()
    r = max(2, round(min(H, W) * 0.04))  # grow over anti-aliased edges
    ink = F.max_pool2d(ink[None, None], 2 * r + 1, stride=1, padding=r)[0, 0]
    keep = 1 - ink
    fill = region.clone()
    k = 3
    while keep.min() < 1 and k < max(H, W) * 4:
        num = _box_blur(fill * keep.unsqueeze(-1), k)
        den = _box_blur(keep, k)
        grown = (den > 1e-3).float()
        new = num / den.clamp(min=1e-6).unsqueeze(-1)
        hole = ((1 - keep) * grown).unsqueeze(-1)
        fill = fill * (1 - hole) + new * hole
        keep = torch.maximum(keep, grown)
        k *= 2
    fill = _box_blur(fill, 5)
    m = _box_blur(_box_blur(ink, r + 1), r + 1).unsqueeze(-1)
    out = img.clone()
    out[y0:y1, x0:x1] = region * (1 - m) + fill * m
    return out


class DetailPrepareLogo:
    """Logo slots only: erase the old print and place a faint guide of your logo before regeneration."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "crop": ("IMAGE",),
                "crop_info": ("STRING", {"forceInput": True}),
                "reference": ("IMAGE",),
                "fit": (["contain", "stretch"], {"default": "contain"}),
                "scale": ("FLOAT", {"default": 0.9, "min": 0.2, "max": 1.2, "step": 0.05}),
                "opacity": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.05}),
                "erase_original": ("BOOLEAN", {"default": True,
                                   "tooltip": "Paint over the garbled logo with the surrounding fabric first, so it can't ghost through."}),
                "erase_grow": ("FLOAT", {"default": 0.15, "min": 0.0, "max": 1.0, "step": 0.05,
                               "tooltip": "How far past the detected box to erase (fraction of box size)."}),
            },
            "optional": {"reference_mask": ("MASK", {"tooltip": "Mask output of Load Image for a transparent PNG logo."})},
        }

    RETURN_TYPES = ("IMAGE", "MASK")
    RETURN_NAMES = ("crop", "logo_mask")
    FUNCTION = "place"
    CATEGORY = "Detail Fixer"
    DESCRIPTION = "Composites the reference logo into the detected box. Follow with a low-denoise pass to match fabric and light."

    def place(self, crop, crop_info, reference, fit, scale, opacity, erase_original=True,
              erase_grow=0.15, reference_mask=None):
        info = json.loads(crop_info)
        c = crop[0].clone()
        oh, ow = c.shape[0], c.shape[1]
        x0, y0, x1, y1 = info["crop"]
        sx, sy = ow / (x1 - x0), oh / (y1 - y0)
        bx0, by0, bx1, by1 = info["box"]
        bx0, bx1 = (bx0 - x0) * sx, (bx1 - x0) * sx
        by0, by1 = (by0 - y0) * sy, (by1 - y0) * sy
        if erase_original:
            gx, gy = (bx1 - bx0) * erase_grow / 2, (by1 - by0) * erase_grow / 2
            c = _erase_box(c, bx0 - gx, by0 - gy, bx1 + gx, by1 + gy)
        bw, bh = (bx1 - bx0) * scale, (by1 - by0) * scale
        ref = reference[0]
        rh, rw = ref.shape[0], ref.shape[1]
        if fit == "contain":
            k = min(bw / rw, bh / rh)
            tw, th = max(1, int(rw * k)), max(1, int(rh * k))
        else:
            tw, th = max(1, int(bw)), max(1, int(bh))
        ref_r = _resize(ref, tw, th)
        if reference_mask is not None:
            alpha = 1.0 - reference_mask[0]  # ComfyUI masks mark transparent areas
            alpha = F.interpolate(alpha[None, None], size=(th, tw), mode="bilinear", align_corners=False)[0, 0]
        else:
            alpha = torch.ones((th, tw))
        alpha = alpha * opacity
        cx, cy = (bx0 + bx1) / 2, (by0 + by1) / 2
        px, py = int(round(cx - tw / 2)), int(round(cy - th / 2))
        ys, xs = slice(max(py, 0), min(py + th, oh)), slice(max(px, 0), min(px + tw, ow))
        ry, rx = slice(ys.start - py, ys.stop - py), slice(xs.start - px, xs.stop - px)
        a = alpha[ry, rx].unsqueeze(-1)
        c[ys, xs] = c[ys, xs] * (1 - a) + ref_r[ry, rx] * a
        m = torch.zeros((oh, ow))
        m[ys, xs] = alpha[ry, rx]
        return (c.unsqueeze(0), m.unsqueeze(0))


class DetailPasteBack:
    """Put every regenerated crop back into the original image with a feathered blend."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "image": ("IMAGE",),
            "fixed_crops": ("IMAGE",),
            "crop_info": ("STRING", {"forceInput": True}),
            "blend_area": (["box", "whole crop"], {"default": "box",
                           "tooltip": "box only changes pixels around the detection, so the rest of the photo stays identical."}),
            "expand": ("FLOAT", {"default": 0.25, "min": 0.0, "max": 1.0, "step": 0.05,
                                 "tooltip": "Grow the blend area beyond the box (fraction of box size)."}),
            "feather": ("INT", {"default": 12, "min": 0, "max": 100}),
        }, "optional": {
            "blend_mask": ("MASK", {"tooltip": "Optional: the inpaint mask used for each crop (same list order). "
                                    "When connected, exactly that area is blended back instead of the box."}),
            "match_color": ("BOOLEAN", {"default": False,
                            "tooltip": "Match the regenerated area's color and contrast to the original (good for faces and skin; leave off for logos)."}),
        }}

    INPUT_IS_LIST = True
    RETURN_TYPES = ("IMAGE", "IMAGE")
    RETURN_NAMES = ("image", "before_after")
    FUNCTION = "paste"
    CATEGORY = "Detail Fixer"
    DESCRIPTION = "Pastes all fixed logos back into the original photo."

    def paste(self, image, fixed_crops, crop_info, blend_area, expand, feather, blend_mask=None, match_color=None):
        orig = _first(image)[0].float().cpu()
        out = orig.clone()
        blend_area, expand, feather = _first(blend_area), _first(expand), _first(feather)
        match_color = bool(_first(match_color)) if match_color is not None else False
        if len(fixed_crops) != len(crop_info):
            raise RuntimeError(f"Detail Paste Back: got {len(fixed_crops)} crops but {len(crop_info)} crop_info items.")
        h, w = out.shape[0], out.shape[1]
        for idx, (crop, info_s) in enumerate(zip(fixed_crops, crop_info)):
            info = json.loads(info_s)
            if info.get("skip"):
                continue
            x0, y0, x1, y1 = info["crop"]
            patch = _resize(crop[0].float().cpu(), x1 - x0, y1 - y0)
            m = torch.zeros((h, w))
            if blend_mask is not None and idx < len(blend_mask):
                bm = blend_mask[idx].float().cpu()
                bm = bm[0] if bm.dim() == 3 else bm
                bm = F.interpolate(bm[None, None], size=(y1 - y0, x1 - x0), mode="bilinear", align_corners=False)[0, 0]
                m[y0:y1, x0:x1] = (bm > 0.5).float()
            elif blend_area == "whole crop":
                m[y0:y1, x0:x1] = 1
            else:
                bx0, by0, bx1, by1 = info["box"]
                # grow by the feather radius too, so the fade happens outside the logo
                # and none of the old logo leaks back in
                gx, gy = (bx1 - bx0) * expand + feather, (by1 - by0) * expand + feather
                m[max(y0, int(by0 - gy)):min(y1, int(by1 + gy)), max(x0, int(bx0 - gx)):min(x1, int(bx1 + gx))] = 1
            if match_color:
                sel = m[y0:y1, x0:x1] > 0.5
                if sel.sum() > 16:
                    src, ref = patch[sel], out[y0:y1, x0:x1][sel]
                    patch = ((patch - src.mean(0)) / src.std(0).clamp(min=1e-4) * ref.std(0) + ref.mean(0)).clamp(0, 1)
            m = _feather(m, feather)
            inner = torch.zeros((h, w))
            inner[y0:y1, x0:x1] = 1
            m = (m * inner).unsqueeze(-1)  # never blend outside the crop
            full = out.clone()
            full[y0:y1, x0:x1] = patch
            out = out * (1 - m) + full * m
        ba = torch.cat([orig, torch.ones((h, max(4, w // 100), 3)), out], dim=1)
        return (out.unsqueeze(0), ba.unsqueeze(0))


class DetailPickModel:
    """Choose which result a fix slot passes on. Only the chosen branch is computed."""

    CHOICES = ["skip", "local", "api 1", "api 2"]

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "use": (cls.CHOICES, {"default": "local",
                    "tooltip": "skip = leave the photo as it is. local / api 1 / api 2 = take that branch's result. "
                               "Branches that aren't chosen don't run, so API credits are only spent when selected."}),
            "original": ("IMAGE", {"lazy": True}),
        }, "optional": {
            "local": ("IMAGE", {"lazy": True}),
            "api_1": ("IMAGE", {"lazy": True}),
            "api_2": ("IMAGE", {"lazy": True}),
        }}

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "pick"
    CATEGORY = "Detail Fixer"

    @staticmethod
    def _key(use):
        return {"skip": "original", "local": "local", "api 1": "api_1", "api 2": "api_2"}[use]

    def check_lazy_status(self, use, original=None, local=None, api_1=None, api_2=None):
        key = self._key(use)
        return [key] if {"original": original, "local": local, "api_1": api_1, "api_2": api_2}[key] is None else []

    def pick(self, use, original=None, local=None, api_1=None, api_2=None):
        out = {"original": original, "local": local, "api_1": api_1, "api_2": api_2}[self._key(use)]
        if out is None:
            raise RuntimeError(f"Detail Pick Model: '{use}' is selected but nothing is connected to it.")
        return (out,)


NODE_CLASS_MAPPINGS = {
    "DetailDetect": DetailDetect,
    "DetailCrop": DetailCrop,
    "DetailPrepareLogo": DetailPrepareLogo,
    "DetailPasteBack": DetailPasteBack,
    "DetailPickModel": DetailPickModel,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "DetailDetect": "Detail Detect",
    "DetailCrop": "Detail Crop",
    "DetailPrepareLogo": "Detail Prepare Logo (erase + guide)",
    "DetailPasteBack": "Detail Paste Back",
    "DetailPickModel": "Detail Pick Model",
}
