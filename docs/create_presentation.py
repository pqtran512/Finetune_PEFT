"""Generate concise professional slides: Fine-tune + Prototype."""
from __future__ import annotations

from pathlib import Path

from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_CONNECTOR, MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.oxml.ns import nsmap, qn
from pptx.util import Emu, Inches, Pt

# --- Design tokens ---
BG = RGBColor(0x0B, 0x12, 0x20)
CARD = RGBColor(0x14, 0x1C, 0x2E)
CARD_ALT = RGBColor(0x1A, 0x24, 0x3A)
ACCENT = RGBColor(0x14, 0xB8, 0xA6)  # teal
ACCENT_DIM = RGBColor(0x0F, 0x76, 0x6E)
WHITE = RGBColor(0xF8, 0xFA, 0xFC)
MUTED = RGBColor(0x94, 0xA3, 0xB8)
SOFT = RGBColor(0xCB, 0xD5, 0xE1)
LINE = RGBColor(0x33, 0x41, 0x55)
STEP_FILL = RGBColor(0x0E, 0x74, 0x9A)
STEP_FILL_2 = RGBColor(0x15, 0x58, 0x75)

OUT = Path(__file__).resolve().parent / "FineTune_Prototype_Slides.pptx"
W, H = Inches(13.333), Inches(7.5)  # 16:9


def _set_run(run, size=18, bold=False, color=WHITE, font="Segoe UI"):
    run.font.name = font
    run.font.size = Pt(size)
    run.font.bold = bold
    run.font.color.rgb = color


def _fill_solid(shape, color: RGBColor):
    shape.fill.solid()
    shape.fill.fore_color.rgb = color
    shape.line.fill.background()


def _add_bg(slide):
    shp = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, W, H)
    _fill_solid(shp, BG)
    # top accent line
    bar = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, W, Inches(0.06))
    _fill_solid(bar, ACCENT)


def _textbox(slide, left, top, width, height, text, size=18, bold=False, color=WHITE, align=PP_ALIGN.LEFT, font="Segoe UI"):
    box = slide.shapes.add_textbox(left, top, width, height)
    tf = box.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.alignment = align
    run = p.add_run()
    run.text = text
    _set_run(run, size=size, bold=bold, color=color, font=font)
    return box


def _bullets(slide, left, top, width, height, items, size=16, color=SOFT, spacing=10):
    box = slide.shapes.add_textbox(left, top, width, height)
    tf = box.text_frame
    tf.word_wrap = True
    for i, item in enumerate(items):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment = PP_ALIGN.LEFT
        p.space_after = Pt(spacing)
        p.level = 0
        run = p.add_run()
        run.text = f"•  {item}"
        _set_run(run, size=size, color=color)
    return box


def _card(slide, left, top, width, height, fill=CARD):
    shp = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, left, top, width, height)
    _fill_solid(shp, fill)
    shp.adjustments[0] = 0.08
    return shp


def _pill(slide, left, top, width, height, text, fill=ACCENT, text_color=BG, size=12, bold=True):
    shp = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, left, top, width, height)
    _fill_solid(shp, fill)
    shp.adjustments[0] = 0.5
    tf = shp.text_frame
    tf.word_wrap = False
    tf.paragraphs[0].alignment = PP_ALIGN.CENTER
    shp.text_frame.auto_size = None
    p = tf.paragraphs[0]
    run = p.add_run()
    run.text = text
    _set_run(run, size=size, bold=bold, color=text_color)
    tf.paragraphs[0].space_before = Pt(0)
    # vertical-ish center via anchor
    try:
        tf._txBody.bodyPr.set("anchor", "ctr")
    except Exception:
        pass
    return shp


def _flow_box(slide, left, top, width, height, title, subtitle=None, fill=STEP_FILL, title_size=12):
    shp = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, left, top, width, height)
    _fill_solid(shp, fill)
    shp.adjustments[0] = 0.15
    tf = shp.text_frame
    tf.word_wrap = True
    try:
        tf._txBody.bodyPr.set("anchor", "ctr")
    except Exception:
        pass
    p = tf.paragraphs[0]
    p.alignment = PP_ALIGN.CENTER
    run = p.add_run()
    run.text = title
    _set_run(run, size=title_size, bold=True, color=WHITE)
    if subtitle:
        p2 = tf.add_paragraph()
        p2.alignment = PP_ALIGN.CENTER
        r2 = p2.add_run()
        r2.text = subtitle
        _set_run(r2, size=10, color=SOFT)
    return shp


def _arrow_right(slide, left, top, width=Inches(0.28), height=Inches(0.22)):
    shp = slide.shapes.add_shape(MSO_SHAPE.RIGHT_ARROW, left, top, width, height)
    _fill_solid(shp, ACCENT)
    return shp


def _arrow_down(slide, left, top, width=Inches(0.22), height=Inches(0.28)):
    shp = slide.shapes.add_shape(MSO_SHAPE.DOWN_ARROW, left, top, width, height)
    _fill_solid(shp, ACCENT)
    return shp


def slide_title(prs):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    _add_bg(slide)

    # left accent panel
    panel = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, Inches(0.18), H)
    _fill_solid(panel, ACCENT)

    _textbox(slide, Inches(0.9), Inches(2.0), Inches(11), Inches(0.4),
             "ĐỒ ÁN / BÁO CÁO", size=14, bold=True, color=ACCENT)
    _textbox(slide, Inches(0.9), Inches(2.45), Inches(11.2), Inches(1.4),
             "Fine-tune CodeLlama cho\nJava Code Completion", size=36, bold=True, color=WHITE)
    _textbox(slide, Inches(0.9), Inches(4.2), Inches(11), Inches(0.5),
             "QLoRA  •  Bayesian Optimization  •  Prototype Web Demo", size=16, color=MUTED)

    # bottom meta chips
    _pill(slide, Inches(0.9), Inches(5.5), Inches(2.4), Inches(0.38), "CodeLlama-7B", fill=CARD_ALT, text_color=SOFT, size=11)
    _pill(slide, Inches(3.5), Inches(5.5), Inches(2.2), Inches(0.38), "HumanEval-Java", fill=CARD_ALT, text_color=SOFT, size=11)
    _pill(slide, Inches(5.9), Inches(5.5), Inches(2.6), Inches(0.38), "Flask Prototype", fill=CARD_ALT, text_color=SOFT, size=11)


def slide_pipeline(prs):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    _add_bg(slide)
    _textbox(slide, Inches(0.7), Inches(0.35), Inches(10), Inches(0.5),
             "01  —  Pipeline Fine-tune", size=26, bold=True)
    _textbox(slide, Inches(0.7), Inches(0.9), Inches(11), Inches(0.35),
             "7 bước end-to-end (tự động hóa bởi run_pipeline.py)", size=14, color=MUTED)

    steps_row1 = [
        ("1. Build\nDataset", "Evol + Magicoder"),
        ("2. EDA &\nLọc data", "Chống nhiễu OOD"),
        ("3. Bayesian\nOpt", "Tìm hyperparams"),
        ("4. Full\nTrain", "QLoRA 4-bit"),
    ]
    steps_row2 = [
        ("5. Merge\nLoRA", "Gắn vào base"),
        ("6. Inference", "HumanEval-Java"),
        ("7. Evaluate", "pass@1"),
    ]

    box_w, box_h = Inches(2.35), Inches(1.25)
    y1 = Inches(1.8)
    gap = Inches(0.35)
    start_x = Inches(0.7)
    fills = [STEP_FILL, STEP_FILL, ACCENT_DIM, STEP_FILL_2]

    for i, ((t, s), fill) in enumerate(zip(steps_row1, fills)):
        x = start_x + i * (box_w + gap)
        _flow_box(slide, x, y1, box_w, box_h, t, s, fill=fill, title_size=13)
        if i < len(steps_row1) - 1:
            _arrow_right(slide, x + box_w + Inches(0.04), y1 + Inches(0.52), Inches(0.28), Inches(0.22))

    # down arrow from last of row1 to center of row2
    _arrow_down(slide, Inches(10.35), Inches(3.15), Inches(0.22), Inches(0.32))

    y2 = Inches(3.65)
    # row2 centered under
    row2_total = 3 * box_w + 2 * gap
    start_x2 = (W - row2_total) / 2
    fills2 = [STEP_FILL_2, STEP_FILL, ACCENT]

    for i, ((t, s), fill) in enumerate(zip(steps_row2, fills2)):
        x = start_x2 + i * (box_w + gap)
        _flow_box(slide, x, y2, box_w, box_h, t, s, fill=fill, title_size=13)
        if i < len(steps_row2) - 1:
            _arrow_right(slide, x + box_w + Inches(0.04), y2 + Inches(0.52), Inches(0.28), Inches(0.22))

    _card(slide, Inches(0.7), Inches(5.35), Inches(11.9), Inches(1.5))
    _textbox(slide, Inches(1.0), Inches(5.55), Inches(11), Inches(0.35),
             "Điểm nhấn", size=14, bold=True, color=ACCENT)
    _bullets(slide, Inches(1.0), Inches(5.95), Inches(11.3), Inches(0.8),
             [
                 "Data completion (prefix/target) — không dùng chat format",
                 "QLoRA trên RTX 3060 12GB  •  Hyperparams từ Optuna BO",
             ], size=14, color=SOFT, spacing=6)


def slide_data(prs):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    _add_bg(slide)
    _textbox(slide, Inches(0.7), Inches(0.35), Inches(11), Inches(0.5),
             "02  —  Dataset & Format", size=26, bold=True)

    # Left card: sources + filters
    _card(slide, Inches(0.7), Inches(1.15), Inches(5.8), Inches(5.6))
    _textbox(slide, Inches(1.0), Inches(1.4), Inches(5.2), Inches(0.4),
             "Nguồn & lọc", size=18, bold=True, color=ACCENT)
    _bullets(slide, Inches(1.0), Inches(2.0), Inches(5.2), Inches(4.2),
             [
                 "Evol-Instruct + Magicoder",
                 "Map → {prefix, target}",
                 "Lọc markdown / naked method",
                 "Bọc class Problem (~80%)",
                 "Lọc helper call lỗi",
                 "Decontaminate HumanEval",
             ], size=15, spacing=12)

    # Right: format diagram
    _card(slide, Inches(6.8), Inches(1.15), Inches(5.8), Inches(5.6), fill=CARD_ALT)
    _textbox(slide, Inches(7.1), Inches(1.4), Inches(5.2), Inches(0.4),
             "Mẫu huấn luyện", size=18, bold=True, color=ACCENT)

    _flow_box(slide, Inches(7.3), Inches(2.15), Inches(4.9), Inches(1.7),
              "PREFIX  (loss off)",
              "imports + class Problem {\nJavadoc + signature + {",
              fill=STEP_FILL, title_size=14)
    _arrow_down(slide, Inches(9.5), Inches(4.0), Inches(0.25), Inches(0.3))
    _flow_box(slide, Inches(7.3), Inches(4.4), Inches(4.9), Inches(1.7),
              "TARGET  (loss on)",
              "method body + }",
              fill=ACCENT_DIM, title_size=14)


def slide_train(prs):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    _add_bg(slide)
    _textbox(slide, Inches(0.7), Inches(0.35), Inches(11), Inches(0.5),
             "03  —  Training & Evaluation", size=26, bold=True)

    # 3 columns
    cols = [
        ("Bayesian Opt", [
            "Optuna TPE",
            "Proxy: ~8% data / 1 epoch",
            "Minimize eval_loss",
            "Xuất best_params.json",
        ], STEP_FILL),
        ("QLoRA Train", [
            "CodeLlama-7B + 4-bit",
            "LoRA: q/k/v/o + MLP",
            "NEFTune • cosine LR",
            "Adapter: evol_completion_bo_v2",
        ], ACCENT_DIM),
        ("Evaluate", [
            "Merge LoRA → base",
            "Inference HumanEval-Java",
            "Đo pass@1",
            "Phân loại: PASS / COMPILE / FAIL",
        ], STEP_FILL_2),
    ]

    x0 = Inches(0.7)
    cw = Inches(3.85)
    gap = Inches(0.35)
    for i, (title, items, fill) in enumerate(cols):
        x = x0 + i * (cw + gap)
        _card(slide, x, Inches(1.2), cw, Inches(5.5))
        header = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, x, Inches(1.2), cw, Inches(0.7))
        _fill_solid(header, fill)
        header.adjustments[0] = 0.1
        tf = header.text_frame
        try:
            tf._txBody.bodyPr.set("anchor", "ctr")
        except Exception:
            pass
        p = tf.paragraphs[0]
        p.alignment = PP_ALIGN.CENTER
        run = p.add_run()
        run.text = title
        _set_run(run, size=16, bold=True, color=WHITE)
        _bullets(slide, x + Inches(0.3), Inches(2.2), cw - Inches(0.5), Inches(4.0),
                 items, size=14, spacing=14)


def slide_prototype_arch(prs):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    _add_bg(slide)
    _textbox(slide, Inches(0.7), Inches(0.35), Inches(11), Inches(0.5),
             "04  —  Prototype: Java AI Code Generator", size=26, bold=True)
    _textbox(slide, Inches(0.7), Inches(0.9), Inches(11), Inches(0.35),
             "Flask Web  •  CodeLlama-7B + LoRA  •  Prompt Enhancer", size=14, color=MUTED)

    nodes = [
        ("User\nInput", "NL / Signature / Java"),
        ("Prompt\nEnhancer", "Chuẩn hóa format"),
        ("Model\nGenerate", "Stream tokens"),
        ("Clean +\njvac check", "Sửa output"),
        ("Run /\nDownload", "Test + .java"),
    ]
    box_w, box_h = Inches(2.0), Inches(1.35)
    y = Inches(2.0)
    start = Inches(0.55)
    gap = Inches(0.42)
    fills = [STEP_FILL, ACCENT_DIM, STEP_FILL_2, STEP_FILL, ACCENT]

    for i, ((t, s), fill) in enumerate(zip(nodes, fills)):
        x = start + i * (box_w + gap)
        _flow_box(slide, x, y, box_w, box_h, t, s, fill=fill, title_size=13)
        if i < len(nodes) - 1:
            _arrow_right(slide, x + box_w + Inches(0.05), y + Inches(0.55), Inches(0.32), Inches(0.24))

    # bottom feature strip
    _card(slide, Inches(0.7), Inches(4.0), Inches(11.9), Inches(2.7))
    _textbox(slide, Inches(1.0), Inches(4.25), Inches(11), Inches(0.4),
             "Khả năng chính", size=16, bold=True, color=ACCENT)

    feats = [
        ("3 kiểu đầu vào", "Ngôn ngữ tự nhiên, chữ ký hàm, class đầy đủ"),
        ("Prompt rules ND4", "Javadoc, class Problem, CoT, language tag"),
        ("Verify thực tế", "Biên dịch javac + chạy java trong sandbox"),
        ("UX demo", "Preset bài mẫu • gợi ý input • tải Problem.java"),
    ]
    for i, (h, d) in enumerate(feats):
        col = i % 2
        row = i // 2
        lx = Inches(1.0) + col * Inches(5.8)
        ly = Inches(4.8) + row * Inches(0.85)
        _textbox(slide, lx, ly, Inches(5.4), Inches(0.3), h, size=14, bold=True, color=WHITE)
        _textbox(slide, lx, ly + Inches(0.3), Inches(5.4), Inches(0.35), d, size=12, color=MUTED)


def slide_summary(prs):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    _add_bg(slide)
    _textbox(slide, Inches(0.7), Inches(0.35), Inches(11), Inches(0.5),
             "05  —  Tóm tắt", size=26, bold=True)

    items = [
        ("Data", "Evol/Magicoder → completion Java, lọc nhiễu, khớp HumanEval"),
        ("Train", "QLoRA CodeLlama-7B + Bayesian Optimization"),
        ("Eval", "pass@1 trên HumanEval-Java"),
        ("Prototype", "Web demo: enhance → generate → compile → run"),
    ]

    for i, (k, v) in enumerate(items):
        y = Inches(1.3) + i * Inches(1.2)
        _card(slide, Inches(0.7), y, Inches(11.9), Inches(1.05))
        num = slide.shapes.add_shape(MSO_SHAPE.OVAL, Inches(1.0), y + Inches(0.28), Inches(0.5), Inches(0.5))
        _fill_solid(num, ACCENT if i == len(items) - 1 else STEP_FILL)
        tf = num.text_frame
        try:
            tf._txBody.bodyPr.set("anchor", "ctr")
        except Exception:
            pass
        p = tf.paragraphs[0]
        p.alignment = PP_ALIGN.CENTER
        run = p.add_run()
        run.text = str(i + 1)
        _set_run(run, size=14, bold=True, color=BG)
        _textbox(slide, Inches(1.8), y + Inches(0.2), Inches(2.2), Inches(0.35), k, size=16, bold=True, color=ACCENT)
        _textbox(slide, Inches(4.1), y + Inches(0.2), Inches(8), Inches(0.65), v, size=15, color=SOFT)


def main():
    prs = Presentation()
    prs.slide_width = W
    prs.slide_height = H

    slide_title(prs)
    slide_pipeline(prs)
    slide_data(prs)
    slide_train(prs)
    slide_prototype_arch(prs)
    slide_summary(prs)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    prs.save(str(OUT))
    print(f"Saved: {OUT}")


if __name__ == "__main__":
    main()
