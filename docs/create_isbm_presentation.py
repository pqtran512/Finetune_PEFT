"""Generate ISBM conference slides (English, ~10 min) — reuses project template."""
from __future__ import annotations

from pathlib import Path

from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import PP_ALIGN
from pptx.util import Inches, Pt

from create_presentation import (
    ACCENT,
    ACCENT_DIM,
    BG,
    CARD,
    CARD_ALT,
    MUTED,
    SOFT,
    STEP_FILL,
    STEP_FILL_2,
    WHITE,
    W,
    H,
    _add_bg,
    _arrow_down,
    _arrow_right,
    _bullets,
    _card,
    _fill_solid,
    _flow_box,
    _pill,
    _set_run,
    _textbox,
)

OUT = Path(__file__).resolve().parent / "ISBM_Java_Code_Generation_Slides.pptx"


def slide_title(prs):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    _add_bg(slide)

    panel = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, Inches(0.18), H)
    _fill_solid(panel, ACCENT)

    _textbox(slide, Inches(0.9), Inches(1.55), Inches(11), Inches(0.4),
             "ISBM CONFERENCE", size=14, bold=True, color=ACCENT)
    _textbox(slide, Inches(0.9), Inches(2.05), Inches(11.2), Inches(1.55),
             "Improving Java Code Generation\nwith Parameter-Efficient Fine-Tuning",
             size=34, bold=True, color=WHITE)
    _textbox(slide, Inches(0.9), Inches(3.85), Inches(11), Inches(0.5),
             "QLoRA  •  Bayesian Hyperparameter Optimization  •  Controlled Repair Pipeline",
             size=16, color=MUTED)
    _textbox(slide, Inches(0.9), Inches(4.55), Inches(11), Inches(0.45),
             "Pham Quynh Tran", size=18, bold=True, color=SOFT)
    _textbox(slide, Inches(0.9), Inches(5.0), Inches(11), Inches(0.4),
             "Ho Chi Minh City University of Technology, VNU-HCM", size=14, color=MUTED)

    _pill(slide, Inches(0.9), Inches(5.75), Inches(2.4), Inches(0.38), "CodeLlama-7B", fill=CARD_ALT, text_color=SOFT, size=11)
    _pill(slide, Inches(3.5), Inches(5.75), Inches(2.8), Inches(0.38), "Qwen2.5-Coder-32B", fill=CARD_ALT, text_color=SOFT, size=11)
    _pill(slide, Inches(6.5), Inches(5.75), Inches(2.4), Inches(0.38), "HumanEval-Java", fill=CARD_ALT, text_color=SOFT, size=11)


def slide_motivation(prs):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    _add_bg(slide)
    _textbox(slide, Inches(0.7), Inches(0.35), Inches(10), Inches(0.5),
             "01  —  Motivation", size=26, bold=True)

    _card(slide, Inches(0.7), Inches(1.15), Inches(5.8), Inches(5.6))
    _textbox(slide, Inches(1.0), Inches(1.4), Inches(5.2), Inches(0.4),
             "Problem", size=18, bold=True, color=ACCENT)
    _bullets(slide, Inches(1.0), Inches(2.0), Inches(5.2), Inches(4.5),
             [
                 "General-purpose LLMs underperform on Java-specific code generation",
                 "Full fine-tuning is costly and impractical on consumer GPUs",
                 "Hyperparameters (LoRA rank, LR, dropout) strongly affect code quality",
                 "Benchmark: MultiPL-E Java — CodeLlama-7B baseline ≈ 29.2% Pass@1",
             ], size=15, spacing=14)

    _card(slide, Inches(6.8), Inches(1.15), Inches(5.8), Inches(5.6), fill=CARD_ALT)
    _textbox(slide, Inches(7.1), Inches(1.4), Inches(5.2), Inches(0.4),
             "Research Goals", size=18, bold=True, color=ACCENT)
    _bullets(slide, Inches(7.1), Inches(2.0), Inches(5.2), Inches(4.5),
             [
                 "Specialize LLMs for Java via PEFT (QLoRA)",
                 "Automate hyperparameter search with Bayesian Optimization",
                 "Build a reproducible train → merge → evaluate pipeline",
                 "Improve compile-time correctness with prompt & repair strategies",
             ], size=15, spacing=14)


def slide_related_work(prs):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    _add_bg(slide)
    _textbox(slide, Inches(0.7), Inches(0.35), Inches(11), Inches(0.5),
             "02  —  Related Work", size=26, bold=True)
    _textbox(slide, Inches(0.7), Inches(0.9), Inches(11), Inches(0.35),
             "PEFT for code LLMs — prior work focuses on Python / C++, not Java", size=14, color=MUTED)

    rows = [
        ("Li et al. (2024–25)", "Secure code generation", "C/C++", "LoRA & IA³ beat full FT"),
        ("Weyssow et al. (2024)", "PEFT survey for code", "Python", "LoRA / QLoRA most effective"),
        ("This work", "Java completion + HPO", "Java", "QLoRA + BO + repair pipeline"),
    ]
    col_w = [Inches(2.6), Inches(3.2), Inches(1.5), Inches(4.0)]
    headers = ["Study", "Focus", "Lang.", "Key finding"]
    x0 = Inches(0.7)
    y_hdr = Inches(1.35)
    hdr_h = Inches(0.55)

    x = x0
    for title, w in zip(headers, col_w):
        shp = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, x, y_hdr, w, hdr_h)
        _fill_solid(shp, ACCENT_DIM)
        tf = shp.text_frame
        p = tf.paragraphs[0]
        p.alignment = PP_ALIGN.CENTER
        run = p.add_run()
        run.text = title
        _set_run(run, size=12, bold=True, color=WHITE)
        x += w

    y = Inches(2.0)
    row_h = Inches(1.05)
    for i, row in enumerate(rows):
        fill = CARD_ALT if i == len(rows) - 1 else CARD
        x = x0
        for j, (cell, w) in enumerate(zip(row, col_w)):
            shp = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, x, y, w, row_h)
            _fill_solid(shp, fill if j > 0 else (ACCENT if i == len(rows) - 1 else STEP_FILL))
            tf = shp.text_frame
            tf.word_wrap = True
            p = tf.paragraphs[0]
            p.alignment = PP_ALIGN.CENTER if j == 2 else PP_ALIGN.LEFT
            run = p.add_run()
            run.text = f"  {cell}" if j != 2 else cell
            color = BG if j == 0 and i == len(rows) - 1 else WHITE
            _set_run(run, size=12, bold=(i == len(rows) - 1), color=color)
            x += w
        y += row_h + Inches(0.08)

    _card(slide, Inches(0.7), Inches(5.35), Inches(11.9), Inches(1.5))
    _textbox(slide, Inches(1.0), Inches(5.55), Inches(11), Inches(0.35),
             "Gap", size=14, bold=True, color=ACCENT)
    _textbox(slide, Inches(1.0), Inches(5.95), Inches(11.3), Inches(0.75),
             "Limited evidence on Java-specific PEFT pipelines with automated hyperparameter tuning "
             "and post-generation compile repair on HumanEval-Java.",
             size=14, color=SOFT)


def slide_architecture(prs):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    _add_bg(slide)
    _textbox(slide, Inches(0.7), Inches(0.35), Inches(11), Inches(0.5),
             "03  —  Proposed Pipeline", size=26, bold=True)

    nodes = [
        ("Input\nPrompt", "Problem description"),
        ("Pre-\nprocessing", "Template & tokenize"),
        ("LLM +\nQLoRA", "Code generation"),
        ("Post-\nprocessing", "Clean & format"),
        ("Evaluate", "javac + unit tests"),
    ]
    box_w, box_h = Inches(2.0), Inches(1.35)
    y = Inches(1.85)
    start = Inches(0.55)
    gap = Inches(0.42)
    fills = [STEP_FILL, ACCENT_DIM, STEP_FILL_2, STEP_FILL, ACCENT]

    for i, ((t, s), fill) in enumerate(zip(nodes, fills)):
        x = start + i * (box_w + gap)
        _flow_box(slide, x, y, box_w, box_h, t, s, fill=fill, title_size=13)
        if i < len(nodes) - 1:
            _arrow_right(slide, x + box_w + Inches(0.05), y + Inches(0.55), Inches(0.32), Inches(0.24))

    _card(slide, Inches(0.7), Inches(3.65), Inches(5.5), Inches(3.2))
    _textbox(slide, Inches(1.0), Inches(3.9), Inches(5), Inches(0.35),
             "Design choices", size=16, bold=True, color=ACCENT)
    _bullets(slide, Inches(1.0), Inches(4.35), Inches(5.0), Inches(2.3),
             [
                 "Base models: CodeLlama-7B → Qwen2.5-Coder-32B",
                 "PEFT: QLoRA (4-bit NF4 + LoRA adapters)",
                 "Swappable LLM module — fair cross-model comparison",
                 "Merge LoRA weights for efficient inference",
             ], size=14, spacing=10)

    _card(slide, Inches(6.5), Inches(3.65), Inches(6.1), Inches(3.2), fill=CARD_ALT)
    _textbox(slide, Inches(6.8), Inches(3.9), Inches(5.5), Inches(0.35),
             "7-step automation (run_pipeline.py)", size=16, bold=True, color=ACCENT)
    _bullets(slide, Inches(6.8), Inches(4.35), Inches(5.5), Inches(2.3),
             [
                 "Build dataset → EDA & filter → Bayesian Opt",
                 "Full QLoRA train → merge LoRA → inference",
                 "Pass@1 on HumanEval-Java",
             ], size=14, spacing=10)


def slide_method_peft(prs):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    _add_bg(slide)
    _textbox(slide, Inches(0.7), Inches(0.35), Inches(11), Inches(0.5),
             "04  —  Method: QLoRA & Bayesian Optimization", size=26, bold=True)

    cols = [
        ("QLoRA", [
            "Freeze 4-bit quantized base weights",
            "Train low-rank adapters: W = W₀ + BA",
            "Target: q/k/v/o projections + MLP",
            "NF4 quantization + paged optimizers",
            "Trains on RTX 3060 12GB / H100 80GB",
        ], STEP_FILL),
        ("Bayesian Optimization", [
            "Optuna TPE over LoRA hyperparams",
            "Search: rank r, alpha, dropout, LR",
            "Proxy trials on ~10% data, 1 epoch",
            "Objective: minimize eval loss",
            "Export best_params.json → full train",
        ], ACCENT_DIM),
        ("Training format", [
            "Evol-Instruct-Code (~13k Java samples)",
            "Completion: prefix (masked) + target",
            "Class Problem wrapper (~80% samples)",
            "Decontaminate vs HumanEval-Java",
            "NEFTune + cosine LR schedule",
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
        p = tf.paragraphs[0]
        p.alignment = PP_ALIGN.CENTER
        run = p.add_run()
        run.text = title
        _set_run(run, size=15, bold=True, color=WHITE)
        _bullets(slide, x + Inches(0.3), Inches(2.2), cw - Inches(0.5), Inches(4.0),
                 items, size=13, spacing=12)


def slide_evaluation(prs):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    _add_bg(slide)
    _textbox(slide, Inches(0.7), Inches(0.35), Inches(11), Inches(0.5),
             "05  —  Evaluation", size=26, bold=True)

    _card(slide, Inches(0.7), Inches(1.15), Inches(5.8), Inches(2.5))
    _textbox(slide, Inches(1.0), Inches(1.4), Inches(5.2), Inches(0.35),
             "Benchmark: HumanEval-Java (MultiPL-E)", size=16, bold=True, color=ACCENT)
    _bullets(slide, Inches(1.0), Inches(1.9), Inches(5.2), Inches(1.6),
             [
                 "161 Java programming problems with unit tests",
                 "Metric: Pass@1 (greedy decoding)",
                 "Aligned with BigCode Models Leaderboard",
             ], size=14, spacing=10)

    _card(slide, Inches(6.8), Inches(1.15), Inches(5.8), Inches(2.5), fill=CARD_ALT)
    _textbox(slide, Inches(7.1), Inches(1.4), Inches(5.2), Inches(0.35),
             "Evaluation pipeline", size=16, bold=True, color=ACCENT)
    _bullets(slide, Inches(7.1), Inches(1.9), Inches(5.2), Inches(1.6),
             [
                 "Generate Java method → post-process output",
                 "Compile with javac → run unit tests",
                 "Classify: PASSED / COMPILE_ERROR / FAILED",
             ], size=14, spacing=10)

    _card(slide, Inches(0.7), Inches(4.0), Inches(11.9), Inches(2.75))
    _textbox(slide, Inches(1.0), Inches(4.25), Inches(11), Inches(0.35),
             "Error analysis drives improvements", size=16, bold=True, color=ACCENT)
    _bullets(slide, Inches(1.0), Inches(4.75), Inches(11.3), Inches(1.8),
             [
                 "Group 1 — COMPILE_ERROR: syntax / structure issues → prompt directive + targeted repair pass",
                 "Group 2 — FAILED / TIMEOUT: logic & reasoning gaps → requires stronger base model or richer training data",
             ], size=14, spacing=12)


def slide_results_codellama(prs):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    _add_bg(slide)
    _textbox(slide, Inches(0.7), Inches(0.35), Inches(11), Inches(0.5),
             "06  —  Results: CodeLlama-7B + QLoRA", size=26, bold=True)
    _textbox(slide, Inches(0.7), Inches(0.9), Inches(11), Inches(0.35),
             "Hardware: RTX 3060 12GB  •  32 GB RAM", size=14, color=MUTED)

    rows = [
        ("CodeLlama-7B (baseline)", "29.20%"),
        ("CodeLlama-13B", "32.23%"),
        ("CodeLlama-34B", "40.19%"),
        ("CodeLlama-7B + QLoRA (ours)", "41.14%"),
        ("CodeLlama-34B-Instruct", "41.53%"),
        ("CodeLlama-70B", "44.72%"),
    ]
    _card(slide, Inches(0.7), Inches(1.4), Inches(6.2), Inches(4.8))
    _textbox(slide, Inches(1.0), Inches(1.65), Inches(5.5), Inches(0.35),
             "Pass@1 on HumanEval-Java", size=16, bold=True, color=ACCENT)

    y = Inches(2.2)
    for i, (model, score) in enumerate(rows):
        highlight = i == 3
        fill = ACCENT if highlight else (CARD_ALT if i % 2 else CARD)
        shp = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(1.0), y, Inches(5.6), Inches(0.62))
        _fill_solid(shp, fill)
        shp.adjustments[0] = 0.12
        _textbox(slide, Inches(1.25), y + Inches(0.12), Inches(3.8), Inches(0.4),
                 model, size=13, bold=highlight, color=BG if highlight else WHITE)
        _textbox(slide, Inches(4.9), y + Inches(0.12), Inches(1.5), Inches(0.4),
                 score, size=14, bold=True, color=BG if highlight else ACCENT, align=PP_ALIGN.RIGHT)
        y += Inches(0.72)

    _card(slide, Inches(7.2), Inches(1.4), Inches(5.4), Inches(4.8), fill=CARD_ALT)
    _textbox(slide, Inches(7.5), Inches(1.65), Inches(4.8), Inches(0.35),
             "Key takeaway", size=16, bold=True, color=ACCENT)
    _bullets(slide, Inches(7.5), Inches(2.2), Inches(4.8), Inches(3.5),
             [
                 "+11.9 pp over 7B baseline",
                 "7B fine-tuned model matches 34B-Instruct",
                 "PEFT on consumer GPU beats much larger frozen models",
                 "Training loss ↓ without overfitting signs",
             ], size=14, spacing=14)

    _pill(slide, Inches(0.7), Inches(6.45), Inches(3.2), Inches(0.38),
          "Target exceeded: 29.2% → 41.14%", fill=ACCENT_DIM, text_color=WHITE, size=11)


def slide_results_qwen(prs):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    _add_bg(slide)
    _textbox(slide, Inches(0.7), Inches(0.35), Inches(11), Inches(0.5),
             "07  —  Scaling Up: Qwen2.5-Coder-32B", size=26, bold=True)
    _textbox(slide, Inches(0.7), Inches(0.9), Inches(11), Inches(0.35),
             "Hardware: H100 80GB  •  Same pipeline, stronger base model", size=14, color=MUTED)

    rows = [
        ("Qwen2.5-Coder-32B (base)", "65.49%"),
        ("+ QLoRA SFT", "68.35%"),
        ("+ Prompt directive", "70.89%"),
        ("Qwen2.5-Coder-32B-Instruct", "73.69%"),
        ("+ QLoRA + directive + repair pass", "80.38%"),
    ]
    _card(slide, Inches(0.7), Inches(1.35), Inches(7.0), Inches(4.2))
    _textbox(slide, Inches(1.0), Inches(1.6), Inches(6.4), Inches(0.35),
             "Incremental improvements", size=16, bold=True, color=ACCENT)

    y = Inches(2.15)
    for i, (pipe, score) in enumerate(rows):
        highlight = i == len(rows) - 1
        fill = ACCENT if highlight else (CARD_ALT if i % 2 else CARD)
        shp = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(1.0), y, Inches(6.4), Inches(0.58))
        _fill_solid(shp, fill)
        shp.adjustments[0] = 0.12
        _textbox(slide, Inches(1.2), y + Inches(0.1), Inches(4.5), Inches(0.4),
                 pipe, size=12, bold=highlight, color=BG if highlight else SOFT)
        _textbox(slide, Inches(5.5), y + Inches(0.1), Inches(1.7), Inches(0.4),
                 score, size=14, bold=True, color=BG if highlight else ACCENT, align=PP_ALIGN.RIGHT)
        y += Inches(0.68)

    _card(slide, Inches(7.95), Inches(1.35), Inches(4.65), Inches(4.2), fill=CARD_ALT)
    _textbox(slide, Inches(8.2), Inches(1.6), Inches(4.1), Inches(0.35),
             "Improvement strategies", size=15, bold=True, color=ACCENT)
    _bullets(slide, Inches(8.2), Inches(2.05), Inches(4.1), Inches(3.2),
             [
                 "Prompt directive: enforce complete Java methods",
                 "Stop sequences to trim unwanted tokens",
                 "Repair pass: fix COMPILE_ERROR only",
                 "No retraining needed for directive",
             ], size=13, spacing=12)

    _card(slide, Inches(0.7), Inches(5.75), Inches(11.9), Inches(1.15))
    _textbox(slide, Inches(1.0), Inches(6.0), Inches(11.3), Inches(0.65),
             "Repair pass yields the largest gain (+9.5 pp) by resolving missing symbols and helper methods — "
             "targeting failure modes beats blind retraining.",
             size=14, color=SOFT)


def slide_prototype(prs):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    _add_bg(slide)
    _textbox(slide, Inches(0.7), Inches(0.35), Inches(11), Inches(0.5),
             "08  —  Demo Prototype", size=26, bold=True)
    _textbox(slide, Inches(0.7), Inches(0.9), Inches(11), Inches(0.35),
             "Flask web app — CodeLlama-7B + LoRA adapter", size=14, color=MUTED)

    nodes = [
        ("User\nInput", "NL / signature"),
        ("Prompt\nEnhancer", "Java template"),
        ("Model\nGenerate", "Stream tokens"),
        ("javac\nVerify", "Compile check"),
        ("Run /\nDownload", "Test + export"),
    ]
    box_w, box_h = Inches(2.0), Inches(1.25)
    y = Inches(1.75)
    start = Inches(0.55)
    gap = Inches(0.42)
    fills = [STEP_FILL, ACCENT_DIM, STEP_FILL_2, STEP_FILL, ACCENT]

    for i, ((t, s), fill) in enumerate(zip(nodes, fills)):
        x = start + i * (box_w + gap)
        _flow_box(slide, x, y, box_w, box_h, t, s, fill=fill, title_size=12)
        if i < len(nodes) - 1:
            _arrow_right(slide, x + box_w + Inches(0.05), y + Inches(0.5), Inches(0.32), Inches(0.22))

    _card(slide, Inches(0.7), Inches(3.35), Inches(11.9), Inches(3.0))
    _textbox(slide, Inches(1.0), Inches(3.6), Inches(11), Inches(0.35),
             "Bridging research and practice", size=16, bold=True, color=ACCENT)
    feats = [
        ("3 input modes", "Natural language, function signature, full Java class"),
        ("Prompt rules", "Javadoc, class Problem wrapper, chain-of-thought hints"),
        ("Real verification", "javac compile + sandboxed java execution"),
        ("Demo UX", "Sample presets, input suggestions, download Problem.java"),
    ]
    for i, (h, d) in enumerate(feats):
        col = i % 2
        row = i // 2
        lx = Inches(1.0) + col * Inches(5.8)
        ly = Inches(4.15) + row * Inches(0.85)
        _textbox(slide, lx, ly, Inches(5.4), Inches(0.3), h, size=14, bold=True, color=WHITE)
        _textbox(slide, lx, ly + Inches(0.3), Inches(5.4), Inches(0.35), d, size=12, color=MUTED)


def slide_conclusion(prs):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    _add_bg(slide)
    _textbox(slide, Inches(0.7), Inches(0.35), Inches(11), Inches(0.5),
             "09  —  Conclusion & Future Work", size=26, bold=True)

    items = [
        ("Contributions", "End-to-end Java PEFT pipeline with Bayesian hyperparameter search and compile-aware repair"),
        ("Results", "41.14% Pass@1 (7B on RTX 3060) → 80.38% (32B + QLoRA + repair on H100)"),
        ("Insight", "Failure-mode analysis + prompt engineering can match gains from larger models"),
        ("Future", "Automated HPO at scale • Java-specific datasets • multi-benchmark evaluation"),
    ]

    for i, (k, v) in enumerate(items):
        y = Inches(1.15) + i * Inches(1.15)
        _card(slide, Inches(0.7), y, Inches(11.9), Inches(1.0))
        num = slide.shapes.add_shape(MSO_SHAPE.OVAL, Inches(1.0), y + Inches(0.25), Inches(0.5), Inches(0.5))
        _fill_solid(num, ACCENT if i == 1 else STEP_FILL)
        tf = num.text_frame
        p = tf.paragraphs[0]
        p.alignment = PP_ALIGN.CENTER
        run = p.add_run()
        run.text = str(i + 1)
        _set_run(run, size=14, bold=True, color=BG)
        _textbox(slide, Inches(1.8), y + Inches(0.18), Inches(2.4), Inches(0.35), k, size=15, bold=True, color=ACCENT)
        _textbox(slide, Inches(4.2), y + Inches(0.18), Inches(8.0), Inches(0.65), v, size=14, color=SOFT)


def slide_thanks(prs):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    _add_bg(slide)

    panel = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, Inches(0.18), H)
    _fill_solid(panel, ACCENT)

    _textbox(slide, Inches(0.9), Inches(2.6), Inches(11), Inches(0.9),
             "Thank You", size=44, bold=True, color=WHITE, align=PP_ALIGN.LEFT)
    _textbox(slide, Inches(0.9), Inches(3.65), Inches(11), Inches(0.5),
             "Questions & Discussion", size=22, color=ACCENT)
    _textbox(slide, Inches(0.9), Inches(4.5), Inches(11), Inches(0.4),
             "Pham Quynh Tran  •  HCMUT, VNU-HCM", size=16, color=MUTED)


def slide_outline(prs):
    """Optional agenda slide — helps audience follow a 10-min talk."""
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    _add_bg(slide)
    _textbox(slide, Inches(0.7), Inches(0.35), Inches(10), Inches(0.5),
             "Outline", size=26, bold=True)

    sections = [
        ("01", "Motivation"),
        ("02", "Related Work"),
        ("03", "Proposed Pipeline"),
        ("04", "QLoRA & Bayesian Optimization"),
        ("05", "Evaluation"),
        ("06", "Results — CodeLlama-7B"),
        ("07", "Results — Qwen2.5-Coder-32B"),
        ("08", "Demo Prototype"),
        ("09", "Conclusion"),
    ]
    for i, (num, title) in enumerate(sections):
        col = i % 3
        row = i // 3
        x = Inches(0.7) + col * Inches(4.1)
        y = Inches(1.2) + row * Inches(1.55)
        _card(slide, x, y, Inches(3.75), Inches(1.25))
        _textbox(slide, x + Inches(0.25), y + Inches(0.22), Inches(0.6), Inches(0.4),
                 num, size=18, bold=True, color=ACCENT)
        _textbox(slide, x + Inches(0.85), y + Inches(0.28), Inches(2.7), Inches(0.5),
                 title, size=15, color=SOFT)


def main():
    prs = Presentation()
    prs.slide_width = W
    prs.slide_height = H

    slide_title(prs)
    slide_outline(prs)
    slide_motivation(prs)
    slide_related_work(prs)
    slide_architecture(prs)
    slide_method_peft(prs)
    slide_evaluation(prs)
    slide_results_codellama(prs)
    slide_results_qwen(prs)
    slide_prototype(prs)
    slide_conclusion(prs)
    slide_thanks(prs)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    prs.save(str(OUT))
    print(f"Saved: {OUT}  ({len(prs.slides)} slides)")


if __name__ == "__main__":
    main()
