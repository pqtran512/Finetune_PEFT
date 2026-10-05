"""Lightweight EN/VI message catalog for the prototype API."""

DEFAULT_LANG = "en"
SUPPORTED_LANGS = ("en", "vi")

MESSAGES = {
    "en": {
        "status.unloaded": "Unloaded",
        "status.loading": "Loading model...",
        "status.ready": "Ready",
        "status.ready_mock": "Ready (Mock Mode)",
        "status.error": "Error loading model: {error}",
        "model.base": "CodeLlama-7B (base)",
        "model.finetuned": "CodeLlama-7B (fine-tuned)",
        "error.unknown_model": "Unknown model: {model}",
        "error.lora_missing": "Fine-tuned adapter is not available on this machine.",
        "gen.model_selected": "Selected model: {model}.",
        "error.empty_prompt": "Prompt must not be empty.",
        "error.empty_code": "Source code must not be empty.",
        "error.no_static_method": "No 'public static' method found in the source code to execute.",
        "error.model_not_loaded": "Model has not been loaded.",
        "error.generate_failed": "System error during code generation: {error}",
        "error.run_timeout": "Execution error: Timed out (Timeout 10s).",
        "error.run_system": "System error while running code: {error}",
        "gen.init": "> Initializing code generation request",
        "gen.params": "Loading inference configuration parameters",
        "gen.enhance_start": "> Applying Prompt Engineering rules",
        "gen.input_type_nl": "Natural Language",
        "gen.input_type_method": "Method signature",
        "gen.input_type_full": "Full Java class",
        "gen.enhance_info": "Detected input format: {input_type}.",
        "gen.enhance_rules": "Successfully applied optimization rules: {rules}.",
        "gen.model_start": "> Connecting to AI server",
        "gen.inference_start": "> Code generation inference progress",
        "gen.progress_1": "[10%] Decomposing the problem and generating tokens...",
        "gen.progress_2": "[35%] Analyzing algorithm logic...",
        "gen.progress_3": "[60%] Preliminary Java syntax check...",
        "gen.progress_4": "[85%] Formatting the complete class structure...",
        "gen.progress_5": "[95%] Streaming code tokens into output buffer...",
        "gen.inference_wait": "Running model inference... Please wait.",
        "gen.compile_start": "> Compile & automatic verification",
        "gen.compile_info": "Compiling Problem.java with javac...",
        "gen.compile_success": "Trial compilation of Problem.java... SUCCESS.",
        "gen.build_ok": "BUILD SUCCESSFUL in {time}",
        "gen.compile_warn": "Compilation warning: Syntax errors in generated source!\n{summary}",
        "gen.build_warn": "BUILD SUCCESSFUL (with syntax warnings)",
        "gen.more_errors": "... (and {n} more error lines)",
    },
    "vi": {
        "status.unloaded": "Chưa tải",
        "status.loading": "Đang tải mô hình...",
        "status.ready": "Sẵn sàng",
        "status.ready_mock": "Sẵn sàng (Mock Mode)",
        "status.error": "Lỗi khi tải mô hình: {error}",
        "model.base": "CodeLlama-7B (gốc)",
        "model.finetuned": "CodeLlama-7B (đã fine-tune)",
        "error.unknown_model": "Không nhận diện được mô hình: {model}",
        "error.lora_missing": "Adapter đã fine-tune không có trên máy này.",
        "gen.model_selected": "Mô hình đang dùng: {model}.",
        "error.empty_prompt": "Prompt không được để trống.",
        "error.empty_code": "Mã nguồn không được để trống.",
        "error.no_static_method": "Không tìm thấy hàm 'public static' nào trong mã nguồn để thực thi.",
        "error.model_not_loaded": "Mô hình chưa được tải lên hệ thống.",
        "error.generate_failed": "Lỗi hệ thống trong quá trình sinh mã: {error}",
        "error.run_timeout": "Lỗi thực thi: Quá thời gian quy định (Timeout 10s).",
        "error.run_system": "Lỗi hệ thống khi chạy thử mã: {error}",
        "gen.init": "> Khởi tạo yêu cầu sinh mã",
        "gen.params": "Đang nạp tham số cấu hình suy luận",
        "gen.enhance_start": "> Áp dụng quy tắc Prompt Engineering",
        "gen.input_type_nl": "Ngôn ngữ tự nhiên (Natural Language)",
        "gen.input_type_method": "Chữ ký hàm (Method signature)",
        "gen.input_type_full": "Mã nguồn Java đầy đủ (Full Java class)",
        "gen.enhance_info": "Định dạng đầu vào phát hiện: {input_type}.",
        "gen.enhance_rules": "Đã áp dụng thành công các quy tắc tối ưu hóa: {rules}.",
        "gen.model_start": "> Kết nối máy chủ AI",
        "gen.inference_start": "> Tiến trình suy luận sinh mã",
        "gen.progress_1": "[10%] Bắt đầu phân rã bài toán và sinh chuỗi token...",
        "gen.progress_2": "[35%] Phân tích logic giải thuật bài toán...",
        "gen.progress_3": "[60%] Rà soát lỗi cú pháp Java sơ bộ...",
        "gen.progress_4": "[85%] Định dạng hoàn chỉnh cấu trúc class...",
        "gen.progress_5": "[95%] Streaming code tokens into output buffer...",
        "gen.inference_wait": "Đang thực hiện suy luận trên mô hình... Vui lòng đợi.",
        "gen.compile_start": "> Biên dịch & Kiểm thử tự động",
        "gen.compile_info": "Đang tiến hành biên dịch thử file Problem.java bằng javac...",
        "gen.compile_success": "Biên dịch thử nghiệm Problem.java... THÀNH CÔNG.",
        "gen.build_ok": "BUILD SUCCESSFUL trong {time}",
        "gen.compile_warn": "Cảnh báo biên dịch: Có lỗi cú pháp trong mã nguồn sinh ra!\n{summary}",
        "gen.build_warn": "BUILD SUCCESSFUL (với cảnh báo lỗi cú pháp)",
        "gen.more_errors": "... (và {n} dòng lỗi khác)",
    },
}


def normalize_lang(lang) -> str:
    if not lang:
        return DEFAULT_LANG
    lang = str(lang).strip().lower()
    if lang.startswith("vi"):
        return "vi"
    if lang.startswith("en"):
        return "en"
    return DEFAULT_LANG if lang not in SUPPORTED_LANGS else lang


def t(lang: str, key: str, **kwargs) -> str:
    lang = normalize_lang(lang)
    catalog = MESSAGES.get(lang) or MESSAGES[DEFAULT_LANG]
    template = catalog.get(key) or MESSAGES[DEFAULT_LANG].get(key) or key
    if kwargs:
        try:
            return template.format(**kwargs)
        except (KeyError, ValueError):
            return template
    return template


def resolve_lang(data=None) -> str:
    """Resolve language from JSON body, query string, or Accept-Language."""
    if data and isinstance(data, dict) and data.get("lang"):
        return normalize_lang(data.get("lang"))
    from flask import request

    q = request.args.get("lang")
    if q:
        return normalize_lang(q)
    header = request.headers.get("Accept-Language", "")
    if header:
        primary = header.split(",")[0].strip()
        return normalize_lang(primary)
    return DEFAULT_LANG
