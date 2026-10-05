/**
 * AJAC prototype i18n — English (default) + Vietnamese.
 * Usage: t('key'), setLang('vi'|'en'), applyI18n(root)
 */
(function (global) {
    const STORAGE_KEY = "ajac_lang";
    const DEFAULT_LANG = "en";

    const TRANSLATIONS = {
        en: {
            "badge.connecting": "Connecting...",
            "badge.status": "Model status: {status}",
            "badge.offline": "Cannot connect to server",
            "panel.config": "Configuration & Request",
            "label.model": "Model:",
            "model.base": "CodeLlama-7B (base)",
            "model.finetuned": "CodeLlama-7B (fine-tuned)",
            "model.switch": "Switch model",
            "model.unavailable": "Fine-tuned adapter is not available on this machine.",
            "alert.model_switch_failed": "Could not switch model: {error}",
            "label.preset": "Sample problems:",
            "preset.nl_prime": "Prime Number",
            "preset.factorial": "Factorial",
            "preset.remove_duplicates": "Remove Duplicates",
            "preset.right_triangle": "Right Triangle",
            "label.prompt": "Requirement description:",
            "prompt.placeholder": "Enter a natural-language description or paste a Java method declaration here...",
            "btn.generate": "Generate Java Code",
            "btn.copy": "Copy",
            "btn.download": "Download",
            "btn.copied": "Copied!",
            "btn.preparing_preview": "Preparing preview...",
            "btn.downloaded": "Downloaded {filename}!",
            "console.title": "Inference Log",
            "empty.title": "Completed Java code will appear here",
            "empty.subtext": 'Choose or enter a request on the left, then click "Generate Java Code".',
            "runner.title": "RUN",
            "btn.run": "Run function",
            "btn.running": "Running...",
            "label.arguments": "Arguments:",
            "label.output": "Output:",
            "label.params": "Parameters:",
            "helper.default": "Example: (1, 2)",
            "args.placeholder": "Enter function arguments, e.g. 1, 2 or (1, 2)",
            "args.placeholder_alt": "Enter arguments, e.g. 1, 2 or [1, 2, 3]",
            "console.placeholder": 'Enter arguments above and click "Run function" to see the result.',
            "console.compiling": "Compiling and executing Java code...",
            "modal.preview": "Preview",
            "modal.close": "Close (Esc)",
            "modal.copy": "Copy code",
            "modal.download": "Download .java file",
            "alert.empty_prompt": "Please enter a prompt or method declaration to generate code!",
            "alert.server_error": "Server error: {status}",
            "alert.no_result": "Did not receive a complete generation result from the server.",
            "alert.generate_failed": "Code generation failed: {error}",
            "alert.copy_failed": "Failed to copy code.",
            "alert.no_code_download": "No source code available to download.",
            "alert.preview_failed": "Could not create a preview of the code.",
            "alert.copy_source_failed": "Could not copy source code.",
            "alert.model_load_error": "Model load error: {status}",
            "alert.unknown_error": "Unknown system error",
            "log.conn_error": "> Connection / System error",
            "log.generate_failed": "Code generation failed: {error}",
            "stats.time": "Time: {time}",
            "run.success": "Run succeeded! Output:\n\n{output}",
            "run.err_compile": "COMPILATION ERROR:",
            "run.err_runtime": "RUNTIME EXCEPTION:",
            "run.err_timeout": "TIMEOUT ERROR:",
            "run.err_system": "SYSTEM ERROR:",
            "run.err_api": "API CONNECTION ERROR:\n\n{error}",
            "suggest.no_params": "Function has no parameters",
            "suggest.no_args": "No arguments needed",
            "suggest.example": "Example: {value}",
            "helper.prime": "Example: 7",
            "helper.prime_ph": "Example: 7 or 10",
            "helper.factorial": "Example: 5",
            "helper.remove_dup": "Example: [1, 2, 3, 2, 4]",
            "helper.triangle": "Example: 3, 4, 5",
            "example.nl_prime": "Write a function that checks whether an integer n is a prime number. Return true if it is prime, otherwise return false.",
            "lang.en": "EN",
            "lang.vi": "VI",
            "lang.switch": "Language",
        },
        vi: {
            "badge.connecting": "Đang kết nối...",
            "badge.status": "Trạng thái mô hình: {status}",
            "badge.offline": "Không kết nối được server",
            "panel.config": "Cấu hình & Yêu cầu",
            "label.model": "Mô hình:",
            "model.base": "CodeLlama-7B (gốc)",
            "model.finetuned": "CodeLlama-7B (đã fine-tune)",
            "model.switch": "Chuyển mô hình",
            "model.unavailable": "Adapter đã fine-tune không có trên máy này.",
            "alert.model_switch_failed": "Không chuyển được mô hình: {error}",
            "label.preset": "Chọn bài toán mẫu:",
            "preset.nl_prime": "Số nguyên tố",
            "preset.factorial": "Factorial",
            "preset.remove_duplicates": "Remove Duplicates",
            "preset.right_triangle": "Right Triangle",
            "label.prompt": "Mô tả yêu cầu:",
            "prompt.placeholder": "Nhập mô tả yêu cầu bằng ngôn ngữ tự nhiên hoặc dán khai báo hàm Java vào đây...",
            "btn.generate": "Bắt đầu sinh mã Java",
            "btn.copy": "Sao chép",
            "btn.download": "Tải xuống",
            "btn.copied": "Đã sao chép!",
            "btn.preparing_preview": "Đang tạo bản xem trước...",
            "btn.downloaded": "Đã tải xuống {filename}!",
            "console.title": "Inference Log",
            "empty.title": "Kết quả code Java hoàn thiện sẽ hiển thị tại đây",
            "empty.subtext": 'Chọn hoặc nhập yêu cầu ở cột bên trái và bấm nút "Bắt đầu sinh mã Java".',
            "runner.title": "RUN",
            "btn.run": "Chạy hàm",
            "btn.running": "Đang chạy...",
            "label.arguments": "Arguments:",
            "label.output": "Output:",
            "label.params": "Tham số:",
            "helper.default": "Ví dụ: (1, 2)",
            "args.placeholder": "Nhập các đối số truyền vào hàm, ví dụ: 1, 2 hoặc (1, 2)",
            "args.placeholder_alt": "Nhập các đối số, ví dụ: 1, 2 hoặc [1, 2, 3]",
            "console.placeholder": 'Nhập đối số đầu vào ở trên và nhấn "Chạy hàm" để xem kết quả tính toán.',
            "console.compiling": "Đang tiến hành biên dịch và thực thi mã Java...",
            "modal.preview": "Xem trước",
            "modal.close": "Đóng (Esc)",
            "modal.copy": "Sao chép mã",
            "modal.download": "Tải xuống file .java",
            "alert.empty_prompt": "Vui lòng nhập prompt hoặc khai báo hàm cần sinh mã!",
            "alert.server_error": "Lỗi từ server: {status}",
            "alert.no_result": "Không nhận được kết quả sinh mã hoàn chỉnh từ máy chủ.",
            "alert.generate_failed": "Lỗi khi sinh code: {error}",
            "alert.copy_failed": "Lỗi khi sao chép code.",
            "alert.no_code_download": "Chưa có mã nguồn để tải xuống.",
            "alert.preview_failed": "Không thể tạo bản code xem trước.",
            "alert.copy_source_failed": "Không thể sao chép mã nguồn.",
            "alert.model_load_error": "Lỗi tải mô hình: {status}",
            "alert.unknown_error": "Lỗi hệ thống không xác định",
            "log.conn_error": "> Lỗi kết nối / Hệ thống",
            "log.generate_failed": "Quá trình sinh mã thất bại: {error}",
            "stats.time": "Thời gian: {time}",
            "run.success": "Chạy thành công! Kết quả đầu ra:\n\n{output}",
            "run.err_compile": "LỖI BIÊN DỊCH (Compilation Error):",
            "run.err_runtime": "LỖI KHI CHẠY (Runtime Exception):",
            "run.err_timeout": "QUÁ THỜI GIAN THỰC THI (Timeout Error):",
            "run.err_system": "LỖI HỆ THỐNG:",
            "run.err_api": "LỖI KẾT NỐI API:\n\n{error}",
            "suggest.no_params": "Hàm không có tham số",
            "suggest.no_args": "Không cần đối số",
            "suggest.example": "Ví dụ: {value}",
            "helper.prime": "Ví dụ: 7",
            "helper.prime_ph": "Ví dụ: 7 hoặc 10",
            "helper.factorial": "Ví dụ: 5",
            "helper.remove_dup": "Ví dụ: [1, 2, 3, 2, 4]",
            "helper.triangle": "Ví dụ: 3, 4, 5",
            "example.nl_prime": "Viết hàm kiểm tra một số nguyên n có phải là số nguyên tố hay không. Trả về true nếu là số nguyên tố, ngược lại trả về false.",
            "lang.en": "EN",
            "lang.vi": "VI",
            "lang.switch": "Ngôn ngữ",
        },
    };

    let currentLang = DEFAULT_LANG;

    function normalizeLang(lang) {
        if (!lang) return DEFAULT_LANG;
        lang = String(lang).toLowerCase();
        if (lang.startsWith("vi")) return "vi";
        if (lang.startsWith("en")) return "en";
        return DEFAULT_LANG;
    }

    function loadLang() {
        try {
            const saved = localStorage.getItem(STORAGE_KEY);
            if (saved) return normalizeLang(saved);
        } catch (_) { /* ignore */ }
        return DEFAULT_LANG;
    }

    function t(key, vars) {
        const catalog = TRANSLATIONS[currentLang] || TRANSLATIONS[DEFAULT_LANG];
        let text = catalog[key] || TRANSLATIONS[DEFAULT_LANG][key] || key;
        if (vars && typeof vars === "object") {
            Object.keys(vars).forEach((k) => {
                text = text.replace(new RegExp("\\{" + k + "\\}", "g"), String(vars[k]));
            });
        }
        return text;
    }

    function applyI18n(root) {
        const scope = root || document;
        scope.querySelectorAll("[data-i18n]").forEach((el) => {
            const key = el.getAttribute("data-i18n");
            if (key) el.textContent = t(key);
        });
        scope.querySelectorAll("[data-i18n-placeholder]").forEach((el) => {
            const key = el.getAttribute("data-i18n-placeholder");
            if (key) el.setAttribute("placeholder", t(key));
        });
        scope.querySelectorAll("[data-i18n-title]").forEach((el) => {
            const key = el.getAttribute("data-i18n-title");
            if (key) el.setAttribute("title", t(key));
        });
        document.documentElement.lang = currentLang;
        document.querySelectorAll(".lang-btn").forEach((btn) => {
            btn.classList.toggle("active", btn.getAttribute("data-lang") === currentLang);
        });
    }

    function setLang(lang, options) {
        const next = normalizeLang(lang);
        const prev = currentLang;
        currentLang = next;
        try {
            localStorage.setItem(STORAGE_KEY, next);
        } catch (_) { /* ignore */ }
        applyI18n(document);
        if (options && typeof options.onChange === "function" && prev !== next) {
            options.onChange(next, prev);
        }
        document.dispatchEvent(new CustomEvent("ajac:langchange", { detail: { lang: next, prev } }));
        return next;
    }

    function getLang() {
        return currentLang;
    }

    currentLang = loadLang();

    global.AJAC_I18N = {
        t,
        setLang,
        getLang,
        applyI18n,
        DEFAULT_LANG,
        TRANSLATIONS,
    };
    global.t = t;
})(window);
