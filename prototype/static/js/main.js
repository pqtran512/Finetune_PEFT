// ĐỊNH NGHĨA CÁC PROMPT MẪU (HUMANEVAL-JAVA & NL PROMPTS)
const EXAMPLES = {
    nl_prime: `Viết hàm kiểm tra một số nguyên n có phải là số nguyên tố hay không. Trả về true nếu là số nguyên tố, ngược lại trả về false.`,

    rolling_max: `import java.util.ArrayList;
import java.util.List;

public class Problem {
    /**
     * From a given list of integers, generate a list of rolling maximum element found until given moment
     * in the sequence.
     */
    public static List<Integer> rollingMax(List<Integer> numbers) {`,

    remove_duplicates: `import java.util.ArrayList;
import java.util.List;

public class Problem {
    /**
     * From a list of integers, remove all elements that occur more than once.
     * Keep order of elements left the same as in the original list.
     */
    public static List<Integer> removeDuplicates(List<Integer> numbers) {`,

    sort_even: `import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

public class Problem {
    /**
     * This function takes a list of numbers and returns a list where the values at even indices are sorted,
     * while the values at odd indices remain unchanged.
     */
    public static List<Integer> sortEven(List<Integer> l) {`
};

const LOADING_MESSAGES = [
    "Đang kết nối tới máy chủ mô hình...",
    "Đang phân tích cấu trúc mã yêu cầu...",
    "Đang thực hiện suy luận sinh mã Java...",
    "Mô hình fine-tuned đang thực hiện chuỗi tokens...",
    "Đang định dạng cấu trúc cú pháp code...",
    "Đang kiểm tra và hoàn tất khối mã..."
];

const PRESET_HELPERS = {
    nl_prime: {
        placeholder: "Ví dụ: 7 hoặc 10",
        helper: "Ví dụ: 7",
        default: "7"
    },
    rolling_max: {
        placeholder: "Ví dụ: [1, 2, 4, 3, 5]",
        helper: "Ví dụ: [1, 2, 4, 3, 5]",
        default: "[1, 2, 4, 3, 5]"
    },
    remove_duplicates: {
        placeholder: "Ví dụ: [1, 2, 3, 2, 4]",
        helper: "Ví dụ: [1, 2, 3, 2, 4]",
        default: "[1, 2, 3, 2, 4]"
    },
    sort_even: {
        placeholder: "Ví dụ: [5, 6, 3, 4, 1, 2]",
        helper: "Ví dụ: [5, 6, 3, 4, 1, 2]",
        default: "[5, 6, 3, 4, 1, 2]"
    }
};

document.addEventListener("DOMContentLoaded", () => {
    // Elements
    const badgeDot = document.getElementById("badge-dot");
    const badgeText = document.getElementById("badge-text");
    const infoBase = document.getElementById("info-base");

    // Preset chips
    const presetChips = document.querySelectorAll(".preset-chip");

    // State cho cấu hình sinh mã
    let currentTemperature = 0.2;
    let currentMaxTokens = 512;

    const promptInput = document.getElementById("prompt-input");
    const inputLineNumbers = document.getElementById("input-line-numbers");

    const btnGenerate = document.getElementById("btn-generate");
    const btnCopy = document.getElementById("btn-copy");
    const btnDownload = document.getElementById("btn-download");

    const statsTime = document.getElementById("stats-time");
    const codeContainer = document.getElementById("code-editor");
    const codeOutput = document.getElementById("code-output");
    const emptyState = document.getElementById("empty-state");
    const loadingOverlay = document.getElementById("loading-overlay");
    const loadingMessage = document.getElementById("loading-message");

    // Khởi tạo Ace Editor
    let editor = null;
    let previewEditor = null;
    let currentDownloadFilename = "Problem.java";
    if (window.ace) {
        ace.config.set('basePath', 'https://cdnjs.cloudflare.com/ajax/libs/ace/1.32.7/');
        editor = ace.edit("code-editor");
        editor.setTheme("ace/theme/chrome");
        editor.session.setMode("ace/mode/java");
        editor.setOptions({
            fontSize: "13px",
            fontFamily: "var(--font-code)",
            showPrintMargin: false,
            useSoftTabs: true,
            tabSize: 4,
            readOnly: false
        });

        // Cập nhật currentFullCode khi người dùng chỉnh sửa trong editor
        editor.on("change", () => {
            currentFullCode = editor.getValue();
        });
        if (document.getElementById("preview-code-editor")) {
            previewEditor = ace.edit("preview-code-editor");
            previewEditor.setTheme("ace/theme/chrome");
            previewEditor.session.setMode("ace/mode/java");
            previewEditor.setOptions({
                fontSize: "13px",
                fontFamily: "var(--font-code)",
                showPrintMargin: false,
                useSoftTabs: true,
                tabSize: 4,
                readOnly: false,
                wrap: true
            });
        }
    }

    // Test runner elements
    const testRunnerSection = document.getElementById("test-runner-section");
    const btnRunCode = document.getElementById("btn-run-code");
    const testInputArgs = document.getElementById("test-input-args");
    const testInputHelper = document.getElementById("test-input-helper");
    const consoleOutput = document.getElementById("console-output");
    const runSpinner = document.querySelector(".run-spinner");
    const runBtnText = document.querySelector(".run-btn-text");

    // Parameter hints elements
    const paramHints = document.getElementById("param-hints");
    const paramHintsChips = document.getElementById("param-hints-chips");

    // Modal Preview Elements
    const previewModal = document.getElementById("preview-modal");
    const previewFilename = document.getElementById("preview-filename");
    const btnCloseModal = document.getElementById("btn-close-modal");
    const btnModalCopy = document.getElementById("btn-modal-copy");
    const btnModalDownload = document.getElementById("btn-modal-download");



    // State
    let currentFullCode = "";
    let loadingMessageInterval = null;

    // Cập nhật số thứ tự dòng cho Textarea
    function updateInputLineNumbers() {
        const text = promptInput.value;
        const lines = text.split("\n");
        const count = Math.max(lines.length, 1);

        let html = "";
        for (let i = 1; i <= count; i++) {
            html += `<span>${i}</span>`;
        }
        inputLineNumbers.innerHTML = html;
    }

    // Lắng nghe sự kiện gõ phím để cập nhật dòng số thứ tự
    promptInput.addEventListener("input", () => {
        updateInputLineNumbers();
        // Bỏ active của các thẻ mẫu nếu người dùng tự sửa đổi code
        presetChips.forEach(c => c.classList.remove("active"));

        // Reset gợi ý test input về mặc định
        testInputArgs.placeholder = "Nhập các đối số, ví dụ: 1, 2 hoặc [1, 2, 3]";
        testInputHelper.textContent = "Ví dụ: (1, 2)";
        paramHints.classList.add("hidden");
        paramHintsChips.innerHTML = "";
    });

    promptInput.addEventListener("scroll", () => {
        // Đồng bộ cuộn giữa line numbers và textarea
        inputLineNumbers.scrollTop = promptInput.scrollTop;
    });

    // Khởi tạo số dòng ban đầu
    updateInputLineNumbers();

    // Xử lý click Preset Chips
    presetChips.forEach(chip => {
        chip.addEventListener("click", () => {
            const val = chip.getAttribute("data-value");

            // Xóa active cũ, thêm active mới
            presetChips.forEach(c => c.classList.remove("active"));
            chip.classList.add("active");

            if (val && EXAMPLES[val]) {
                promptInput.value = EXAMPLES[val];
            } else {
                promptInput.value = "";
            }

            updateInputLineNumbers();
            // Cuộn về đầu
            promptInput.scrollTop = 0;
            inputLineNumbers.scrollTop = 0;

            // Cập nhật gợi ý test input và giá trị mặc định cho từng preset bài toán
            if (val && PRESET_HELPERS[val]) {
                testInputArgs.placeholder = PRESET_HELPERS[val].placeholder;
                testInputHelper.textContent = PRESET_HELPERS[val].helper;
                testInputArgs.value = PRESET_HELPERS[val].default;
            } else {
                testInputArgs.placeholder = "Nhập các đối số, ví dụ: 1, 2 hoặc (1, 2)";
                testInputHelper.textContent = "Ví dụ: (1, 2)";
                testInputArgs.value = "";
            }
        });
    });

    // Xử lý Segmented Controls (Độ sáng tạo)
    const creativitySegments = document.querySelectorAll("#creativity-segments .segment-btn");
    creativitySegments.forEach(btn => {
        btn.addEventListener("click", () => {
            creativitySegments.forEach(b => b.classList.remove("active"));
            btn.classList.add("active");

            const val = parseFloat(btn.getAttribute("data-value"));
            currentTemperature = isNaN(val) ? 0.2 : val;
        });
    });

    // Xử lý Segmented Controls (Độ dài tối đa)
    const lengthSegments = document.querySelectorAll("#length-segments .segment-btn");
    lengthSegments.forEach(btn => {
        btn.addEventListener("click", () => {
            lengthSegments.forEach(b => b.classList.remove("active"));
            btn.classList.add("active");

            const val = parseInt(btn.getAttribute("data-value"));
            currentMaxTokens = isNaN(val) ? 512 : val;
        });
    });

    // Hàm render code
    function renderCode() {
        if (editor) {
            editor.setValue(currentFullCode, -1);
        } else if (codeOutput) {
            codeOutput.textContent = currentFullCode;
            Prism.highlightElement(codeOutput);
        }
    }

    // Hàm cập nhật trạng thái mô hình từ API
    let checkInterval = null;
    const modelBadge = document.querySelector(".model-badge");

    function checkModelStatus() {
        fetch("/api/status")
            .then(res => res.json())
            .then(data => {
                if (modelBadge) modelBadge.setAttribute("title", `Trạng thái mô hình: ${data.status}`);
                if (badgeText) badgeText.textContent = "";
                if (infoBase) infoBase.textContent = data.base_model || "-";

                const status = data.status || "";
                const isReady =
                    status.startsWith("Sẵn sàng") ||
                    status.startsWith("Ready") ||
                    status.includes("Mock");
                const isError =
                    status.includes("Lỗi") ||
                    status.includes("Loi") ||
                    status.toLowerCase().startsWith("error");

                if (isReady) {
                    badgeDot.className = "dot online";
                    btnGenerate.disabled = false;
                    clearInterval(checkInterval);
                } else if (isError) {
                    badgeDot.className = "dot offline";

                    btnGenerate.disabled = true;
                    clearInterval(checkInterval);
                    alert("Lỗi tải mô hình: " + status);
                } else {
                    if (badgeDot) badgeDot.className = "dot pending";
                    btnGenerate.disabled = true;
                }
            })
            .catch(err => {
                if (badgeDot) badgeDot.className = "dot offline";
                if (modelBadge) modelBadge.setAttribute("title", "Không kết nối được server");
                btnGenerate.disabled = true;
                console.error("Lỗi kết nối API status:", err);
            });
    }

    // Bắt đầu kiểm tra trạng thái
    checkModelStatus();
    checkInterval = setInterval(checkModelStatus, 3000);

    function startConsoleLog() {
        const logBody = document.getElementById("console-log-body");
        const progressFill = document.getElementById("progress-bar-fill");
        if (logBody) logBody.innerHTML = "";
        if (progressFill) progressFill.style.width = "0%";
    }

    function addLogLine(text, style = "normal") {
        const logBody = document.getElementById("console-log-body");
        if (!logBody) return;
        const line = document.createElement("div");
        line.className = `console-log-line ${style}`;
        
        if (style === "error") {
            line.style.whiteSpace = "pre-wrap";
            line.style.fontFamily = "var(--font-mono)";
        }
        
        line.textContent = text;
        logBody.appendChild(line);
        logBody.scrollTop = logBody.scrollHeight;
    }

    function updateProgressBar(step) {
        const progressFill = document.getElementById("progress-bar-fill");
        if (!progressFill) return;
        
        let percent = 0;
        switch (step) {
            case "init": percent = 5; break;
            case "params": percent = 10; break;
            case "enhance_start": percent = 20; break;
            case "enhance_info": percent = 30; break;
            case "enhance_rules": percent = 45; break;
            case "model_start": percent = 55; break;
            case "model_info": percent = 65; break;
            case "inference_start": percent = 72; break;
            case "inference_progress": percent = 80; break;
            case "inference_progress_1": percent = 75; break;
            case "inference_progress_2": percent = 80; break;
            case "inference_progress_3": percent = 84; break;
            case "inference_progress_4": percent = 88; break;
            case "inference_progress_5": percent = 92; break;
            case "compile_start": percent = 94; break;
            case "compile_info": percent = 96; break;
            case "compile_success": percent = 98; break;
            case "compile_fail": percent = 98; break;
            case "build_finished": percent = 100; break;
            default: percent = 95;
        }
        progressFill.style.width = `${percent}%`;
    }

    // Xử lý sự kiện Sinh code
    btnGenerate.addEventListener("click", async () => {
        const prompt = promptInput.value.trim();
        if (!prompt) {
            alert("Vui lòng nhập prompt hoặc khai báo hàm cần sinh mã!");
            return;
        }

        // Hiện overlay loading IDE Console
        loadingOverlay.classList.remove("hidden");
        btnGenerate.disabled = true;
        startConsoleLog();

        const payload = {
            prompt: prompt,
            temperature: currentTemperature,
            max_new_tokens: currentMaxTokens,
            enable_cot: false, // Loại bỏ quy tắc 14
            enable_language_tag: true, // Luôn áp dụng quy tắc 5
            input_args: testInputArgs ? testInputArgs.value.trim() : ""
        };

        try {
            const response = await fetch("/api/generate", {
                method: "POST",
                headers: {
                    "Content-Type": "application/json"
                },
                body: JSON.stringify(payload)
            });

            if (!response.ok) {
                const data = await response.json().catch(() => ({}));
                throw new Error(data.error || `Lỗi từ server: ${response.status}`);
            }

            const reader = response.body.getReader();
            const decoder = new TextDecoder("utf-8");
            let buffer = "";
            let finalResult = null;

            while (true) {
                const { done, value } = await reader.read();
                if (done) break;

                buffer += decoder.decode(value, { stream: true });
                const lines = buffer.split("\n");
                buffer = lines.pop(); // Giữ lại phần thừa (nếu có) để xử lý trong chunk sau

                for (const line of lines) {
                    const trimmed = line.trim();
                    if (!trimmed) continue;

                    try {
                        const data = JSON.parse(trimmed);
                        if (data.step === "result") {
                            finalResult = data;
                        } else if (data.step === "error") {
                            addLogLine(data.text, "error");
                        } else if (data.text) {
                            addLogLine(data.text, data.type || "normal");
                            updateProgressBar(data.step);
                        }
                    } catch (e) {
                        console.error("Lỗi parse chunk JSON:", e, trimmed);
                    }
                }
            }

            // Chờ 800ms để người dùng xem dòng chữ BUILD SUCCESSFUL/WARNING
            setTimeout(() => {
                loadingOverlay.classList.add("hidden");
                btnGenerate.disabled = false;

                if (finalResult) {
                    // Hiển thị code kết quả
                    emptyState.classList.add("hidden");
                    codeContainer.classList.remove("hidden");

                    // Lấy code hoàn chỉnh từ backend (không bao gồm hàm main kiểm thử trong editor)
                    let fullCode = finalResult.enhanced_prompt 
                        ? (finalResult.enhanced_prompt + finalResult.generated_code) 
                        : (finalResult.full_code || finalResult.generated_code);
                    if (!finalResult.enhanced_prompt) {
                        if (!fullCode.includes("class Problem") && !fullCode.startsWith(prompt)) {
                            fullCode = prompt + fullCode;
                        }
                    }

                    currentFullCode = fullCode;
                    renderCode();

                    statsTime.textContent = `Thời gian: ${finalResult.time_taken}`;
                    statsTime.classList.remove("hidden");
                    btnCopy.disabled = false;
                    btnDownload.disabled = false;

                    testRunnerSection.classList.remove("hidden");
                    btnRunCode.disabled = false;
                    testInputArgs.disabled = false;
                    consoleOutput.innerHTML = '<span class="console-placeholder">Nhập đối số đầu vào ở trên và nhấn "Chạy hàm" để xem kết quả tính toán.</span>';

                    fetchSuggestedInput(currentFullCode);
                } else {
                    alert("Không nhận được kết quả sinh mã hoàn chỉnh từ máy chủ.");
                }
            }, 800);

        } catch (err) {
            addLogLine("> Task: Lỗi kết nối / Hệ thống", "task");
            addLogLine("Quá trình sinh mã thất bại: " + err.message, "error");
            
            setTimeout(() => {
                loadingOverlay.classList.add("hidden");
                btnGenerate.disabled = false;
                alert("Lỗi khi sinh code: " + err.message);
            }, 2000);
        }
    });

    // Xử lý sự kiện Sao chép (Copy)
    btnCopy.addEventListener("click", () => {
        const codeText = editor ? editor.getValue() : (codeOutput ? codeOutput.textContent : currentFullCode);
        if (!codeText) return;

        navigator.clipboard.writeText(codeText)
            .then(() => {
                const originalText = btnCopy.innerHTML;
                btnCopy.innerHTML = `
                    <svg viewBox="0 0 24 24" width="13" height="13" fill="none" stroke="currentColor" stroke-width="2">
                        <polyline points="20 6 9 17 4 12"></polyline>
                    </svg>
                    <span>Đã sao chép!</span>
                `;
                btnCopy.style.borderColor = "var(--color-success)";
                btnCopy.style.background = "var(--color-success-glow)";
                btnCopy.style.color = "var(--color-success)";

                setTimeout(() => {
                    btnCopy.innerHTML = originalText;
                    btnCopy.style.borderColor = "";
                    btnCopy.style.background = "";
                    btnCopy.style.color = "";
                }, 2000);
            })
            .catch(err => {
                console.error("Không thể sao chép code:", err);
                alert("Lỗi khi sao chép code.");
            });
    });

    // Hàm mở modal xem trước bản code đầy đủ dùng để test kết quả
    function openPreviewModal() {
        const codeText = editor ? editor.getValue() : currentFullCode;
        if (!codeText) {
            alert("Chưa có mã nguồn để tải xuống.");
            return;
        }

        const inputArgs = testInputArgs ? testInputArgs.value.trim() : "";

        // Hiển thị trạng thái đang chuẩn bị preview
        btnDownload.disabled = true;
        const origBtnText = btnDownload.innerHTML;
        btnDownload.innerHTML = `
            <span class="loader-spinner" style="width:12px;height:12px;border-width:2px;border-top-color:var(--color-primary);border-color:rgba(79,70,229,0.2);"></span>
            <span>Đang tạo bản xem trước...</span>
        `;

        fetch("/api/preview-full-code", {
            method: "POST",
            headers: {
                "Content-Type": "application/json"
            },
            body: JSON.stringify({
                code: codeText,
                input_args: inputArgs
            })
        })
            .then(res => {
                if (!res.ok) throw new Error("Không thể tạo bản code xem trước.");
                return res.json();
            })
            .then(data => {
                currentDownloadFilename = data.filename || "Problem.java";
                if (previewFilename) previewFilename.textContent = currentDownloadFilename;

                if (previewEditor) {
                    previewEditor.setValue(data.full_code || codeText, -1);
                }

                // Mở modal
                if (previewModal) {
                    previewModal.classList.remove("hidden");
                    document.body.style.overflow = "hidden";
                }

                // Render lại kích thước editor trong modal
                setTimeout(() => {
                    if (previewEditor) {
                        previewEditor.resize();
                        previewEditor.focus();
                    }
                }, 100);
            })
            .catch(err => {
                console.error("Lỗi preview full code:", err);
                // Fallback: Mở modal với code hiện tại
                currentDownloadFilename = "Problem.java";
                if (previewFilename) previewFilename.textContent = currentDownloadFilename;
                if (previewEditor) previewEditor.setValue(codeText, -1);
                if (previewModal) {
                    previewModal.classList.remove("hidden");
                    document.body.style.overflow = "hidden";
                }
            })
            .finally(() => {
                btnDownload.disabled = false;
                btnDownload.innerHTML = origBtnText;
            });
    }

    function closePreviewModal() {
        if (previewModal) {
            previewModal.classList.add("hidden");
            document.body.style.overflow = "";
        }
    }

    // Sự kiện nút Tải xuống ở Header panel -> Mở modal xem trước bản code đầy đủ
    btnDownload.addEventListener("click", openPreviewModal);

    // Đóng modal
    if (btnCloseModal) {
        btnCloseModal.addEventListener("click", closePreviewModal);
    }

    if (previewModal) {
        previewModal.addEventListener("click", (e) => {
            if (e.target === previewModal) {
                closePreviewModal();
            }
        });
    }

    document.addEventListener("keydown", (e) => {
        if (e.key === "Escape" && previewModal && !previewModal.classList.contains("hidden")) {
            closePreviewModal();
        }
    });

    // Sao chép code từ Modal preview
    if (btnModalCopy) {
        btnModalCopy.addEventListener("click", () => {
            const codeToCopy = previewEditor ? previewEditor.getValue() : "";
            if (!codeToCopy) return;

            navigator.clipboard.writeText(codeToCopy)
                .then(() => {
                    const originalHTML = btnModalCopy.innerHTML;
                    btnModalCopy.innerHTML = `
                        <svg viewBox="0 0 24 24" width="13" height="13" fill="none" stroke="currentColor" stroke-width="2">
                            <polyline points="20 6 9 17 4 12"></polyline>
                        </svg>
                        <span>Đã sao chép!</span>
                    `;
                    btnModalCopy.style.borderColor = "var(--color-success)";
                    btnModalCopy.style.background = "var(--color-success-glow)";
                    btnModalCopy.style.color = "var(--color-success)";

                    setTimeout(() => {
                        btnModalCopy.innerHTML = originalHTML;
                        btnModalCopy.style.borderColor = "";
                        btnModalCopy.style.background = "";
                        btnModalCopy.style.color = "";
                    }, 2000);
                })
                .catch(err => {
                    console.error("Lỗi khi sao chép:", err);
                    alert("Không thể sao chép mã nguồn.");
                });
        });
    }

    // Tải xuống file .java từ Modal preview
    if (btnModalDownload) {
        btnModalDownload.addEventListener("click", () => {
            const codeToDownload = previewEditor ? previewEditor.getValue() : "";
            if (!codeToDownload) return;

            const filename = currentDownloadFilename || "Problem.java";
            const blob = new Blob([codeToDownload], { type: "text/plain;charset=utf-8" });
            const url = URL.createObjectURL(blob);

            const a = document.createElement("a");
            a.href = url;
            a.download = filename;
            document.body.appendChild(a);
            a.click();

            // Hiệu ứng phản hồi thành công
            const origHTML = btnModalDownload.innerHTML;
            btnModalDownload.innerHTML = `
                <svg viewBox="0 0 24 24" width="14" height="14" fill="none" stroke="currentColor" stroke-width="2">
                    <polyline points="20 6 9 17 4 12"></polyline>
                </svg>
                <span>Đã tải xuống ${filename}!</span>
            `;

            setTimeout(() => {
                btnModalDownload.innerHTML = origHTML;
            }, 2200);

            setTimeout(() => {
                document.body.removeChild(a);
                window.URL.revokeObjectURL(url);
            }, 100);
        });
    }

    // Xử lý sự kiện chạy thử nghiệm mã Java
    btnRunCode.addEventListener("click", () => {
        const inputArgs = testInputArgs.value.trim();

        // Trạng thái Loading của runner
        btnRunCode.disabled = true;
        testInputArgs.disabled = true;
        runSpinner.classList.remove("hidden");
        runBtnText.textContent = "Đang chạy...";
        consoleOutput.innerHTML = '<span class="console-placeholder">Đang tiến hành biên dịch và thực thi mã Java...</span>';

        const payload = {
            code: editor ? editor.getValue() : currentFullCode,
            input_args: inputArgs
        };

        fetch("/api/run", {
            method: "POST",
            headers: {
                "Content-Type": "application/json"
            },
            body: JSON.stringify(payload)
        })
            .then(res => {
                if (!res.ok) {
                    return res.json().then(data => { throw new Error(data.error || "Lỗi hệ thống không xác định"); });
                }
                return res.json();
            })
            .then(data => {
                if (data.success) {
                    // Thành công
                    consoleOutput.innerHTML = `<span class="console-success">Chạy thành công! Kết quả đầu ra:\n\n${escapeHtml(data.output)}</span>`;
                } else {
                    // Thất bại (Lỗi biên dịch / Runtime / Timeout)
                    let errorTitle = "";
                    if (data.stage === "compile") {
                        errorTitle = "LỖI BIÊN DỊCH (Compilation Error):";
                    } else if (data.stage === "runtime") {
                        errorTitle = "LỖI KHI CHẠY (Runtime Exception):";
                    } else if (data.stage === "timeout") {
                        errorTitle = "QUÁ THỜI GIAN THỰC THI (Timeout Error):";
                    } else {
                        errorTitle = "LỖI HỆ THỐNG:";
                    }

                    consoleOutput.innerHTML = `<span class="console-error">${errorTitle}\n\n${escapeHtml(data.output)}</span>`;
                }
            })
            .catch(err => {
                consoleOutput.innerHTML = `<span class="console-error">LỖI KẾT NỐI API:\n\n${escapeHtml(err.message)}</span>`;
            })
            .finally(() => {
                btnRunCode.disabled = false;
                testInputArgs.disabled = false;
                runSpinner.classList.add("hidden");
                runBtnText.textContent = "Chạy hàm";
            });
    });

    // Hàm phụ trợ để tránh lỗi XSS/HTML Injection trong console
    function escapeHtml(text) {
        if (!text) return "";
        return text
            .replace(/&/g, "&amp;")
            .replace(/</g, "&lt;")
            .replace(/>/g, "&gt;")
            .replace(/"/g, "&quot;")
            .replace(/'/g, "&#039;");
    }

    // =====================================================
    //  GỢI Ý INPUT THÔNG MINH (SUGGEST INPUT)
    // =====================================================
    function fetchSuggestedInput(code) {
        // Kiểm tra nếu đang dùng preset đã có gợi ý sẵn thì bỏ qua
        const activePreset = document.querySelector(".preset-chip.active");
        if (activePreset) {
            const presetVal = activePreset.getAttribute("data-value");
            if (presetVal && PRESET_HELPERS[presetVal]) {
                // Vẫn gọi API để hiển thị param hints chips
                _callSuggestAPI(code, true);
                return;
            }
        }

        _callSuggestAPI(code, false);
    }

    function _callSuggestAPI(code, skipAutoFill) {
        fetch("/api/suggest-input", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ code: code })
        })
            .then(res => res.json())
            .then(data => {
                if (!data.params || data.params.length === 0) {
                    paramHints.classList.add("hidden");
                    paramHintsChips.innerHTML = "";
                    if (!skipAutoFill) {
                        testInputArgs.placeholder = "Hàm không có tham số";
                        testInputHelper.textContent = "Không cần đối số";
                        testInputArgs.value = "";
                    }
                    return;
                }


                // Cập nhật helper badge với thông tin kiểu tham số
                const paramSummary = data.params.map(p => `${p.type} ${p.name}`).join(", ");
                testInputHelper.textContent = paramSummary;

                // Auto-fill giá trị gợi ý nếu chưa có preset hoạt động
                if (!skipAutoFill && data.suggested_input) {
                    // Bỏ ngoặc tròn ngoài để hiện thị thân thiện hơn
                    let displayValue = data.suggested_input;
                    if (displayValue.startsWith("(") && displayValue.endsWith(")")) {
                        displayValue = displayValue.slice(1, -1);
                    }
                    testInputArgs.value = displayValue;
                    testInputArgs.placeholder = `Ví dụ: ${displayValue}`;

                    // Hiệu ứng flash nhấp nháy khi auto-fill
                    testInputArgs.classList.add("auto-filled");
                    setTimeout(() => testInputArgs.classList.remove("auto-filled"), 1000);
                }
            })
            .catch(err => {
                console.error("Lỗi khi gọi API suggest-input:", err);
                // Không hiển thị lỗi cho user, chỉ log
            });
    }
});
