// ĐỊNH NGHĨA CÁC PROMPT MẪU (HUMANEVAL-JAVA)
const EXAMPLES = {
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
    public static List<Integer> sortEven(List<Integer> l) {`,

    will_it_fly: `import java.util.List;

public class Problem {
    /**
     * Write a function that returns true if the object q will fly, and false otherwise.
     * The object q will fly if it's balanced (it is a palindromic list) and the sum of its elements
     * is less than or equal to the maximum weight w.
     */
    public static boolean willItFly(List<Integer> q, int w) {`
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
    rolling_max: {
        placeholder: "Ví dụ: List.of(1, 2, 4, 3, 5)",
        helper: "Ví dụ: (List.of(1, 2, 4, 3, 5))",
        default: "(List.of(1, 2, 4, 3, 5))"
    },
    remove_duplicates: {
        placeholder: "Ví dụ: List.of(1, 2, 3, 2, 4)",
        helper: "Ví dụ: (List.of(1, 2, 3, 2, 4))",
        default: "(List.of(1, 2, 3, 2, 4))"
    },
    sort_even: {
        placeholder: "Ví dụ: List.of(5, 6, 3, 4, 1, 2)",
        helper: "Ví dụ: (List.of(5, 6, 3, 4, 1, 2))",
        default: "(List.of(5, 6, 3, 4, 1, 2))"
    },
    will_it_fly: {
        placeholder: "Ví dụ: List.of(3, 2, 3), 10",
        helper: "Ví dụ: (List.of(3, 2, 3), 10)",
        default: "(List.of(3, 2, 3), 10)"
    }
};

document.addEventListener("DOMContentLoaded", () => {
    // Elements
    const badgeDot = document.getElementById("badge-dot");
    const badgeText = document.getElementById("badge-text");
    const infoBase = document.getElementById("info-base");
    const infoLora = document.getElementById("info-lora");
    const infoDevice = document.getElementById("info-device");
    
    // Preset chips
    const presetChips = document.querySelectorAll(".preset-chip");
    
    // Raw inputs inside details
    const temperatureInput = document.getElementById("temperature");
    const tempVal = document.getElementById("temp-val");
    const maxTokensInput = document.getElementById("max-tokens");
    const tokensVal = document.getElementById("tokens-val");
    
    const promptInput = document.getElementById("prompt-input");
    const inputLineNumbers = document.getElementById("input-line-numbers");
    
    const btnGenerate = document.getElementById("btn-generate");
    const btnCopy = document.getElementById("btn-copy");
    const btnDownload = document.getElementById("btn-download");
    
    const statsTime = document.getElementById("stats-time");
    const codeContainer = document.getElementById("code-container");
    const codeOutput = document.getElementById("code-output");
    const emptyState = document.getElementById("empty-state");
    const loadingOverlay = document.getElementById("loading-overlay");
    const loadingMessage = document.getElementById("loading-message");



    // Test runner elements
    const testRunnerSection = document.getElementById("test-runner-section");
    const btnRunCode = document.getElementById("btn-run-code");
    const testInputArgs = document.getElementById("test-input-args");
    const testInputHelper = document.getElementById("test-input-helper");
    const consoleOutput = document.getElementById("console-output");
    const runSpinner = document.querySelector(".run-spinner");
    const runBtnText = document.querySelector(".run-btn-text");

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
        testInputArgs.placeholder = "Nhập các đối số, ví dụ: 1, 2 hoặc (1, 2)";
        testInputHelper.textContent = "Ví dụ: (1, 2)";
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
            
            const val = btn.getAttribute("data-value");
            temperatureInput.value = val;
            tempVal.textContent = val;
        });
    });

    // Xử lý Segmented Controls (Độ dài tối đa)
    const lengthSegments = document.querySelectorAll("#length-segments .segment-btn");
    lengthSegments.forEach(btn => {
        btn.addEventListener("click", () => {
            lengthSegments.forEach(b => b.classList.remove("active"));
            btn.classList.add("active");
            
            const val = btn.getAttribute("data-value");
            maxTokensInput.value = val;
            tokensVal.textContent = val;
        });
    });

    // Đồng bộ ngược từ Sliders nâng cao về Segmented Controls
    temperatureInput.addEventListener("input", (e) => {
        const val = parseFloat(e.target.value);
        tempVal.textContent = val;
        
        creativitySegments.forEach(b => {
            if (parseFloat(b.getAttribute("data-value")) === val) {
                b.classList.add("active");
            } else {
                b.classList.remove("active");
            }
        });
    });

    maxTokensInput.addEventListener("input", (e) => {
        const val = parseInt(e.target.value);
        tokensVal.textContent = val;
        
        lengthSegments.forEach(b => {
            if (parseInt(b.getAttribute("data-value")) === val) {
                b.classList.add("active");
            } else {
                b.classList.remove("active");
            }
        });
    });

    // Hàm render code
    function renderCode() {
        codeOutput.textContent = currentFullCode;
        Prism.highlightElement(codeOutput);
    }

    // Hàm cập nhật trạng thái mô hình từ API
    let checkInterval = null;
    
    function checkModelStatus() {
        fetch("/api/status")
            .then(res => res.json())
            .then(data => {
                badgeText.textContent = data.status;
                infoBase.textContent = data.base_model;
                
                // Hiển thị tên thư mục cuối của adapter LoRA
                if (data.lora_path) {
                    const parts = data.lora_path.split(/[\\/]/);
                    infoLora.textContent = parts[parts.length - 1] || data.lora_path;
                } else {
                    infoLora.textContent = "Không có";
                }
                
                infoDevice.textContent = data.device;

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
                    badgeDot.className = "dot pending";
                    btnGenerate.disabled = true;
                }
            })
            .catch(err => {
                badgeDot.className = "dot offline";
                badgeText.textContent = "Không kết nối được server";
                btnGenerate.disabled = true;
                console.error("Lỗi kết nối API status:", err);
            });
    }

    // Bắt đầu kiểm tra trạng thái
    checkModelStatus();
    checkInterval = setInterval(checkModelStatus, 3000);

    // Bắt đầu chu kỳ thay đổi thông điệp loading động
    function startLoadingMessageCycle() {
        let index = 0;
        loadingMessage.textContent = LOADING_MESSAGES[0];
        loadingMessageInterval = setInterval(() => {
            index = (index + 1) % LOADING_MESSAGES.length;
            loadingMessage.textContent = LOADING_MESSAGES[index];
        }, 2000);
    }

    function stopLoadingMessageCycle() {
        if (loadingMessageInterval) {
            clearInterval(loadingMessageInterval);
            loadingMessageInterval = null;
        }
    }

    // Xử lý sự kiện Sinh code
    btnGenerate.addEventListener("click", () => {
        const prompt = promptInput.value.trim();
        if (!prompt) {
            alert("Vui lòng nhập prompt hoặc khai báo hàm cần sinh mã!");
            return;
        }

        // Show loading state
        loadingOverlay.classList.remove("hidden");
        btnGenerate.disabled = true;
        startLoadingMessageCycle();
        
        const payload = {
            prompt: prompt,
            temperature: parseFloat(temperatureInput.value),
            max_new_tokens: parseInt(maxTokensInput.value)
        };

        fetch("/api/generate", {
            method: "POST",
            headers: {
                "Content-Type": "application/json"
            },
            body: JSON.stringify(payload)
        })
        .then(res => {
            if (!res.ok) {
                return res.json().then(data => { throw new Error(data.error || "Lỗi không xác định"); });
            }
            return res.json();
        })
        .then(data => {
            // Hiển thị code kết quả
            emptyState.classList.add("hidden");
            codeContainer.classList.remove("hidden");
            
            // Xử lý ghép prompt vào code sinh ra nếu model trả về phần thân
            let rawCode = data.generated_code;
            if (!rawCode.startsWith(prompt)) {
                rawCode = prompt + rawCode;
            }

            // Lưu trữ code
            currentFullCode = rawCode;

            // Render
            renderCode();

            // Cập nhật stats
            statsTime.textContent = `Thời gian sinh: ${data.time_taken}`;
            statsTime.classList.remove("hidden");
            btnCopy.disabled = false;
            btnDownload.disabled = false;

            // Hiển thị và kích hoạt khu vực Chạy thử nghiệm (Playground)
            testRunnerSection.classList.remove("hidden");
            btnRunCode.disabled = false;
            testInputArgs.disabled = false;
            consoleOutput.innerHTML = '<span class="console-placeholder">Nhập đối số đầu vào ở trên và nhấn "Chạy hàm" để xem kết quả tính toán.</span>';
        })
        .catch(err => {
            alert("Lỗi khi sinh code: " + err.message);
        })
        .finally(() => {
            loadingOverlay.classList.add("hidden");
            btnGenerate.disabled = false;
            stopLoadingMessageCycle();
        });
    });

    // Xử lý sự kiện Sao chép (Copy)
    btnCopy.addEventListener("click", () => {
        const codeText = codeOutput.textContent;
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

    // Xử lý sự kiện Tải về file Java
    btnDownload.addEventListener("click", () => {
        const codeText = codeOutput.textContent;
        if (!codeText) return;

        const filename = "Output.java";
        const blob = new Blob([codeText], { type: "text/plain;charset=utf-8" });
        const url = URL.createObjectURL(blob);
        
        const a = document.createElement("a");
        a.href = url;
        a.download = filename;
        document.body.appendChild(a);
        a.click();
        
        // Dọn dẹp
        setTimeout(() => {
            document.body.removeChild(a);
            window.URL.revokeObjectURL(url);
        }, 100);
    });

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
            code: currentFullCode,
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
});
