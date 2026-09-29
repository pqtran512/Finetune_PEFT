// Sample prompts (HumanEval-Java & NL). NL prime text follows current UI language.
const EXAMPLES_STATIC = {
    factorial: `import java.util.ArrayList;
import java.util.List;

public class Problem {
    /**
     * Return the factorial of a non-negative integer n.
     * Example: factorial(5) returns 120.
     */
    public static long factorial(int n) {`,

    remove_duplicates: `import java.util.ArrayList;
import java.util.List;

public class Problem {
    /**
     * From a list of integers, remove all elements that occur more than once.
     * Keep order of elements left the same as in the original list.
     */
    public static List<Integer> removeDuplicates(List<Integer> numbers) {`,

    right_triangle: `import java.util.ArrayList;
import java.util.List;

public class Problem {
    /**
     * Given the lengths of the three sides of a triangle, return true if they form
     * a right-angled triangle, false otherwise.
     * Example: rightAngleTriangle(3, 4, 5) returns true.
     */
    public static boolean rightAngleTriangle(int a, int b, int c) {`,
};

function getExamples() {
    return {
        nl_prime: t("example.nl_prime"),
        ...EXAMPLES_STATIC,
    };
}

function getPresetHelpers() {
    return {
        nl_prime: {
            placeholder: t("helper.prime_ph"),
            helper: t("helper.prime"),
            default: "7",
        },
        factorial: {
            placeholder: t("helper.factorial"),
            helper: t("helper.factorial"),
            default: "5",
        },
        remove_duplicates: {
            placeholder: t("helper.remove_dup"),
            helper: t("helper.remove_dup"),
            default: "[1, 2, 3, 2, 4]",
        },
        right_triangle: {
            placeholder: t("helper.triangle"),
            helper: t("helper.triangle"),
            default: "3, 4, 5",
        },
    };
}

function currentLang() {
    return (window.AJAC_I18N && AJAC_I18N.getLang()) || "en";
}

document.addEventListener("DOMContentLoaded", () => {
    if (window.AJAC_I18N) {
        AJAC_I18N.applyI18n(document);
    }

    const badgeDot = document.getElementById("badge-dot");
    const badgeText = document.getElementById("badge-text");
    const infoBase = document.getElementById("info-base");

    const presetChips = document.querySelectorAll(".preset-chip");
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

    let editor = null;
    let previewEditor = null;
    let currentDownloadFilename = "Problem.java";
    if (window.ace) {
        ace.config.set("basePath", "https://cdnjs.cloudflare.com/ajax/libs/ace/1.32.7/");
        editor = ace.edit("code-editor");
        editor.setTheme("ace/theme/chrome");
        editor.session.setMode("ace/mode/java");
        editor.setOptions({
            fontSize: "13px",
            fontFamily: "var(--font-code)",
            showPrintMargin: false,
            useSoftTabs: true,
            tabSize: 4,
            readOnly: false,
        });

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
                wrap: true,
            });
        }
    }

    const testRunnerSection = document.getElementById("test-runner-section");
    const btnRunCode = document.getElementById("btn-run-code");
    const testInputArgs = document.getElementById("test-input-args");
    const testInputHelper = document.getElementById("test-input-helper");
    const consoleOutput = document.getElementById("console-output");
    const runSpinner = document.querySelector(".run-spinner");
    const runBtnText = document.querySelector(".run-btn-text");

    const paramHints = document.getElementById("param-hints");
    const paramHintsChips = document.getElementById("param-hints-chips");

    const previewModal = document.getElementById("preview-modal");
    const previewFilename = document.getElementById("preview-filename");
    const btnCloseModal = document.getElementById("btn-close-modal");
    const btnModalCopy = document.getElementById("btn-modal-copy");
    const btnModalDownload = document.getElementById("btn-modal-download");

    let currentFullCode = "";
    let checkInterval = null;
    const modelBadge = document.querySelector(".model-badge");

    // Language toggle
    document.querySelectorAll(".lang-btn").forEach((btn) => {
        btn.addEventListener("click", () => {
            const lang = btn.getAttribute("data-lang");
            if (!lang || !window.AJAC_I18N) return;
            AJAC_I18N.setLang(lang);

            // Refresh active preset prompt / helpers for the new language
            const activePreset = document.querySelector(".preset-chip.active");
            if (activePreset) {
                const val = activePreset.getAttribute("data-value");
                const examples = getExamples();
                const helpers = getPresetHelpers();
                if (val && examples[val]) {
                    promptInput.value = examples[val];
                    updateInputLineNumbers();
                }
                if (val && helpers[val]) {
                    testInputArgs.placeholder = helpers[val].placeholder;
                    testInputHelper.textContent = helpers[val].helper;
                }
            } else if (testInputHelper && !testInputHelper.dataset.locked) {
                testInputArgs.placeholder = t("args.placeholder");
                testInputHelper.textContent = t("helper.default");
            }

            // Refresh console placeholder if still in empty placeholder state
            const placeholder = consoleOutput && consoleOutput.querySelector(".console-placeholder");
            if (placeholder && !placeholder.classList.contains("console-success") && !placeholder.classList.contains("console-error")) {
                placeholder.textContent = t("console.placeholder");
            }

            checkModelStatus();
        });
    });

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

    promptInput.addEventListener("input", () => {
        updateInputLineNumbers();
        presetChips.forEach((c) => c.classList.remove("active"));

        testInputArgs.placeholder = t("args.placeholder_alt");
        testInputHelper.textContent = t("helper.default");
        testInputHelper.dataset.locked = "";
        paramHints.classList.add("hidden");
        paramHintsChips.innerHTML = "";
    });

    promptInput.addEventListener("scroll", () => {
        inputLineNumbers.scrollTop = promptInput.scrollTop;
    });

    updateInputLineNumbers();

    presetChips.forEach((chip) => {
        chip.addEventListener("click", () => {
            const val = chip.getAttribute("data-value");
            const examples = getExamples();
            const helpers = getPresetHelpers();

            presetChips.forEach((c) => c.classList.remove("active"));
            chip.classList.add("active");

            if (val && examples[val]) {
                promptInput.value = examples[val];
            } else {
                promptInput.value = "";
            }

            updateInputLineNumbers();
            promptInput.scrollTop = 0;
            inputLineNumbers.scrollTop = 0;

            if (val && helpers[val]) {
                testInputArgs.placeholder = helpers[val].placeholder;
                testInputHelper.textContent = helpers[val].helper;
                testInputArgs.value = helpers[val].default;
                testInputHelper.dataset.locked = "1";
            } else {
                testInputArgs.placeholder = t("args.placeholder");
                testInputHelper.textContent = t("helper.default");
                testInputArgs.value = "";
                testInputHelper.dataset.locked = "";
            }
        });
    });

    function renderCode() {
        if (editor) {
            editor.setValue(currentFullCode, -1);
        } else if (codeOutput) {
            codeOutput.textContent = currentFullCode;
            Prism.highlightElement(codeOutput);
        }
    }

    function checkModelStatus() {
        fetch(`/api/status?lang=${encodeURIComponent(currentLang())}`)
            .then((res) => res.json())
            .then((data) => {
                if (modelBadge) {
                    modelBadge.setAttribute("title", t("badge.status", { status: data.status }));
                }
                if (badgeText) badgeText.textContent = "";
                if (infoBase) infoBase.textContent = data.base_model || "-";

                const status = data.status || "";
                const statusKey = data.status_key || "";
                const isReady =
                    statusKey === "ready" ||
                    statusKey === "ready_mock" ||
                    status.startsWith("Sẵn sàng") ||
                    status.startsWith("Ready") ||
                    status.includes("Mock");
                const isError =
                    statusKey === "error" ||
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
                    alert(t("alert.model_load_error", { status }));
                } else {
                    if (badgeDot) badgeDot.className = "dot pending";
                    btnGenerate.disabled = true;
                }
            })
            .catch((err) => {
                if (badgeDot) badgeDot.className = "dot offline";
                if (modelBadge) modelBadge.setAttribute("title", t("badge.offline"));
                btnGenerate.disabled = true;
                console.error("Status API error:", err);
            });
    }

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

    btnGenerate.addEventListener("click", async () => {
        const prompt = promptInput.value.trim();
        if (!prompt) {
            alert(t("alert.empty_prompt"));
            return;
        }

        loadingOverlay.classList.remove("hidden");
        btnGenerate.disabled = true;
        startConsoleLog();

        const payload = {
            prompt: prompt,
            enable_cot: false,
            enable_language_tag: true,
            input_args: testInputArgs ? testInputArgs.value.trim() : "",
            lang: currentLang(),
        };

        try {
            const response = await fetch("/api/generate", {
                method: "POST",
                headers: {
                    "Content-Type": "application/json",
                    "Accept-Language": currentLang(),
                },
                body: JSON.stringify(payload),
            });

            if (!response.ok) {
                const data = await response.json().catch(() => ({}));
                throw new Error(data.error || t("alert.server_error", { status: response.status }));
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
                buffer = lines.pop();

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
                        console.error("JSON chunk parse error:", e, trimmed);
                    }
                }
            }

            setTimeout(() => {
                loadingOverlay.classList.add("hidden");
                btnGenerate.disabled = false;

                if (finalResult) {
                    emptyState.classList.add("hidden");
                    codeContainer.classList.remove("hidden");

                    let fullCode = finalResult.enhanced_prompt
                        ? finalResult.enhanced_prompt + finalResult.generated_code
                        : finalResult.full_code || finalResult.generated_code;
                    if (!finalResult.enhanced_prompt) {
                        if (!fullCode.includes("class Problem") && !fullCode.startsWith(prompt)) {
                            fullCode = prompt + fullCode;
                        }
                    }

                    currentFullCode = fullCode;
                    renderCode();

                    statsTime.textContent = t("stats.time", { time: finalResult.time_taken });
                    statsTime.classList.remove("hidden");
                    btnCopy.disabled = false;
                    btnDownload.disabled = false;

                    testRunnerSection.classList.remove("hidden");
                    btnRunCode.disabled = false;
                    testInputArgs.disabled = false;
                    consoleOutput.innerHTML = `<span class="console-placeholder">${escapeHtml(t("console.placeholder"))}</span>`;

                    fetchSuggestedInput(currentFullCode);
                } else {
                    alert(t("alert.no_result"));
                }
            }, 800);
        } catch (err) {
            addLogLine(t("log.conn_error"), "task");
            addLogLine(t("log.generate_failed", { error: err.message }), "error");

            setTimeout(() => {
                loadingOverlay.classList.add("hidden");
                btnGenerate.disabled = false;
                alert(t("alert.generate_failed", { error: err.message }));
            }, 2000);
        }
    });

    btnCopy.addEventListener("click", () => {
        const codeText = editor ? editor.getValue() : codeOutput ? codeOutput.textContent : currentFullCode;
        if (!codeText) return;

        navigator.clipboard
            .writeText(codeText)
            .then(() => {
                const originalText = btnCopy.innerHTML;
                btnCopy.innerHTML = `
                    <svg viewBox="0 0 24 24" width="13" height="13" fill="none" stroke="currentColor" stroke-width="2">
                        <polyline points="20 6 9 17 4 12"></polyline>
                    </svg>
                    <span>${escapeHtml(t("btn.copied"))}</span>
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
            .catch((err) => {
                console.error("Copy failed:", err);
                alert(t("alert.copy_failed"));
            });
    });

    function openPreviewModal() {
        const codeText = editor ? editor.getValue() : currentFullCode;
        if (!codeText) {
            alert(t("alert.no_code_download"));
            return;
        }

        const inputArgs = testInputArgs ? testInputArgs.value.trim() : "";

        btnDownload.disabled = true;
        const origBtnText = btnDownload.innerHTML;
        btnDownload.innerHTML = `
            <span class="loader-spinner" style="width:12px;height:12px;border-width:2px;border-top-color:var(--color-primary);border-color:rgba(79,70,229,0.2);"></span>
            <span>${escapeHtml(t("btn.preparing_preview"))}</span>
        `;

        fetch("/api/preview-full-code", {
            method: "POST",
            headers: {
                "Content-Type": "application/json",
                "Accept-Language": currentLang(),
            },
            body: JSON.stringify({
                code: codeText,
                input_args: inputArgs,
                lang: currentLang(),
            }),
        })
            .then((res) => {
                if (!res.ok) throw new Error(t("alert.preview_failed"));
                return res.json();
            })
            .then((data) => {
                currentDownloadFilename = data.filename || "Problem.java";
                if (previewFilename) previewFilename.textContent = currentDownloadFilename;

                if (previewEditor) {
                    previewEditor.setValue(data.full_code || codeText, -1);
                }

                if (previewModal) {
                    previewModal.classList.remove("hidden");
                    document.body.style.overflow = "hidden";
                }

                setTimeout(() => {
                    if (previewEditor) {
                        previewEditor.resize();
                        previewEditor.focus();
                    }
                }, 100);
            })
            .catch((err) => {
                console.error("Preview full code error:", err);
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

    btnDownload.addEventListener("click", openPreviewModal);

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

    if (btnModalCopy) {
        btnModalCopy.addEventListener("click", () => {
            const codeToCopy = previewEditor ? previewEditor.getValue() : "";
            if (!codeToCopy) return;

            navigator.clipboard
                .writeText(codeToCopy)
                .then(() => {
                    const originalHTML = btnModalCopy.innerHTML;
                    btnModalCopy.innerHTML = `
                        <svg viewBox="0 0 24 24" width="13" height="13" fill="none" stroke="currentColor" stroke-width="2">
                            <polyline points="20 6 9 17 4 12"></polyline>
                        </svg>
                        <span>${escapeHtml(t("btn.copied"))}</span>
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
                .catch((err) => {
                    console.error("Copy error:", err);
                    alert(t("alert.copy_source_failed"));
                });
        });
    }

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

            const origHTML = btnModalDownload.innerHTML;
            btnModalDownload.innerHTML = `
                <svg viewBox="0 0 24 24" width="14" height="14" fill="none" stroke="currentColor" stroke-width="2">
                    <polyline points="20 6 9 17 4 12"></polyline>
                </svg>
                <span>${escapeHtml(t("btn.downloaded", { filename }))}</span>
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

    btnRunCode.addEventListener("click", () => {
        const inputArgs = testInputArgs.value.trim();

        btnRunCode.disabled = true;
        testInputArgs.disabled = true;
        runSpinner.classList.remove("hidden");
        runBtnText.textContent = t("btn.running");
        consoleOutput.innerHTML = `<span class="console-placeholder">${escapeHtml(t("console.compiling"))}</span>`;

        const payload = {
            code: editor ? editor.getValue() : currentFullCode,
            input_args: inputArgs,
            lang: currentLang(),
        };

        fetch("/api/run", {
            method: "POST",
            headers: {
                "Content-Type": "application/json",
                "Accept-Language": currentLang(),
            },
            body: JSON.stringify(payload),
        })
            .then((res) => {
                if (!res.ok) {
                    return res.json().then((data) => {
                        throw new Error(data.error || t("alert.unknown_error"));
                    });
                }
                return res.json();
            })
            .then((data) => {
                if (data.success) {
                    consoleOutput.innerHTML = `<span class="console-success">${escapeHtml(
                        t("run.success", { output: data.output })
                    )}</span>`;
                } else {
                    let errorTitle = "";
                    if (data.stage === "compile") {
                        errorTitle = t("run.err_compile");
                    } else if (data.stage === "runtime") {
                        errorTitle = t("run.err_runtime");
                    } else if (data.stage === "timeout") {
                        errorTitle = t("run.err_timeout");
                    } else {
                        errorTitle = t("run.err_system");
                    }

                    consoleOutput.innerHTML = `<span class="console-error">${escapeHtml(errorTitle)}\n\n${escapeHtml(
                        data.output
                    )}</span>`;
                }
            })
            .catch((err) => {
                consoleOutput.innerHTML = `<span class="console-error">${escapeHtml(
                    t("run.err_api", { error: err.message })
                )}</span>`;
            })
            .finally(() => {
                btnRunCode.disabled = false;
                testInputArgs.disabled = false;
                runSpinner.classList.add("hidden");
                runBtnText.textContent = t("btn.run");
            });
    });

    function escapeHtml(text) {
        if (!text) return "";
        return String(text)
            .replace(/&/g, "&amp;")
            .replace(/</g, "&lt;")
            .replace(/>/g, "&gt;")
            .replace(/"/g, "&quot;")
            .replace(/'/g, "&#039;");
    }

    function fetchSuggestedInput(code) {
        const activePreset = document.querySelector(".preset-chip.active");
        if (activePreset) {
            const presetVal = activePreset.getAttribute("data-value");
            if (presetVal && getPresetHelpers()[presetVal]) {
                _callSuggestAPI(code, true);
                return;
            }
        }

        _callSuggestAPI(code, false);
    }

    function _callSuggestAPI(code, skipAutoFill) {
        fetch("/api/suggest-input", {
            method: "POST",
            headers: {
                "Content-Type": "application/json",
                "Accept-Language": currentLang(),
            },
            body: JSON.stringify({ code: code, lang: currentLang() }),
        })
            .then((res) => res.json())
            .then((data) => {
                if (!data.params || data.params.length === 0) {
                    paramHints.classList.add("hidden");
                    paramHintsChips.innerHTML = "";
                    if (!skipAutoFill) {
                        testInputArgs.placeholder = t("suggest.no_params");
                        testInputHelper.textContent = t("suggest.no_args");
                        testInputArgs.value = "";
                    }
                    return;
                }

                const paramSummary = data.params.map((p) => `${p.type} ${p.name}`).join(", ");
                testInputHelper.textContent = paramSummary;

                if (!skipAutoFill && data.suggested_input) {
                    let displayValue = data.suggested_input;
                    if (displayValue.startsWith("(") && displayValue.endsWith(")")) {
                        displayValue = displayValue.slice(1, -1);
                    }
                    testInputArgs.value = displayValue;
                    testInputArgs.placeholder = t("suggest.example", { value: displayValue });

                    testInputArgs.classList.add("auto-filled");
                    setTimeout(() => testInputArgs.classList.remove("auto-filled"), 1000);
                }
            })
            .catch((err) => {
                console.error("suggest-input API error:", err);
            });
    }
});
