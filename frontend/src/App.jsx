import { useEffect, useMemo, useRef, useState } from 'react'
import CodeMirror from '@uiw/react-codemirror'
import { java } from '@codemirror/lang-java'
import { vscodeDark, vscodeLight } from '@uiw/codemirror-theme-vscode'
import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import {
  Activity,
  AlertTriangle,
  Bot,
  CheckCircle2,
  ChevronRight,
  Clock3,
  Code2,
  Copy,
  Cpu,
  Download,
  FileJson,
  Moon,
  Network,
  Play,
  RefreshCw,
  ScanLine,
  Sun,
  Terminal,
  X,
  XCircle,
} from 'lucide-react'
import { presets } from './config/presets'
import './App.css'

const javaExtensions = [java()]

const API_BASE = import.meta.env.VITE_API_BASE_URL || '/api'
const THEME_STORAGE_KEY = 'ajac-theme'

function getInitialTheme() {
  try {
    const stored = localStorage.getItem(THEME_STORAGE_KEY)
    if (stored === 'light' || stored === 'dark') return stored
  } catch {
    /* ignore */
  }
  if (typeof window !== 'undefined' && window.matchMedia('(prefers-color-scheme: dark)').matches) {
    return 'dark'
  }
  return 'light'
}

const tabs = [
  { id: 'monitor', label: 'Pipeline Monitor', icon: Activity },
  { id: 'code', label: 'Generated Code', icon: Code2 },
  { id: 'xray', label: 'Execution Trace', icon: ScanLine },
  { id: 'raw', label: 'Raw Data', icon: FileJson },
  { id: 'theory', label: 'Architecture', icon: Network },
]

function isReadyStatus(status) {
  const key = status?.status_key || ''
  const text = status?.status || ''
  return (
    key === 'ready' ||
    key === 'ready_mock' ||
    text.startsWith('Ready') ||
    text.startsWith('Sẵn sàng') ||
    text.includes('Mock')
  )
}

function isErrorStatus(status) {
  const key = status?.status_key || ''
  const text = (status?.status || '').toLowerCase()
  return key === 'error' || text.includes('error') || text.includes('lỗi') || text.includes('loi')
}

function buildFullCode(result, prompt) {
  if (!result) return ''
  if (result.enhanced_prompt) {
    return `${result.enhanced_prompt}${result.generated_code || ''}`
  }
  let full = result.full_code || result.generated_code || ''
  if (!full.includes('class Problem') && prompt && !full.startsWith(prompt)) {
    full = prompt + full
  }
  return full
}

function StatusBadge({ status }) {
  const success = status === 'success' || status === 'passed' || status === 'ready'
  const warning = status === 'loading' || status === 'cancelled'
  return (
    <span className={`status-badge ${success ? 'status-success' : warning ? 'status-warning' : 'status-error'}`}>
      {success ? <CheckCircle2 size={13} /> : warning ? <AlertTriangle size={13} /> : <XCircle size={13} />}
      {status || 'idle'}
    </span>
  )
}

function Metric({ label, value, icon: Icon }) {
  return (
    <div className="metric-card">
      <div className="metric-label"><Icon size={14} /> {label}</div>
      <strong>{value}</strong>
    </div>
  )
}

function App() {
  const [presetId, setPresetId] = useState(presets[0].id)
  const [prompt, setPrompt] = useState(presets[0].prompt)
  const [inputArgs, setInputArgs] = useState(presets[0].defaultArgs)
  const [argsHint, setArgsHint] = useState(presets[0].argsHint)
  const [modelId, setModelId] = useState('finetuned')
  const [models, setModels] = useState([
    { id: 'base', label: 'CodeLlama-7B (base)', available: true },
    { id: 'finetuned', label: 'CodeLlama-7B + LoRA', available: true },
  ])
  const [serverStatus, setServerStatus] = useState(null)
  const [activeTab, setActiveTab] = useState('monitor')
  const [loading, setLoading] = useState(false)
  const [running, setRunning] = useState(false)
  const [switchingModel, setSwitchingModel] = useState(false)
  const [error, setError] = useState('')
  const [result, setResult] = useState(null)
  const [fullCode, setFullCode] = useState('')
  const [executionSteps, setExecutionSteps] = useState([])
  const [xrayEvents, setXrayEvents] = useState([])
  const [runOutput, setRunOutput] = useState(null)
  const [paramHints, setParamHints] = useState([])
  const [previewOpen, setPreviewOpen] = useState(false)
  const [previewLoading, setPreviewLoading] = useState(false)
  const [previewFilename, setPreviewFilename] = useState('Problem.java')
  const [previewCode, setPreviewCode] = useState('')
  const [theme, setTheme] = useState(() => {
    const initial = getInitialTheme()
    document.documentElement.setAttribute('data-theme', initial)
    return initial
  })
  const consoleRef = useRef(null)
  const abortControllerRef = useRef(null)
  const editorTheme = theme === 'dark' ? vscodeDark : vscodeLight

  useEffect(() => {
    document.documentElement.setAttribute('data-theme', theme)
    try {
      localStorage.setItem(THEME_STORAGE_KEY, theme)
    } catch {
      /* ignore */
    }
  }, [theme])

  const toggleTheme = () => setTheme((current) => (current === 'dark' ? 'light' : 'dark'))

  const preset = useMemo(
    () => presets.find((item) => item.id === presetId) || presets[0],
    [presetId],
  )
  const modelReady = isReadyStatus(serverStatus)
  const modelError = isErrorStatus(serverStatus)

  useEffect(() => {
    const el = consoleRef.current
    if (el) el.scrollTop = el.scrollHeight
  }, [executionSteps])

  useEffect(() => () => abortControllerRef.current?.abort(), [])

  useEffect(() => {
    let cancelled = false
    const poll = async () => {
      try {
        const response = await fetch(`${API_BASE}/status`)
        if (!response.ok) throw new Error(`HTTP ${response.status}`)
        const data = await response.json()
        if (cancelled) return
        setServerStatus(data)
        if (Array.isArray(data.models) && data.models.length) setModels(data.models)
        if (data.active_model && !switchingModel) setModelId(data.active_model)
      } catch {
        if (!cancelled) {
          setServerStatus({ status: 'Offline', status_key: 'error' })
        }
      }
    }
    poll()
    const timer = setInterval(poll, 3000)
    return () => {
      cancelled = true
      clearInterval(timer)
    }
  }, [switchingModel])

  const selectPreset = (event) => {
    const next = presets.find((item) => item.id === event.target.value) || presets[0]
    setPresetId(next.id)
    setPrompt(next.prompt)
    setInputArgs(next.defaultArgs)
    setArgsHint(next.argsHint)
    setError('')
    setParamHints([])
  }

  const changeModel = async (event) => {
    const next = event.target.value
    if (!next || next === modelId || switchingModel) return
    const previous = modelId
    setSwitchingModel(true)
    setModelId(next)
    try {
      const response = await fetch(`${API_BASE}/model`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ model: next, lang: 'en' }),
      })
      const data = await response.json().catch(() => ({}))
      if (!response.ok) throw new Error(data.error || `HTTP ${response.status}`)
      setModelId(data.active_model || next)
      if (Array.isArray(data.models)) setModels(data.models)
    } catch (err) {
      setModelId(previous)
      setError(err instanceof Error ? err.message : 'Could not switch model.')
    } finally {
      setSwitchingModel(false)
    }
  }

  const cancelGenerate = () => abortControllerRef.current?.abort()

  const generateCode = async () => {
    if (!prompt.trim() || loading || !modelReady) return
    abortControllerRef.current?.abort()
    const controller = new AbortController()
    abortControllerRef.current = controller

    setLoading(true)
    setError('')
    setResult(null)
    setFullCode('')
    setExecutionSteps([])
    setXrayEvents([])
    setRunOutput(null)
    setParamHints([])
    setActiveTab('monitor')

    const pendingXray = {
      enhance: null,
      model: null,
      inference: null,
      compile: null,
    }

    try {
      const response = await fetch(`${API_BASE}/generate`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          Accept: 'application/x-ndjson',
        },
        signal: controller.signal,
        body: JSON.stringify({
          prompt: prompt.trim(),
          enable_language_tag: true,
          enable_cot: false,
          input_args: inputArgs.trim(),
          lang: 'en',
          model: modelId,
        }),
      })

      if (!response.ok) {
        const data = await response.json().catch(() => ({}))
        throw new Error(data.error || `Generate failed (HTTP ${response.status})`)
      }
      if (!response.body) throw new Error('Browser does not support reading the response stream.')

      const reader = response.body.getReader()
      const decoder = new TextDecoder()
      let buffer = ''
      let finalPayload = null

      const handleLine = (line) => {
        if (!line.trim()) return
        let event
        try {
          event = JSON.parse(line)
        } catch {
          setExecutionSteps((steps) => [...steps, '⚠️ Received invalid NDJSON chunk.'])
          return
        }

        if (event.step === 'result') {
          finalPayload = event
          return
        }
        if (event.step === 'error') {
          throw new Error(event.text || 'Code generation pipeline failed.')
        }

        if (event.text) {
          setExecutionSteps((steps) => [...steps, event.text])
        }

        const step = event.step || ''
        if (step.startsWith('enhance')) {
          pendingXray.enhance = {
            step: 'prompt_enhance',
            target: 'PromptEnhancer',
            execution_detail: event.text,
            input: prompt.trim(),
          }
        } else if (step.startsWith('model')) {
          pendingXray.model = {
            step: 'model_select',
            target: modelId,
            execution_detail: event.text,
            input: { model: modelId, enable_language_tag: true, enable_cot: false },
          }
        } else if (step.startsWith('inference')) {
          pendingXray.inference = {
            step: 'inference',
            target: 'CodeLlama generate()',
            execution_detail: event.text,
          }
        } else if (step.startsWith('compile') || step === 'build_finished') {
          const failed = step === 'compile_fail'
          pendingXray.compile = {
            step: 'compile_smoke',
            target: 'javac',
            execution_detail: event.text,
            error: failed ? event.text : undefined,
          }
        }
      }

      while (true) {
        const { value, done } = await reader.read()
        buffer += decoder.decode(value || new Uint8Array(), { stream: !done })
        const lines = buffer.split('\n')
        buffer = lines.pop() || ''
        lines.forEach(handleLine)
        if (done) break
      }
      if (buffer.trim()) handleLine(buffer)

      if (!finalPayload) throw new Error('No final result received from the pipeline.')

      const code = buildFullCode(finalPayload, prompt.trim())
      setResult(finalPayload)
      setFullCode(code)
      setActiveTab('code')

      const xray = []
      if (pendingXray.enhance) {
        xray.push({
          ...pendingXray.enhance,
          output: {
            input_type: finalPayload.input_type,
            enhanced_prompt: finalPayload.enhanced_prompt,
          },
        })
      }
      if (pendingXray.model) xray.push(pendingXray.model)
      if (pendingXray.inference) {
        xray.push({
          ...pendingXray.inference,
          output: finalPayload.generated_code,
        })
      }
      if (pendingXray.compile) xray.push(pendingXray.compile)
      xray.push({
        step: 'result',
        target: 'AJAC pipeline',
        execution_detail: `Completed in ${finalPayload.time_taken || '?'}`,
        output: {
          model: finalPayload.model,
          input_type: finalPayload.input_type,
          time_taken: finalPayload.time_taken,
        },
      })
      setXrayEvents(xray)

      try {
        const suggestRes = await fetch(`${API_BASE}/suggest-input`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ code, lang: 'en' }),
        })
        if (suggestRes.ok) {
          const suggest = await suggestRes.json()
          if (suggest.suggested_input && !inputArgs.trim()) {
            setInputArgs(suggest.suggested_input.replace(/^\(|\)$/g, ''))
          }
          if (Array.isArray(suggest.params)) setParamHints(suggest.params)
        }
      } catch {
        /* suggest is best-effort */
      }
    } catch (requestError) {
      if (requestError?.name === 'AbortError') {
        setError('Pipeline cancelled.')
        setExecutionSteps((steps) => [...steps, '⛔ Pipeline cancelled.'])
      } else {
        setError(requestError instanceof Error ? requestError.message : 'Could not connect to the code generation API.')
      }
      setResult(null)
    } finally {
      if (abortControllerRef.current === controller) abortControllerRef.current = null
      setLoading(false)
    }
  }

  const runCode = async () => {
    if (!fullCode.trim() || running) return
    setRunning(true)
    setRunOutput(null)
    try {
      const response = await fetch(`${API_BASE}/run`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          code: fullCode,
          input_args: inputArgs.trim(),
          lang: 'en',
        }),
      })
      const data = await response.json().catch(() => ({}))
      if (!response.ok && !data.output) {
        throw new Error(data.error || `Run failed (HTTP ${response.status})`)
      }
      setRunOutput(data)
      setXrayEvents((events) => [
        ...events,
        {
          step: 'run_test',
          target: 'javac + java TestRunner',
          execution_detail: data.success ? 'Runtime OK' : `Failed at ${data.stage || 'unknown'}`,
          input: inputArgs.trim(),
          output: data.output,
          error: data.success ? undefined : data.output,
        },
      ])
    } catch (err) {
      setRunOutput({
        success: false,
        stage: 'system',
        output: err instanceof Error ? err.message : 'Run failed',
      })
    } finally {
      setRunning(false)
    }
  }

  const copyCode = async () => {
    if (!fullCode) return
    try {
      await navigator.clipboard.writeText(fullCode)
    } catch {
      setError('Could not copy code to the clipboard.')
    }
  }

  const openPreview = async () => {
    if (!fullCode || previewLoading) return
    setPreviewLoading(true)
    setError('')
    try {
      const response = await fetch(`${API_BASE}/preview-full-code`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ code: fullCode, input_args: inputArgs.trim(), lang: 'en' }),
      })
      const data = await response.json().catch(() => ({}))
      if (!response.ok) throw new Error(data.error || 'Could not create preview.')
      setPreviewFilename(data.filename || 'Problem.java')
      setPreviewCode(data.full_code || fullCode)
      setPreviewOpen(true)
    } catch (err) {
      setPreviewFilename('Problem.java')
      setPreviewCode(fullCode)
      setPreviewOpen(true)
      setError(err instanceof Error ? err.message : 'Preview failed — showing editor code.')
    } finally {
      setPreviewLoading(false)
    }
  }

  const closePreview = () => setPreviewOpen(false)

  const copyPreviewCode = async () => {
    if (!previewCode) return
    try {
      await navigator.clipboard.writeText(previewCode)
    } catch {
      setError('Could not copy code to the clipboard.')
    }
  }

  const downloadPreviewFile = () => {
    if (!previewCode) return
    const blob = new Blob([previewCode], { type: 'text/x-java' })
    const url = URL.createObjectURL(blob)
    const anchor = document.createElement('a')
    anchor.href = url
    anchor.download = previewFilename || 'Problem.java'
    anchor.click()
    URL.revokeObjectURL(url)
  }

  useEffect(() => {
    if (!previewOpen) return undefined
    const onKey = (event) => {
      if (event.key === 'Escape') setPreviewOpen(false)
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [previewOpen])

  const prettyRaw = JSON.stringify(result ?? {}, null, 2)
  const statusChipClass = modelReady ? 'ready' : modelError ? 'error' : 'loading'

  return (
    <main className="app-shell">
      <header className="topbar">
        <div className="brand">
          <div className="brand-mark"><img src="/java.png" alt="Java" className="brand-logo" /></div>
          <div>
            <div className="eyebrow">JAVA COMPLETION</div>
            <h1>AJAC — Automatic Java Code Generator</h1>
          </div>
        </div>
        <div className="topbar-status">
          <button
            type="button"
            className="theme-toggle"
            onClick={toggleTheme}
            aria-label={theme === 'dark' ? 'Switch to light mode' : 'Switch to dark mode'}
            title={theme === 'dark' ? 'Light mode' : 'Dark mode'}
          >
            {theme === 'dark' ? <Sun size={15} /> : <Moon size={15} />}
          </button>
          <span className={`status-chip ${statusChipClass}`}>
            <span className="pulse" />
            {(serverStatus?.status || 'Connecting...').slice(0, 42)}
          </span>
          <span className="topbar-divider" />
          <span>CODEGEN CONSOLE</span>
          <span className="topbar-divider" />
          <span>v2.0.0</span>
        </div>
      </header>

      <section className="dashboard-grid">
        <aside className="left-pane">
          <div className="pane-heading">
            <div>
              <div className="eyebrow">INPUT / CONTROLS</div>
              <h2>Generation Console</h2>
            </div>
            <Cpu size={18} className="muted" />
          </div>

          <div className="control-grid control-grid--secondary">
            <label>
              MODEL
              <select value={modelId} onChange={changeModel} disabled={switchingModel || !modelReady}>
                {models.map((item) => (
                  <option key={item.id} value={item.id} disabled={item.available === false}>
                    {item.label || item.id}
                  </option>
                ))}
              </select>
            </label>
            <label>
              PROBLEM PRESET
              <select value={presetId} onChange={selectPreset}>
                {presets.map((item) => (
                  <option key={item.id} value={item.id}>{item.title}</option>
                ))}
              </select>
            </label>
          </div>

          <div className="intel-card intel-card--muted">
            <div className="card-kicker"><AlertTriangle size={14} /> PIPELINE BRIEF</div>
            <div className="intel-title">{preset.group}</div>
            <p>{preset.description}</p>
          </div>

          <label className="query-label prompt-stage">
            <span className="prompt-stage-label">PROBLEM / PREFIX</span>
            <textarea
              className="prompt-textarea"
              value={prompt}
              onChange={(event) => {
                setPrompt(event.target.value)
                setParamHints([])
              }}
              placeholder="Enter an NL description or Java prefix (signature + {)..."
            />
          </label>

          <div className="action-row">
            <button
              className="execute-button"
              onClick={generateCode}
              disabled={loading || !prompt.trim() || !modelReady}
            >
              {loading
                ? <><RefreshCw size={20} className="spin" /> GENERATING...</>
                : <><Play size={20} /> GENERATE JAVA</>}
            </button>
            {loading && (
              <button type="button" className="cancel-button" onClick={cancelGenerate}>
                <XCircle size={17} /> CANCEL
              </button>
            )}
          </div>

          {error && (
            <div className="error-banner">
              <XCircle size={17} />
              <div>
                <strong>{error === 'Pipeline cancelled.' ? 'Cancelled' : 'Request failed'}</strong>
                <span>{error}</span>
              </div>
            </div>
          )}

          <div className="run-card">
            <div className="card-kicker"><Terminal size={14} fontWeight={900} /> RUN GENERATED FUNCTION</div>

            <div className="run-block">
              <div className="run-label">ARGUMENTS</div>
              <div className="run-args-row">
                <input
                  className="run-args"
                  value={inputArgs}
                  onChange={(event) => setInputArgs(event.target.value)}
                  placeholder="e.g. 5  ·  3, 4, 5  ·  [1, 2, 3]"
                  disabled={!fullCode}
                  aria-label="Function arguments"
                />
                <button className="run-button" onClick={runCode} disabled={!fullCode || running}>
                  {running
                    ? <><RefreshCw size={14} className="spin" /> …</>
                    : <><Play size={14} /> RUN</>}
                </button>
              </div>
              {paramHints.length > 0 ? (
                <div className="run-params">
                  <span className="run-params-label">Parameters</span>
                  <div className="param-chips">
                    {paramHints.map((param) => (
                      <span className="param-chip" key={`${param.name}-${param.type}`}>
                        {param.type} {param.name}
                      </span>
                    ))}
                  </div>
                </div>
              ) : (
                <p className="run-hint">{argsHint}</p>
              )}
            </div>

            <div className={`run-terminal ${runOutput ? (runOutput.success ? 'ok' : 'fail') : ''}`}>
              <div className="run-terminal-header">
                <span className="console-dot console-red" />
                <span className="console-dot console-yellow" />
                <span className="console-dot console-green" />
                <span className="run-terminal-title"><Terminal size={12} /> CONSOLE</span>
                <span className={`run-status ${running ? 'running' : runOutput ? (runOutput.success ? 'ok' : 'fail') : 'idle'}`}>
                  {running ? 'running' : runOutput ? (runOutput.success ? 'ok' : 'error') : 'idle'}
                </span>
              </div>
              <pre className="run-console">
                {runOutput
                  ? (runOutput.success
                    ? (runOutput.output || '(no output)')
                    : `[${runOutput.stage || 'error'}]\n${runOutput.output || 'failed'}`)
                  : 'Ready'}
              </pre>
            </div>
          </div>
        </aside>

        <section className="right-pane">
          <div className="pane-heading monitor-heading">
            <div>
              <div className="eyebrow">OBSERVABILITY / TRACE</div>
              <h2>Pipeline Monitor</h2>
            </div>
            <div className="live-indicator"><span className="pulse" /> LIVE</div>
          </div>

          <nav className="tabs">
            {tabs.map(({ id, label, icon: Icon }) => (
              <button
                key={id}
                className={activeTab === id ? 'tab active' : 'tab'}
                onClick={() => setActiveTab(id)}
              >
                <Icon size={15} /> {label}
              </button>
            ))}
          </nav>

          {!result && !loading && activeTab === 'monitor' && (
            <div className="welcome-panel">
              <div className="welcome-icon"><Terminal size={25} /></div>
              <h3>Ready for Java completion</h3>
              <p>Pick a preset or enter a problem, then Generate to watch <br /> prompt normalization → code generation → post-processing.</p>
              <div className="flow-hint">
                <span>PROMPT</span><ChevronRight size={13} />
                <span>GENERATE CODE</span><ChevronRight size={13} />
                <span>POST-PROCESS</span><ChevronRight size={13} />
                <span>OUTPUT</span>
              </div>
            </div>
          )}

          {loading && <ThinkingConsole steps={executionSteps} consoleRef={consoleRef} />}

          {activeTab === 'monitor' && result && !loading && <Monitor result={result} />}
          {activeTab === 'code' && (
            <CodePanel
              code={fullCode}
              onChange={setFullCode}
              onCopy={copyCode}
              onDownload={openPreview}
              downloadLabel={previewLoading ? 'PREPARING...' : 'DOWNLOAD'}
              downloadDisabled={previewLoading || !fullCode.trim()}
              editorTheme={editorTheme}
            />
          )}
          {activeTab === 'xray' && <ExecutionTrace events={xrayEvents} loading={loading} />}
          {activeTab === 'raw' && <pre className="raw-panel">{prettyRaw}</pre>}
          {activeTab === 'theory' && <Theory />}
        </section>
      </section>

      {previewOpen && (
        <PreviewModal
          filename={previewFilename}
          code={previewCode}
          onChange={setPreviewCode}
          onClose={closePreview}
          onCopy={copyPreviewCode}
          onDownload={downloadPreviewFile}
          editorTheme={editorTheme}
        />
      )}
    </main>
  )
}

function TraceValue({ label, value, code = false, json = false }) {
  if (value === undefined || value === null || value === '') return null
  const serialized = typeof value === 'string' ? value : JSON.stringify(value, null, 2)
  return (
    <div className="trace-field">
      <div className="trace-label">{label}</div>
      {json && typeof value === 'object'
        ? <pre className="trace-json">{serialized}</pre>
        : code
          ? <pre className="trace-code">{serialized}</pre>
          : <pre className="trace-text">{serialized}</pre>}
    </div>
  )
}

function ExecutionTrace({ events, loading }) {
  return (
    <div className="execution-trace">
      <div className="trace-summary">
        <div>
          <div className="eyebrow">X-RAY / STRUCTURED EVENTS</div>
          <h2>Execution Trace</h2>
        </div>
        <span className="count-chip">{events.length} EVENTS</span>
      </div>
      {!events.length && (
        <div className="empty-state">
          {loading ? 'Waiting for completed steps to capture X-Ray...' : 'Run Generate to inspect the detailed trace.'}
        </div>
      )}
      <div className="trace-list">
        {events.map((event, index) => {
          const failed = Boolean(event.error)
          return (
            <details
              className={`trace-card ${failed ? 'trace-failed' : ''}`}
              open={index === events.length - 1}
              key={`${event.step}-${index}`}
            >
              <summary>
                <span className="trace-index">{String(index + 1).padStart(2, '0')}</span>
                <span>{event.step}</span>
                <span className={`trace-state ${failed ? 'trace-state-error' : 'trace-state-ok'}`}>
                  {failed ? 'ERROR' : 'CAPTURED'}
                </span>
              </summary>
              <div className="trace-content">
                <TraceValue label="Target" value={event.target} />
                <TraceValue label="Execution Detail" value={event.execution_detail} />
                <TraceValue
                  label="Input"
                  value={event.input}
                  code={typeof event.input === 'string'}
                  json={typeof event.input === 'object'}
                />
                {event.output !== undefined && (
                  <TraceValue
                    label="Output"
                    value={event.output}
                    code={typeof event.output === 'string'}
                    json={typeof event.output === 'object'}
                  />
                )}
                {event.error && <TraceValue label="Error" value={event.error} />}
              </div>
            </details>
          )
        })}
      </div>
    </div>
  )
}

function ThinkingConsole({ steps, consoleRef }) {
  return (
    <div className="thinking-console">
      <div className="console-header">
        <span className="console-dot console-red" />
        <span className="console-dot console-yellow" />
        <span className="console-dot console-green" />
        <span className="console-title"><Terminal size={13} /> AJAC PIPELINE / LIVE TRACE</span>
      </div>
      <div className="console-body" ref={consoleRef} role="log" aria-live="polite">
        {steps.length === 0 && <div className="console-line console-muted">▌ Initializing code generation pipeline...</div>}
        {steps.map((step, index) => (
          <div className="console-line" key={`${index}-${step.slice(0, 24)}`}>
            <span className="console-index">{String(index + 1).padStart(2, '0')}</span>
            <span>{step}</span>
          </div>
        ))}
        <div className="console-cursor">▌</div>
      </div>
    </div>
  )
}

function Monitor({ result }) {
  const summary = [
    `**Model:** \`${result.model || '—'}\``,
    `**Input type:** \`${result.input_type || '—'}\``,
    `**Latency:** ${result.time_taken || '—'}`,
    '',
    'Code was assembled from the enhanced prompt + generated continuation. Open the **Generated Code** tab to review/edit, then **Run** from the left panel.',
  ].join('\n')

  return (
    <div className="monitor-content">
      <div className="metrics-grid">
        <Metric label="STATUS" value={<StatusBadge status="success" />} icon={CheckCircle2} />
        <Metric label="MODEL" value={result.model || '—'} icon={Cpu} />
        <Metric label="LATENCY" value={result.time_taken || '—'} icon={Clock3} />
      </div>
      <div className="answer-card">
        <div className="card-kicker"><Bot size={14} /> GENERATION SUMMARY</div>
        <div className="markdown-body">
          <ReactMarkdown remarkPlugins={[remarkGfm]}>{summary}</ReactMarkdown>
        </div>
      </div>
    </div>
  )
}

function CodePanel({ code, onChange, onCopy, onDownload, downloadLabel = 'DOWNLOAD', downloadDisabled = false, editorTheme }) {
  const hasCode = Boolean(code?.trim())
  return (
    <div className="code-panel">
      <div className="code-toolbar">
        <div className="eyebrow" style={{ margin: 0 }}>
          {hasCode ? 'PROBLEM.JAVA / EDITABLE' : 'PROBLEM.JAVA / SCRATCHPAD'}
        </div>
        <div className="code-toolbar-actions">
          <button type="button" className="ghost-button" onClick={onCopy} disabled={!hasCode}>
            <Copy size={13} /> COPY
          </button>
          <button type="button" className="ghost-button" onClick={onDownload} disabled={downloadDisabled || !hasCode}>
            <Download size={13} /> {downloadLabel}
          </button>
        </div>
      </div>
      <CodeMirror
        className="code-body"
        value={code}
        height="62vh"
        minHeight="360px"
        theme={editorTheme}
        extensions={javaExtensions}
        onChange={onChange}
        placeholder="// Enter or paste Java code here — Generate will overwrite when a result arrives"
        basicSetup={{
          lineNumbers: true,
          highlightActiveLine: true,
          foldGutter: true,
          bracketMatching: true,
        }}
        aria-label="Java source editor"
      />
    </div>
  )
}

function PreviewModal({ filename, code, onChange, onClose, onCopy, onDownload, editorTheme }) {
  return (
    <div className="modal-overlay" onClick={onClose} role="presentation">
      <div
        className="modal-card"
        role="dialog"
        aria-modal="true"
        aria-label="Preview full Java file"
        onClick={(event) => event.stopPropagation()}
      >
        <div className="modal-header">
          <div>
            <div className="eyebrow">DOWNLOAD PREVIEW</div>
            <h3>{filename}</h3>
            <p className="modal-hint">Full runnable source (with main) from /api/preview-full-code. Edit before copy/download.</p>
          </div>
          <button type="button" className="modal-close" onClick={onClose} aria-label="Close preview">
            <X size={16} />
          </button>
        </div>
        <CodeMirror
          className="modal-code"
          value={code}
          height="52vh"
          minHeight="360px"
          theme={editorTheme}
          extensions={javaExtensions}
          onChange={onChange}
          basicSetup={{
            lineNumbers: true,
            highlightActiveLine: true,
            foldGutter: true,
            bracketMatching: true,
          }}
          aria-label="Preview Java source"
        />
        <div className="modal-footer">
          <button type="button" className="ghost-button" onClick={onCopy}><Copy size={13} /> COPY CODE</button>
          <button type="button" className="modal-download" onClick={onDownload}>
            <Download size={13} /> DOWNLOAD .JAVA
          </button>
        </div>
      </div>
    </div>
  )
}

const ARCH_STEPS = [
  {
    id: 'input',
    num: '01',
    titleVi: 'Đề bài / đầu vào',
    tech: 'Input prompt',
    body: 'Người dùng nhập đề bài dạng ngôn ngữ tự nhiên hoặc khung mã Java cần hoàn thiện.',
    variant: null,
  },
  {
    id: 'prompt',
    num: '02',
    titleVi: 'Chuẩn hóa prompt',
    tech: 'Prompt processing',
    body: 'Hệ thống chuẩn hóa đề bài về dạng prompt thống nhất trước khi đưa vào mô hình.',
    variant: null,
  },
  {
    id: 'generate',
    num: '03',
    titleVi: 'Sinh mã bằng LLM',
    tech: 'LLM code generation',
    body: 'Mô hình người dùng chọn (base hoặc fine-tune) sinh phần mã Java còn thiếu.',
    variant: 'gen',
  },
  {
    id: 'post',
    num: '04',
    titleVi: 'Hậu xử lý',
    tech: null,
    body: 'Làm sạch kết quả: cân bằng ngoặc, cắt phần thừa, chuẩn bị mã sẵn sàng hiển thị.',
    variant: null,
  },
  {
    id: 'output',
    num: '05',
    titleVi: 'Kết quả mã Java',
    tech: 'Output',
    body: 'Mã hoàn chỉnh được đưa ra panel Generated Code để xem, sao chép hoặc chạy thử.',
    variant: 'out',
  },
]

function ArchitectureFlow({ steps }) {
  return (
    <ol className="arch-flow" aria-label="Architecture pipeline">
      {steps.map((step, index) => (
        <li key={step.id} className="arch-flow-item">
          <div className={`arch-node${step.variant ? ` arch-node--${step.variant}` : ''}`}>
            <span className="arch-num">{step.num}</span>
            <strong>{step.titleVi}</strong>
            {step.tech ? <code>{step.tech}</code> : null}
          </div>
          {index < steps.length - 1 ? (
            <span className="arch-arrow" aria-hidden="true">
              <ChevronRight size={18} />
            </span>
          ) : null}
        </li>
      ))}
    </ol>
  )
}

function Theory() {
  return (
    <div className="theory-panel">
      <ArchitectureFlow steps={ARCH_STEPS} />
      <div className="section-title">
        <span>PIPELINE STAGES</span>
        <span className="count-chip">5 STEPS</span>
      </div>
      <div className="theory-grid">
        {ARCH_STEPS.map((step) => (
          <article key={step.id}>
            <span className="step-number">{step.num}</span>
            <div>
              <h3>
                {step.titleVi}
                {step.tech ? <span className="step-tech">{step.tech}</span> : null}
              </h3>
              <p>{step.body}</p>
            </div>
          </article>
        ))}
      </div>
    </div>
  )
}

export default App
