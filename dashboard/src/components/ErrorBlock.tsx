import { memo, useState } from "react"
import { AlertTriangle, Check, ChevronDown, ChevronRight, Copy } from "lucide-react"

import { useCopyToClipboard } from "../hooks/useCopyToClipboard"

type ErrorBlockProps = {
  errorType?: string
  errorMessage?: string
  errorStack?: string
}

export const ErrorBlock = memo(function ErrorBlock({ errorType, errorMessage, errorStack }: ErrorBlockProps) {
  const [stackExpanded, setStackExpanded] = useState(false)
  const stackCopy = useCopyToClipboard()

  return (
    <div className="mx-4 mt-3 space-y-2 rounded-md border border-destructive/25 bg-destructive/5 px-3 py-2">
      <div className="flex items-start gap-2">
        <AlertTriangle className="mt-0.5 size-3.5 shrink-0 text-destructive" />
        <div className="min-w-0 flex-1 text-xs">
          {errorType && (
            <span className="font-semibold text-destructive">{errorType}</span>
          )}
          {errorType && errorMessage && <span className="text-destructive/70">: </span>}
          {errorMessage && (
            <span className="text-destructive/80">{errorMessage}</span>
          )}
        </div>
        {errorStack && (
          <button
            type="button"
            className="shrink-0 rounded p-1 text-destructive/40 hover:text-destructive/80 hover:bg-destructive/10 transition-colors"
            onClick={() => stackCopy.copy(errorStack)}
            aria-label="Copy stack trace"
          >
            {stackCopy.copied
              ? <Check className="size-3" />
              : <Copy className="size-3" />}
          </button>
        )}
      </div>
      {errorStack && (
        <button
          type="button"
          onClick={() => setStackExpanded((prev) => !prev)}
          className="flex items-center gap-1 text-[11px] text-destructive/60 hover:text-destructive/90 transition-colors"
        >
          {stackExpanded ? (
            <ChevronDown className="size-3" />
          ) : (
            <ChevronRight className="size-3" />
          )}
          {stackExpanded ? "Hide stack trace" : "Show stack trace"}
        </button>
      )}
      {errorStack && stackExpanded && (
        <pre className="overflow-auto rounded bg-destructive/10 px-2.5 py-2 font-mono text-[10px] leading-relaxed text-destructive/80 whitespace-pre-wrap max-h-64">
          {errorStack}
        </pre>
      )}
    </div>
  )
})
