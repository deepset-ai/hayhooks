import { memo, useMemo } from "react"
import type { ReactNode } from "react"

import { cn } from "@/lib/utils"

export type StatTone = "default" | "destructive"

type StatProps = {
  label: string
  value: ReactNode
  hint?: string
  tone?: StatTone
  sparkline?: number[]
  title?: string
}

const TONE_CLASS: Record<StatTone, string> = {
  default: "text-foreground",
  destructive: "text-destructive",
}

export const Stat = memo(function Stat({
  label,
  value,
  hint,
  tone = "default",
  sparkline,
  title,
}: StatProps) {
  return (
    <div className="bg-card px-4 py-3" title={title}>
      <p className="text-xs text-muted-foreground">
        {label}
        {hint !== undefined && (
          <span className="ml-1 text-muted-foreground/70">· {hint}</span>
        )}
      </p>
      <div className="mt-0.5 flex items-baseline gap-2">
        <p className={cn("text-lg font-semibold tabular-nums leading-tight", TONE_CLASS[tone])}>
          {value}
        </p>
        {sparkline !== undefined && sparkline.length >= 2 && (
          <Sparkline values={sparkline} className="text-primary/70" />
        )}
      </div>
    </div>
  )
})

type SparklineProps = {
  values: number[]
  width?: number
  height?: number
  className?: string
}

export const Sparkline = memo(function Sparkline({
  values,
  width = 64,
  height = 18,
  className,
}: SparklineProps) {
  const { polyline, dot } = useMemo(() => {
    if (values.length < 2) return { polyline: "", dot: null as { x: number; y: number } | null }
    let min = Infinity
    let max = -Infinity
    for (const v of values) {
      if (v < min) min = v
      if (v > max) max = v
    }
    const range = max - min || 1
    const lastIdx = values.length - 1
    const points = values.map((v, i) => {
      const x = (i / lastIdx) * (width - 2) + 1
      const y = height - 1 - ((v - min) / range) * (height - 2)
      return `${x.toFixed(1)},${y.toFixed(1)}`
    })
    const last = values[lastIdx]
    return {
      polyline: points.join(" "),
      dot: {
        x: width - 1,
        y: height - 1 - ((last - min) / range) * (height - 2),
      },
    }
  }, [values, width, height])

  if (polyline === "") return null

  return (
    <svg
      width={width}
      height={height}
      viewBox={`0 0 ${width} ${height}`}
      role="img"
      aria-label="Recent duration trend"
      className={cn("inline-block shrink-0", className)}
    >
      <polyline
        fill="none"
        stroke="currentColor"
        strokeWidth="1.25"
        strokeLinecap="round"
        strokeLinejoin="round"
        points={polyline}
      />
      {dot !== null && (
        <circle cx={dot.x} cy={dot.y} r="1.75" fill="currentColor" />
      )}
    </svg>
  )
})

type StatStripProps = {
  children: ReactNode
}

export const StatStrip = memo(function StatStrip({ children }: StatStripProps) {
  return (
    <div className="grid grid-cols-2 gap-px overflow-hidden rounded-lg bg-border ring-1 ring-border sm:grid-cols-4">
      {children}
    </div>
  )
})
