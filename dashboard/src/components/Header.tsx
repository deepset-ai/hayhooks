import { memo } from "react"
import { Moon, RefreshCw, Sun, Trash2 } from "lucide-react"

import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import { Separator } from "@/components/ui/separator"
import { cn } from "@/lib/utils"
import haystackIcon from "@/assets/haystack-icon.png"
import { useTraceActions, useTraceStatus } from "../hooks/useTracesContext"
import { fmtTime } from "../utils/formatting"

type HeaderProps = {
  dark: boolean
  onToggleDark: () => void
}

export const Header = memo(function Header({
  dark,
  onToggleDark,
}: HeaderProps) {
  const { updatedAt, error, refreshing, clearing } = useTraceStatus()
  const { refresh, clear } = useTraceActions()
  const busy = refreshing || clearing
  return (
    <header className="sticky top-0 z-30 border-b bg-background/95 backdrop-blur supports-[backdrop-filter]:bg-background/80">
      <div className="mx-auto flex h-14 max-w-7xl items-center justify-between gap-4 px-6">
        <div className="flex items-center gap-3">
          <img src={haystackIcon} alt="Haystack" className="size-8 rounded-lg" />
          <div>
            <h1 className="text-sm font-semibold leading-none">Hayhooks</h1>
            <p className="mt-0.5 text-xs text-muted-foreground">Trace Dashboard</p>
          </div>
        </div>
        <div className="flex items-center gap-2">
          {updatedAt !== null && (
            <span className="hidden items-center gap-1.5 text-xs text-muted-foreground sm:flex">
              <span className="live-dot" />
              Updated {fmtTime(updatedAt)}
            </span>
          )}
          {error !== null && (
            <Badge variant="destructive" className="text-xs">{error}</Badge>
          )}
          <Separator orientation="vertical" className="mx-1 h-6" />
          <Button variant="ghost" size="sm" onClick={() => void clear()} disabled={busy} className="gap-1.5 text-xs">
            <Trash2 className="size-3.5" />
            Clear
          </Button>
          <Button variant="ghost" size="sm" onClick={() => void refresh()} disabled={busy} className="gap-1.5 text-xs">
            <RefreshCw className={cn("size-3.5", refreshing && "animate-spin")} />
            Refresh
          </Button>
          <Button variant="ghost" size="icon" className="size-8" onClick={onToggleDark} aria-label="Toggle dark mode">
            {dark ? <Sun className="size-3.5" /> : <Moon className="size-3.5" />}
          </Button>
        </div>
      </div>
    </header>
  )
})
