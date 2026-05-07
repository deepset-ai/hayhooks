import { memo, useState } from "react"
import { Moon, RefreshCw, Sun, Trash2 } from "lucide-react"

import {
  AlertDialog,
  AlertDialogContent,
  AlertDialogDescription,
  AlertDialogFooter,
  AlertDialogHeader,
  AlertDialogTitle,
} from "@/components/ui/alert-dialog"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import { Separator } from "@/components/ui/separator"
import { cn } from "@/lib/utils"
import haystackIcon from "@/assets/haystack-icon.png"
import { useTraceActions, useTraceData, useTraceStatus } from "../hooks/useTracesContext"
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
  const { traces } = useTraceData()
  const [confirmOpen, setConfirmOpen] = useState(false)
  const busy = refreshing || clearing
  const hasTraces = traces.length > 0
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
          <Button
            variant="ghost"
            size="sm"
            onClick={() => setConfirmOpen(true)}
            disabled={busy || !hasTraces}
            className="gap-1.5 text-xs"
            title="Clear all captured traces"
          >
            <Trash2 className="size-3.5" />
            Clear
          </Button>
          <Button
            variant="ghost"
            size="sm"
            onClick={() => void refresh()}
            disabled={busy}
            className="gap-1.5 text-xs"
            title="Refresh traces now"
          >
            <RefreshCw className={cn("size-3.5", refreshing && "animate-spin")} />
            Refresh
          </Button>
          <Button
            variant="ghost"
            size="icon"
            className="size-8"
            onClick={onToggleDark}
            aria-label={dark ? "Switch to light mode" : "Switch to dark mode"}
            title={dark ? "Switch to light mode" : "Switch to dark mode"}
          >
            {dark ? <Sun className="size-3.5" /> : <Moon className="size-3.5" />}
          </Button>
        </div>
      </div>

      <AlertDialog open={confirmOpen} onOpenChange={setConfirmOpen}>
        <AlertDialogContent>
          <AlertDialogHeader>
            <AlertDialogTitle>Clear all traces?</AlertDialogTitle>
            <AlertDialogDescription>
              This removes every trace from the live buffer. You can&apos;t undo it,
              but new traces will appear as soon as your pipelines handle requests.
            </AlertDialogDescription>
          </AlertDialogHeader>
          <AlertDialogFooter>
            <Button
              variant="ghost"
              size="sm"
              onClick={() => setConfirmOpen(false)}
            >
              Cancel
            </Button>
            <Button
              variant="destructive"
              size="sm"
              onClick={() => {
                setConfirmOpen(false)
                void clear()
              }}
            >
              <Trash2 className="size-3.5" />
              Clear traces
            </Button>
          </AlertDialogFooter>
        </AlertDialogContent>
      </AlertDialog>
    </header>
  )
})
