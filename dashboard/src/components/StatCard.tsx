import { memo } from "react"
import type { ReactNode } from "react"
import { Card, CardContent } from "@/components/ui/card"

type StatCardProps = {
  label: string
  value: string | number
  icon: ReactNode
  title?: string
}

export const StatCard = memo(function StatCard({ label, value, icon, title }: StatCardProps) {
  return (
    <Card className="shadow-none" title={title}>
      <CardContent className="flex items-center gap-3 py-3">
        <div className="flex size-9 items-center justify-center rounded-md bg-muted text-muted-foreground">
          {icon}
        </div>
        <div>
          <p className="text-xs text-muted-foreground">{label}</p>
          <p className="text-lg font-semibold tabular-nums leading-tight">{value}</p>
        </div>
      </CardContent>
    </Card>
  )
})
