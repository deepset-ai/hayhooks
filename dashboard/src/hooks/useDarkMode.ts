import { useCallback, useState } from "react"

export function useDarkMode() {
  const [dark, setDark] = useState(() => document.documentElement.classList.contains("dark"))

  const toggle = useCallback(() => {
    setDark((prev) => {
      const next = !prev
      document.documentElement.classList.toggle("dark", next)
      return next
    })
  }, [])

  return { dark, toggle } as const
}
