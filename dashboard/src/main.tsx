import '@fontsource/inter'
import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import './index.css'
import App from './App.tsx'
import { TracesProvider } from './hooks/TracesProvider'

createRoot(document.getElementById('root')!).render(
  <StrictMode>
    <TracesProvider>
      <App />
    </TracesProvider>
  </StrictMode>,
)
