import '@fontsource/inter/400.css'
import '@fontsource/inter/500.css'
import '@fontsource/inter/600.css'
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
