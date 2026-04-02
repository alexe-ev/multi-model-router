import { useState, useEffect } from 'react'
import KPICards from './components/KPICards'
import CostChart from './components/CostChart'
import ModelDistribution from './components/ModelDistribution'
import RequestLog from './components/RequestLog'

interface Stats {
  total_requests: number
  total_cost: number
  avg_latency_ms: number
  fallback_count: number
  total_tokens_in: number
  total_tokens_out: number
  model_distribution: Record<string, { count: number; cost: number }>
  savings: {
    actual_cost: number
    baseline_cost: number
    savings: number
    savings_pct: number
  }
}

export default function App() {
  const [stats, setStats] = useState<Stats | null>(null)
  const [daily, setDaily] = useState<any[]>([])
  const [models, setModels] = useState<any[]>([])
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    Promise.all([
      fetch('/api/stats').then(r => r.json()),
      fetch('/api/stats/daily').then(r => r.json()),
      fetch('/api/models').then(r => r.json()),
    ])
      .then(([s, d, m]) => {
        setStats(s)
        setDaily(d)
        setModels(m)
      })
      .catch(e => setError(e.message))
  }, [])

  if (error) return <div className="dashboard"><div className="error">Error: {error}</div></div>
  if (!stats) return <div className="dashboard"><div className="loading">Loading...</div></div>

  return (
    <div className="dashboard">
      <h1>mmrouter dashboard</h1>
      <KPICards stats={stats} />
      <div className="charts-grid">
        <CostChart data={daily} />
        <ModelDistribution data={models} />
      </div>
      <RequestLog />
    </div>
  )
}
