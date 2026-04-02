import { useState, useEffect } from 'react'

interface RequestItem {
  id: number
  timestamp: string
  prompt_hash: string
  complexity: string
  category: string
  confidence: number
  model: string
  tokens_in: number
  tokens_out: number
  cost: number
  latency_ms: number
  fallback_used: number
}

interface RequestsResponse {
  total: number
  items: RequestItem[]
  limit: number
  offset: number
}

function shortModelName(model: string): string {
  if (model.includes('haiku')) return 'Haiku'
  if (model.includes('sonnet')) return 'Sonnet'
  if (model.includes('opus')) return 'Opus'
  if (model.includes('gpt-4o-mini')) return 'GPT-4o-mini'
  if (model.includes('gpt-4o')) return 'GPT-4o'
  if (model.includes('gemini')) return 'Gemini'
  return model
}

export default function RequestLog() {
  const [data, setData] = useState<RequestsResponse | null>(null)
  const [page, setPage] = useState(0)
  const [model, setModel] = useState('')
  const [complexity, setComplexity] = useState('')
  const limit = 20

  useEffect(() => {
    const params = new URLSearchParams({ limit: String(limit), offset: String(page * limit) })
    if (model) params.set('model', model)
    if (complexity) params.set('complexity', complexity)
    fetch(`/api/requests?${params}`)
      .then(r => r.json())
      .then(setData)
      .catch(() => {})
  }, [page, model, complexity])

  const totalPages = data ? Math.ceil(data.total / limit) : 0

  return (
    <div className="request-log">
      <h2>Request Log</h2>
      <div className="filters">
        <select value={complexity} onChange={e => { setComplexity(e.target.value); setPage(0) }}>
          <option value="">All complexity</option>
          <option value="simple">Simple</option>
          <option value="medium">Medium</option>
          <option value="complex">Complex</option>
        </select>
        <select value={model} onChange={e => { setModel(e.target.value); setPage(0) }}>
          <option value="">All models</option>
          <option value="claude-haiku-4-5-20251001">Haiku</option>
          <option value="claude-sonnet-4-6">Sonnet</option>
          <option value="claude-opus-4-6">Opus</option>
        </select>
      </div>
      {!data ? (
        <div className="loading">Loading...</div>
      ) : (
        <>
          <table>
            <thead>
              <tr>
                <th>Time</th>
                <th>Complexity</th>
                <th>Category</th>
                <th>Model</th>
                <th>Tokens</th>
                <th>Cost</th>
                <th>Latency</th>
                <th>Fallback</th>
              </tr>
            </thead>
            <tbody>
              {data.items.map(item => (
                <tr key={item.id}>
                  <td>{new Date(item.timestamp).toLocaleString()}</td>
                  <td>{item.complexity}</td>
                  <td>{item.category}</td>
                  <td>{shortModelName(item.model)}</td>
                  <td>{item.tokens_in}+{item.tokens_out}</td>
                  <td>${item.cost.toFixed(6)}</td>
                  <td>{item.latency_ms.toFixed(0)}ms</td>
                  <td>{item.fallback_used ? 'Yes' : ''}</td>
                </tr>
              ))}
            </tbody>
          </table>
          <div className="pagination">
            <button disabled={page === 0} onClick={() => setPage(p => p - 1)}>Prev</button>
            <span>Page {page + 1} of {totalPages || 1} ({data.total} total)</span>
            <button disabled={page + 1 >= totalPages} onClick={() => setPage(p => p + 1)}>Next</button>
          </div>
        </>
      )}
    </div>
  )
}
