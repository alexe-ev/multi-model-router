import { useState, useEffect } from 'react'

interface Incident {
  id: number
  rule_name: string
  severity: string
  message: string
  details: Record<string, unknown>
  first_seen: string
  last_seen: string
  fire_count: number
  cleared_at: string | null
  handled_at: string | null
}

interface AlertsResponse {
  items: Incident[]
  total: number
  open_count: number
  handled_count: number
  limit: number
}

function when(iso: string): string {
  return new Date(iso).toLocaleString([], {
    month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit',
  })
}

function clockOnly(iso: string): string {
  return new Date(iso).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
}

function tzLabel(): string {
  const mins = -new Date().getTimezoneOffset()
  const sign = mins < 0 ? '-' : '+'
  const h = Math.floor(Math.abs(mins) / 60)
  const m = Math.abs(mins) % 60
  return `UTC${sign}${h}${m ? ':' + String(m).padStart(2, '0') : ''}`
}

export default function AlertPanel() {
  const [data, setData] = useState<AlertsResponse | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [markError, setMarkError] = useState<string | null>(null)
  const [showHandled, setShowHandled] = useState(false)
  const [pending, setPending] = useState<number | null>(null)

  useEffect(() => {
    fetch('/api/alerts')
      .then(r => { if (!r.ok) throw new Error(`HTTP ${r.status}`); return r.json() })
      .then(setData)
      .catch(e => setError(e.message))
  }, [])

  async function setHandled(id: number, handled: boolean) {
    setPending(id)
    setMarkError(null)
    try {
      const r = await fetch(`/api/alerts/${id}/handled`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ handled }),
      })
      if (!r.ok) throw new Error(`HTTP ${r.status}`)
      const updated: Incident = await r.json()
      setData(d => {
        if (!d) return d
        // Derive the delta from what the row actually was, not from the button
        // we pressed: a stale tab can mark an already-handled row, and a blind
        // +/-1 would drift the header counts away from the database.
        const prev = d.items.find(i => i.id === updated.id)
        const wasHandled = !!prev && prev.handled_at !== null
        const nowHandled = updated.handled_at !== null
        const delta = wasHandled === nowHandled ? 0 : nowHandled ? 1 : -1
        return {
          ...d,
          items: d.items.map(i => (i.id === updated.id ? updated : i)),
          open_count: d.open_count - delta,
          handled_count: d.handled_count + delta,
        }
      })
    } catch {
      setMarkError(handled ? 'Could not mark as handled. Try again.' : 'Could not unmark. Try again.')
    } finally {
      setPending(null)
    }
  }

  if (error) {
    return (
      <div className="alert-panel">
        <div className="head"><h2>Alerts</h2></div>
        <div className="error">Error: {error}</div>
      </div>
    )
  }
  if (!data) {
    return (
      <div className="alert-panel">
        <div className="head"><h2>Alerts</h2></div>
        <div className="loading">Loading...</div>
      </div>
    )
  }

  const visible = data.items.filter(i => (showHandled ? true : i.handled_at === null))

  return (
    <div className="alert-panel">
      <div className="head">
        <h2>Alerts</h2>
        {data.total > 0 && <span className="open-count">{data.open_count} open</span>}
        {data.handled_count > 0 && (
          <label className="toggle">
            <input
              type="checkbox"
              checked={showHandled}
              onChange={e => setShowHandled(e.target.checked)}
            />
            Show handled ({data.handled_count})
          </label>
        )}
      </div>
      {data.total === 0 ? (
        <div className="empty">No alerts yet.</div>
      ) : visible.length === 0 ? (
        <div className="empty">
          {data.open_count === 0
            ? `Nothing open. ${data.handled_count} handled alerts hidden.`
            : `No open alerts among the ${data.items.length} most recent. ` +
              `${data.open_count} open alerts are older than that.`}
        </div>
      ) : (
        <>
          <div className="tz">Times in local time ({tzLabel()})</div>
          <table>
            <thead>
              <tr>
                <th>Severity</th><th>Rule</th><th>Fires</th>
                <th>First seen</th><th>Last seen</th><th>State</th><th></th>
              </tr>
            </thead>
            <tbody>
              {visible.map(i => (
                <tr key={i.id} className={i.handled_at ? 'handled' : undefined}>
                  <td><span className={`sev ${i.severity}`}>{i.severity}</span></td>
                  <td>
                    <div className="rule">{i.rule_name}</div>
                    <div className="msg">{i.message}</div>
                  </td>
                  <td className="num">{i.fire_count}</td>
                  <td className="when">{when(i.first_seen)}</td>
                  <td className="when">{when(i.last_seen)}</td>
                  <td className="state">
                    {i.handled_at && (
                      <><span className="marked">handled {when(i.handled_at)}</span><br /></>
                    )}
                    {i.cleared_at ? (
                      <span className="cleared">cleared {clockOnly(i.cleared_at)}</span>
                    ) : (
                      <span className="firing">{i.handled_at ? 'still firing' : 'firing'}</span>
                    )}
                  </td>
                  <td>
                    <button
                      disabled={pending === i.id}
                      onClick={() => setHandled(i.id, i.handled_at === null)}
                    >
                      {i.handled_at ? 'Unmark' : 'Mark handled'}
                    </button>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </>
      )}
      {data.total > data.items.length && (
        <div className="msg">
          Showing the {data.items.length} most recent of {data.total}.
        </div>
      )}
      {markError && <div className="error">{markError}</div>}
    </div>
  )
}
