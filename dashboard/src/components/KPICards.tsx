interface Props {
  stats: {
    total_requests: number
    total_cost: number
    avg_latency_ms: number
    fallback_count: number
    savings: { savings_pct: number }
  }
}

export default function KPICards({ stats }: Props) {
  return (
    <div className="kpi-grid">
      <div className="kpi-card">
        <div className="label">Total Requests</div>
        <div className="value">{stats.total_requests.toLocaleString()}</div>
      </div>
      <div className="kpi-card">
        <div className="label">Total Cost</div>
        <div className="value">${stats.total_cost.toFixed(4)}</div>
      </div>
      <div className="kpi-card">
        <div className="label">Cost Savings</div>
        <div className="value green">{stats.savings.savings_pct.toFixed(1)}%</div>
      </div>
      <div className="kpi-card">
        <div className="label">Avg Latency</div>
        <div className="value">{stats.avg_latency_ms.toFixed(0)}ms</div>
      </div>
      <div className="kpi-card">
        <div className="label">Fallback Rate</div>
        <div className="value">
          {stats.total_requests > 0
            ? ((stats.fallback_count / stats.total_requests) * 100).toFixed(1)
            : '0'}%
        </div>
      </div>
    </div>
  )
}
