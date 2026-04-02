import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, Legend } from 'recharts'

interface DailyEntry {
  date: string
  model: string
  request_count: number
  total_cost: number
}

interface Props {
  data: DailyEntry[]
}

export default function CostChart({ data }: Props) {
  // Pivot: group by date, with cost per model as separate keys
  const models = [...new Set(data.map(d => d.model))]
  const byDate: Record<string, any> = {}
  for (const entry of data) {
    if (!byDate[entry.date]) byDate[entry.date] = { date: entry.date }
    byDate[entry.date][entry.model] = (byDate[entry.date][entry.model] || 0) + entry.total_cost
  }
  const chartData = Object.values(byDate).sort((a: any, b: any) => a.date.localeCompare(b.date))

  const colors = ['#6366f1', '#22d3ee', '#f59e0b', '#ef4444', '#10b981']

  return (
    <div className="chart-card">
      <h2>Daily Cost by Model</h2>
      {chartData.length === 0 ? (
        <div className="loading">No data</div>
      ) : (
        <ResponsiveContainer width="100%" height={250}>
          <BarChart data={chartData}>
            <XAxis dataKey="date" tick={{ fill: '#888', fontSize: 11 }} />
            <YAxis tick={{ fill: '#888', fontSize: 11 }} tickFormatter={v => `$${v}`} />
            <Tooltip
              contentStyle={{ background: '#1a1d27', border: '1px solid #2a2d37' }}
              formatter={(value: number) => [`$${value.toFixed(6)}`, '']}
            />
            <Legend />
            {models.map((model, i) => (
              <Bar key={model} dataKey={model} stackId="cost" fill={colors[i % colors.length]} />
            ))}
          </BarChart>
        </ResponsiveContainer>
      )}
    </div>
  )
}
