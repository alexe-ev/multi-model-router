import { PieChart, Pie, Cell, Tooltip, ResponsiveContainer } from 'recharts'

interface ModelEntry {
  model: string
  count: number
  total_cost: number
  avg_latency_ms: number
}

interface Props {
  data: ModelEntry[]
}

const COLORS = ['#6366f1', '#22d3ee', '#f59e0b', '#ef4444', '#10b981']

function shortModelName(model: string): string {
  if (model.includes('haiku')) return 'Haiku'
  if (model.includes('sonnet')) return 'Sonnet'
  if (model.includes('opus')) return 'Opus'
  if (model.includes('gpt-4o-mini')) return 'GPT-4o-mini'
  if (model.includes('gpt-4o')) return 'GPT-4o'
  if (model.includes('gemini')) return 'Gemini'
  return model
}

export default function ModelDistribution({ data }: Props) {
  return (
    <div className="chart-card">
      <h2>Model Distribution</h2>
      {data.length === 0 ? (
        <div className="loading">No data</div>
      ) : (
        <ResponsiveContainer width="100%" height={250}>
          <PieChart>
            <Pie
              data={data}
              dataKey="count"
              nameKey="model"
              cx="50%"
              cy="50%"
              outerRadius={80}
              label={({ model, percent }) => `${shortModelName(model)} ${(percent * 100).toFixed(0)}%`}
              labelLine={false}
            >
              {data.map((_, i) => (
                <Cell key={i} fill={COLORS[i % COLORS.length]} />
              ))}
            </Pie>
            <Tooltip
              contentStyle={{ background: '#1a1d27', border: '1px solid #2a2d37' }}
              formatter={(value: number, name: string) => [value, name]}
            />
          </PieChart>
        </ResponsiveContainer>
      )}
    </div>
  )
}
