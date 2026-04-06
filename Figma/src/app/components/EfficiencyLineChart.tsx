import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts';

interface EfficiencyLineChartProps {
  data: Array<{
    case: string;
    score: number;
  }>;
}

export function EfficiencyLineChart({ data }: EfficiencyLineChartProps) {
  return (
    <ResponsiveContainer width="100%" height={300}>
      <LineChart data={data}>
        <CartesianGrid key="grid" strokeDasharray="3 3" stroke="#e5e7eb" />
        <XAxis key="xaxis" dataKey="case" tick={{ fill: '#6b7280', fontSize: 12 }} />
        <YAxis key="yaxis" tick={{ fill: '#6b7280', fontSize: 12 }} domain={[0, 100]} label={{ value: 'Score %', angle: -90, position: 'insideLeft', fill: '#6b7280' }} />
        <Tooltip key="tooltip" />
        <Line key="score" type="monotone" dataKey="score" stroke="#22c55e" strokeWidth={2} dot={{ fill: '#22c55e', r: 4 }} name="Efficiency Score" />
      </LineChart>
    </ResponsiveContainer>
  );
}
