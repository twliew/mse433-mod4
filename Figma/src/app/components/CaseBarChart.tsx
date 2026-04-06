import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';

interface CaseBarChartProps {
  data: Array<{
    case: string;
    productive: number;
    nonValue: number;
  }>;
}

export function CaseBarChart({ data }: CaseBarChartProps) {
  return (
    <ResponsiveContainer width="100%" height={300}>
      <BarChart data={data}>
        <CartesianGrid key="grid" strokeDasharray="3 3" stroke="#e5e7eb" />
        <XAxis key="xaxis" dataKey="case" tick={{ fill: '#6b7280', fontSize: 12 }} />
        <YAxis key="yaxis" tick={{ fill: '#6b7280', fontSize: 12 }} label={{ value: 'Minutes', angle: -90, position: 'insideLeft', fill: '#6b7280' }} />
        <Tooltip key="tooltip" />
        <Legend key="legend" />
        <Bar key="productive" dataKey="productive" fill="#3b82f6" name="Productive Time" stackId="a" />
        <Bar key="nonValue" dataKey="nonValue" fill="#ef4444" name="Non-Value-Added" stackId="a" />
      </BarChart>
    </ResponsiveContainer>
  );
}
