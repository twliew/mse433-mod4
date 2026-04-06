interface StackedBarChartProps {
  phases: Array<{
    name: string;
    productive: number;
    nonValue: number;
  }>;
}

export function StackedBarChart({ phases }: StackedBarChartProps) {
  return (
    <div className="space-y-4">
      {phases.map((phase) => {
        const total = phase.productive + phase.nonValue;
        const productivePercent = (phase.productive / total) * 100;
        const nonValuePercent = (phase.nonValue / total) * 100;

        return (
          <div key={phase.name} className="space-y-2">
            <div className="flex justify-between text-sm text-gray-600">
              <span>{phase.name}</span>
              <span>{total.toFixed(1)}m</span>
            </div>
            <div className="flex h-8 rounded overflow-hidden">
              <div
                className="bg-blue-500 flex items-center justify-center text-white text-xs"
                style={{ width: `${productivePercent}%` }}
              >
                {productivePercent > 15 && `${productivePercent.toFixed(0)}%`}
              </div>
              <div
                className="bg-red-500 flex items-center justify-center text-white text-xs"
                style={{ width: `${nonValuePercent}%` }}
              >
                {nonValuePercent > 15 && `${nonValuePercent.toFixed(0)}%`}
              </div>
            </div>
          </div>
        );
      })}
      <div className="flex gap-4 pt-2 text-sm">
        <div className="flex items-center gap-2">
          <div className="w-3 h-3 bg-blue-500 rounded"></div>
          <span className="text-gray-600">Productive Time</span>
        </div>
        <div className="flex items-center gap-2">
          <div className="w-3 h-3 bg-red-500 rounded"></div>
          <span className="text-gray-600">Non-Value-Added</span>
        </div>
      </div>
    </div>
  );
}
