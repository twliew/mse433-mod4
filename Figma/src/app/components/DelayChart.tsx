interface DelayChartProps {
  delays: Array<{
    transition: string;
    avgDelay: number;
    longestDelay: number;
  }>;
}

export function DelayChart({ delays }: DelayChartProps) {
  const maxDelay = Math.max(...delays.map(d => d.longestDelay));

  return (
    <div className="space-y-4">
      {delays.map((delay) => {
        const barWidth = (delay.avgDelay / maxDelay) * 100;

        return (
          <div key={delay.transition} className="space-y-1">
            <div className="text-sm text-gray-700">{delay.transition}</div>
            <div className="flex items-center gap-3">
              <div className="flex-1">
                <div className="bg-gray-200 h-7 rounded overflow-hidden">
                  <div
                    className="bg-orange-400 h-full flex items-center px-2 text-white text-xs"
                    style={{ width: `${barWidth}%` }}
                  >
                    {delay.avgDelay.toFixed(1)}m
                  </div>
                </div>
              </div>
            </div>
            <div className="flex gap-4 text-xs text-gray-500">
              <span>Avg delay: {delay.avgDelay.toFixed(1)}m</span>
              <span>Longest: {delay.longestDelay.toFixed(1)}m</span>
            </div>
          </div>
        );
      })}
    </div>
  );
}
