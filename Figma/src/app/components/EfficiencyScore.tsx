interface EfficiencyScoreProps {
  score: number;
}

export function EfficiencyScore({ score }: EfficiencyScoreProps) {
  const getColor = (score: number) => {
    if (score >= 80) return '#22c55e';
    if (score >= 60) return '#eab308';
    return '#ef4444';
  };

  const getLabel = (score: number) => {
    if (score >= 80) return 'Excellent';
    if (score >= 60) return 'Good';
    return 'Needs Improvement';
  };

  const color = getColor(score);
  const label = getLabel(score);

  return (
    <div className="flex flex-col items-center justify-center py-12">
      <div className="text-8xl mb-4" style={{ color }}>
        {score}%
      </div>
      <div className="text-xl text-gray-600 mb-6">{label}</div>
      <div className="flex gap-6 text-sm">
        <div className="flex items-center gap-2">
          <div className="w-3 h-3 bg-green-500 rounded-full"></div>
          <span className="text-gray-600">Excellent (80+)</span>
        </div>
        <div className="flex items-center gap-2">
          <div className="w-3 h-3 bg-yellow-500 rounded-full"></div>
          <span className="text-gray-600">Good (60-79)</span>
        </div>
        <div className="flex items-center gap-2">
          <div className="w-3 h-3 bg-red-500 rounded-full"></div>
          <span className="text-gray-600">&lt;60</span>
        </div>
      </div>
    </div>
  );
}
