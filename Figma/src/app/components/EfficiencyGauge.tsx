interface EfficiencyGaugeProps {
  score: number;
}

export function EfficiencyGauge({ score }: EfficiencyGaugeProps) {
  const size = 200;
  const strokeWidth = 24;
  const radius = (size - strokeWidth * 2) / 2;
  const centerX = size / 2;
  const centerY = size / 2 + strokeWidth / 2;

  const getColor = (score: number) => {
    if (score >= 80) return '#22c55e';
    if (score >= 60) return '#eab308';
    return '#ef4444';
  };

  const color = getColor(score);

  // Convert angle to coordinates
  // Start at 180° (left), end at 0° (right) for a semicircle
  const polarToCartesian = (angleInDegrees: number) => {
    const angleInRadians = ((angleInDegrees - 90) * Math.PI) / 180;
    return {
      x: centerX + radius * Math.cos(angleInRadians),
      y: centerY + radius * Math.sin(angleInRadians)
    };
  };

  const startPoint = polarToCartesian(180); // Left side
  const endPoint = polarToCartesian(0);     // Right side

  // Background semicircle from 180° to 0° (counterclockwise, sweepFlag = 1)
  const backgroundPath = `M ${startPoint.x},${startPoint.y} A ${radius},${radius} 0 0,1 ${endPoint.x},${endPoint.y}`;

  // Score arc: starts at 180° and goes counterclockwise based on score percentage
  const scoreAngleDegrees = 180 - (score / 100) * 180;
  const scoreEndPoint = polarToCartesian(scoreAngleDegrees);
  const largeArcFlag = score > 50 ? 1 : 0;
  const scorePath = `M ${startPoint.x},${startPoint.y} A ${radius},${radius} 0 ${largeArcFlag},1 ${scoreEndPoint.x},${scoreEndPoint.y}`;

  return (
    <div className="flex flex-col items-center justify-center py-6">
      <div className="relative" style={{ width: size, height: size / 2 + 60 }}>
        <svg
          width={size}
          height={size / 2 + strokeWidth}
          viewBox={`0 0 ${size} ${size / 2 + strokeWidth}`}
        >
          {/* Background arc */}
          <path
            d={backgroundPath}
            fill="none"
            stroke="#e5e7eb"
            strokeWidth={strokeWidth}
            strokeLinecap="round"
          />
          {/* Score arc */}
          <path
            d={scorePath}
            fill="none"
            stroke={color}
            strokeWidth={strokeWidth}
            strokeLinecap="round"
          />
        </svg>
        <div className="absolute inset-0 flex items-center justify-center" style={{ paddingTop: '30px' }}>
          <div className="text-center">
            <div className="text-5xl mb-1" style={{ color }}>{score}%</div>
            <div className="text-sm text-gray-500">Efficiency Score</div>
          </div>
        </div>
      </div>
      <div className="flex gap-4 mt-2 text-xs flex-wrap justify-center">
        <div className="flex items-center gap-1">
          <div className="w-3 h-3 bg-green-500 rounded"></div>
          <span className="text-gray-600">Excellent (80+)</span>
        </div>
        <div className="flex items-center gap-1">
          <div className="w-3 h-3 bg-yellow-500 rounded"></div>
          <span className="text-gray-600">Good (60-79)</span>
        </div>
        <div className="flex items-center gap-1">
          <div className="w-3 h-3 bg-red-500 rounded"></div>
          <span className="text-gray-600">Needs Improvement (&lt;60)</span>
        </div>
      </div>
    </div>
  );
}
