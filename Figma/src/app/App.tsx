import { FilterButton } from './components/FilterButton';
import { MetricCard } from './components/MetricCard';
import { KPICard } from './components/KPICard';
import { StackedBarChart } from './components/StackedBarChart';
import { DelayChart } from './components/DelayChart';
import { CaseBarChart } from './components/CaseBarChart';
import { EfficiencyLineChart } from './components/EfficiencyLineChart';
import {
  physicians,
  caseComplexityOptions,
  dateRangeOptions,
  phaseData,
  delayData,
  caseTimeData,
  efficiencyScoreData,
  dashboardSummary,
} from './data/sampleData';

export default function App() {
  // Brand color
  const brandPurple = '#5B4B8A';

  const getEfficiencyColor = (score: number) => {
    if (score >= 80) return '#22c55e';
    if (score >= 60) return '#eab308';
    return '#ef4444';
  };

  const getEfficiencyLabel = (score: number) => {
    if (score >= 80) return 'Excellent';
    if (score >= 60) return 'Good';
    return 'Needs Improvement';
  };

  const avgCaseTime = caseTimeData.reduce((sum, c) => sum + c.productive + c.nonValue, 0) / caseTimeData.length;
  const avgDelay = delayData.reduce((sum, d) => sum + d.avgDelay, 0) / delayData.length;
  const efficiencyScore = dashboardSummary.efficiencyScore;

  return (
    <div className="min-h-screen bg-gray-50">
      <div className="px-8 py-4" style={{ background: 'linear-gradient(135deg, #5B4B8A 0%, #7B6BA8 100%)' }}>
        <div className="max-w-[1440px] mx-auto">
          <h1 className="text-white">AFib Ablation Efficiency Dashboard</h1>
          <p className="text-white/90 text-sm">EP Lab Performance Metrics</p>
        </div>
      </div>

      <div className="max-w-[1440px] mx-auto p-6">
        <div className="flex gap-4 mb-6">
          <FilterButton label={`Physician: ${physicians[0]}`} />
          <FilterButton label={`Complexity: ${caseComplexityOptions[1]}`} />
          <FilterButton label={`Range: ${dateRangeOptions[0]}`} />
        </div>

        <div className="grid grid-cols-4 gap-4 mb-6">
          <KPICard
            title="Overall Efficiency Score"
            value={`${efficiencyScore}%`}
            subtitle={getEfficiencyLabel(efficiencyScore)}
            color={getEfficiencyColor(efficiencyScore)}
            trend={{ value: 3.2, isPositive: true }}
          />
          <KPICard
            title="Avg Case Duration"
            value={`${avgCaseTime.toFixed(0)}m`}
            subtitle="Total procedure time"
            color={brandPurple}
          />
          <KPICard
            title="Avg Transition Delay"
            value={`${avgDelay.toFixed(1)}m`}
            subtitle="Between procedure steps"
            color={brandPurple}
          />
          <KPICard
            title="Cases Analyzed"
            value={caseTimeData.length}
            subtitle="Current filter period"
            color={brandPurple}
          />
        </div>

        <div className="mb-6">
          <h2 className="mb-3 text-base" style={{ color: brandPurple }}>Process Analysis</h2>
          <div className="grid grid-cols-2 gap-4">
            <MetricCard title="Time Lost Within Each Phase">
              <StackedBarChart phases={phaseData} />
            </MetricCard>

            <MetricCard title="Delay Between Steps">
              <DelayChart delays={delayData} />
            </MetricCard>
          </div>
        </div>

        <div>
          <h2 className="mb-3 text-base" style={{ color: brandPurple }}>Case-Level Trends</h2>
          <div className="grid grid-cols-2 gap-4">
            <MetricCard title="Time Lost Across Cases">
              <CaseBarChart data={caseTimeData} />
            </MetricCard>

            <MetricCard title="Efficiency Score Trend">
              <EfficiencyLineChart data={efficiencyScoreData} />
            </MetricCard>
          </div>
        </div>
      </div>
    </div>
  );
}