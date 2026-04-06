export const physicians = [
  'Dr. Patel',
  'Dr. Lopez',
  'Dr. Chen',
  'Dr. Hernandez',
];

export const caseComplexityOptions = ['Low', 'Medium', 'High'];
export const dateRangeOptions = ['Last 7 Days', 'Last 30 Days', 'This Quarter'];

export const phaseData = [
  { name: 'Prep & Setup', productive: 18, nonValue: 7 },
  { name: 'Mapping', productive: 27, nonValue: 6 },
  { name: 'Ablation', productive: 34, nonValue: 11 },
  { name: 'Verification', productive: 14, nonValue: 4 },
  { name: 'Closure', productive: 12, nonValue: 6 },
];

export const delayData = [
  { transition: 'Prep → Mapping', avgDelay: 5.1, longestDelay: 12.0 },
  { transition: 'Mapping → Ablation', avgDelay: 7.5, longestDelay: 15.2 },
  { transition: 'Ablation → Verification', avgDelay: 4.3, longestDelay: 9.5 },
  { transition: 'Verification → Closure', avgDelay: 6.5, longestDelay: 11.8 },
];

export const caseTimeData = [
  { case: 'Case 1', productive: 36, nonValue: 9 },
  { case: 'Case 2', productive: 38, nonValue: 11 },
  { case: 'Case 3', productive: 35, nonValue: 12 },
  { case: 'Case 4', productive: 40, nonValue: 10 },
  { case: 'Case 5', productive: 34, nonValue: 8 },
  { case: 'Case 6', productive: 37, nonValue: 9 },
];

export const efficiencyScoreData = [
  { case: 'Case 1', score: 76 },
  { case: 'Case 2', score: 79 },
  { case: 'Case 3', score: 74 },
  { case: 'Case 4', score: 82 },
  { case: 'Case 5', score: 78 },
  { case: 'Case 6', score: 80 },
];

export const dashboardSummary = {
  efficiencyScore: 79,
  avgCaseDuration: 43,
  avgTransitionDelay: 5.8,
  casesAnalyzed: 24,
};
