import { ReactNode } from 'react';

interface MetricCardProps {
  title: string;
  children: ReactNode;
  className?: string;
}

export function MetricCard({ title, children, className = '' }: MetricCardProps) {
  return (
    <div className={`bg-white rounded-xl shadow-md p-4 ${className}`}>
      <h3 className="text-gray-800 mb-3 text-sm">{title}</h3>
      {children}
    </div>
  );
}