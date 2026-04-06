import { ChevronDown } from 'lucide-react';

interface FilterButtonProps {
  label: string;
}

export function FilterButton({ label }: FilterButtonProps) {
  return (
    <button className="flex items-center gap-2 px-6 py-3 bg-white border-2 rounded-lg transition-all hover:shadow-md" style={{ borderColor: '#5B4B8A', color: '#5B4B8A' }}>
      <span>{label}</span>
      <ChevronDown className="w-4 h-4" />
    </button>
  );
}
