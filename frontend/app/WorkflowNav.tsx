import Link from 'next/link';

type WorkflowPage = 'overview' | 'gym' | 'generate' | 'console';

const STEPS: Array<{ id: WorkflowPage; href: string; label: string; step?: string }> = [
  { id: 'overview', href: '/', label: 'Overview' },
  { id: 'gym', href: '/mission-control', label: 'Training Gym', step: '01' },
  { id: 'generate', href: '/generate', label: 'Scene Generator', step: '02' },
  { id: 'console', href: '/console', label: 'Interactive Console', step: '03' },
];

export default function WorkflowNav({ active }: { active: WorkflowPage }) {
  return (
    <nav className="nav-tabs" aria-label="Battle Angel workflow">
      {STEPS.map((item) => (
        <Link
          key={item.id}
          href={item.href}
          className={`nav-tab${active === item.id ? ' is-active' : ''}`}
          aria-current={active === item.id ? 'page' : undefined}
        >
          {item.label}
        </Link>
      ))}
    </nav>
  );
}
