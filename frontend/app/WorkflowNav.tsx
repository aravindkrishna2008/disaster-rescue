'use client';

import Link from 'next/link';
import { useEffect, useState } from 'react';
import { consoleHref, loadWorkflowSession, subscribeWorkflowSession } from './workflowSession';

type WorkflowPage = 'overview' | 'gym' | 'generate' | 'console';

const STEPS: Array<{ id: WorkflowPage; href: string | ((sceneId: string | null) => string); label: string }> = [
  { id: 'overview', href: '/', label: 'Overview' },
  { id: 'gym', href: '/mission-control', label: 'Training Gym' },
  { id: 'generate', href: '/generate', label: 'Scene Generator' },
  { id: 'console', href: (sceneId) => consoleHref(sceneId), label: 'Interactive Console' },
];

export default function WorkflowNav({ active }: { active: WorkflowPage }) {
  const [activeSceneId, setActiveSceneId] = useState<string | null>(null);

  useEffect(() => {
    setActiveSceneId(loadWorkflowSession().activeSceneId);
    return subscribeWorkflowSession(() => {
      setActiveSceneId(loadWorkflowSession().activeSceneId);
    });
  }, []);

  return (
    <nav className="nav-tabs" aria-label="Battle Angel workflow">
      {STEPS.map((item) => {
        const href = typeof item.href === 'function' ? item.href(activeSceneId) : item.href;
        return (
          <Link
            key={item.id}
            href={href}
            className={`nav-tab${active === item.id ? ' is-active' : ''}`}
            aria-current={active === item.id ? 'page' : undefined}
          >
            {item.label}
          </Link>
        );
      })}
    </nav>
  );
}
