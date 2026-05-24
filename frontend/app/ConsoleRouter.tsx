'use client';

import { useEffect, useState } from 'react';
import Console from './Console';
import GeneratedConsole from './GeneratedConsole';
import { loadWorkflowSession } from './workflowSession';

export default function ConsoleRouter({ sceneIdFromUrl }: { sceneIdFromUrl?: string }) {
  const [sceneId, setSceneId] = useState<string | undefined>(sceneIdFromUrl);

  useEffect(() => {
    if (sceneIdFromUrl) {
      setSceneId(sceneIdFromUrl);
      return;
    }
    const active = loadWorkflowSession().activeSceneId;
    if (active) setSceneId(active);
  }, [sceneIdFromUrl]);

  if (sceneId) return <GeneratedConsole sceneId={sceneId} />;
  return <Console />;
}
