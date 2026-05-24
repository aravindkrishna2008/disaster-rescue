/** Cross-page workflow state (gym runs + active generated scene). */

export type GymRunRecord =
  | { status: 'idle' }
  | { status: 'done'; result: unknown; finishedAt: number; cacheBust: number }
  | { status: 'error'; message: string }
  | { status: 'cancelled' };

export type WorkflowSession = {
  activeSceneId: string | null;
  gymRuns: Record<string, GymRunRecord>;
};

const STORAGE_KEY = 'battle-angel-workflow-v1';
const WORKFLOW_EVENT = 'battle-angel-workflow';

const EMPTY: WorkflowSession = { activeSceneId: null, gymRuns: {} };

export function loadWorkflowSession(): WorkflowSession {
  if (typeof window === 'undefined') return EMPTY;
  try {
    const raw = window.localStorage.getItem(STORAGE_KEY);
    if (!raw) return EMPTY;
    const parsed = JSON.parse(raw) as Partial<WorkflowSession>;
    return {
      activeSceneId: typeof parsed.activeSceneId === 'string' ? parsed.activeSceneId : null,
      gymRuns: parsed.gymRuns && typeof parsed.gymRuns === 'object' ? parsed.gymRuns : {},
    };
  } catch {
    return EMPTY;
  }
}

export function saveWorkflowSession(next: WorkflowSession): void {
  if (typeof window === 'undefined') return;
  window.localStorage.setItem(STORAGE_KEY, JSON.stringify(next));
  window.dispatchEvent(new Event(WORKFLOW_EVENT));
}

export function patchWorkflowSession(patch: Partial<WorkflowSession>): WorkflowSession {
  const merged = { ...loadWorkflowSession(), ...patch };
  saveWorkflowSession(merged);
  return merged;
}

export function setActiveSceneId(sceneId: string | null): void {
  patchWorkflowSession({ activeSceneId: sceneId });
}

export function consoleHref(sceneId?: string | null): string {
  const id = sceneId ?? (typeof window !== 'undefined' ? loadWorkflowSession().activeSceneId : null);
  return id ? `/console?scene_id=${encodeURIComponent(id)}` : '/console';
}

export function subscribeWorkflowSession(listener: () => void): () => void {
  if (typeof window === 'undefined') return () => {};
  const handler = () => listener();
  window.addEventListener(WORKFLOW_EVENT, handler);
  window.addEventListener('storage', handler);
  return () => {
    window.removeEventListener(WORKFLOW_EVENT, handler);
    window.removeEventListener('storage', handler);
  };
}

export function serializeGymRuns(
  runs: Record<number, { status: string; result?: unknown; finishedAt?: number; cacheBust?: number; message?: string }>,
): Record<string, GymRunRecord> {
  const out: Record<string, GymRunRecord> = {};
  for (const [idx, state] of Object.entries(runs)) {
    if (state.status === 'done' && state.result) {
      out[idx] = {
        status: 'done',
        result: state.result,
        finishedAt: state.finishedAt ?? Date.now(),
        cacheBust: state.cacheBust ?? Date.now(),
      };
    } else if (state.status === 'error') {
      out[idx] = { status: 'error', message: state.message ?? 'Run failed' };
    } else if (state.status === 'cancelled') {
      out[idx] = { status: 'cancelled' };
    }
  }
  return out;
}

export function saveGymRuns(
  runs: Record<number, { status: string; result?: unknown; finishedAt?: number; cacheBust?: number; message?: string }>,
): void {
  patchWorkflowSession({ gymRuns: serializeGymRuns(runs) });
}

export function hydrateGymRuns(stored: Record<string, GymRunRecord>): Record<number, GymRunRecord> {
  const out: Record<number, GymRunRecord> = {};
  for (const [idx, state] of Object.entries(stored)) {
    const n = Number(idx);
    if (!Number.isNaN(n)) out[n] = state;
  }
  return out;
}
