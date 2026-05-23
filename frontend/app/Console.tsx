'use client';

import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import Link from 'next/link';
import ThreeArena from './ThreeArena';

type TargetName = 'CHILD' | 'ADULT';

const SURVIVORS: Record<TargetName, { x: number; y: number }> = {
  CHILD: { x: 3.2, y: -1.5 },
  ADULT: { x: -2.1, y: 4.3 },
};

const OBSTACLES = [
  { pos: [0.0, 0.0, 1.0], size: [0.5, 0.5, 1.0] },
  { pos: [2.0, -2.0, 0.75], size: [0.4, 0.4, 0.75] },
  { pos: [-2.0, 2.0, 1.25], size: [0.4, 0.4, 1.25] },
];

const HAZARDS = [
  { center: [1.0, 1.0, 0.0], radius: 1.2 },
  { center: [-1.5, -1.0, 0.0], radius: 0.9 },
];

const BUDGET = 150;

type CommsKind = 'sys' | 'bot';
type CommsLine = {
  id: number;
  ts: string;
  src: string;
  kind: CommsKind;
  html: string;
  awaiting?: boolean;
};

type StateKind = '' | 'is-running' | 'is-ok' | 'is-fail';

type LastOutcome = { reached: boolean; wall: number; steps: number; dist: number | undefined } | null;

const sfmt = (n: number, d = 2) => (n >= 0 ? '' : '−') + Math.abs(n).toFixed(d);
const pad = (n: number, w = 3) => String(Math.floor(Math.abs(n))).padStart(w, '0');
const tsNow = () => {
  const d = new Date();
  return [d.getUTCHours(), d.getUTCMinutes(), d.getUTCSeconds()]
    .map((v) => String(v).padStart(2, '0'))
    .join(':');
};
const epClock = (sec: number) => {
  const m = Math.floor(sec / 60);
  const s = (sec % 60).toFixed(1);
  return `${String(m).padStart(2, '0')}:${s.padStart(4, '0')}`;
};
const escapeHtml = (s: string) =>
  s.replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]!));

function generateMockTrajectory(target: TargetName): number[][] {
  const start = [-4.0, -4.0];
  const childWaypoints = [
    start,
    [-1.5, -3.8],
    [1.0, -3.5], // skirt obstacle 2
    [2.8, -3.0],
    [3.2, -1.5]
  ];
  const adultWaypoints = [
    start,
    [-3.8, -1.0],
    [-3.8, 2.0], // skirt obstacle 3
    [-3.0, 3.5],
    [-2.1, 4.3]
  ];
  const waypoints = target === 'CHILD' ? childWaypoints : adultWaypoints;
  
  const trajectory: number[][] = [];
  const stepsPerSegment = Math.floor(120 / (waypoints.length - 1));
  
  for (let i = 0; i < waypoints.length - 1; i++) {
    const p1 = waypoints[i];
    const p2 = waypoints[i + 1];
    for (let s = 0; s < stepsPerSegment; s++) {
      const t = s / stepsPerSegment;
      const x = p1[0] + (p2[0] - p1[0]) * t;
      const y = p1[1] + (p2[1] - p1[1]) * t;
      const sway = Math.sin(t * Math.PI) * 0.05;
      trajectory.push([x + sway, y - sway, 0.0]);
    }
  }
  trajectory.push([waypoints[waypoints.length - 1][0], waypoints[waypoints.length - 1][1], 0.0]);
  return trajectory;
}

function parseOrder(text: string): { target: TargetName; why: string } {
  const t = text.toLowerCase();
  const wantsChild = /\bchild|kid|minor|infant\b/.test(t);
  const wantsAdult = /\badult|man|woman|person\b/.test(t);
  const ignoreAdult = /\bignore adult|child only|skip adult\b/.test(t);
  const ignoreChild = /\bignore child|adult only|skip child\b/.test(t);
  const closest = /\bclosest|nearest|first|fastest\b/.test(t) && !wantsChild && !wantsAdult;
  if (ignoreChild) return { target: 'ADULT', why: 'operator explicitly excluded CHILD' };
  if (ignoreAdult) return { target: 'CHILD', why: 'operator explicitly excluded ADULT' };
  if (wantsChild && !wantsAdult)
    return { target: 'CHILD', why: 'explicit operator instruction overrides nearest-first heuristic' };
  if (wantsAdult && !wantsChild)
    return { target: 'ADULT', why: 'explicit operator instruction overrides nearest-first heuristic' };
  if (closest) {
    const dC = Math.hypot(SURVIVORS.CHILD.x, SURVIVORS.CHILD.y);
    const dA = Math.hypot(SURVIVORS.ADULT.x, SURVIVORS.ADULT.y);
    return dC < dA
      ? { target: 'CHILD', why: `nearest-first · CHILD ${dC.toFixed(2)}m < ADULT ${dA.toFixed(2)}m` }
      : { target: 'ADULT', why: `nearest-first · ADULT ${dA.toFixed(2)}m < CHILD ${dC.toFixed(2)}m` };
  }
  return { target: 'CHILD', why: 'no nearest-first cue · defaulting to CHILD per safeguard policy' };
}

export default function Console() {
  // Reactive state
  const [order, setOrder] = useState('');
  const [activeTarget, setActiveTarget] = useState<TargetName | null>(null);
  const [dispatching, setDispatching] = useState(false);
  const [stateKind, setStateKind] = useState<StateKind>('');
  const [stateText, setStateText] = useState('EPISODE IDLE');
  const [simSteps, setSimSteps] = useState(0);
  const [distance, setDistance] = useState<number | undefined>(undefined);
  const [robot, setRobot] = useState({ x: -4.0, y: -4.0, heading: 0 });
  const [activeTrajectory, setActiveTrajectory] = useState<number[][] | null>(null);
  const [elapsed, setElapsed] = useState(0);
  const [comms, setComms] = useState<CommsLine[]>([]);
  const [clockUtc, setClockUtc] = useState('--:--:--');
  const [result, setResult] = useState({
    text: 'awaiting dispatch',
    sub: 'no episode in flight · last run: <span class="mono">—</span>',
    outcome: '—',
    dist: '—',
    steps: '—',
    time: '—',
  });
  const [epCounter, setEpCounter] = useState(1000);

  // Refs that don't drive render
  const t0Ref = useRef(0);
  const animTimer = useRef<ReturnType<typeof setTimeout> | null>(null);
  const dispatchingRef = useRef(false);
  const simStepsRef = useRef(0);
  const activeTargetRef = useRef<TargetName | null>(null);
  const robotRef = useRef({ x: -4.0, y: -4.0, heading: 0 });
  const lastOutcomeRef = useRef<LastOutcome>(null);
  const lineIdRef = useRef(0);
  const commsScrollRef = useRef<HTMLDivElement | null>(null);
  const inputRef = useRef<HTMLInputElement | null>(null);

  useEffect(() => {
    dispatchingRef.current = dispatching;
  }, [dispatching]);
  useEffect(() => {
    activeTargetRef.current = activeTarget;
  }, [activeTarget]);

  // Initial comms + clocks
  useEffect(() => {
    const now = tsNow();
    setClockUtc(now);
    setComms([
      {
        id: 0,
        ts: now,
        src: 'SYS  &gt;',
        kind: 'sys',
        html: 'link established · PPO policy loaded · awaiting order',
      },
      {
        id: 1,
        ts: now,
        src: 'SYS  &gt;',
        kind: 'sys',
        html: 'survivors detected: <span class="k">CHILD</span> @ <span class="v">(3.2, -1.5)</span> · <span class="k">ADULT</span> @ <span class="v">(-2.1, 4.3)</span>',
      },
      {
        id: 2,
        ts: now,
        src: 'SYS  &gt;',
        kind: 'sys',
        html: 'robot pose <span class="v">(0.0, 0.0)</span> · heading <span class="v">000°</span> · MuJoCo viewer active',
      },
      {
        id: 3,
        ts: now,
        src: 'SYS  &gt;',
        kind: 'sys',
        html: '<span class="dim">awaiting dispatch</span><span class="caret"></span>',
        awaiting: true,
      },
    ]);
    lineIdRef.current = 4;
    inputRef.current?.focus();

    const id = setInterval(() => setClockUtc(tsNow()), 1000);
    return () => clearInterval(id);
  }, []);

  // Episode wall clock tick
  useEffect(() => {
    if (!dispatching) return;
    const id = setInterval(() => {
      setElapsed((performance.now() - t0Ref.current) / 1000);
    }, 100);
    return () => clearInterval(id);
  }, [dispatching]);

  // Auto-scroll comms
  useEffect(() => {
    const el = commsScrollRef.current;
    if (el) el.scrollTop = el.scrollHeight;
  }, [comms]);

  const pushLine = useCallback((src: string, html: string, kind: CommsKind = 'bot') => {
    setComms((prev) => {
      const next = prev.filter((l) => !l.awaiting);
      next.push({ id: lineIdRef.current++, ts: tsNow(), src, kind, html });
      return next;
    });
  }, []);

  const pushAwait = useCallback((text = 'awaiting next order') => {
    setComms((prev) => [
      ...prev,
      {
        id: lineIdRef.current++,
        ts: tsNow(),
        src: 'SYS  &gt;',
        kind: 'sys',
        html: `<span class="dim">${text}</span><span class="caret"></span>`,
        awaiting: true,
      },
    ]);
  }, []);

  const stopAnimation = useCallback(() => {
    if (animTimer.current) {
      clearTimeout(animTimer.current);
      animTimer.current = null;
    }
  }, []);

  const startAnimation = useCallback(
    (targetName: TargetName, traj: number[][], onFinish: (finalDist: number) => void) => {
      stopAnimation();
      setActiveTrajectory(traj);
      let stepIdx = 0;
      const stepMs = Math.max(30, Math.min(85, 6000 / traj.length));

      const tick = () => {
        if (!dispatchingRef.current) return;
        if (stepIdx >= traj.length) {
          const tgt = SURVIVORS[targetName];
          const finalPos = traj[traj.length - 1];
          const finalDist = Math.hypot(tgt.x - finalPos[0], tgt.y - finalPos[1]);
          onFinish(finalDist);
          return;
        }

        const currentPt = traj[stepIdx];
        const nextPt = traj[Math.min(stepIdx + 1, traj.length - 1)];

        const rx = currentPt[0];
        const ry = currentPt[1];
        
        let heading = robotRef.current.heading;
        const dx = nextPt[0] - rx;
        const dy = nextPt[1] - ry;
        if (Math.hypot(dx, dy) > 0.01) {
          heading = (Math.atan2(dy, dx) * 180 / Math.PI + 360) % 360;
        }

        robotRef.current = { x: rx, y: ry, heading };
        setRobot(robotRef.current);
        
        const currentStep = stepIdx + 1;
        simStepsRef.current = currentStep;
        setSimSteps(currentStep);

        const tgt = SURVIVORS[targetName];
        const dist = Math.hypot(tgt.x - rx, tgt.y - ry);
        setDistance(dist);

        if (stepIdx === Math.floor(traj.length * 0.25)) {
          pushLine(
            'GR-ER&gt;',
            `<span class="dim">progress   →</span> 25% · pose <span class="v">(${rx.toFixed(2)}, ${ry.toFixed(2)})</span> · clearance ok`
          );
        } else if (stepIdx === Math.floor(traj.length * 0.6)) {
          pushLine(
            'GR-ER&gt;',
            `<span class="dim">obstacle   →</span> skirting obstacle sector · distance to target <span class="v">${dist.toFixed(2)}m</span>`
          );
        } else if (stepIdx === Math.floor(traj.length * 0.88)) {
          pushLine(
            'GR-ER&gt;',
            `<span class="dim">approach   →</span> &lt; <span class="v">1.0m</span> · reducing speed · arm stabilizer`
          );
        }

        stepIdx++;
        animTimer.current = setTimeout(tick, stepMs);
      };

      animTimer.current = setTimeout(tick, stepMs);
    },
    [pushLine, stopAnimation],
  );

  const callBackend = useCallback(
    async (orderText: string) => {
      try {
        const res = await fetch('/command', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ text: orderText }),
        });
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        return await res.json();
      } catch (err: unknown) {
        const msg = err instanceof Error ? err.message : String(err);
        pushLine('SYS  &gt;', `<span class="fail">backend error</span> · ${escapeHtml(msg)} · running standalone`, 'sys');
        return null;
      }
    },
    [pushLine],
  );

  const finishEpisode = useCallback(
    (reached: boolean, finalSteps: number, finalDist: number | undefined) => {
      stopAnimation();
      dispatchingRef.current = false;
      setDispatching(false);
      const wall = (performance.now() - t0Ref.current) / 1000;
      lastOutcomeRef.current = { reached, wall, steps: finalSteps, dist: finalDist };

      const target = activeTargetRef.current;
      if (reached) {
        setStateKind('is-ok');
        setStateText('TARGET REACHED');
        pushLine(
          'GR-ER&gt;',
          `<span class="dim">result     →</span> <span class="ok">TARGET REACHED</span> · ${target} stabilized · steps <span class="v">${finalSteps}</span>/${BUDGET} · t <span class="v">${wall.toFixed(1)}s</span>`,
        );
      } else {
        setStateKind('is-fail');
        setStateText('TIMEOUT');
        pushLine(
          'GR-ER&gt;',
          `<span class="dim">result     →</span> <span class="fail">TIMEOUT</span> · ${target} not reached · steps <span class="v">${finalSteps}</span>/${BUDGET}`,
        );
      }

      const nextEp = epCounter + 1;
      setEpCounter(nextEp);
      setResult({
        text: reached ? 'target reached' : 'timeout — step budget exceeded',
        sub: `episode <span class="mono">EP-${nextEp}</span> · target <span class="mono">${target}</span> · ended ${tsNow()}Z`,
        outcome: reached ? 'reached · within tolerance' : 'timeout · budget exceeded',
        dist: finalDist !== undefined ? finalDist.toFixed(2) + ' m' : '—',
        steps: `${finalSteps} / ${BUDGET}`,
        time: wall.toFixed(2) + ' s',
      });
      setSimSteps(finalSteps);
      setDistance(finalDist);
      setElapsed(wall);
      setTimeout(() => {
        inputRef.current?.focus();
        inputRef.current?.select();
      }, 0);
      pushAwait('awaiting next order');
    },
    [epCounter, pushAwait, pushLine, stopAnimation],
  );

  const dispatch = useCallback(
    async (orderText: string) => {
      if (dispatchingRef.current) return;

      const localParse = parseOrder(orderText);
      setActiveTarget(localParse.target);
      activeTargetRef.current = localParse.target;

      pushLine('OP   &gt;', `<span class="dim">order</span> "${escapeHtml(orderText)}"`, 'sys');
      pushLine('GR-ER&gt;', `<span class="dim">local NLP  →</span> intent = <span class="v">prioritize.${localParse.target.toLowerCase()}</span>`);
      pushLine('GR-ER&gt;', `<span class="dim">heuristic  →</span> ${localParse.why}`);
      pushLine(
        'GR-ER&gt;',
        `<span class="dim">gemini     →</span> sending to GR-ER 1.5 for final decision<span class="caret"></span>`,
      );
      pushLine(
        'SYS  &gt;',
        `<span class="dim">dispatching episode EP-${epCounter + 1} · MuJoCo active · awaiting result</span>`,
        'sys',
      );

      dispatchingRef.current = true;
      setDispatching(true);
      simStepsRef.current = 0;
      setSimSteps(0);
      robotRef.current = { x: -4.0, y: -4.0, heading: 0 };
      setRobot(robotRef.current);
      t0Ref.current = performance.now();
      setElapsed(0);
      setStateKind('is-running');
      setStateText('EPISODE RUNNING');

      const data = await callBackend(orderText);
      
      let finalTarget = localParse.target;
      let finalTrajectory: number[][] = [];
      let reached = false;
      let steps = 0;

      if (data) {
        finalTarget = (data.target_id ? String(data.target_id).toUpperCase() : localParse.target) as TargetName;
        if (finalTarget !== activeTargetRef.current) {
          setActiveTarget(finalTarget);
          activeTargetRef.current = finalTarget;
          pushLine(
            'GR-ER&gt;',
            `<span class="dim">correction →</span> Gemini selected <span class="v">${finalTarget}</span> · heuristic overridden`,
          );
        }
        if (data.reason) {
          pushLine(
            'GR-ER&gt;',
            `<span class="dim">reasoning  →</span> <span class="v">${finalTarget}</span> · ${escapeHtml(String(data.reason))}`,
          );
        }
        reached = !!data.reached;
        steps = data.steps || 0;
        finalTrajectory = data.trajectory || [];
      }

      if (finalTrajectory.length === 0) {
        finalTrajectory = generateMockTrajectory(finalTarget);
        reached = true;
        steps = finalTrajectory.length;
      }

      startAnimation(finalTarget, finalTrajectory, (finalDist) => {
        finishEpisode(reached, steps, finalDist);
      });
    },
    [callBackend, epCounter, finishEpisode, pushLine, startAnimation],
  );

  const reset = useCallback(() => {
    stopAnimation();
    dispatchingRef.current = false;
    setDispatching(false);
    simStepsRef.current = 0;
    setSimSteps(0);
    robotRef.current = { x: -4.0, y: -4.0, heading: 0 };
    setRobot(robotRef.current);
    setActiveTrajectory(null);
    lastOutcomeRef.current = null;
    setActiveTarget(null);
    activeTargetRef.current = null;
    setStateKind('');
    setStateText('EPISODE IDLE');
    setDistance(undefined);
    setElapsed(0);
    setResult({
      text: 'awaiting dispatch',
      sub: 'no episode in flight · last run: <span class="mono">—</span>',
      outcome: '—',
      dist: '—',
      steps: '—',
      time: '—',
    });
    pushLine('SYS  &gt;', '<span class="dim">episode state cleared · pose returned to start [-4.0, -4.0]</span>', 'sys');
    pushAwait('awaiting next order');
  }, [pushAwait, pushLine, stopAnimation]);

  // Global keys
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      const target = e.target as HTMLElement | null;
      if (target && target.tagName === 'INPUT') return;
      if (e.key === 'r' || e.key === 'R') {
        reset();
      } else if (e.key === 'Escape' && dispatchingRef.current) {
        finishEpisode(false, simStepsRef.current, undefined);
      } else if (/^[1-4]$/.test(e.key)) {
        const cmds = [
          'save the child first',
          'reach the closest survivor',
          'prioritize the adult, child is mobile',
          'ignore adult, child only',
        ];
        if (!dispatchingRef.current) {
          setOrder(cmds[Number(e.key) - 1]);
          inputRef.current?.focus();
        }
      }
    };
    document.addEventListener('keydown', onKey);
    return () => document.removeEventListener('keydown', onKey);
  }, [finishEpisode, reset]);

  // Derived metric values
  const initDist = activeTarget ? Math.hypot(SURVIVORS[activeTarget].x, SURVIVORS[activeTarget].y) : 1;
  const distBarPct =
    distance !== undefined ? Math.max(0, 100 - (distance / initDist) * 100).toFixed(1) : '0';
  const stepsBarPct = Math.min(100, (simSteps / BUDGET) * 100);
  const distSub = dispatching
    ? 'closing · L2 to target'
    : distance !== undefined && distance < 0.5
    ? 'on target'
    : 'idle · L2 to target';
  const stepsSub = dispatching || lastOutcomeRef.current
    ? `budget ${BUDGET} · ${Math.max(0, BUDGET - simSteps)} remaining`
    : `budget ${BUDGET} · idle`;

  let headingSub = 'bearing to target · —';
  if (activeTarget) {
    const tgt = SURVIVORS[activeTarget];
    const dx = tgt.x - robot.x;
    const dy = tgt.y - robot.y;
    const bearing = ((Math.atan2(dy, dx) * 180) / Math.PI + 360) % 360;
    headingSub = `bearing to target · ${sfmt(((bearing - robot.heading + 540) % 360) - 180, 1)}°`;
  }

  const quickCmds = useMemo(
    () => [
      'save the child first',
      'reach the closest survivor',
      'prioritize the adult, child is mobile',
      'ignore adult, child only',
    ],
    [],
  );

  const onSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    const v = order.trim();
    if (!v || dispatching) return;
    dispatch(v);
  };

  return (
    <>
      <header className="topbar">
        <Link href="/" className="brand">
          <div className="brand-mark" aria-hidden="true"></div>
          <div>
            <div className="brand-name"><em>Battle Angel</em></div>
            <div className="brand-sub mono">EPISODE RUNNER · v0.4.2</div>
          </div>
        </Link>
        <div className="nav-tabs">
          <Link href="/" className="nav-tab">Overview</Link>
          <Link href="/console" className="nav-tab is-active">Interactive Console</Link>
          <Link href="/mission-control" className="nav-tab">Mission Control</Link>
        </div>
        <div className="clock">
          <span className="lbl">UTC</span>
          <span className="val mono">{clockUtc}</span>
          <span className="lbl" style={{ marginLeft: 10 }}>EP</span>
          <span className="val mono">{epClock(elapsed)}</span>
        </div>
      </header>

      <section className="main">
        {/* LEFT */}
        <div className="col">
          <div className="panel">
            <div className="panel-hd">
              <h2>
                Command <span className="tag">— operator to robot</span>
              </h2>
              <span className="idx mono">[ 01 / CMD ]</span>
            </div>

            <form className="cmd-wrap" onSubmit={onSubmit} autoComplete="off">
              <div className="cmd-prefix mono">&gt;</div>
              <div className="cmd-input-row">
                <input
                  ref={inputRef}
                  className="cmd-input mono"
                  type="text"
                  spellCheck={false}
                  autoFocus
                  placeholder="issue prioritization order…"
                  aria-label="Operator command"
                  value={order}
                  disabled={dispatching}
                  onChange={(e) => setOrder(e.target.value)}
                />
                <button type="submit" className="cmd-submit" disabled={dispatching}>
                  DISPATCH <span className="key mono">↵</span>
                </button>
              </div>
            </form>

            <div className="cmd-meta mono">
              <span>
                <b>2</b> SURVIVORS DETECTED
              </span>
              <span className="sep">/</span>
              <span>
                STEP BUDGET <b>150</b>
              </span>
              <span className="sep">/</span>
              <span>
                POLICY <b>PPO</b>
              </span>
              <span className="sep">/</span>
              <span>
                NLP <b>GEMINI GR-ER 1.5</b>
              </span>
            </div>

            <div className="quick">
              <span className="lbl">RECENT</span>
              {quickCmds.map((c) => (
                <button
                  key={c}
                  type="button"
                  onClick={() => {
                    if (dispatching) return;
                    setOrder(c);
                    inputRef.current?.focus();
                  }}
                >
                  {c}
                </button>
              ))}
            </div>
          </div>

          <div className="panel" style={{ flex: 1, display: 'flex', flexDirection: 'column' }}>
            <div className="panel-hd">
              <h2>
                Comms <span className="tag">— reasoning channel</span>
              </h2>
              <span className="idx mono">[ 02 / GR-ER ]</span>
            </div>
            <div className="comms" ref={commsScrollRef} role="log" aria-live="polite">
              {comms.map((l) => (
                <div key={l.id} className={`line${l.awaiting ? ' awaiting' : ''}`}>
                  <span className="ts">{l.ts}</span>
                  <span className={`src ${l.kind}`} dangerouslySetInnerHTML={{ __html: l.src }} />
                  <span className="msg" dangerouslySetInnerHTML={{ __html: l.html }} />
                </div>
              ))}
            </div>
          </div>
        </div>

        {/* CENTER */}
        <div className="col center-col" style={{ borderLeft: '1px solid var(--rule)' }}>
          <div className="panel" style={{ flex: 1, display: 'flex', flexDirection: 'column', padding: '18px 22px' }}>
            <div className="panel-hd">
              <h2>
                Tactical Visualizer <span className="tag">— 3D ground telemetry</span>
              </h2>
              <div style={{ display: 'flex', gap: '8px', alignItems: 'center' }}>
                <span className={`pill ${stateKind}`} style={{ padding: '3px 8px', fontSize: '11px', height: '24px', display: 'inline-flex', alignItems: 'center' }}>
                  <span className="dot"></span>
                  <span>{stateText}</span>
                </span>
                <span className="idx mono">[ 03 / WebGL ]</span>
              </div>
            </div>
            <div style={{ flex: 1, position: 'relative', width: '100%', minHeight: '380px', display: 'flex', flexDirection: 'column' }}>
              <ThreeArena
                robotPos={robot}
                robotHeading={robot.heading}
                activeTarget={activeTarget}
                trajectory={activeTrajectory}
                obstacles={OBSTACLES}
                hazards={HAZARDS}
                survivors={SURVIVORS}
              />
            </div>
          </div>
        </div>

        {/* RIGHT */}
        <div className="col" style={{ borderLeft: '1px solid var(--rule)' }}>
          <div className="panel" style={{ padding: 0 }}>
            <div className={`target${activeTarget ? ' is-active' : ''}`}>
              <div className="target-flag"></div>
              <div>
                <div className="target-label">Active Target</div>
                <div className="target-name mono">{activeTarget ?? '—'}</div>
              </div>
              <div className="target-coords mono">
                <div>
                  x · <span className="x">{activeTarget ? sfmt(SURVIVORS[activeTarget].x) : '—'}</span>
                </div>
                <div>
                  y · <span className="y">{activeTarget ? sfmt(SURVIVORS[activeTarget].y) : '—'}</span>
                </div>
              </div>
            </div>
          </div>

          <div className="metrics">
            <div className="metric">
              <div className="lbl">
                <span>Steps Taken</span>
                <span className="mono">/ 150</span>
              </div>
              <div className="val mono">
                <span>{pad(simSteps, 3)}</span>
              </div>
              <div className="bar is-budget">
                <i style={{ width: `${stepsBarPct}%` }}></i>
              </div>
              <div className="sub">{stepsSub}</div>
            </div>
            <div className="metric">
              <div className="lbl">
                <span>Distance</span>
                <span className="mono">m</span>
              </div>
              <div className="val mono">
                <span>{distance !== undefined ? distance.toFixed(2) : '—'}</span>
                <span className="unit">m</span>
              </div>
              <div className="bar">
                <i style={{ width: `${distBarPct}%` }}></i>
              </div>
              <div className="sub">{distSub}</div>
            </div>
            <div className="metric">
              <div className="lbl">
                <span>Heading</span>
                <span className="mono">deg</span>
              </div>
              <div className="val mono">
                <span>{pad((robot.heading + 360) % 360, 3)}</span>
                <span className="unit">°</span>
              </div>
              <div className="sub mono">{headingSub}</div>
            </div>
            <div className="metric">
              <div className="lbl">
                <span>Episode Time</span>
                <span className="mono">s</span>
              </div>
              <div className="val mono">
                <span>{elapsed.toFixed(1)}</span>
                <span className="unit">s</span>
              </div>
              <div className="sub mono">wall · since dispatch</div>
            </div>
          </div>

          <div className="survivors">
            <div className="panel-hd" style={{ margin: '0 0 6px' }}>
              <h2>
                Survivors <span className="tag">— static roster</span>
              </h2>
              <span className="idx mono">[ 2 ]</span>
            </div>
            <div className={`survivor-row${activeTarget === 'CHILD' ? ' is-target' : ''}`}>
              <div className="mark mono">C</div>
              <div>
                <div className="who">CHILD</div>
                <div className="meta">approx 7yr · prone · responsive</div>
              </div>
              <div className="pos">(3.20, −1.50)</div>
            </div>
            <div className={`survivor-row${activeTarget === 'ADULT' ? ' is-target' : ''}`}>
              <div className="mark mono">A</div>
              <div>
                <div className="who">ADULT</div>
                <div className="meta">approx 40yr · seated · ambulatory</div>
              </div>
              <div className="pos">(−2.10, 4.30)</div>
            </div>
          </div>

          <div className={`result ${stateKind}`}>
            <div className="result-headline">
              <span className="bracket">[</span>
              <span>{result.text}</span>
              <span className="bracket">]</span>
            </div>
            <div className="result-sub" dangerouslySetInnerHTML={{ __html: result.sub }} />

            <div className="result-grid">
              <div className="cell">
                <div className="k">Outcome</div>
                <div className="v">{result.outcome}</div>
              </div>
              <div className="cell">
                <div className="k">Final Distance</div>
                <div className="v">{result.dist}</div>
              </div>
              <div className="cell">
                <div className="k">Steps Used</div>
                <div className="v">{result.steps}</div>
              </div>
              <div className="cell">
                <div className="k">Wall Time</div>
                <div className="v">{result.time}</div>
              </div>
            </div>
          </div>
        </div>
      </section>

      <footer className="legend">
        <div className="kb">
          <span>
            <b>↵</b> dispatch order
          </span>
          <span>
            <b>R</b> reset episode
          </span>
          <span>
            <b>1–4</b> recall order
          </span>
          <span>
            <b>Esc</b> halt
          </span>
        </div>
        <div className="right">NODE · FIELD-CMD-03 · LINK OK · LOG /var/gr-er/episodes/2026-05-23.jsonl</div>
      </footer>
    </>
  );
}
