import Console from '../Console';
import GeneratedConsole from '../GeneratedConsole';

export default function ConsolePage({
  searchParams,
}: {
  searchParams?: { scene_id?: string };
}) {
  const sceneId = searchParams?.scene_id;
  return sceneId ? <GeneratedConsole sceneId={sceneId} /> : <Console />;
}
