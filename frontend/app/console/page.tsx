import ConsoleRouter from '../ConsoleRouter';

export default function ConsolePage({
  searchParams,
}: {
  searchParams?: { scene_id?: string };
}) {
  return <ConsoleRouter sceneIdFromUrl={searchParams?.scene_id} />;
}
