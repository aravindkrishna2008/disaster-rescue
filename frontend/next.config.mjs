/** @type {import('next').NextConfig} */
const nextConfig = {
  experimental: {
    // Episode rollouts can take several minutes; avoid dev-proxy timeouts.
    proxyTimeout: 300_000,
  },
  async rewrites() {
    const backend = process.env.BACKEND_URL || 'http://127.0.0.1:8000';
    return [
      { source: '/command', destination: `${backend}/command` },
      { source: '/scenes', destination: `${backend}/scenes` },
      { source: '/scene/:idx/run', destination: `${backend}/scene/:idx/run` },
      { source: '/gifs/:path*', destination: `${backend}/gifs/:path*` },
      { source: '/runs', destination: `${backend}/runs` },
      { source: '/runs/cancel', destination: `${backend}/runs/cancel` },
      { source: '/runs/:id/eval_live', destination: `${backend}/runs/:id/eval_live` },
      { source: '/runs/:id/episodes', destination: `${backend}/runs/:id/episodes` },
      { source: '/runs/:id/episodes/:scene', destination: `${backend}/runs/:id/episodes/:scene` },
      { source: '/runs/:id/episodes/:scene/replay', destination: `${backend}/runs/:id/episodes/:scene/replay` },
      { source: '/runs/:id', destination: `${backend}/runs/:id` },
      { source: '/train', destination: `${backend}/train` },
      { source: '/train/status', destination: `${backend}/train/status` },
      { source: '/train/cancel', destination: `${backend}/train/cancel` },
      { source: '/run-gifs/:path*', destination: `${backend}/run-gifs/:path*` },
      { source: '/health', destination: `${backend}/health` },
      { source: '/generate-scene', destination: `${backend}/generate-scene` },
      { source: '/generate-prompt', destination: `${backend}/generate-prompt` },
      { source: '/generated-scenes/:sceneId/run', destination: `${backend}/generated-scenes/:sceneId/run` },
      { source: '/generated-scenes/:sceneId', destination: `${backend}/generated-scenes/:sceneId` },
    ];
  },
};

export default nextConfig;
