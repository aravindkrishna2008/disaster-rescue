/** @type {import('next').NextConfig} */
const nextConfig = {
  async rewrites() {
    const backend = process.env.BACKEND_URL || 'http://127.0.0.1:8000';
    return [
      { source: '/command', destination: `${backend}/command` },
      { source: '/scenes', destination: `${backend}/scenes` },
      { source: '/scene/:idx/run', destination: `${backend}/scene/:idx/run` },
      { source: '/gifs/:path*', destination: `${backend}/gifs/:path*` },
      { source: '/health', destination: `${backend}/health` },
    ];
  },
};

export default nextConfig;
