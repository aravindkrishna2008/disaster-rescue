/** @type {import('next').NextConfig} */
const nextConfig = {
  async rewrites() {
    const backend = process.env.BACKEND_URL || 'http://127.0.0.1:8000';
    return [
      { source: '/command', destination: `${backend}/command` },
    ];
  },
};

export default nextConfig;
