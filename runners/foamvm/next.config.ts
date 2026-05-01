import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  // Allow large streaming responses for CFD runs
  experimental: {
    serverActions: {
      bodySizeLimit: '50mb',
    },
  },
};

export default nextConfig;
