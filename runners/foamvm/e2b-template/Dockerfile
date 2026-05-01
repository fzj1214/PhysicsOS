FROM fzj1214/openfoam-env:latest

# Install git and xz-utils via apt (uses HTTP apt sources, no curl/SSL needed)
RUN apt-get update && \
    apt-get install -y git xz-utils && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Download Node.js 20 LTS via Python's urllib
# (Python's SSL is compiled into its own binary, bypasses broken system curl)
RUN python3 -c "\
import urllib.request; \
urllib.request.urlretrieve( \
  'https://nodejs.org/dist/v20.19.0/node-v20.19.0-linux-x64.tar.xz', \
  '/tmp/node.tar.xz')"

# Extract Node.js into /usr/local
RUN tar -xJf /tmp/node.tar.xz -C /usr/local --strip-components=1 && \
    rm /tmp/node.tar.xz && \
    node --version && npm --version

# Install Claude Code CLI
RUN npm install -g @anthropic-ai/claude-code && \
    claude --version || true

# Create /workspace with open permissions so any sandbox user can write
RUN mkdir -p /workspace && chmod 777 /workspace

WORKDIR /workspace
CMD ["/bin/bash"]
