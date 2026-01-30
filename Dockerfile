FROM node:22-bookworm

# Install Bun (required for build scripts)
RUN curl -fsSL https://bun.sh/install | bash
ENV PATH="/root/.bun/bin:${PATH}"

RUN corepack enable

WORKDIR /app

ARG CLAWDBOT_DOCKER_APT_PACKAGES=""
RUN if [ -n "$CLAWDBOT_DOCKER_APT_PACKAGES" ]; then \
      apt-get update && \
      DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends $CLAWDBOT_DOCKER_APT_PACKAGES && \
      apt-get clean && \
      rm -rf /var/lib/apt/lists/* /var/cache/apt/archives/*; \
    fi

COPY package.json pnpm-lock.yaml pnpm-workspace.yaml .npmrc ./
COPY ui/package.json ./ui/package.json
COPY patches ./patches
COPY scripts ./scripts

RUN pnpm install --frozen-lockfile

COPY . .
RUN CLAWDBOT_A2UI_SKIP_MISSING=1 pnpm build
# Force pnpm for UI build (Bun may fail on ARM/Synology architectures)
ENV CLAWDBOT_PREFER_PNPM=1
RUN echo "auto-install-peers=false" >> .npmrc && pnpm ui:install
RUN pnpm ui:build

ENV NODE_ENV=production

# Railway-specific: default gateway config (bind must be "lan" not "0.0.0.0")
RUN mkdir -p /root/.clawdbot && \
    echo '{"gateway":{"mode":"local","bind":"lan","trustedProxies":["100.64.0.0/10","10.0.0.0/8"],"controlUi":{"allowInsecureAuth":true}}}' > /root/.clawdbot/moltbot.json

# Fix stale config: delete old config and write fresh one
ENTRYPOINT ["sh", "-c", "mkdir -p /data/.clawdbot; rm -f /data/.clawdbot/moltbot.json; echo '{\"gateway\":{\"mode\":\"local\",\"bind\":\"lan\",\"trustedProxies\":[\"100.64.0.0/10\",\"10.0.0.0/8\"],\"controlUi\":{\"allowInsecureAuth\":true}}}' > /data/.clawdbot/moltbot.json; exec node dist/index.js gateway run --bind 0.0.0.0 --port $PORT --allow-unconfigured"]
