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

# Railway-specific: default gateway config
RUN mkdir -p /root/.clawdbot && \
    echo '{"gateway":{"mode":"local","bind":"0.0.0.0","trustedProxies":["100.64.0.0/10","10.0.0.0/8"],"controlUi":{"allowInsecureAuth":true}}}' > /root/.clawdbot/moltbot.json

ENTRYPOINT ["sh", "-c", "DIR=${CLAWDBOT_STATE_DIR:-/root/.clawdbot}; mkdir -p \"$DIR\"; [ -f \"$DIR/moltbot.json\" ] || cp /root/.clawdbot/moltbot.json \"$DIR/moltbot.json\"; exec node dist/index.js gateway run --bind 0.0.0.0 --allow-unconfigured"]
