export type EmbeddedRunHandle = {
  queueMessage: (text: string) => Promise<void>;
  isStreaming: () => boolean;
  isCompacting: () => boolean;
  abort: () => void;
};

type EmbeddedRunWaiter = {
  resolve: (ended: boolean) => void;
  timer: NodeJS.Timeout;
};

const ACTIVE_EMBEDDED_RUNS = new Map<string, EmbeddedRunHandle>();
const EMBEDDED_RUN_WAITERS = new Map<string, Set<EmbeddedRunWaiter>>();

export function registerEmbeddedRun(
  sessionId: string,
  handle: EmbeddedRunHandle,
): void {
  if (!sessionId) return;
  ACTIVE_EMBEDDED_RUNS.set(sessionId, handle);
}

export function unregisterEmbeddedRun(
  sessionId: string,
  handle?: EmbeddedRunHandle,
): void {
  if (!sessionId) return;
  const current = ACTIVE_EMBEDDED_RUNS.get(sessionId);
  if (!current) return;
  if (handle && current !== handle) return;
  ACTIVE_EMBEDDED_RUNS.delete(sessionId);
  notifyEmbeddedRunEnded(sessionId);
}

export function queueEmbeddedRunMessage(
  sessionId: string,
  text: string,
): boolean {
  const handle = ACTIVE_EMBEDDED_RUNS.get(sessionId);
  if (!handle) return false;
  if (!handle.isStreaming()) return false;
  if (handle.isCompacting()) return false;
  void handle.queueMessage(text);
  return true;
}

export function abortEmbeddedRun(sessionId: string): boolean {
  const handle = ACTIVE_EMBEDDED_RUNS.get(sessionId);
  if (!handle) return false;
  handle.abort();
  return true;
}

export function isEmbeddedRunActive(sessionId: string): boolean {
  return ACTIVE_EMBEDDED_RUNS.has(sessionId);
}

export function isEmbeddedRunStreaming(sessionId: string): boolean {
  const handle = ACTIVE_EMBEDDED_RUNS.get(sessionId);
  if (!handle) return false;
  return handle.isStreaming();
}

export function waitForEmbeddedRunEnd(
  sessionId: string,
  timeoutMs = 15_000,
): Promise<boolean> {
  if (!sessionId || !ACTIVE_EMBEDDED_RUNS.has(sessionId))
    return Promise.resolve(true);
  return new Promise((resolve) => {
    const waiters = EMBEDDED_RUN_WAITERS.get(sessionId) ?? new Set();
    const waiter: EmbeddedRunWaiter = {
      resolve,
      timer: setTimeout(
        () => {
          waiters.delete(waiter);
          if (waiters.size === 0) EMBEDDED_RUN_WAITERS.delete(sessionId);
          resolve(false);
        },
        Math.max(100, timeoutMs),
      ),
    };
    waiters.add(waiter);
    EMBEDDED_RUN_WAITERS.set(sessionId, waiters);
    if (!ACTIVE_EMBEDDED_RUNS.has(sessionId)) {
      waiters.delete(waiter);
      if (waiters.size === 0) EMBEDDED_RUN_WAITERS.delete(sessionId);
      clearTimeout(waiter.timer);
      resolve(true);
    }
  });
}

function notifyEmbeddedRunEnded(sessionId: string) {
  const waiters = EMBEDDED_RUN_WAITERS.get(sessionId);
  if (!waiters || waiters.size === 0) return;
  EMBEDDED_RUN_WAITERS.delete(sessionId);
  for (const waiter of waiters) {
    clearTimeout(waiter.timer);
    waiter.resolve(true);
  }
}
