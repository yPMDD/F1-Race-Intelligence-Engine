/**
 * Incremental Server-Sent Events reader.
 *
 * The backend emits `data: {"token": "..."}` frames terminated by a blank
 * line, then `data: [DONE]`.
 *
 * This is deliberately a plain factory with no DOM or React dependency so it
 * can be unit-tested directly against chunked input.
 */

export const DONE = '[DONE]';

/**
 * @param {object} handlers
 * @param {(token: string) => void} handlers.onToken
 * @param {(message: string) => void} [handlers.onError]  server-sent error frame
 * @returns {{push: (chunk: string) => void, end: () => void, sawToken: () => boolean, isDone: () => boolean}}
 */
export function createSseReader({ onToken, onError = () => {} }) {
  let buffer = '';
  let sawToken = false;
  let done = false;

  function handleLine(line) {
    if (done) return;
    // Strip a trailing \r so CRLF streams work too.
    const clean = line.endsWith('\r') ? line.slice(0, -1) : line;

    // SSE comments / other fields (event:, id:, retry:) are ignored.
    if (!clean.startsWith('data:')) return;

    const payload = clean.slice(5).trim();
    if (!payload) return;

    if (payload === DONE) {
      done = true;
      return;
    }

    let parsed;
    try {
      parsed = JSON.parse(payload);
    } catch {
      // A malformed frame must not tear down the stream. The previous code
      // rethrew any parse failure because its guard compared against the
      // literal string 'Unexpected token', whereas V8 reports
      // "Unexpected token 'x', ... is not valid JSON" -- so the guard never
      // fired and one bad frame killed the response mid-flight.
      return;
    }

    if (typeof parsed.error === 'string' && parsed.error) {
      onError(parsed.error);
      return;
    }
    if (typeof parsed.token === 'string' && parsed.token !== '') {
      sawToken = true;
      onToken(parsed.token);
    }
  }

  return {
    push(chunk) {
      buffer += chunk;
      let idx;
      while ((idx = buffer.indexOf('\n')) !== -1) {
        const line = buffer.slice(0, idx);
        buffer = buffer.slice(idx + 1);
        handleLine(line);
      }
    },
    /** Flush any trailing partial line once the stream closes. */
    end() {
      if (buffer) {
        handleLine(buffer);
        buffer = '';
      }
    },
    sawToken: () => sawToken,
    isDone: () => done,
  };
}

/**
 * Batches token appends so a fast stream does not trigger one React render per
 * token. Groq emits tokens far faster than 60fps, so without this the chat
 * pane re-rendered hundreds of times per response.
 */
export function createTokenBuffer(flushFn, { intervalMs = 50 } = {}) {
  let pending = '';
  let timer = null;

  const flush = () => {
    if (timer !== null) {
      clearTimeout(timer);
      timer = null;
    }
    if (pending) {
      const chunk = pending;
      pending = '';
      flushFn(chunk);
    }
  };

  return {
    push(token) {
      pending += token;
      if (timer === null) {
        timer = setTimeout(flush, intervalMs);
      }
    },
    flush,
    get pending() {
      return pending;
    },
  };
}
