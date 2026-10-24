import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { createSseReader, createTokenBuffer, DONE } from '../src/lib/sse';

function frame(obj) {
  return `data: ${JSON.stringify(obj)}\n\n`;
}

describe('createSseReader', () => {
  it('emits tokens from a complete frame', () => {
    const tokens = [];
    const r = createSseReader({ onToken: (t) => tokens.push(t) });
    r.push(frame({ token: 'Hello' }));
    expect(tokens).toEqual(['Hello']);
    expect(r.sawToken()).toBe(true);
  });

  it('reassembles a frame split across chunk boundaries', () => {
    const tokens = [];
    const r = createSseReader({ onToken: (t) => tokens.push(t) });
    r.push('data: {"tok');
    r.push('en": "split');
    r.push('"}');
    r.push('\n\n');
    expect(tokens).toEqual(['split']);
  });

  it('reassembles a token split mid-multibyte-character', () => {
    // "é" is 2 bytes in UTF-8. The component uses a streaming TextDecoder
    // (decode(value, {stream:true})) precisely so a chunk boundary landing
    // between the two bytes does not corrupt the character. Mirror that here.
    const payload = frame({ token: 'café au lait' });
    const bytes = new TextEncoder().encode(payload);
    const decoder = new TextDecoder();

    // Split at a byte offset that falls inside the two-byte "é".
    const cut = payload.indexOf('caf') + 4;
    expect(cut < bytes.length);

    const tokens = [];
    const r = createSseReader({ onToken: (t) => tokens.push(t) });
    r.push(decoder.decode(bytes.slice(0, cut), { stream: true }));
    r.push(decoder.decode(bytes.slice(cut), { stream: true }));

    expect(tokens).toEqual(['café au lait']);
  });

  it('handles several frames in one chunk', () => {
    const tokens = [];
    const r = createSseReader({ onToken: (t) => tokens.push(t) });
    r.push(frame({ token: 'a' }) + frame({ token: 'b' }) + frame({ token: 'c' }));
    expect(tokens).toEqual(['a', 'b', 'c']);
  });

  it('stops at the DONE sentinel', () => {
    const tokens = [];
    const r = createSseReader({ onToken: (t) => tokens.push(t) });
    r.push(frame({ token: 'a' }) + `data: ${DONE}\n\n` + frame({ token: 'ignored' }));
    expect(tokens).toEqual(['a']);
    expect(r.isDone()).toBe(true);
  });

  it('surfaces server-sent errors', () => {
    const errors = [];
    const r = createSseReader({ onToken: () => {}, onError: (e) => errors.push(e) });
    r.push(frame({ error: 'model overloaded' }));
    expect(errors).toEqual(['model overloaded']);
  });

  // --- the regression this module exists for ---------------------------
  it('skips a malformed frame WITHOUT killing the stream', () => {
    const tokens = [];
    const r = createSseReader({ onToken: (t) => tokens.push(t) });
    r.push('data: {not json at all\n\n');
    r.push(frame({ token: 'still streaming' }));
    expect(tokens).toEqual(['still streaming']);
  });

  it('ignores blank lines and non-data fields', () => {
    const tokens = [];
    const r = createSseReader({ onToken: (t) => tokens.push(t) });
    r.push(`\n\n: keepalive comment\nevent: message\nid: 7\n${frame({ token: 'ok' })}`);
    expect(tokens).toEqual(['ok']);
  });

  it('handles CRLF line endings', () => {
    const tokens = [];
    const r = createSseReader({ onToken: (t) => tokens.push(t) });
    r.push('data: {"token": "crlf"}\r\n\r\n');
    expect(tokens).toEqual(['crlf']);
  });

  it('ignores empty-token frames', () => {
    const tokens = [];
    const r = createSseReader({ onToken: (t) => tokens.push(t) });
    r.push(frame({ token: '' }));
    expect(tokens).toEqual([]);
    expect(r.sawToken()).toBe(false);
  });

  it('flushes a trailing line without a newline on end()', () => {
    const tokens = [];
    const r = createSseReader({ onToken: (t) => tokens.push(t) });
    r.push('data: {"token": "tail"}');
    expect(tokens).toEqual([]);
    r.end();
    expect(tokens).toEqual(['tail']);
  });
});

describe('createTokenBuffer', () => {
  beforeEach(() => vi.useFakeTimers());
  afterEach(() => vi.useRealTimers());

  it('coalesces many pushes into one flush', () => {
    const flush = vi.fn();
    const b = createTokenBuffer(flush, { intervalMs: 50 });
    'hello'.split('').forEach((c) => b.push(c));
    expect(flush).not.toHaveBeenCalled();
    vi.advanceTimersByTime(50);
    expect(flush).toHaveBeenCalledTimes(1);
    expect(flush).toHaveBeenCalledWith('hello');
  });

  it('flush() emits immediately and clears the timer', () => {
    const flush = vi.fn();
    const b = createTokenBuffer(flush, { intervalMs: 50 });
    b.push('abc');
    b.flush();
    expect(flush).toHaveBeenCalledWith('abc');
    vi.advanceTimersByTime(100);
    expect(flush).toHaveBeenCalledTimes(1);
  });

  it('does not call flush for an empty buffer', () => {
    const flush = vi.fn();
    const b = createTokenBuffer(flush);
    b.flush();
    expect(flush).not.toHaveBeenCalled();
  });
});
