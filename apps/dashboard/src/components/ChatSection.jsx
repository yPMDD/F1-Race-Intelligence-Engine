import React, { useState, useRef, useEffect, useCallback } from 'react';
import { Send, Bot, User, Loader2, Sparkles, RefreshCw, WifiOff } from 'lucide-react';
import { createSseReader, createTokenBuffer } from '../lib/sse';

const API_BASE = import.meta.env.VITE_API_URL || 'http://localhost:8000';

// Keep in sync with DEFAULT_MODEL in nlp/agents/graph.py.
const MODEL_LABEL = 'qwen3.8-27b';

const ChatSection = () => {
  const [messages, setMessages] = useState([
    { role: 'assistant', content: 'Race Intelligence Engine online. Standing by for telemetry analysis or regulation queries for the 2026 season.' }
  ]);
  const [input, setInput] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [apiStatus, setApiStatus] = useState('checking'); // 'online' | 'offline' | 'checking'
  const [lastFailedQuery, setLastFailedQuery] = useState(null);
  const scrollRef = useRef(null);
  const abortRef = useRef(null);

  // Health check - polls every 10 seconds
  const checkHealth = useCallback(async () => {
    try {
      const res = await fetch(`${API_BASE}/health`, { signal: AbortSignal.timeout(3000) });
      setApiStatus(res.ok ? 'online' : 'offline');
    } catch {
      setApiStatus('offline');
    }
  }, []);

  useEffect(() => {
    checkHealth();
    const interval = setInterval(checkHealth, 10000);
    return () => clearInterval(interval);
  }, [checkHealth]);

  // Abort any in-flight stream if the component unmounts.
  useEffect(() => () => abortRef.current?.abort(), []);

  useEffect(() => {
    if (scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
    }
  }, [messages]);

  const appendToLast = (chunk) => {
    setMessages(prev => {
      if (prev.length === 0) return prev;
      const updated = [...prev];
      const last = updated[updated.length - 1];
      updated[updated.length - 1] = { ...last, content: last.content + chunk };
      return updated;
    });
  };

  const handleSend = async (overrideQuery = null) => {
    const queryText = overrideQuery ?? input;
    if (!queryText.trim() || isLoading) return;

    const userMessage = { role: 'user', content: queryText };
    setMessages(prev => overrideQuery
      // On retry: replace the last error msg with a fresh attempt
      ? [...prev.slice(0, -2), userMessage, { role: 'assistant', content: '' }]
      : [...prev, userMessage, { role: 'assistant', content: '' }]
    );
    if (!overrideQuery) setInput('');
    setLastFailedQuery(null);
    setIsLoading(true);

    const controller = new AbortController();
    abortRef.current = controller;

    let serverError = null;
    const buf = createTokenBuffer(appendToLast);

    try {
      let response;
      try {
        response = await fetch(`${API_BASE}/agent/chat`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ prompt: queryText }),
          signal: controller.signal,
        });
      } catch (e) {
        // Transport-level failure => the backend really is unreachable.
        if (e.name !== 'AbortError') setApiStatus('offline');
        throw e;
      }

      if (!response.ok) throw new Error(`API returned ${response.status}`);
      if (!response.body) throw new Error('The server did not return a stream.');

      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      const sse = createSseReader({
        onToken: (t) => buf.push(t),
        onError: (m) => { serverError = m; },
      });

      // `sse.isDone()` terminates the loop on [DONE] instead of merely
      // breaking the inner frame loop, which previously left the reader
      // running until the server happened to close the connection.
      while (!sse.isDone()) {
        const { done, value } = await reader.read();
        if (done) break;
        sse.push(decoder.decode(value, { stream: true }));
      }
      sse.end();
      buf.flush();
      try { await reader.cancel(); } catch { /* already closed */ }

      if (serverError) throw new Error(serverError);
      if (!sse.sawToken()) {
        throw new Error('The strategist returned an empty response. Please retry.');
      }
    } catch (error) {
      buf.flush();
      if (error.name === 'AbortError') return;

      const errMsg = error.message || 'Unknown error';
      setLastFailedQuery(queryText);
      // NOTE: deliberately does NOT set apiStatus to 'offline'. A failed agent
      // call (unknown track, model timeout, empty response) is not the backend
      // being unreachable; marking it offline disabled the input box and told
      // the user to restart services.
      setMessages(prev => {
        if (prev.length === 0) return prev;
        const updated = [...prev];
        updated[updated.length - 1] = {
          role: 'assistant',
          content: `⚠️ ${errMsg}`,
          isError: true,
        };
        return updated;
      });
    } finally {
      buf.flush();
      abortRef.current = null;
      setIsLoading(false);
    }
  };

  const statusConfig = {
    online:   { color: 'bg-emerald-500', text: 'text-emerald-400', label: 'ONLINE', pulse: true },
    offline:  { color: 'bg-red-500',     text: 'text-red-400',     label: 'OFFLINE', pulse: false },
    checking: { color: 'bg-yellow-500',  text: 'text-yellow-400',  label: 'CHECKING', pulse: true },
  }[apiStatus];

  return (
    <div className="flex flex-col h-full overflow-hidden bg-transparent">
      {/* Header */}
      <div className="p-4 border-b border-white/5 flex items-center justify-between bg-white/[0.02]">
        <div className="flex items-center gap-2">
          <Sparkles size={18} className="text-f1-red" />
          <h2 className="text-xs font-black text-white uppercase tracking-widest">Race Strategist AI</h2>
        </div>
        <div className="flex items-center gap-2">
          {apiStatus === 'offline' && (
            <button
              onClick={checkHealth}
              title="Retry connection"
              className="p-1 rounded text-slate-500 hover:text-white transition-colors"
            >
              <RefreshCw size={11} />
            </button>
          )}
          <div className="flex items-center gap-1.5">
            <div className={`w-1.5 h-1.5 rounded-full ${statusConfig.color} ${statusConfig.pulse ? 'animate-pulse' : ''}`} />
            <span className={`text-[10px] font-bold uppercase tracking-tighter ${statusConfig.text}`}>
              {apiStatus === 'offline' ? `API OFFLINE · ${statusConfig.label}` : `Groq ${MODEL_LABEL} · ${statusConfig.label}`}
            </span>
          </div>
        </div>
      </div>

      {/* Offline banner */}
      {apiStatus === 'offline' && (
        <div className="px-4 py-2 bg-red-950/40 border-b border-red-500/20 flex items-center gap-2 text-red-400 text-[10px] font-mono">
          <WifiOff size={11} />
          <span>Backend unreachable at <code className="bg-black/40 px-1 rounded">{API_BASE}</code>. Start it with <code className="bg-black/40 px-1 rounded">.\start.ps1</code>.</span>
        </div>
      )}

      {/* Message List */}
      <div
        ref={scrollRef}
        className="flex-1 overflow-y-auto p-4 space-y-4 font-mono text-xs leading-relaxed custom-scrollbar"
      >
        {messages.map((msg, idx) => (
          <div key={idx} className={`flex gap-3 ${msg.role === 'user' ? 'justify-end' : 'justify-start'}`}>
            {msg.role === 'assistant' && (
              <div className={`shrink-0 w-6 h-6 rounded flex items-center justify-center
                ${msg.isError
                  ? 'bg-red-500/10 border border-red-500/30 text-red-400'
                  : 'bg-f1-red/10 border border-f1-red/20 text-f1-red'}`}>
                {msg.isError ? <WifiOff size={12} /> : <Bot size={14} />}
              </div>
            )}
            <div className={`max-w-[85%] rounded-lg border ${
              msg.role === 'user'
                ? 'p-3 bg-blue-600/10 border-blue-500/30 text-blue-100'
                : msg.isError
                  ? 'p-3 bg-red-950/30 border-red-500/20 text-red-300'
                  : 'p-3 bg-white/[0.02] border-white/5 text-slate-300'
            }`}>
              <div className="whitespace-pre-wrap">{msg.content || (isLoading && idx === messages.length - 1 ? '' : '…')}</div>

              {/* Retry button on error messages */}
              {msg.isError && lastFailedQuery && idx === messages.length - 1 && (
                <button
                  onClick={() => handleSend(lastFailedQuery)}
                  className="mt-2 flex items-center gap-1.5 text-[10px] text-red-400 hover:text-red-200 transition-colors border border-red-500/30 rounded px-2 py-1 hover:bg-red-500/10"
                >
                  <RefreshCw size={10} />
                  Retry query
                </button>
              )}
            </div>
            {msg.role === 'user' && (
              <div className="shrink-0 w-6 h-6 rounded bg-blue-600/10 border border-blue-600/20 flex items-center justify-center text-blue-400">
                <User size={14} />
              </div>
            )}
          </div>
        ))}
        {isLoading && (
          <div className="flex gap-3 justify-start">
            <div className="shrink-0 w-6 h-6 rounded bg-white/[0.02] flex items-center justify-center">
              <Loader2 size={14} className="animate-spin text-slate-500" />
            </div>
            <div className="max-w-[85%] p-3 rounded-lg border bg-white/[0.02] border-white/5 text-slate-500 italic animate-pulse">
              Analyzing telemetry and regulations...
            </div>
          </div>
        )}
      </div>

      {/* Input Area */}
      <div className="p-3 bg-black/20 border-t border-white/5">
        <div className="relative flex items-center">
          <input
            type="text"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={(e) => e.key === 'Enter' && handleSend()}
            placeholder={apiStatus === 'offline' ? 'API offline — start services first...' : 'Ask the strategist...'}
            disabled={apiStatus === 'offline'}
            className="w-full bg-[#0d0d12] border border-white/10 rounded-lg py-2.5 pl-4 pr-12 text-xs font-mono text-white placeholder-slate-600 focus:outline-none focus:border-f1-red transition-colors shadow-inner disabled:opacity-40 disabled:cursor-not-allowed"
          />
          <button
            onClick={() => handleSend()}
            disabled={isLoading || !input.trim() || apiStatus === 'offline'}
            className="absolute right-2 p-1.5 rounded-md bg-f1-red text-white hover:bg-red-700 transition-colors disabled:opacity-50"
          >
            <Send size={14} />
          </button>
        </div>
        <div className="mt-2 flex justify-between items-center px-1">
          <span className="text-[9px] text-slate-500 uppercase font-bold tracking-widest">System Prompt: Race_Strategist_v3</span>
          <span className="text-[9px] text-slate-500 font-mono">HYBRID_RAG · TRUE_STREAM · SEMANTIC_CACHE</span>
        </div>
      </div>
    </div>
  );
};

export default ChatSection;
