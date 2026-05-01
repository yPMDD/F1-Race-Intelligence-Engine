import React, { useState, useRef, useEffect } from 'react';
import { Send, Bot, User, Loader2, Sparkles } from 'lucide-react';

const ChatSection = () => {
  const [messages, setMessages] = useState([
    { role: 'assistant', content: 'Race Intelligence Engine online. Standing by for telemetry analysis or regulation queries for the 2026 season.' }
  ]);
  const [input, setInput] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const scrollRef = useRef(null);

  useEffect(() => {
    if (scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
    }
  }, [messages]);

  const handleSend = async () => {
    if (!input.trim() || isLoading) return;

    const userMessage = { role: 'user', content: input };
    setMessages(prev => [...prev, userMessage]);
    setInput('');
    setIsLoading(true);

    try {
      const response = await fetch('http://localhost:8000/agent/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ prompt: input }),
      });

      if (!response.ok) throw new Error('Agent offline');

      const data = await response.json();
      setMessages(prev => [...prev, { role: 'assistant', content: data.answer }]);
    } catch (error) {
      setMessages(prev => [...prev, { role: 'assistant', content: 'ERROR: Intelligence link severed. Ensure Ollama/API is running.' }]);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="bg-[#1f1f27] rounded-xl border border-slate-800 shadow-sm flex flex-col h-full overflow-hidden">
      <div className="p-4 border-b border-slate-800 flex items-center justify-between bg-slate-800/20">
        <div className="flex items-center gap-2">
          <Sparkles size={18} className="text-f1-red" />
          <h2 className="text-sm font-display text-white uppercase tracking-wider">Race Strategist AI</h2>
        </div>
        <div className="flex items-center gap-1.5">
          <div className="w-1.5 h-1.5 bg-emerald-500 rounded-full animate-pulse"></div>
          <span className="text-[10px] font-bold text-emerald-500 uppercase tracking-tighter">Ollama Llama3</span>
        </div>
      </div>

      {/* Message List */}
      <div 
        ref={scrollRef}
        className="flex-1 overflow-y-auto p-4 space-y-4 font-mono text-xs leading-relaxed"
      >
        {messages.map((msg, idx) => (
          <div key={idx} className={`flex gap-3 ${msg.role === 'user' ? 'justify-end' : 'justify-start'}`}>
            {msg.role === 'assistant' && (
              <div className="shrink-0 w-6 h-6 rounded bg-f1-red/10 border border-f1-red/20 flex items-center justify-center text-f1-red">
                <Bot size={14} />
              </div>
            )}
            <div className={`max-w-[85%] p-3 rounded-lg border ${
              msg.role === 'user' 
                ? 'bg-blue-600/10 border-blue-500/30 text-blue-100' 
                : 'bg-slate-800/40 border-slate-700/50 text-slate-300'
            }`}>
              <div className="whitespace-pre-wrap">{msg.content}</div>
            </div>
            {msg.role === 'user' && (
              <div className="shrink-0 w-6 h-6 rounded bg-blue-600/10 border border-blue-600/20 flex items-center justify-center text-blue-400">
                <User size={14} />
              </div>
            )}
          </div>
        ))}
        {isLoading && (
          <div className="flex gap-3 justify-start animate-pulse">
            <div className="shrink-0 w-6 h-6 rounded bg-slate-800 flex items-center justify-center">
              <Loader2 size={14} className="animate-spin text-slate-500" />
            </div>
            <div className="max-w-[85%] p-3 rounded-lg border bg-slate-800/40 border-slate-700/50 text-slate-500 italic">
              Analyzing telemetry and regulations...
            </div>
          </div>
        )}
      </div>

      {/* Input Area */}
      <div className="p-3 bg-slate-900/50 border-t border-slate-800">
        <div className="relative flex items-center">
          <input
            type="text"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={(e) => e.key === 'Enter' && handleSend()}
            placeholder="Ask the strategist..."
            className="w-full bg-[#15151e] border border-slate-700 rounded-lg py-2.5 pl-4 pr-12 text-xs font-mono text-white placeholder-slate-600 focus:outline-none focus:border-f1-red transition-colors"
          />
          <button 
            onClick={handleSend}
            disabled={isLoading || !input.trim()}
            className="absolute right-2 p-1.5 rounded-md bg-f1-red text-white hover:bg-red-700 transition-colors disabled:opacity-50 disabled:hover:bg-f1-red"
          >
            <Send size={14} />
          </button>
        </div>
        <div className="mt-2 flex justify-between items-center px-1">
          <span className="text-[9px] text-slate-500 uppercase font-bold tracking-widest">System Prompt: Race_Strategist_v2</span>
          <span className="text-[9px] text-slate-500 font-mono">2026_REGS_LOADED</span>
        </div>
      </div>
    </div>
  );
};

export default ChatSection;
