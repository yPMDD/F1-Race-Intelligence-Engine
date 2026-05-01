import React, { useState, useEffect } from 'react';
import { Flag, Activity, Clock, Zap, Database, Cpu, MessageSquare, Sparkles } from 'lucide-react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts';
import ChatSection from './components/ChatSection';

function App() {
  const [telemetry, setTelemetry] = useState([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    // Mock high-frequency telemetry data for the chart
    const generateTelemetry = () => {
      const data = [];
      let speed = 280;
      for (let i = 0; i < 50; i++) {
        speed += Math.random() * 20 - 10;
        data.push({
          time: `${i}s`,
          speed: Math.max(80, Math.min(340, speed)),
          throttle: Math.random() > 0.3 ? 100 : Math.random() * 100
        });
      }
      return data;
    };

    setTelemetry(generateTelemetry());
    setLoading(false);
  }, []);

  const CustomTooltip = ({ active, payload, label }) => {
    if (active && payload && payload.length) {
      return (
        <div className="bg-[#1f1f27] border border-slate-700 p-3 rounded-lg shadow-lg">
          <p className="text-slate-400 text-xs font-semibold mb-2">T+{label}</p>
          {payload.map((entry, index) => (
            <div key={index} className="flex items-center gap-2 text-sm">
              <div className="w-2 h-2 rounded-full" style={{ backgroundColor: entry.color }}></div>
              <span className="font-medium text-slate-300">{entry.name}:</span>
              <span className="font-semibold text-white">{entry.value.toFixed(1)}</span>
            </div>
          ))}
        </div>
      );
    }
    return null;
  };

  return (
    <div className="min-h-screen bg-[#15151e] text-slate-300 flex flex-col font-sans selection:bg-f1-red selection:text-white">
      {/* Top Navbar */}
      <nav className="h-14 bg-[#1f1f27] border-b border-slate-800 flex items-center justify-between px-6 shrink-0 sticky top-0 z-50">
        <div className="flex items-center gap-3">
          <div className="bg-f1-red text-white p-1.5 rounded-md shadow-[0_0_15px_rgba(255,24,1,0.3)]">
            <Flag size={18} />
          </div>
          <span className="font-display text-lg tracking-wide text-white uppercase italic">Race Intelligence</span>
        </div>
        <div className="flex items-center gap-6 text-[10px] text-slate-400 font-bold uppercase tracking-widest">
          <div className="flex items-center gap-2">
            <Database size={14} className="text-blue-500" />
            <span>PostgreSQL: 5433</span>
          </div>
          <div className="flex items-center gap-2">
            <Cpu size={14} className="text-purple-500" />
            <span>Ollama: Llama3</span>
          </div>
          <div className="flex items-center gap-2 text-emerald-400 bg-emerald-900/20 px-3 py-1 rounded border border-emerald-800/50">
            <div className="w-1.5 h-1.5 bg-emerald-500 rounded-full animate-pulse shadow-[0_0_8px_rgba(16,185,129,0.5)]"></div>
            <span>System Active</span>
          </div>
        </div>
      </nav>

      {/* Main Content */}
      <main className="flex-1 max-w-[1600px] w-full mx-auto p-6 flex flex-col gap-6 overflow-hidden">
        
        {/* Header Section */}
        <div className="flex items-end justify-between border-l-4 border-f1-red pl-4">
          <div>
            <h1 className="text-3xl font-display text-white tracking-tighter uppercase italic">Control Room_v0.2</h1>
            <p className="text-slate-500 text-[10px] font-bold uppercase tracking-[0.2em] mt-1">
              Data Ingestion / Vector RAG / Agentic Inference
            </p>
          </div>
          <div className="text-right hidden md:block">
            <div className="text-xs font-mono text-slate-500 uppercase tracking-tighter">Session ID: #F1-2026-INTEL-001</div>
            <div className="text-[10px] font-mono text-slate-600 mt-1 uppercase italic">Encryption: AES-256-GCM</div>
          </div>
        </div>

        {/* Top Stats Row */}
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
          {[
            { label: 'Events Stream', value: '1,248,392', icon: Database, color: 'text-blue-400' },
            { label: 'Ingestion Lag', value: '12ms', icon: Zap, color: 'text-amber-400' },
            { label: 'RAG Context', value: '2026_REGS', icon: Sparkles, color: 'text-purple-400' },
            { label: 'Agent State', value: 'NOMINAL', icon: Cpu, color: 'text-emerald-400' }
          ].map((stat, i) => (
            <div key={i} className="bg-[#1f1f27] p-4 rounded-xl border border-slate-800/60 shadow-sm group hover:border-slate-700 transition-colors">
              <div className="flex items-center gap-2 text-slate-500 mb-2">
                <stat.icon size={14} className={stat.color} />
                <span className="text-[10px] font-bold uppercase tracking-wider">{stat.label}</span>
              </div>
              <div className="text-2xl font-display text-white tracking-tighter">{stat.value}</div>
            </div>
          ))}
        </div>

        {/* Main Grid: Telemetry & AI Chat */}
        <div className="grid grid-cols-1 lg:grid-cols-12 gap-6 flex-1 min-h-0">
          
          {/* Left Column: Telemetry & Data Table */}
          <div className="lg:col-span-8 flex flex-col gap-6 min-h-0">
            
            {/* Telemetry Chart */}
            <div className="bg-[#1f1f27] rounded-xl border border-slate-800/60 shadow-sm flex flex-col flex-1 min-h-[350px]">
              <div className="p-4 border-b border-slate-800 flex items-center justify-between bg-slate-800/10">
                <div className="flex items-center gap-2">
                  <Activity size={16} className="text-f1-red" />
                  <h2 className="text-sm font-display text-white uppercase italic tracking-wider">Live Telemetry Stream</h2>
                </div>
                <div className="text-[10px] font-mono text-slate-500">SOURCE: FASTF1_INGESTION_WORKER</div>
              </div>
              <div className="p-6 flex-1">
                <ResponsiveContainer width="100%" height="100%">
                  <LineChart data={telemetry}>
                    <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="#2a2a35" />
                    <XAxis dataKey="time" axisLine={false} tickLine={false} tick={{fill: '#475569', fontSize: 10}} />
                    <YAxis axisLine={false} tickLine={false} tick={{fill: '#475569', fontSize: 10}} domain={[0, 350]} />
                    <Tooltip content={<CustomTooltip />} />
                    <Line 
                      type="monotone" 
                      name="Speed (km/h)" 
                      dataKey="speed" 
                      stroke="#ff1801" 
                      strokeWidth={3} 
                      dot={false} 
                      activeDot={{ r: 4, fill: '#ff1801', stroke: '#fff', strokeWidth: 2 }} 
                    />
                  </LineChart>
                </ResponsiveContainer>
              </div>
            </div>

            {/* Vector Index Table */}
            <div className="bg-[#1f1f27] rounded-xl border border-slate-800/60 shadow-sm overflow-hidden h-[200px] flex flex-col">
              <div className="p-4 border-b border-slate-800 flex items-center justify-between bg-slate-800/10 shrink-0">
                <div className="flex items-center gap-2">
                  <Database size={16} className="text-blue-500" />
                  <h2 className="text-sm font-display text-white uppercase italic tracking-wider">Vector Index Status</h2>
                </div>
              </div>
              <div className="flex-1 overflow-auto">
                <table className="w-full text-left text-xs font-mono">
                  <thead className="bg-[#15151e] border-b border-slate-800 text-slate-500 sticky top-0">
                    <tr>
                      <th className="font-bold p-3 px-5 uppercase tracking-tighter">Document Registry</th>
                      <th className="font-bold p-3 px-5 text-right uppercase tracking-tighter">Chunks</th>
                      <th className="font-bold p-3 px-5 text-right uppercase tracking-tighter">Status</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-slate-800/30">
                    {[
                      { name: 'FIA_2026_F1_TECH_REGS.pdf', chunks: '2,840', status: 'INDEXED' },
                      { name: 'FIA_2026_F1_SPORTING_REGS.pdf', chunks: '1,120', status: 'INDEXED' },
                      { name: 'FIA_2026_SECTION_A_GEN.pdf', chunks: '412', status: 'INDEXED' }
                    ].map((row, i) => (
                      <tr key={i} className="hover:bg-slate-800/30 transition-colors">
                        <td className="p-3 px-5 text-slate-300 italic">{row.name}</td>
                        <td className="p-3 px-5 text-right text-slate-500 font-bold">{row.chunks}</td>
                        <td className="p-3 px-5 text-right">
                          <span className="bg-emerald-900/20 text-emerald-500 px-2 py-0.5 rounded border border-emerald-500/20 text-[9px] font-bold">
                            {row.status}
                          </span>
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          </div>

          {/* Right Column: AI Strategist Chat */}
          <div className="lg:col-span-4 min-h-[500px]">
             <ChatSection />
          </div>

        </div>
      </main>

      {/* Status Bar */}
      <footer className="h-8 bg-[#1f1f27] border-t border-slate-800 px-6 flex items-center justify-between text-[10px] font-mono text-slate-500 shrink-0">
        <div className="flex items-center gap-4">
          <div className="flex items-center gap-1.5">
            <div className="w-1.5 h-1.5 bg-emerald-500 rounded-full"></div>
            <span>BACKEND: ONLINE</span>
          </div>
          <div className="flex items-center gap-1.5">
            <div className="w-1.5 h-1.5 bg-blue-500 rounded-full"></div>
            <span>POSTGRES: UP</span>
          </div>
          <div className="flex items-center gap-1.5">
            <div className="w-1.5 h-1.5 bg-purple-500 rounded-full"></div>
            <span>RAG_CORE: READY</span>
          </div>
        </div>
        <div className="flex items-center gap-2 italic">
          v0.2.0-STABLE | © 2026 F1_INTELLIGENCE_ENGINE
        </div>
      </footer>
    </div>
  );
}

export default App;
