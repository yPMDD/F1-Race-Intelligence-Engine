import React, { useState, useEffect } from 'react';
import { Flag, Activity, Clock, Zap, Database, Cpu, MessageSquare, Sparkles, Filter, ChevronDown, ListFilter, User, Brain } from 'lucide-react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts';
import ChatSection from './components/ChatSection';

const API_BASE = 'http://localhost:8000';

function App() {
  const [years, setYears] = useState([]);
  const [races, setRaces] = useState([]);
  const [drivers, setDrivers] = useState([]);
  const [laps, setLaps] = useState([]);
  
  const [selectedYear, setSelectedYear] = useState(2024);
  const [selectedRace, setSelectedRace] = useState(null);
  const [selectedDriver, setSelectedDriver] = useState(null);
  
  const [loading, setLoading] = useState(true);
  const [timeLeft, setTimeLeft] = useState({ days: 0, hours: 0, mins: 0, secs: 0 });
  const [prediction, setPrediction] = useState(null);
  const [predicting, setPredicting] = useState(false);

  // 1. Initial Load: Fetch Available Years
  useEffect(() => {
    fetch(`${API_BASE}/races/years`)
      .then(res => res.json())
      .then(data => {
        setYears(data.sort((a, b) => b - a));
        if (data.includes(2024)) setSelectedYear(2024);
      })
      .catch(err => console.error("Failed to fetch years:", err));
  }, []);

  // 2. Fetch Races when Year changes
  useEffect(() => {
    if (!selectedYear) return;
    fetch(`${API_BASE}/races/by-year/${selectedYear}`)
      .then(res => res.json())
      .then(data => {
        setRaces(data);
        if (data.length > 0) setSelectedRace(data[0]);
      });
  }, [selectedYear]);

  // 3. Fetch Drivers and Laps when Race changes
  useEffect(() => {
    if (!selectedRace) return;
    setLoading(true);
    setPrediction(null); // Clear old prediction
    fetch(`${API_BASE}/races/${selectedRace.id}/laps`)
      .then(res => res.json())
      .then(data => {
        setLaps(data);
        const uniqueDrivers = [...new Set(data.map(l => l.driver_id))];
        setDrivers(uniqueDrivers);
        if (uniqueDrivers.length > 0) setSelectedDriver(uniqueDrivers[0]);
        setLoading(false);
      });
  }, [selectedRace]);

  // 4. Countdown Timer Logic
  useEffect(() => {
    if (!selectedRace) return;
    const timer = setInterval(() => {
      const raceDate = new Date(selectedRace.date);
      const diff = raceDate.getTime() - new Date().getTime();
      
      if (diff > 0) {
        setTimeLeft({
          days: Math.floor(diff / (1000 * 60 * 60 * 24)),
          hours: Math.floor((diff / (1000 * 60 * 60)) % 24),
          mins: Math.floor((diff / 1000 / 60) % 60),
          secs: Math.floor((diff / 1000) % 60)
        });
      } else {
        setTimeLeft({ days: 0, hours: 0, mins: 0, secs: 0 });
      }
    }, 1000);
    return () => clearInterval(timer);
  }, [selectedRace]);

  const handlePredict = async () => {
    if (!selectedRace) return;
    setPredicting(true);
    try {
      const res = await fetch(`${API_BASE}/races/${selectedRace.id}/predict`, { method: 'POST' });
      const data = await res.json();
      setPrediction(data.prediction);
    } catch (err) {
      console.error("Prediction failed:", err);
    } finally {
      setPredicting(false);
    }
  };

  const currentDriverStats = laps.filter(l => l.driver_id === selectedDriver);
  const bestLap = currentDriverStats.length > 0 ? Math.min(...currentDriverStats.map(l => l.lap_time_ms)) : 0;

  return (
    <div className="min-h-screen bg-[#0d0d12] text-slate-300 flex flex-col font-sans selection:bg-f1-red selection:text-white">
      {/* Top Navbar */}
      <nav className="h-14 bg-[#15151e] border-b border-white/5 flex items-center justify-between px-6 shrink-0 sticky top-0 z-50">
        <div className="flex items-center gap-3">
          <div className="bg-f1-red text-white p-1 rounded shadow-[0_0_15px_rgba(255,24,1,0.2)]">
            <Flag size={16} />
          </div>
          <span className="font-display text-base tracking-widest text-white uppercase italic font-black">F1 ENGINE</span>
        </div>
        
        <div className="flex items-center gap-2 bg-slate-900/50 p-1 rounded-lg border border-white/5">
          <div className="flex items-center px-3 border-r border-white/5 gap-2">
            <Clock size={12} className="text-slate-500" />
            <select 
              value={selectedYear} 
              onChange={(e) => setSelectedYear(parseInt(e.target.value))}
              className="bg-transparent text-[10px] font-bold text-white focus:outline-none cursor-pointer uppercase"
            >
              {years.map(y => <option key={y} value={y} className="bg-[#15151e]">{y}</option>)}
            </select>
          </div>
          <div className="flex items-center px-3 border-r border-white/5 gap-2">
            <Activity size={12} className="text-slate-500" />
            <select 
              value={selectedRace?.id} 
              onChange={(e) => setSelectedRace(races.find(r => r.id === parseInt(e.target.value)))}
              className="bg-transparent text-[10px] font-bold text-white focus:outline-none cursor-pointer uppercase max-w-[150px]"
            >
              {races.map(r => <option key={r.id} value={r.id} className="bg-[#15151e]">{r.name}</option>)}
            </select>
          </div>
          <div className="flex items-center px-3 gap-2">
            <User size={12} className="text-slate-500" />
            <select 
              value={selectedDriver || ''} 
              onChange={(e) => setSelectedDriver(e.target.value)}
              className="bg-transparent text-[10px] font-bold text-white focus:outline-none cursor-pointer uppercase"
            >
              {drivers.map(d => <option key={d} value={d} className="bg-[#15151e]">{d}</option>)}
            </select>
          </div>
        </div>

        <div className="flex items-center gap-4 text-[10px] text-slate-500 font-bold uppercase tracking-widest">
           <div className="flex items-center gap-2 text-emerald-400 bg-emerald-900/10 px-3 py-1 rounded border border-emerald-800/20">
            <div className="w-1 h-1 bg-emerald-500 rounded-full animate-pulse"></div>
            <span>LIVE: SYSTEM READY</span>
          </div>
        </div>
      </nav>

      <main className="flex-1 max-w-[1800px] w-full mx-auto p-4 flex flex-col gap-4 overflow-hidden">
        <div className="flex items-end justify-between border-l-2 border-f1-red pl-4 py-1">
          <div>
            <h1 className="text-2xl font-display text-white tracking-tighter uppercase italic font-black">
              {selectedRace?.name || 'Loading...'} <span className="text-f1-red opacity-50 ml-2">[{selectedYear}]</span>
            </h1>
            <div className="flex gap-4 mt-1">
              <span className="text-slate-500 text-[9px] font-bold uppercase tracking-[0.2em]">Telemetry Analysis</span>
              <span className="text-slate-500 text-[9px] font-bold uppercase tracking-[0.2em]">Vector RAG</span>
            </div>
          </div>
          <div className="flex gap-3">
             <div className="bg-[#15151e] px-4 py-2 rounded border border-white/5">
                <div className="text-[8px] text-slate-500 uppercase font-bold mb-1">Fastest Lap</div>
                <div className="text-lg font-mono text-white font-bold">{bestLap ? (bestLap / 1000).toFixed(3) + 's' : 'N/A'}</div>
             </div>
             <div className="bg-[#15151e] px-4 py-2 rounded border border-white/5">
                <div className="text-[8px] text-slate-500 uppercase font-bold mb-1">Total Laps</div>
                <div className="text-lg font-mono text-white font-bold">{currentDriverStats.length}</div>
             </div>
          </div>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-12 gap-4 flex-1 min-h-0">
          <div className="lg:col-span-8 flex flex-col gap-4 min-h-0">
            <div className="bg-[#15151e] rounded-lg border border-white/5 shadow-2xl flex flex-col flex-1 min-h-[400px]">
              <div className="p-3 border-b border-white/5 flex items-center justify-between bg-white/[0.02]">
                <div className="flex items-center gap-2">
                  <Activity size={14} className="text-f1-red" />
                  <h2 className="text-xs font-bold text-white uppercase tracking-widest">Lap Performance Comparison</h2>
                </div>
              </div>
              <div className="p-4 flex-1 flex items-center justify-center relative">
                {currentDriverStats.length > 0 ? (
                  <ResponsiveContainer width="100%" height="100%">
                    <LineChart data={currentDriverStats}>
                      <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="#1f1f27" />
                      <XAxis dataKey="lap_number" axisLine={false} tickLine={false} tick={{fill: '#475569', fontSize: 9}} />
                      <YAxis axisLine={false} tickLine={false} tick={{fill: '#475569', fontSize: 9}} domain={['auto', 'auto']} />
                      <Tooltip contentStyle={{backgroundColor: '#1f1f27', border: '1px solid #334155'}} />
                      <Line type="monotone" dataKey="lap_time_ms" stroke="#ff1801" strokeWidth={2} dot={false} />
                    </LineChart>
                  </ResponsiveContainer>
                ) : (
                  <div className="flex flex-col items-center gap-8">
                    <div className="grid grid-cols-4 gap-4 text-center">
                      {[['Days', timeLeft.days], ['Hours', timeLeft.hours], ['Mins', timeLeft.mins], ['Secs', timeLeft.secs]].map(([label, val]) => (
                        <div key={label}>
                          <div className="w-16 h-16 bg-[#0d0d12] rounded border border-white/5 flex items-center justify-center text-3xl font-black text-white">
                            {val.toString().padStart(2, '0')}
                          </div>
                          <div className="text-[7px] font-bold text-slate-500 uppercase mt-2 tracking-widest">{label}</div>
                        </div>
                      ))}
                    </div>
                    <div className="flex flex-col items-center">
                      <div className="flex items-center gap-2 mb-2">
                         <div className="w-2 h-2 bg-f1-red rounded-full animate-ping"></div>
                         <h3 className="text-sm font-black text-white tracking-[0.3em] uppercase italic">RACE COUNTDOWN</h3>
                      </div>
                      <p className="text-[10px] text-slate-500 uppercase font-bold tracking-widest">Session starting soon • Live telemetry pending</p>
                    </div>
                  </div>
                )}
              </div>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-2 gap-4 h-[300px]">
              <div className="bg-[#15151e] rounded-lg border border-white/5 shadow-sm overflow-hidden flex flex-col">
                <div className="p-3 border-b border-white/5 flex items-center justify-between bg-white/[0.02] shrink-0">
                  <div className="flex items-center gap-2">
                    <Flag size={14} className="text-f1-red" />
                    <h2 className="text-xs font-bold text-white uppercase tracking-widest">Race Classification</h2>
                  </div>
                  {(!selectedRace?.results || selectedRace.results.length === 0) && (
                    <button 
                      onClick={handlePredict}
                      disabled={predicting}
                      className="flex items-center gap-1.5 px-2 py-1 bg-f1-red/10 hover:bg-f1-red/20 border border-f1-red/20 rounded text-[9px] font-black text-f1-red uppercase tracking-widest transition-all disabled:opacity-50"
                    >
                      {predicting ? <Cpu size={10} className="animate-spin" /> : <Sparkles size={10} />}
                      {predicting ? "Simulating..." : "Predict Results"}
                    </button>
                  )}
                </div>
                <div className="flex-1 overflow-auto custom-scrollbar">
                  {prediction ? (
                    <div className="p-4 bg-f1-red/5 h-full">
                       <div className="flex items-center gap-2 mb-3">
                          <Brain size={12} className="text-f1-red" />
                          <span className="text-[10px] font-black text-white uppercase tracking-widest">AI Strategic Forecast</span>
                       </div>
                       <div className="text-[10px] font-mono leading-relaxed text-slate-300 whitespace-pre-wrap">
                          {prediction}
                       </div>
                    </div>
                  ) : selectedRace?.results && selectedRace.results.length > 0 ? (
                    <table className="w-full text-left text-[10px] font-mono">
                      <thead className="bg-black/40 border-b border-white/5 text-slate-500 sticky top-0">
                        <tr>
                          <th className="p-2 uppercase tracking-tighter w-12 text-center">Pos</th>
                          <th className="p-2 uppercase tracking-tighter">Driver</th>
                          <th className="p-2 uppercase tracking-tighter text-right">Points</th>
                          <th className="p-2 uppercase tracking-tighter text-right">Status</th>
                        </tr>
                      </thead>
                      <tbody className="divide-y divide-white/[0.02]">
                        {selectedRace?.results?.sort((a,b) => a.position - b.position).map((res, i) => (
                          <tr key={i} className={`hover:bg-white/[0.02] transition-colors ${res.position <= 3 ? 'bg-f1-red/5' : ''}`}>
                            <td className="p-2 text-center font-bold text-white">{res.position}</td>
                            <td className="p-2 text-white font-bold">{res.driver_id}</td>
                            <td className="p-2 text-right text-slate-400">{res.points}</td>
                            <td className="p-2 text-right">
                              <span className="text-[8px] uppercase text-slate-500">{res.status}</span>
                            </td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  ) : (
                    <div className="h-full flex flex-col items-center justify-center p-8 text-center opacity-40">
                      <Zap size={24} className="text-slate-500 mb-2" />
                      <p className="text-[10px] font-bold uppercase tracking-widest">No Classification Data</p>
                      <p className="text-[8px] uppercase mt-1">Predictions available via AI Strategist</p>
                    </div>
                  )}
                </div>
              </div>

              <div className="bg-[#15151e] rounded-lg border border-white/5 shadow-sm overflow-hidden flex flex-col">
                <div className="p-3 border-b border-white/5 flex items-center justify-between bg-white/[0.02] shrink-0">
                  <div className="flex items-center gap-2">
                    <ListFilter size={14} className={currentDriverStats.length > 0 ? "text-blue-500" : "text-f1-red"} />
                    <h2 className="text-xs font-bold text-white uppercase tracking-widest">
                      {currentDriverStats.length > 0 ? "Telemetry Log" : "Circuit Intelligence"}
                    </h2>
                  </div>
                </div>
                <div className="flex-1 overflow-auto custom-scrollbar">
                  {currentDriverStats.length > 0 ? (
                    <table className="w-full text-left text-[10px] font-mono">
                      <thead className="bg-black/40 border-b border-white/5 text-slate-500 sticky top-0">
                        <tr>
                          <th className="p-2 uppercase tracking-tighter">Lap</th>
                          <th className="p-2 uppercase tracking-tighter">Time</th>
                          <th className="p-2 uppercase tracking-tighter">Compound</th>
                        </tr>
                      </thead>
                      <tbody className="divide-y divide-white/[0.02]">
                        {currentDriverStats.sort((a,b) => b.lap_number - a.lap_number).map((lap, i) => (
                          <tr key={i} className="hover:bg-white/[0.02] transition-colors">
                            <td className="p-2 text-white font-bold">{lap.lap_number}</td>
                            <td className="p-2 text-f1-red font-bold">{(lap.lap_time_ms/1000).toFixed(3)}s</td>
                            <td className="p-2">
                              <span className={`px-1.5 py-0.5 rounded text-[7px] font-black border ${
                                lap.compound === 'SOFT' ? 'bg-red-900/20 text-red-500 border-red-500/20' : 
                                lap.compound === 'MEDIUM' ? 'bg-yellow-900/20 text-yellow-500 border-yellow-500/20' : 
                                'bg-slate-700/20 text-slate-300 border-slate-500/20'
                              }`}>
                                {lap.compound}
                              </span>
                            </td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  ) : (
                    <div className="p-4 space-y-4">
                      <div className="grid grid-cols-2 gap-2">
                        <div className="bg-white/[0.02] p-2 rounded border border-white/5">
                          <div className="text-[7px] text-slate-500 uppercase font-black tracking-widest">Circuit Length</div>
                          <div className="text-xs text-white font-bold">5.412 KM</div>
                        </div>
                        <div className="bg-white/[0.02] p-2 rounded border border-white/5">
                          <div className="text-[7px] text-slate-500 uppercase font-black tracking-widest">Turns</div>
                          <div className="text-xs text-white font-bold">15 (High Speed)</div>
                        </div>
                      </div>
                      
                      <div className="bg-f1-red/5 p-3 rounded border border-f1-red/10">
                        <div className="flex items-center gap-2 mb-2">
                          <Zap size={10} className="text-f1-red" />
                          <span className="text-[8px] font-black text-white uppercase tracking-widest">Historical Context</span>
                        </div>
                        <div className="space-y-2">
                          <div className="flex justify-between items-center border-b border-white/5 pb-2">
                            <span className="text-[8px] text-slate-400 uppercase">2025 Winner</span>
                            <span className="text-[10px] text-white font-black uppercase tracking-tighter">MAX VERSTAPPEN</span>
                          </div>
                          <div className="flex justify-between items-center">
                            <span className="text-[8px] text-slate-400 uppercase">Track Record</span>
                            <span className="text-[10px] text-f1-red font-black">1:27.264s</span>
                          </div>
                        </div>
                      </div>
                    </div>
                  )}
                </div>
              </div>
            </div>
          </div>

          <div className="lg:col-span-4 flex flex-col gap-4">
             <div className="flex-1 bg-[#15151e] rounded-lg border border-white/5 overflow-hidden flex flex-col shadow-2xl">
                <ChatSection />
             </div>
          </div>
        </div>
      </main>

      <footer className="h-8 bg-[#15151e] border-t border-white/5 px-6 flex items-center justify-between text-[9px] font-mono text-slate-500 shrink-0">
        <div className="flex items-center gap-6">
          <div className="flex items-center gap-1.5"><div className="w-1 h-1 bg-emerald-500 rounded-full"></div> ENGINE_v2: READY</div>
          <div className="flex items-center gap-1.5"><div className="w-1 h-1 bg-blue-500 rounded-full"></div> RAG_INDEX: MULTI_ERA_READY</div>
          <div className="flex items-center gap-1.5 text-f1-red font-bold"><Zap size={10}/> LATENCY: 8ms</div>
        </div>
        <div className="tracking-widest uppercase opacity-40">© 2026 F1_INTELLIGENCE_SYSTEM</div>
      </footer>
    </div>
  );
}

export default App;
