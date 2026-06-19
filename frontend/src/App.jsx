import React, { useState, useEffect } from 'react';
import axios from 'axios';
import Navbar from './components/Navbar';
import ChatInterface from './components/ChatInterface';
import RiskChart from './components/UncertaintyChart';
import Guidance from './components/Guidance';
import { motion } from 'framer-motion';

// ngrok-free serves an HTML interstitial to browsers unless this header is present.
axios.defaults.headers.common['ngrok-skip-browser-warning'] = 'true';

// Permanent backend address: the reserved ngrok STATIC domain the Colab backend always opens on. The deployed
// Vercel site targets this by default, so there is nothing to paste; the top-bar field still overrides it.
const DEFAULT_BACKEND_URL = 'https://declared-angular-matchbox.ngrok-free.dev';

// Resolve the backend URL: saved override -> (Vite dev :5173 -> localhost:8000) -> committed default -> same origin.
const defaultBackendUrl = () => {
  const saved = localStorage.getItem('halluscan_backend_url');
  if (saved) return saved;
  if (window.location.port === '5173') return 'http://localhost:8000';
  if (DEFAULT_BACKEND_URL) return DEFAULT_BACKEND_URL;
  return window.location.origin;
};

const tierClasses = (tier) => {
  if (tier === 'high') return { text: 'text-red-600', sub: 'text-red-500', icon: '🚨' };
  if (tier === 'medium') return { text: 'text-amber-500', sub: 'text-amber-600', icon: '⚠️' };
  if (tier === 'ok') return { text: 'text-green-600', sub: 'text-green-600', icon: '🛡️' };
  return { text: 'text-gray-400', sub: 'text-gray-400', icon: '•' };
};

function App() {
  const [messages, setMessages] = useState([]);
  const [history, setHistory] = useState([]);       // {question, risk} per turn (for the chart)
  const [loading, setLoading] = useState(false);
  const [status, setStatus] = useState({ model_loaded: false });
  const [showGuidance, setShowGuidance] = useState(false);
  const [backendUrl, setBackendUrlState] = useState(defaultBackendUrl);
  const [highlightEnabled, setHighlightEnabled] = useState(() => {
    const saved = localStorage.getItem('highlight_enabled');
    return saved !== null ? JSON.parse(saved) : true;
  });

  const setBackendUrl = (url) => {
    const clean = (url || '').trim().replace(/\/+$/, '');
    setBackendUrlState(clean);
    localStorage.setItem('halluscan_backend_url', clean);
  };

  useEffect(() => {
    const saved = localStorage.getItem('chat_history');
    if (saved) {
      const parsed = JSON.parse(saved);
      setMessages(parsed.messages || []);
      setHistory(parsed.history || []);
    }
  }, []);

  useEffect(() => {
    const check = async () => {
      try {
        const res = await axios.get(`${backendUrl}/status`, { timeout: 4000 });
        setStatus(res.data);
      } catch {
        setStatus({ model_loaded: false });
      }
    };
    check();
    const interval = setInterval(check, 2500);
    return () => clearInterval(interval);
  }, [backendUrl]);

  useEffect(() => {
    localStorage.setItem('chat_history', JSON.stringify({ messages, history }));
  }, [messages, history]);

  useEffect(() => {
    localStorage.setItem('highlight_enabled', JSON.stringify(highlightEnabled));
  }, [highlightEnabled]);

  const handleSendMessage = async (text) => {
    setMessages(prev => [...prev, { role: 'user', content: text, timestamp: new Date() }]);
    setLoading(true);
    try {
      const res = await axios.post(`${backendUrl}/infer`, { question: text });
      const { answer, aggregate, sentences } = res.data;
      setMessages(prev => [...prev, {
        role: 'assistant', content: answer || 'Error generating response',
        aggregate, sentences, timestamp: new Date(),
      }]);
      setHistory(prev => [...prev, {
        question: text.length > 18 ? text.slice(0, 18) + '…' : text,
        risk: aggregate?.fused ?? null,
      }]);
    } catch (e) {
      const detail = e?.response?.data?.detail || 'Could not reach the backend. Check the URL in the top bar.';
      setMessages(prev => [...prev, { role: 'assistant', content: `Error: ${detail}`, isError: true }]);
    } finally {
      setLoading(false);
    }
  };

  const clearHistory = () => {
    if (window.confirm('Clear all history?')) {
      setMessages([]); setHistory([]);
      localStorage.removeItem('chat_history');
    }
  };

  const last = messages.length > 0 && messages[messages.length - 1].role === 'assistant'
    && !messages[messages.length - 1].isError ? messages[messages.length - 1] : null;
  const agg = last?.aggregate;

  return (
    <div className="min-h-screen flex flex-col bg-[#f5f5f7] text-[#1d1d1f]">
      <Navbar
        status={status}
        onClear={clearHistory}
        onShowGuidance={() => setShowGuidance(true)}
        highlightEnabled={highlightEnabled}
        onToggleHighlight={() => setHighlightEnabled(p => !p)}
        backendUrl={backendUrl}
        onSetBackendUrl={setBackendUrl}
      />

      {showGuidance && <Guidance onClose={() => setShowGuidance(false)} />}

      <main className="flex-1 flex max-w-[1600px] mx-auto w-full p-6 gap-6 h-[calc(100vh-80px)]">
        {/* Left: Chat */}
        <div className="flex-[2] flex flex-col glass rounded-3xl overflow-hidden shadow-sm h-full max-w-[65%]">
          <ChatInterface messages={messages} loading={loading} onSend={handleSendMessage}
            highlightEnabled={highlightEnabled} />
        </div>

        {/* Right: Risk panel */}
        <div className="flex-1 flex flex-col gap-6 h-full min-w-[350px]">
          <div className="glass rounded-3xl p-6 flex-1 shadow-sm">
            <h2 className="text-xl font-semibold mb-4">Hallucination Risk (per turn)</h2>
            <div className="h-full max-h-[360px]"><RiskChart data={history} /></div>
          </div>

          <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }}
            className="glass rounded-3xl p-6 shadow-sm">
            <h3 className="text-gray-500 text-sm font-medium uppercase tracking-wider mb-3">Last Answer — Risk</h3>
            {agg ? (
              <div className="space-y-4">
                <div className="p-4 rounded-2xl bg-white/50 text-center">
                  <div className="text-sm text-gray-400 mb-1">Fused hallucination probability</div>
                  <div className={`text-3xl font-bold ${tierClasses(agg.tier).text}`}>
                    {((agg.fused ?? 0) * 100).toFixed(1)}%
                  </div>
                  <div className={`text-xs mt-1 font-medium ${tierClasses(agg.tier).sub}`}>
                    {tierClasses(agg.tier).icon} {agg.label}
                  </div>
                </div>

                {/* Per-detector breakdown */}
                <div className="space-y-2">
                  <DetectorBar label="SEP (entropy)" value={agg.sep_entropy} />
                  <DetectorBar label="HalluShift" value={agg.hallushift} />
                  <DetectorRow label="TSV (margin)" value={agg.tsv_margin} />
                </div>

                <div className="text-xs text-gray-500 text-center pt-1">
                  {agg.n_flagged} of {agg.n_sentences} claim sentence{agg.n_sentences === 1 ? '' : 's'} flagged high-risk
                </div>
              </div>
            ) : (
              <div className="text-gray-400 italic">No data yet — ask a question.</div>
            )}
          </motion.div>
        </div>
      </main>
    </div>
  );
}

// 0..1 probability rendered as a bar (higher = more hallucinated -> redder)
const DetectorBar = ({ label, value }) => {
  const v = Math.max(0, Math.min(1, value ?? 0));
  const color = v >= 0.5 ? 'bg-red-400' : v >= 0.33 ? 'bg-amber-400' : 'bg-green-400';
  return (
    <div>
      <div className="flex justify-between text-xs text-gray-500 mb-1">
        <span>{label}</span><span>{(v * 100).toFixed(0)}%</span>
      </div>
      <div className="h-2 rounded-full bg-gray-100 overflow-hidden">
        <div className={`h-full ${color}`} style={{ width: `${v * 100}%` }} />
      </div>
    </div>
  );
};

// TSV margin is a cosine difference (~ -1..1), not a probability — show the raw signed value.
const DetectorRow = ({ label, value }) => (
  <div className="flex justify-between text-xs text-gray-500">
    <span>{label}</span>
    <span className={(value ?? 0) >= 0 ? 'text-red-500' : 'text-green-600'}>
      {(value ?? 0) >= 0 ? '+' : ''}{(value ?? 0).toFixed(3)}
    </span>
  </div>
);

export default App;
