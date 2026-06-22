import React, { useState, useEffect } from 'react';
import { ShieldCheck, HelpCircle, Eye, EyeOff, Server, Cpu } from 'lucide-react';

const Navbar = ({ status, onClear, onShowGuidance, highlightEnabled, onToggleHighlight,
                  backendUrl, onSetBackendUrl, availableModels = [], selectedModel, onSelectModel }) => {
    const [draft, setDraft] = useState(backendUrl || '');
    useEffect(() => { setDraft(backendUrl || ''); }, [backendUrl]);

    const online = !!status.model_loaded;

    return (
        <nav className="h-[80px] flex items-center justify-between px-8 bg-white/80 backdrop-blur-md border-b border-gray-100 sticky top-0 z-40">
            <div className="flex items-center gap-3">
                <div className="w-10 h-10 bg-[#1d1d1f] rounded-xl flex items-center justify-center text-white shadow-lg shadow-black/10">
                    <ShieldCheck size={24} />
                </div>
                <div>
                    <h1 className="font-semibold text-lg tracking-tight text-[#1d1d1f]">HalluScan</h1>
                    <div className="flex items-center gap-2 text-xs">
                        <span className={`w-2 h-2 rounded-full ${online ? 'bg-green-500 shadow-[0_0_8px_rgba(34,197,94,0.6)]' : 'bg-red-500'}`}></span>
                        <span className="text-gray-500 font-medium">
                            {online ? 'online · fused hallucination detector' : 'backend offline'}
                        </span>
                    </div>
                </div>
            </div>

            <div className="flex items-center gap-3">
                {/* Backend URL field (paste the Colab/ngrok URL here at the viva) */}
                <div className="hidden md:flex items-center gap-2 bg-gray-50 border border-gray-200 rounded-full pl-3 pr-1 py-1">
                    <Server size={14} className="text-gray-400" />
                    <input
                        value={draft}
                        onChange={(e) => setDraft(e.target.value)}
                        onBlur={() => onSetBackendUrl(draft)}
                        onKeyDown={(e) => { if (e.key === 'Enter') { onSetBackendUrl(draft); e.target.blur(); } }}
                        placeholder="https://xxxx.ngrok-free.app"
                        spellCheck={false}
                        className="bg-transparent text-xs text-gray-600 w-[230px] focus:outline-none"
                        title="Backend URL (Colab + ngrok). Saved in your browser."
                    />
                    <button onClick={() => onSetBackendUrl(draft)}
                        className="text-xs font-medium px-2 py-1 rounded-full bg-[#0071e3] text-white hover:bg-[#0077ED] transition-colors">
                        Set
                    </button>
                </div>

                {/* Model selector — pick which model answers + is scored. Switching loads its weights, so the
                    select is disabled while a swap is in flight (status.loading). Shown only when >1 is available. */}
                {availableModels.length >= 2 && (
                    <div className="flex items-center gap-2 bg-gray-50 border border-gray-200 rounded-full pl-3 pr-2 py-1"
                        title="Model that answers and is scored (switching loads its weights)">
                        <Cpu size={14} className="text-gray-400" />
                        <select
                            value={selectedModel || ''}
                            onChange={(e) => onSelectModel(e.target.value)}
                            disabled={!!status.loading}
                            className="bg-transparent text-xs text-gray-600 focus:outline-none cursor-pointer pr-1 disabled:opacity-50">
                            {availableModels.map((m) => {
                                // Make the resident/active model explicit so the dropdown can't disagree with
                                // what the backend actually has loaded (current_model). · loading… = mid-swap.
                                const tag = m.key === status.loading ? ' · loading…'
                                    : m.key === status.current_model ? ' · loaded' : '';
                                return <option key={m.key} value={m.key}>{m.label}{tag}</option>;
                            })}
                        </select>
                    </div>
                )}

                <button onClick={onToggleHighlight}
                    className={`flex items-center gap-2 px-3 py-1.5 rounded-full text-sm font-medium transition-all duration-300 border ${highlightEnabled ? 'bg-blue-50 text-blue-600 border-blue-200 hover:bg-blue-100' : 'bg-gray-50 text-gray-400 border-gray-200 hover:bg-gray-100'}`}
                    title={highlightEnabled ? 'Sentence highlighting is ON' : 'Sentence highlighting is OFF'}>
                    {highlightEnabled ? <Eye size={14} /> : <EyeOff size={14} />}
                    <span className="hidden sm:inline">{highlightEnabled ? 'Highlights On' : 'Highlights Off'}</span>
                </button>

                <button onClick={onShowGuidance}
                    className="p-2 text-gray-400 hover:text-[#0071e3] transition-colors rounded-full hover:bg-gray-100"
                    title="Help & Guidance">
                    <HelpCircle size={22} />
                </button>

                <button onClick={onClear}
                    className="px-4 py-2 text-sm font-medium text-red-500 hover:bg-red-50 rounded-lg transition-colors">
                    Clear History
                </button>
            </div>
        </nav>
    );
};

export default Navbar;
