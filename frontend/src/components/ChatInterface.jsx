import React, { useEffect, useRef } from 'react';
import { Send, Bot, User, AlertCircle, Loader2 } from 'lucide-react';
import { motion } from 'framer-motion';

// Demo starter questions, validated against the live detector (see tools/check_suggestions.py):
// 2 "clean" (expected low risk) + 2 "caught" (model hallucinates -> flagged high).
// Quick = single-fact short answer · Detailed = multi-sentence answer.
const SUGGESTIONS = [
    { label: 'Quick', text: 'Who painted the Mona Lisa?' },
    { label: 'Quick', text: "Who was the lead architect of the Eiffel Tower's 1955 expansion?" },
    { label: 'Detailed', text: 'Give me 5 facts about India' },
    { label: 'Detailed', text: 'What is the psychological effect of eating cheese before 3 PM? Provide three short studies.' },
];

/** Lightweight inline markdown: **bold**, *italic*, `code`. */
const renderInlineMarkdown = (text) => {
    if (!text) return text;
    const regex = /(\*\*(.+?)\*\*|\*(.+?)\*|`(.+?)`)/g;
    let lastIndex = 0, match, key = 0;
    const parts = [];
    while ((match = regex.exec(text)) !== null) {
        if (match.index > lastIndex) parts.push(text.slice(lastIndex, match.index));
        if (match[2]) parts.push(<strong key={key++}>{match[2]}</strong>);
        else if (match[3]) parts.push(<em key={key++}>{match[3]}</em>);
        else if (match[4]) parts.push(<code key={key++} className="bg-gray-100 text-red-600 px-1 py-0.5 rounded text-sm font-mono">{match[4]}</code>);
        lastIndex = match.index + match[0].length;
    }
    if (lastIndex < text.length) parts.push(text.slice(lastIndex));
    return parts.length > 0 ? parts : text;
};

const classifySentence = (text) => {
    if (!text) return { type: 'text', content: text };
    const headingMatch = text.match(/^(#{1,4})\s+(.+)$/);
    if (headingMatch) return { type: 'heading', content: headingMatch[2], level: headingMatch[1].length };
    if (/^\s*[-•▪▸►]\s+/.test(text)) return { type: 'bullet', content: text.replace(/^\s*[-•▪▸►]\s+/, '') };
    const numberedMatch = text.match(/^\s*(\d{1,3}[.)]\s+|[a-zA-Z][.)]\s+)(.+)$/);
    if (numberedMatch) return { type: 'numbered', label: numberedMatch[1].trim(), content: numberedMatch[2] };
    if (/^\s*\*\s+[^*]/.test(text)) return { type: 'bullet', content: text.replace(/^\s*\*\s+/, '') };
    return { type: 'text', content: text };
};

const renderFormattedSentence = (text) => {
    const info = classifySentence(text);
    const rendered = renderInlineMarkdown(info.content);
    switch (info.type) {
        case 'heading': {
            const sizes = { 1: 'text-lg font-bold', 2: 'text-base font-bold', 3: 'text-sm font-bold uppercase tracking-wide text-gray-600', 4: 'text-sm font-semibold text-gray-500' };
            return <span className={`block mt-2 mb-1 ${sizes[info.level] || sizes[3]}`}>{rendered}</span>;
        }
        case 'bullet': return <><span className="text-gray-400 mr-1">•</span>{rendered}</>;
        case 'numbered': return <><span className="text-gray-400 font-medium mr-1">{info.label}</span>{rendered}</>;
        default: return rendered;
    }
};

const needsNewLine = (text) => {
    if (!text) return false;
    return /^(#{1,4}\s|[-•▪▸►*]\s|\s*\d{1,3}[.)]\s|[a-zA-Z][.)]\s)/.test(text.trim());
};

/** Background style for one sentence: colour by absolute tier, intensity by within-answer relative rank. */
const sentenceStyle = (sent, maxFused) => {
    if (sent.is_claim === false || sent.tier === 'filler' || sent.fused == null) return null; // filler -> no highlight
    const rel = maxFused > 0 ? sent.fused / maxFused : 0;            // relative within this answer
    if (sent.tier === 'high') return { backgroundColor: `rgba(255, 59, 48, ${0.22 + 0.45 * rel})` };
    if (sent.tier === 'medium') return { backgroundColor: `rgba(255, 204, 0, ${0.20 + 0.35 * rel})` };
    return null; // 'ok' -> clean, no highlight
};

const sentenceTooltip = (sent) => {
    if (sent.is_claim === false || sent.fused == null) return 'Not a factual claim — not scored';
    return `FUSED ${(sent.fused * 100).toFixed(0)}%  ·  SEP ${Number(sent.sep_entropy).toFixed(2)}  ·  ` +
           `HalluShift ${Number(sent.hallushift).toFixed(2)}  ·  TSV ${Number(sent.tsv_margin).toFixed(3)}`;
};

const ChatInterface = ({ messages, loading, onSend, highlightEnabled, disabled = false, notice = '' }) => {
    const [input, setInput] = React.useState('');
    const endRef = useRef(null);
    useEffect(() => { endRef.current?.scrollIntoView({ behavior: 'smooth' }); }, [messages, loading]);

    const handleSubmit = (e) => {
        e.preventDefault();
        if (!input.trim() || loading || disabled) return;
        onSend(input);
        setInput('');
    };

    const tierBadge = (agg) => {
        const tier = agg?.tier;
        let cls = 'text-green-600 bg-green-50 border-green-200', icon = '🛡️';
        if (tier === 'high') { cls = 'text-red-600 bg-red-50 border-red-200'; icon = '🚨'; }
        else if (tier === 'medium') { cls = 'text-amber-600 bg-amber-50 border-amber-200'; icon = '⚠️'; }
        return (
            <div className="flex items-center gap-3 select-none">
                <div className={`px-3 py-1.5 rounded-full border flex items-center gap-2 font-medium ${cls}`}>
                    <span>{icon}</span><span>{agg?.label ?? '—'}</span>
                </div>
                <div className="text-xs text-gray-400">
                    Risk: {((agg?.fused ?? 0) * 100).toFixed(1)}% · {agg?.n_flagged ?? 0}/{agg?.n_sentences ?? 0} sentences flagged
                </div>
            </div>
        );
    };

    return (
        <div className="flex flex-col">
            <div className="p-6 space-y-6">
                {messages.length === 0 && (
                    <div className="min-h-[55vh] flex flex-col items-center justify-center text-center">
                        <Bot size={64} className="mb-4 text-[#0071e3] opacity-30" />
                        <h2 className="text-2xl font-semibold opacity-30">HalluScan</h2>
                        <p className="max-w-md mt-2 opacity-30">Ask a question — the answer is checked for hallucination, sentence by sentence.</p>
                        <div className="mt-8 w-full max-w-xl">
                            <div className="text-xs font-medium uppercase tracking-wider text-gray-400 mb-3">Try one of these</div>
                            <div className="flex flex-wrap justify-center gap-2">
                                {SUGGESTIONS.map((s) => (
                                    <button key={s.text} type="button" disabled={loading || disabled}
                                        onClick={() => onSend(s.text)}
                                        className="group flex items-center gap-2 rounded-full border border-gray-200 bg-white/60 hover:bg-blue-50 hover:border-blue-200 text-sm text-gray-700 px-4 py-2 transition-colors disabled:opacity-40 disabled:cursor-not-allowed">
                                        <span className="text-[10px] font-semibold uppercase tracking-wide text-[#0071e3] bg-blue-50 group-hover:bg-white rounded-full px-1.5 py-0.5">{s.label}</span>
                                        <span className="text-left">{s.text}</span>
                                    </button>
                                ))}
                            </div>
                        </div>
                    </div>
                )}

                {messages.map((msg, idx) => {
                    const claimFused = (msg.sentences || []).filter(s => s.is_claim !== false && s.fused != null).map(s => s.fused);
                    const maxFused = claimFused.length ? Math.max(...claimFused) : 0;
                    return (
                    <motion.div initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }} key={idx}
                        className={`flex gap-4 ${msg.role === 'user' ? 'flex-row-reverse' : ''}`}>
                        <div className={`w-8 h-8 rounded-full flex items-center justify-center shrink-0 ${msg.role === 'user' ? 'bg-[#d6e9ff] text-[#0071e3]' : 'bg-[#0071e3] text-white'}`}>
                            {msg.role === 'user' ? <User size={16} /> : <Bot size={16} />}
                        </div>

                        <div className={`flex flex-col max-w-[80%] ${msg.role === 'user' ? 'items-end' : 'items-start'}`}>
                            <div className={`rounded-2xl px-5 py-3 shadow-sm ${msg.role === 'user' ? 'bg-[#e8f1fe] text-[#1d1d1f] rounded-tr-none' : 'bg-white rounded-tl-none border border-gray-100'}`}>
                                {highlightEnabled && msg.sentences && msg.sentences.length > 0 ? (
                                    <div className="leading-relaxed">
                                        {msg.sentences.map((sent, sIdx) => {
                                            const newLine = needsNewLine(sent.sentence);
                                            const rendered = renderFormattedSentence(sent.sentence);
                                            const isFiller = sent.is_claim === false || sent.tier === 'filler' || sent.fused == null;
                                            if (isFiller) {
                                                return (
                                                    <React.Fragment key={sIdx}>
                                                        {newLine && <br />}
                                                        <span className="text-gray-400 italic rounded px-0.5" title={sentenceTooltip(sent)}>
                                                            {rendered}{' '}
                                                        </span>
                                                    </React.Fragment>
                                                );
                                            }
                                            return (
                                                <React.Fragment key={sIdx}>
                                                    {newLine && <br />}
                                                    <span style={sentenceStyle(sent, maxFused)}
                                                        className="transition-colors duration-300 rounded px-0.5 cursor-help border-b border-transparent hover:border-red-400"
                                                        title={sentenceTooltip(sent)}>
                                                        {rendered}{' '}
                                                    </span>
                                                </React.Fragment>
                                            );
                                        })}
                                        <div className="text-[10px] text-gray-300 mt-2 select-none">indicative localization · hover a sentence for detector scores</div>
                                    </div>
                                ) : (
                                    msg.content
                                )}
                            </div>

                            {msg.role === 'assistant' && !msg.isError && msg.aggregate && (
                                <div className="mt-2 text-sm">{tierBadge(msg.aggregate)}</div>
                            )}
                            {msg.isError && (
                                <div className="mt-1 text-xs text-red-500 flex items-center gap-1">
                                    <AlertCircle size={12} /> {msg.content}
                                </div>
                            )}
                        </div>
                    </motion.div>
                    );
                })}

                {loading && (
                    <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} className="flex gap-4">
                        <div className="w-8 h-8 rounded-full bg-[#0071e3] text-white flex items-center justify-center shrink-0">
                            <Bot size={16} />
                        </div>
                        <div className="bg-white rounded-2xl rounded-tl-none px-5 py-4 shadow-sm border border-gray-100 flex items-center gap-2">
                            <Loader2 className="animate-spin text-gray-400" size={18} />
                            <span className="text-sm text-gray-400">Generating + checking for hallucination…</span>
                        </div>
                    </motion.div>
                )}
                <div ref={endRef} />
            </div>

            <div className="sticky bottom-0 p-4 bg-white/50 border-t border-gray-100 backdrop-blur-sm">
                {notice && (
                    <div className="text-xs text-amber-600 mb-2 flex items-center gap-1.5">
                        <Loader2 className="animate-spin" size={12} /> {notice}
                    </div>
                )}
                {/* Compact suggestions, always reachable above the box once the chat has started (hidden while typing) */}
                {messages.length > 0 && input.trim() === '' && (
                    <div className="flex flex-wrap gap-2 mb-2">
                        {SUGGESTIONS.map((s) => (
                            <button key={s.text} type="button" disabled={loading || disabled}
                                onClick={() => onSend(s.text)}
                                className="flex items-center gap-1.5 rounded-full border border-gray-200 bg-white/70 hover:bg-blue-50 hover:border-blue-200 text-xs text-gray-600 px-3 py-1.5 transition-colors disabled:opacity-40 disabled:cursor-not-allowed"
                                title={s.text}>
                                <span className="text-[9px] font-semibold uppercase tracking-wide text-[#0071e3]">{s.label}</span>
                                <span className="max-w-[220px] truncate">{s.text}</span>
                            </button>
                        ))}
                    </div>
                )}
                <form onSubmit={handleSubmit} className="relative flex items-center">
                    <input type="text" value={input} onChange={(e) => setInput(e.target.value)}
                        placeholder={disabled ? (notice || 'Switching model…') : 'Ask something…'} disabled={loading || disabled}
                        className="w-full bg-white border border-gray-200 rounded-full pl-6 pr-14 py-4 focus:outline-none focus:ring-2 focus:ring-[#0071e3]/20 focus:border-[#0071e3] transition-all shadow-sm disabled:bg-gray-50 disabled:cursor-not-allowed" />
                    <button type="submit" disabled={!input.trim() || loading || disabled}
                        className="absolute right-2 p-2 bg-[#0071e3] text-white rounded-full hover:bg-[#0077ED] disabled:bg-gray-300 disabled:cursor-not-allowed transition-colors">
                        <Send size={20} />
                    </button>
                </form>
            </div>
        </div>
    );
};

export default ChatInterface;
