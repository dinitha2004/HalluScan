import React from 'react';
import { X, Info, AlertTriangle, ShieldCheck } from 'lucide-react';

const Guidance = ({ onClose }) => {
    return (
        <div className="fixed inset-0 bg-black/50 backdrop-blur-sm z-50 flex items-center justify-center p-4">
            <div className="bg-white rounded-3xl max-w-2xl w-full shadow-2xl overflow-hidden animate-in fade-in zoom-in duration-200">
                <div className="p-6 border-b border-gray-100 flex justify-between items-center bg-[#f5f5f7]">
                    <h2 className="text-xl font-semibold flex items-center gap-2">
                        <Info className="text-[#0071e3]" size={24} /> Understanding the Risk Score
                    </h2>
                    <button onClick={onClose} className="p-2 hover:bg-gray-200 rounded-full transition-colors"><X size={20} /></button>
                </div>

                <div className="p-6 space-y-6 overflow-y-auto max-h-[80vh]">
                    <section>
                        <h3 className="text-lg font-semibold mb-2 text-gray-800">How HalluScan works</h3>
                        <div className="bg-blue-50 border border-blue-100 rounded-2xl p-4">
                            <p className="text-gray-700 leading-relaxed">
                                HalluScan fuses <strong>three</strong> hallucination signals on a single answer:
                                <strong> SEP</strong> (semantic-entropy probe), <strong>HalluShift</strong> (cross-layer
                                distribution shift), and <strong>TSV</strong> (truthfulness steering direction). A small
                                meta-classifier combines them into one <strong>fused hallucination probability</strong>.
                            </p>
                            <p className="text-gray-600 mt-2 text-sm">
                                The big number is the <strong>answer-level</strong> (calibrated) risk. Long answers are also
                                split into sentences; only <strong>factual-claim</strong> sentences are scored and highlighted —
                                fillers, meta-commentary and questions are skipped.
                            </p>
                        </div>
                    </section>

                    <section>
                        <h3 className="text-lg font-semibold mb-2 text-gray-800">Risk bands</h3>
                        <div className="grid grid-cols-1 md:grid-cols-3 gap-3">
                            <div className="p-4 border rounded-xl bg-green-50/50 border-green-100 text-center">
                                <div className="font-medium text-green-800 mb-1 flex items-center justify-center gap-2"><ShieldCheck size={16} /> Reliable</div>
                                <div className="text-2xl font-bold text-green-600 my-1">&lt; 50%</div>
                                <div className="text-xs text-green-700">Low hallucination risk.</div>
                            </div>
                            <div className="p-4 border rounded-xl bg-amber-50/50 border-amber-100 text-center">
                                <div className="font-medium text-amber-800 mb-1 flex items-center justify-center gap-2"><AlertTriangle size={16} /> Uncertain</div>
                                <div className="text-2xl font-bold text-amber-600 my-1">50–74%</div>
                                <div className="text-xs text-amber-700">Verify with another source.</div>
                            </div>
                            <div className="p-4 border rounded-xl bg-red-50/50 border-red-100 text-center">
                                <div className="font-medium text-red-800 mb-1 flex items-center justify-center gap-2"><AlertTriangle size={16} /> Likely Hallucinated</div>
                                <div className="text-2xl font-bold text-red-600 my-1">&ge; 74%</div>
                                <div className="text-xs text-red-700">Do not trust without checking.</div>
                            </div>
                        </div>
                    </section>

                    <section>
                        <h3 className="text-lg font-semibold mb-2 text-gray-800">Sentence highlights</h3>
                        <div className="bg-gray-50 border border-gray-200 rounded-2xl p-4">
                            <p className="text-gray-600 text-sm leading-relaxed">
                                Yellow = medium-risk claim, red = high-risk claim; the most suspicious sentence is the most
                                saturated. Hover any highlighted sentence to see the individual SEP / HalluShift / TSV / FUSED
                                scores. Sentence highlights are <strong>indicative localization</strong> — there is no
                                per-sentence ground truth, so the calibrated number is the answer-level risk above.
                            </p>
                        </div>
                    </section>

                    <section>
                        <h3 className="text-lg font-semibold mb-2 text-gray-800">Connecting the backend</h3>
                        <div className="bg-gray-50 border border-gray-200 rounded-2xl p-4">
                            <p className="text-gray-600 text-sm leading-relaxed">
                                The model runs on a separate machine (your PC, or a Colab GPU exposed via ngrok). Paste that
                                backend URL into the field in the top bar — it is saved in your browser. Running locally, leave
                                it as <code className="bg-gray-100 px-1 rounded">localhost:8000</code>.
                            </p>
                        </div>
                    </section>
                </div>

                <div className="p-4 bg-gray-50 border-t flex justify-end">
                    <button onClick={onClose} className="px-6 py-2 bg-[#1d1d1f] text-white rounded-full hover:bg-black transition-all font-medium">Got it</button>
                </div>
            </div>
        </div>
    );
};

export default Guidance;
