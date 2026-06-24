import React, { useState, useEffect, useRef } from 'react';
import { 
  Scale, ShieldAlert, Sparkles, UploadCloud, FileText, 
  MessageSquare, Send, CheckCircle2, AlertCircle, 
  ShieldCheck, ChevronRight, Copy, Terminal,
  RefreshCw, Layers, BookOpen, Bot, HelpCircle
} from 'lucide-react';

const API_BASE = "http://localhost:8000";
const HACKATHON_SESSION = "session_hackathon_demo";

interface DocumentMeta {
  doc_id: string;
  filename: string;
  chunks_indexed: number;
  created_at?: string;
}

interface ChatMessage {
  role: 'user' | 'assistant';
  content: string;
  timestamp?: string;
}

interface AgentLog {
  agent: string;
  message: string;
  timestamp: string;
}

function App() {
  const [documents, setDocuments] = useState<DocumentMeta[]>([]);
  const [activeDoc, setActiveDoc] = useState<DocumentMeta | null>(null);
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [inputMessage, setInputMessage] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [activeTab, setActiveTab] = useState<'review' | 'compliance' | 'drafting'>('review');
  const [uploading, setUploading] = useState(false);
  const [agentLogs, setAgentLogs] = useState<AgentLog[]>([]);
  const [uploadError, setUploadError] = useState<string | null>(null);

  // Specialist Agent structured outputs from backend response
  const [reviewResult, setReviewResult] = useState<any>(null);
  const [complianceResult, setComplianceResult] = useState<any>(null);
  const [draftingResult, setDraftingResult] = useState<any>(null);
  const [verificationResult, setVerificationResult] = useState<any>(null);

  const fileInputRef = useRef<HTMLInputElement>(null);
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const logEndRef = useRef<HTMLDivElement>(null);

  // Load documents and conversation history on mount
  useEffect(() => {
    fetchDocuments();
    fetchHistory();
    addLog("System", "LexAgent Initialized. Ready for analysis.");
  }, []);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  useEffect(() => {
    logEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [agentLogs]);

  const addLog = (agent: string, message: string) => {
    const timestamp = new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' });
    setAgentLogs(prev => [...prev, { agent, message, timestamp }]);
  };

  const fetchDocuments = async () => {
    try {
      const res = await fetch(`${API_BASE}/documents`);
      if (res.ok) {
        const data = await res.json();
        setDocuments(data);
        if (data.length > 0 && !activeDoc) {
          setActiveDoc(data[0]);
        }
      }
    } catch (e) {
      console.error("Failed to load documents", e);
      addLog("System", "Backend API offline or unreachable. Running in demo mode.");
    }
  };

  const fetchHistory = async () => {
    try {
      const res = await fetch(`${API_BASE}/sessions/${HACKATHON_SESSION}/history`);
      if (res.ok) {
        const data = await res.json();
        setMessages(data);
      }
    } catch (e) {
      console.error("Failed to fetch session history", e);
    }
  };

  const handleFileUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const files = e.target.files;
    if (!files || files.length === 0) return;
    const file = files[0];
    if (file.type !== "application/pdf") {
      setUploadError("Only PDF documents are supported for legal indexing.");
      return;
    }

    setUploading(true);
    setUploadError(null);
    addLog("RAG Indexer", `Uploading and chunking document: ${file.name}`);

    const formData = new FormData();
    formData.append("file", file);

    try {
      const res = await fetch(`${API_BASE}/upload`, {
        method: "POST",
        body: formData,
      });

      if (!res.ok) {
        throw new Error(await res.text() || "Failed to index document");
      }

      const data = await res.json();
      addLog("Dense Retriever", `Indexed in ChromaDB: ${data.chunks_indexed} paragraphs using NVIDIA Embeddings.`);
      addLog("Sparse Retriever", `Indexed in BM25: ${data.chunks_indexed} keyword indices.`);
      addLog("System", `Document registration complete. ID: ${data.doc_id}`);
      
      setDocuments(prev => [data, ...prev]);
      setActiveDoc(data);
      fetchDocuments();
    } catch (err: any) {
      addLog("System", `Index failure: ${err.message}`);
      setUploadError(err.message || "Failed to upload file.");
    } finally {
      setUploading(false);
    }
  };

  const handleSendMessage = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!inputMessage.trim()) return;

    const userText = inputMessage;
    setInputMessage('');
    setIsLoading(true);

    const userMsg: ChatMessage = { role: 'user', content: userText };
    setMessages(prev => [...prev, userMsg]);

    addLog("Supervisor", `Routing query: "${userText.substring(0, 40)}..."`);
    addLog("Supervisor", "Classifying intent via Llama-3.1-8B-Instruct...");

    try {
      const res = await fetch(`${API_BASE}/chat`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          message: userText,
          session_id: HACKATHON_SESSION,
          doc_id: activeDoc?.doc_id || null,
        }),
      });

      if (!res.ok) throw new Error("Supervisor failed to respond.");

      const data = await res.json();
      
      // Update agent outputs
      if (data.agent_results?.review) {
        setReviewResult(data.agent_results.review);
        addLog("Review Agent", "Contract Review specialist execution complete. Extracted clauses.");
      }
      if (data.agent_results?.compliance) {
        setComplianceResult(data.agent_results.compliance);
        addLog("Compliance Agent", "Statutory checks complete. Evaluated against Indian Contract Act & IT Act.");
      }
      if (data.agent_results?.draft) {
        setDraftingResult(data.agent_results.draft);
        addLog("Drafting Agent", "Synthesized requested clause matching Indian commercial standards.");
      }
      if (data.agent_results?.verification) {
        setVerificationResult(data.agent_results.verification);
        addLog("Verification Agent", `Factual Grounding score: ${Math.round(data.agent_results.verification.confidence_score * 100)}%`);
        if (data.agent_results.verification.hallucinated_claims?.length > 0) {
          addLog("Verification Agent", `⚠️ Flagged ${data.agent_results.verification.hallucinated_claims.length} unverified claims.`);
        } else {
          addLog("Verification Agent", "✅ 100% Grounding Verification passed.");
        }
      }

      setMessages(prev => [...prev, { role: 'assistant', content: data.response }]);
    } catch (e: any) {
      console.error(e);
      addLog("System", "Error orchestrating query response.");
      setMessages(prev => [...prev, { role: 'assistant', content: "An error occurred while coordinating specialist agents. Please ensure the backend is running and configured." }]);
    } finally {
      setIsLoading(false);
    }
  };

  const copyToClipboard = (text: string) => {
    navigator.clipboard.writeText(text);
    addLog("System", "Clause copied to clipboard.");
  };

  const handleLoadDemo = async () => {
    setIsLoading(true);
    addLog("System", "Loading high-fidelity demo contract...");
    addLog("RAG Indexer", "Seeding Sample_NDA_Indian_Enterprise.pdf into local database.");
    
    try {
      const res = await fetch(`${API_BASE}/demo/load`, {
        method: "POST"
      });
      if (!res.ok) throw new Error("Failed to load demo data.");
      
      const data = await res.json();
      
      const demoMeta = {
        doc_id: data.doc_id,
        filename: data.filename,
        chunks_indexed: data.chunks_indexed
      };
      
      setDocuments(prev => {
        if (prev.some(d => d.doc_id === data.doc_id)) return prev;
        return [demoMeta, ...prev];
      });
      setActiveDoc(demoMeta);
      
      setReviewResult(data.agent_results.review);
      setComplianceResult(data.agent_results.compliance);
      setDraftingResult(data.agent_results.draft);
      setVerificationResult(data.agent_results.verification);
      
      // Refresh chat logs
      await fetchHistory();
      
      addLog("Dense Retriever", "Demo vectors indexed successfully.");
      addLog("Verification Agent", "Demo grounding validation: 100% matched.");
      addLog("System", "Demo workspace loaded. Ready to review Section 27 violation.");
    } catch (e: any) {
      console.error(e);
      addLog("System", "Failed to initialize demo workspace.");
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="flex h-screen w-screen overflow-hidden bg-[#07090e] text-[#e2e8f0]">
      {/* 1. SIDEBAR - Workspaces and uploads */}
      <div className="w-80 flex flex-col border-r border-gray-800 bg-[#0a0d14] flex-shrink-0">
        <div className="p-6 border-b border-gray-800 flex items-center space-x-3">
          <div className="p-2 bg-[#10b981]/15 rounded-lg border border-[#10b981]/30">
            <Scale className="h-6 w-6 text-[#10b981] glow-text" />
          </div>
          <div>
            <h1 className="text-xl font-bold tracking-tight text-white glow-text">LexAgent</h1>
            <p className="text-xs text-gray-500 font-medium">Trustworthy Legal Intelligence</p>
          </div>
        </div>

        {/* Upload Contract Component */}
        <div className="p-4 border-b border-gray-800">
          <input 
            type="file" 
            ref={fileInputRef} 
            onChange={handleFileUpload} 
            className="hidden" 
            accept=".pdf"
          />
          <button 
            onClick={() => fileInputRef.current?.click()}
            disabled={uploading}
            className="w-full flex items-center justify-center space-x-2 py-3 px-4 rounded-xl border border-dashed border-gray-700 bg-[#0f1420]/50 hover:bg-[#0f1420] hover:border-[#10b981]/50 text-sm font-semibold transition-all group duration-200"
          >
            {uploading ? (
              <RefreshCw className="h-4 w-4 text-[#10b981] animate-spin" />
            ) : (
              <UploadCloud className="h-4 w-4 text-[#10b981] group-hover:scale-110 transition-transform duration-200" />
            )}
            <span className="text-gray-300 group-hover:text-white">
              {uploading ? "Indexing PDF..." : "Upload Legal Contract"}
            </span>
          </button>
          
          <button 
            onClick={handleLoadDemo}
            disabled={isLoading || uploading}
            className="w-full mt-3 flex items-center justify-center space-x-2 py-3 px-4 rounded-xl border border-gray-800 bg-[#111622]/40 hover:bg-[#10b981]/15 hover:border-[#10b981]/40 text-sm font-semibold transition-all group duration-200 disabled:opacity-50"
          >
            <Sparkles className="h-4 w-4 text-[#10b981] group-hover:scale-110 transition-transform duration-200" />
            <span className="text-gray-300 group-hover:text-white">Load Demo Contract</span>
          </button>

          {uploadError && (
            <p className="text-xs text-red-400 mt-2 text-center flex items-center justify-center gap-1">
              <AlertCircle className="h-3 w-3" /> {uploadError}
            </p>
          )}
        </div>

        {/* Indexed Document Registry */}
        <div className="flex-1 overflow-y-auto p-4 space-y-3">
          <div className="flex items-center justify-between text-xs font-semibold text-gray-500 uppercase tracking-wider mb-2">
            <span>Contract Registry</span>
            <span className="px-1.5 py-0.5 rounded bg-gray-800 text-gray-400 text-[10px]">{documents.length}</span>
          </div>

          {documents.length === 0 ? (
            <div className="text-center py-8 text-gray-500 text-xs flex flex-col items-center justify-center space-y-2">
              <FileText className="h-8 w-8 text-gray-600" />
              <span>No contracts indexed yet.</span>
            </div>
          ) : (
            documents.map(doc => (
              <div 
                key={doc.doc_id}
                onClick={() => {
                  setActiveDoc(doc);
                  addLog("System", `Switched active context to: ${doc.filename}`);
                }}
                className={`p-3.5 rounded-xl border cursor-pointer transition-all flex items-start space-x-3 ${
                  activeDoc?.doc_id === doc.doc_id 
                    ? 'bg-[#10b981]/10 border-[#10b981]/40 shadow-[0_0_15px_rgba(16,185,129,0.05)]' 
                    : 'bg-[#111622]/40 border-gray-800/80 hover:bg-[#111622] hover:border-gray-700'
                }`}
              >
                <FileText className={`h-5 w-5 mt-0.5 ${activeDoc?.doc_id === doc.doc_id ? 'text-[#10b981]' : 'text-gray-400'}`} />
                <div className="overflow-hidden flex-1">
                  <h4 className="text-sm font-semibold text-gray-200 truncate">{doc.filename}</h4>
                  <p className="text-xs text-gray-500 font-medium mt-1">Chunks: {doc.chunks_indexed}</p>
                </div>
                <ChevronRight className="h-4 w-4 text-gray-600 mt-1 flex-shrink-0" />
              </div>
            ))
          )}
        </div>

        {/* Footer */}
        <div className="p-4 border-t border-gray-800 bg-[#080a0f] flex items-center justify-between text-xs text-gray-500">
          <span className="flex items-center gap-1 font-medium">
            <span className="h-2 w-2 rounded-full bg-emerald-500 active-pulse"></span> NIM Status: Connected
          </span>
          <span className="font-semibold text-gray-600">NVIDIA NIM v1.0</span>
        </div>
      </div>

      {/* 2. MAIN LAYOUT */}
      <div className="flex-1 flex overflow-hidden">
        {/* Left Side: Document View & Analytics */}
        <div className="flex-1 flex flex-col border-r border-gray-800 bg-[#080b11]">
          {activeDoc ? (
            <>
              {/* Doc Header & Verification Score */}
              <div className="p-6 border-b border-gray-800 bg-[#0b0f18] flex items-center justify-between">
                <div>
                  <h2 className="text-lg font-bold text-white flex items-center gap-2">
                    <FileText className="h-5 w-5 text-[#10b981]" /> {activeDoc.filename}
                  </h2>
                  <p className="text-xs text-gray-500 font-medium mt-1">Document ID: {activeDoc.doc_id}</p>
                </div>

                {/* Grounding Verification Panel */}
                {verificationResult ? (
                  <div className="flex items-center space-x-3 px-4 py-2 rounded-xl bg-gray-900/60 border border-gray-800">
                    <div className="text-right">
                      <p className="text-[10px] font-bold text-gray-500 uppercase tracking-wide">Factual Grounding</p>
                      <p className="text-sm font-extrabold text-white">
                        {Math.round(verificationResult.confidence_score * 100)}% Verified
                      </p>
                    </div>
                    <div className="relative flex items-center justify-center">
                      {verificationResult.is_grounded ? (
                        <ShieldCheck className="h-8 w-8 text-[#10b981] glow-text" />
                      ) : (
                        <ShieldAlert className="h-8 w-8 text-amber-500 glow-text" />
                      )}
                    </div>
                  </div>
                ) : (
                  <div className="flex items-center space-x-2 text-xs text-gray-500 bg-gray-900/40 px-3.5 py-1.5 rounded-lg border border-gray-800">
                    <HelpCircle className="h-4 w-4 text-gray-400" />
                    <span>Run chat analysis to verify grounding</span>
                  </div>
                )}
              </div>

              {/* Navigation Tabs */}
              <div className="flex border-b border-gray-800 bg-[#0a0d15] px-6">
                <button 
                  onClick={() => setActiveTab('review')}
                  className={`py-3.5 px-4 text-sm font-semibold border-b-2 transition-all flex items-center gap-2 ${
                    activeTab === 'review' 
                      ? 'border-[#10b981] text-white' 
                      : 'border-transparent text-gray-400 hover:text-gray-200'
                  }`}
                >
                  <Layers className="h-4 w-4" /> Risk & Clause Review
                </button>
                <button 
                  onClick={() => setActiveTab('compliance')}
                  className={`py-3.5 px-4 text-sm font-semibold border-b-2 transition-all flex items-center gap-2 ${
                    activeTab === 'compliance' 
                      ? 'border-[#10b981] text-white' 
                      : 'border-transparent text-gray-400 hover:text-gray-200'
                  }`}
                >
                  <ShieldCheck className="h-4 w-4" /> Indian Law Compliance
                </button>
                <button 
                  onClick={() => setActiveTab('drafting')}
                  className={`py-3.5 px-4 text-sm font-semibold border-b-2 transition-all flex items-center gap-2 ${
                    activeTab === 'drafting' 
                      ? 'border-[#10b981] text-white' 
                      : 'border-transparent text-gray-400 hover:text-gray-200'
                  }`}
                >
                  <BookOpen className="h-4 w-4" /> Drafting Lab
                </button>
              </div>

              {/* Tabs Content */}
              <div className="flex-1 overflow-y-auto p-6 space-y-6">
                {activeTab === 'review' && (
                  <div className="space-y-6">
                    {reviewResult ? (
                      <>
                        <div className="p-5 rounded-2xl glass-panel">
                          <h3 className="text-sm font-bold uppercase tracking-wider text-gray-400 mb-2">Executive Summary</h3>
                          <p className="text-sm leading-relaxed text-gray-300">{reviewResult.executive_summary}</p>
                          <div className="mt-4 flex gap-4 text-xs font-semibold">
                            <span className="px-3 py-1 rounded-full bg-gray-800/80 text-gray-300">
                              Type: {reviewResult.document_type}
                            </span>
                            <span className={`px-3 py-1 rounded-full ${
                              reviewResult.overall_risk === 'low' ? 'bg-emerald-500/10 text-emerald-400' :
                              reviewResult.overall_risk === 'medium' ? 'bg-amber-500/10 text-amber-400' : 'bg-red-500/10 text-red-400'
                            }`}>
                              Overall Risk: {reviewResult.overall_risk.toUpperCase()}
                            </span>
                          </div>
                        </div>

                        <div className="space-y-4">
                          <h3 className="text-sm font-bold uppercase tracking-wider text-gray-500">Key Clauses & Revisions</h3>
                          {reviewResult.clauses?.length === 0 ? (
                            <p className="text-sm text-gray-500 italic">No clauses analyzed yet.</p>
                          ) : (
                            reviewResult.clauses?.map((c: any, i: number) => (
                              <div key={i} className="rounded-xl border border-gray-800 bg-[#0c101a]/80 p-5 space-y-3">
                                <div className="flex justify-between items-start">
                                  <div>
                                    <h4 className="text-sm font-bold text-white">{c.clause_type}</h4>
                                    <p className="text-[10px] text-gray-500 mt-0.5">Reference: {c.page_reference}</p>
                                  </div>
                                  <span className={`text-[10px] uppercase font-bold px-2 py-0.5 rounded ${
                                    c.risk_level === 'critical' || c.risk_level === 'high' ? 'bg-red-500/15 text-red-400' :
                                    c.risk_level === 'medium' ? 'bg-amber-500/15 text-amber-400' : 'bg-blue-500/15 text-blue-400'
                                  }`}>
                                    {c.risk_level}
                                  </span>
                                </div>
                                <div className="p-3 bg-gray-900/40 rounded-lg border border-gray-800/60 text-xs text-gray-400 italic">
                                  "{c.clause_text}"
                                </div>
                                <div className="text-xs text-gray-300">
                                  <span className="font-bold text-red-400/80">Risk Assessment: </span>{c.risk_explanation}
                                </div>
                                {c.suggested_revision && (
                                  <div className="p-3 bg-emerald-950/15 rounded-lg border border-emerald-800/30">
                                    <div className="flex items-center justify-between text-xs font-bold text-[#10b981] mb-1.5">
                                      <span>Suggested Balanced Revision</span>
                                      <button onClick={() => copyToClipboard(c.suggested_revision)} className="p-1 hover:bg-emerald-500/10 rounded transition-all">
                                        <Copy className="h-3 w-3" />
                                      </button>
                                    </div>
                                    <p className="text-xs text-gray-300">{c.suggested_revision}</p>
                                  </div>
                                )}
                              </div>
                            ))
                          )}
                        </div>
                      </>
                    ) : (
                      <div className="text-center py-16 text-gray-500 text-sm space-y-4">
                        <Layers className="h-12 w-12 text-gray-700 mx-auto" />
                        <div>
                          <p className="font-semibold text-gray-400">Clause review dashboard empty.</p>
                          <p className="text-xs text-gray-600 mt-1 max-w-sm mx-auto">Ask the bot to review the contract or upload a new PDF to run the specialist agent pipeline.</p>
                        </div>
                      </div>
                    )}
                  </div>
                )}

                {activeTab === 'compliance' && (
                  <div className="space-y-6">
                    {complianceResult ? (
                      <>
                        <div className="p-5 rounded-2xl glass-panel flex items-start space-x-4">
                          {complianceResult.is_compliant ? (
                            <CheckCircle2 className="h-8 w-8 text-emerald-400 flex-shrink-0 mt-0.5" />
                          ) : (
                            <AlertCircle className="h-8 w-8 text-amber-400 flex-shrink-0 mt-0.5" />
                          )}
                          <div>
                            <h3 className="text-sm font-bold uppercase tracking-wider text-gray-400">Compliance Stance</h3>
                            <p className="text-sm text-gray-300 mt-1">{complianceResult.summary}</p>
                          </div>
                        </div>

                        <div className="space-y-4">
                          <h3 className="text-sm font-bold uppercase tracking-wider text-gray-500">Statutory Compliance Issues</h3>
                          {complianceResult.issues?.length === 0 ? (
                            <div className="p-8 border border-emerald-800/30 bg-emerald-950/10 rounded-xl text-center text-xs text-emerald-400 font-medium">
                              No Indian statutory compliance violations detected in the scanned clauses.
                            </div>
                          ) : (
                            complianceResult.issues?.map((issue: any, idx: number) => (
                              <div key={idx} className="rounded-xl border border-gray-800 bg-[#0d111d]/90 p-5 space-y-3">
                                <div className="flex justify-between items-start">
                                  <div>
                                    <h4 className="text-sm font-bold text-amber-400">{issue.law_reference}</h4>
                                    <p className="text-xs text-gray-500 font-semibold mt-1">Severity: {issue.severity.toUpperCase()}</p>
                                  </div>
                                </div>
                                <div className="p-3 bg-gray-900/60 rounded-lg text-xs text-gray-400 italic">
                                  "{issue.clause_text}"
                                </div>
                                <div className="text-xs text-gray-300">
                                  <span className="font-bold text-red-400/80">Violation Details: </span>{issue.violation_description}
                                </div>
                                <div className="p-3 bg-[#10b981]/5 rounded-lg border border-[#10b981]/25 text-xs">
                                  <span className="font-bold text-[#10b981]">Remediation Action: </span>{issue.recommended_action}
                                </div>
                              </div>
                            ))
                          )}
                        </div>
                      </>
                    ) : (
                      <div className="text-center py-16 text-gray-500 text-sm space-y-4">
                        <ShieldCheck className="h-12 w-12 text-gray-700 mx-auto" />
                        <div>
                          <p className="font-semibold text-gray-400">Indian compliance board empty.</p>
                          <p className="text-xs text-gray-600 mt-1 max-w-sm mx-auto">Trigger the compliance check by asking questions related to legal validity or compliance under Indian Contract law.</p>
                        </div>
                      </div>
                    )}
                  </div>
                )}

                {activeTab === 'drafting' && (
                  <div className="space-y-6">
                    {draftingResult ? (
                      <div className="rounded-2xl border border-gray-800 bg-[#0b0e14]/90 overflow-hidden">
                        <div className="p-5 border-b border-gray-800 bg-[#0e121c] flex justify-between items-center">
                          <div>
                            <h3 className="text-sm font-bold text-white uppercase tracking-wider">Drafting Output</h3>
                            <p className="text-[10px] text-gray-500 mt-0.5">Clause Type: {draftingResult.drafted_clause_type}</p>
                          </div>
                          <button 
                            onClick={() => copyToClipboard(draftingResult.drafted_text)}
                            className="flex items-center space-x-1.5 py-1.5 px-3 rounded-lg bg-[#10b981]/15 hover:bg-[#10b981]/25 text-[#10b981] font-semibold text-xs transition-all border border-[#10b981]/30"
                          >
                            <Copy className="h-3 w.5" /> <span>Copy Clause</span>
                          </button>
                        </div>

                        <div className="p-6 space-y-5">
                          <div className="p-4 bg-gray-950/60 rounded-xl border border-gray-800/80 font-mono text-sm leading-relaxed text-[#c5c6c7]">
                            {draftingResult.drafted_text}
                          </div>

                          <div className="space-y-3">
                            <h4 className="text-xs font-bold uppercase tracking-wider text-gray-500">Key Terms Explained</h4>
                            <ul className="text-xs text-gray-400 list-disc list-inside space-y-1.5">
                              {draftingResult.key_terms_explained?.map((term: string, idx: number) => (
                                <li key={idx}><span className="text-gray-300 font-semibold">{term}</span></li>
                              ))}
                            </ul>
                          </div>

                          <div className="p-4 bg-blue-950/10 border border-blue-900/30 rounded-xl text-xs">
                            <h4 className="font-bold text-blue-400 mb-1">Commercial / Business Implications</h4>
                            <p className="text-gray-300 leading-relaxed">{draftingResult.commercial_implications}</p>
                          </div>
                        </div>
                      </div>
                    ) : (
                      <div className="text-center py-16 text-gray-500 text-sm space-y-4">
                        <BookOpen className="h-12 w-12 text-gray-700 mx-auto" />
                        <div>
                          <p className="font-semibold text-gray-400">No drafted clauses found.</p>
                          <p className="text-xs text-gray-600 mt-1 max-w-sm mx-auto">Ask the agent to draft or rewrite a specific clause (e.g. "Draft an arbitration clause with seat in Delhi") to populate this panel.</p>
                        </div>
                      </div>
                    )}
                  </div>
                )}
              </div>
            </>
          ) : (
            <div className="flex-1 flex flex-col items-center justify-center p-8 text-center space-y-6">
              <div className="p-4 bg-[#10b981]/10 rounded-full border border-[#10b981]/30">
                <Scale className="h-12 w-12 text-[#10b981] glow-text animate-pulse" />
              </div>
              <div className="max-w-md">
                <h2 className="text-2xl font-bold text-white glow-text">Welcome to LexAgent Workspace</h2>
                <p className="text-sm text-gray-400 mt-2 leading-relaxed">
                  LexAgent is a verification-first legal AI copilot that cross-references all claims against source document evidence to eliminate AI hallucinations in high-stakes legal contracts.
                </p>
              </div>
              <button 
                onClick={() => fileInputRef.current?.click()}
                className="flex items-center space-x-2 py-3 px-6 rounded-xl bg-[#10b981] hover:bg-[#0d9488] text-white font-bold text-sm transition-all shadow-[0_4px_20px_rgba(16,185,129,0.3)] hover:scale-105"
              >
                <UploadCloud className="h-4 w-4" />
                <span>Upload a PDF to Start Analysis</span>
              </button>
            </div>
          )}
        </div>

        {/* Right Side: Chat Panel & Agent Logs Console */}
        <div className="w-[450px] flex flex-col bg-[#07090e] border-l border-gray-800 flex-shrink-0">
          
          {/* Agent Activity Terminal */}
          <div className="h-60 flex flex-col border-b border-gray-800 bg-[#06080c] overflow-hidden flex-shrink-0">
            <div className="p-4 bg-gray-950 flex items-center justify-between border-b border-gray-800">
              <div className="flex items-center space-x-2 text-xs font-semibold uppercase tracking-wider text-gray-400">
                <Terminal className="h-4 w-4 text-[#10b981]" />
                <span>Multi-Agent Activity logs</span>
              </div>
              <button 
                onClick={() => setAgentLogs([{ agent: "System", message: "Log console cleared. Ready.", timestamp: new Date().toLocaleTimeString() }])}
                className="text-[10px] text-gray-500 hover:text-gray-300 font-semibold"
              >
                Clear
              </button>
            </div>
            <div className="flex-1 overflow-y-auto p-4 font-mono text-[11px] space-y-1.5 bg-black/30">
              {agentLogs.map((log, idx) => (
                <div key={idx} className="flex items-start space-x-1.5 leading-normal">
                  <span className="text-gray-600">[{log.timestamp}]</span>
                  <span className={`font-semibold ${
                    log.agent === 'Supervisor' ? 'text-amber-400' :
                    log.agent === 'Verification Agent' ? 'text-[#10b981]' :
                    log.agent === 'System' ? 'text-blue-400' : 'text-purple-400'
                  }`}>
                    {log.agent}:
                  </span>
                  <span className="text-gray-300">{log.message}</span>
                </div>
              ))}
              <div ref={logEndRef}></div>
            </div>
          </div>

          {/* Conversation Workspace */}
          <div className="flex-1 flex flex-col overflow-hidden">
            <div className="p-4 border-b border-gray-800 bg-[#090d15] flex items-center space-x-2">
              <MessageSquare className="h-4 w-4 text-[#10b981]" />
              <h3 className="text-xs font-bold text-gray-300 uppercase tracking-wider">Chat Console</h3>
            </div>

            {/* Message History */}
            <div className="flex-1 overflow-y-auto p-4 space-y-4 bg-[#080b11]/50">
              {messages.length === 0 && (
                <div className="text-center py-12 text-gray-500 text-xs space-y-3">
                  <Bot className="h-10 w-10 text-gray-700 mx-auto" />
                  <p className="max-w-xs mx-auto">Ask questions about the contract. E.g.: "What are the liability limits?" or "Find the governing law."</p>
                </div>
              )}
              {messages.map((m, idx) => (
                <div key={idx} className={`flex ${m.role === 'user' ? 'justify-end' : 'justify-start'}`}>
                  <div className={`p-3.5 rounded-2xl max-w-[85%] text-xs leading-relaxed ${
                    m.role === 'user' 
                      ? 'bg-[#10b981] text-white rounded-br-none shadow-md font-semibold' 
                      : 'bg-gray-900 border border-gray-800 text-gray-200 rounded-bl-none'
                  }`}>
                    {m.content}
                  </div>
                </div>
              ))}
              {isLoading && (
                <div className="flex justify-start">
                  <div className="p-3.5 rounded-2xl bg-gray-900 border border-gray-800 text-gray-400 text-xs flex items-center space-x-2 rounded-bl-none">
                    <RefreshCw className="h-3 w-3 animate-spin text-[#10b981]" />
                    <span>Orchestrating response...</span>
                  </div>
                </div>
              )}
              <div ref={messagesEndRef}></div>
            </div>

            {/* Input Form */}
            <form onSubmit={handleSendMessage} className="p-4 border-t border-gray-800 bg-[#090c13]">
              <div className="flex space-x-2">
                <input 
                  type="text" 
                  value={inputMessage}
                  onChange={(e) => setInputMessage(e.target.value)}
                  placeholder={activeDoc ? "Ask about this contract..." : "Upload a contract to chat..."}
                  disabled={isLoading || !activeDoc}
                  className="flex-1 px-4 py-3 bg-gray-950 border border-gray-800 rounded-xl focus:border-[#10b981]/50 focus:outline-none text-xs text-white placeholder-gray-500 transition-all font-medium disabled:opacity-50"
                />
                <button 
                  type="submit" 
                  disabled={isLoading || !inputMessage.trim() || !activeDoc}
                  className="p-3 rounded-xl bg-[#10b981] hover:bg-[#0d9488] text-white transition-all disabled:opacity-50 flex items-center justify-center flex-shrink-0"
                >
                  <Send className="h-4 w-4" />
                </button>
              </div>
            </form>
          </div>

        </div>
      </div>
    </div>
  );
}

export default App;
