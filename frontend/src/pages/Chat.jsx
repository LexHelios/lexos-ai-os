import { useEffect, useMemo, useState } from "react";
import { Toaster, toast } from "sonner";
import { aiChat, getProvidersStatus } from "../api";

const ProviderPill = ({ name, enabled }) => (
  <span className={`px-2 py-1 rounded text-xs ${enabled ? "bg-emerald-600" : "bg-gray-700"}`}>{name}</span>
);

export default function ChatPage() {
  const [providers, setProviders] = useState({ local: false, together: false, openrouter: false });
  const [provider, setProvider] = useState("auto");
  const [model, setModel] = useState("default");
  const [temperature, setTemperature] = useState(0.7);
  const [maxTokens, setMaxTokens] = useState(500);
  const [prompt, setPrompt] = useState("");
  const [imageUrl, setImageUrl] = useState("");
  const [answer, setAnswer] = useState("");
  const [answerMeta, setAnswerMeta] = useState({ provider: "-", model: "-" });
  const [loading, setLoading] = useState(false);

  async function refreshProviders() {
    try {
      const s = await getProvidersStatus();
      setProviders(s);
    } catch (e) {
      // eslint-disable-next-line no-console
      console.error(e);
    }
  }

  useEffect(() => {
    refreshProviders();
  }, []);

  const onSend = async (e) => {
    e.preventDefault();
    if (!prompt.trim() && !imageUrl.trim()) {
      toast.error("Enter a prompt or image URL");
      return;
    }
    setLoading(true);
    setAnswer("");
    setAnswerMeta({ provider: "-", model: "-" });
    try {
      const messages = [{ role: "user", content: prompt.trim() }];
      const usedProvider = provider === "auto" ? null : provider;
      const res = await aiChat({ messages, model, temperature, max_tokens: maxTokens, provider: usedProvider, image_url: imageUrl || null });
      setAnswer(res.content);
      setAnswerMeta({ provider: res.provider, model: res.model });
    } catch (e) {
      // eslint-disable-next-line no-console
      console.error(e);
      toast.error(e?.detail || "Chat failed");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-[#0f0f10] text-white">
      <div className="max-w-5xl mx-auto px-4 py-10">
        <h1 className="text-2xl font-semibold">Multimodal Chat</h1>
        <p className="text-gray-400 mt-1">Primary: Local H100 (OpenAI/Ollama), Fallback: Together.ai → OpenRouter</p>

        <div className="mt-4 flex gap-2 items-center flex-wrap">
          <ProviderPill name="Local" enabled={providers.local} />
          <ProviderPill name="Together" enabled={providers.together} />
          <ProviderPill name="OpenRouter" enabled={providers.openrouter} />
          <button className="ml-2 text-xs underline text-emerald-400" onClick={refreshProviders}>refresh</button>
        </div>

        <form onSubmit={onSend} className="mt-6 space-y-4 bg-[#151517] rounded-lg p-5 border border-gray-800">
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            <div>
              <label className="block text-sm text-gray-400 mb-1">Provider</label>
              <select className="w-full rounded border border-gray-700 bg-transparent px-3 py-2" value={provider} onChange={(e) => setProvider(e.target.value)}>
                <option className="text-black" value="auto">Auto</option>
                <option className="text-black" value="local">Local</option>
                <option className="text-black" value="together">Together</option>
                <option className="text-black" value="openrouter">OpenRouter</option>
              </select>
            </div>
            <div>
              <label className="block text-sm text-gray-400 mb-1">Model</label>
              <input className="w-full rounded border border-gray-700 bg-transparent px-3 py-2" value={model} onChange={(e) => setModel(e.target.value)} placeholder="default or provider model id" />
            </div>
            <div className="flex items-center gap-3">
              <div className="flex-1">
                <label className="block text-sm text-gray-400 mb-1">Temperature: {temperature}</label>
                <input type="range" min="0" max="2" step="0.1" value={temperature} onChange={(e) => setTemperature(Number(e.target.value))} className="w-full" />
              </div>
              <div className="w-40">
                <label className="block text-sm text-gray-400 mb-1">Max tokens</label>
                <input type="number" min="50" max="4000" className="w-full rounded border border-gray-700 bg-transparent px-3 py-2" value={maxTokens} onChange={(e) => setMaxTokens(Number(e.target.value))} />
              </div>
            </div>
          </div>

          <div>
            <label className="block text-sm text-gray-400 mb-1">Prompt</label>
            <textarea className="w-full rounded border border-gray-700 bg-transparent px-3 py-2 min-h-28" value={prompt} onChange={(e) => setPrompt(e.target.value)} placeholder="Ask anything..." />
          </div>

          <div>
            <label className="block text-sm text-gray-400 mb-1">Image URL (optional)</label>
            <input className="w-full rounded border border-gray-700 bg-transparent px-3 py-2" value={imageUrl} onChange={(e) => setImageUrl(e.target.value)} placeholder="https://..." />
          </div>

          <div className="flex justify-end gap-3">
            <button type="submit" disabled={loading} className="px-4 py-2 rounded bg-emerald-600 hover:bg-emerald-500 disabled:opacity-50">{loading ? "Sending..." : "Send"}</button>
          </div>
        </form>

        <section className="mt-6">
          <div className="text-sm text-gray-400">Used provider: <span className="text-white">{answerMeta.provider}</span> · Model: <span className="text-white">{answerMeta.model}</span></div>
          <div className="mt-2 p-4 bg-black/30 rounded border border-gray-800 whitespace-pre-wrap min-h-24">{answer || ""}</div>
        </section>
      </div>
      <Toaster richColors position="top-right" />
    </div>
  );
}