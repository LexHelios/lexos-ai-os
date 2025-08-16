import axios from "axios";

const BACKEND_URL = process.env.REACT_APP_BACKEND_URL;
if (!BACKEND_URL) {
  console.warn("REACT_APP_BACKEND_URL is not set. API calls will fail.");
}

export const api = axios.create({ baseURL: `${BACKEND_URL}/api` });

// Core
export async function getHealth() { const { data } = await api.get(`/health`); return data; }
export async function getVersion() { const { data } = await api.get(`/version`); return data; }
export async function getConfig() { const { data } = await api.get(`/config`); return data; }
export async function getProvidersStatus() { const { data } = await api.get(`/providers/status`); return data; }
export async function getProvidersTelemetry() { const { data } = await api.get(`/providers/telemetry`); return data; }

// Status
export async function createStatus(payload) { const { data } = await api.post(`/status`, payload); return data; }
export async function listStatus({ limit = 10, offset = 0, q = "" } = {}) {
  const params = new URLSearchParams(); params.set("limit", String(limit)); params.set("offset", String(offset)); if (q) params.set("q", q);
  const { data } = await api.get(`/status?${params.toString()}`); return data;
}
export async function getStatusCounts() { const { data } = await api.get(`/status/count`); return data; }
export function exportStatusCSV(client = "") { const url = new URL(`${BACKEND_URL}/api/status/export`); if (client) url.searchParams.set("client", client); window.location.href = url.toString(); }
export async function purgeStatus({ client_name = "", older_than_hours = null } = {}) { const { data } = await api.post(`/status/purge`, { client_name, older_than_hours }); return data; }

// AI
export async function aiSummarize({ hours = 24, limit = 200, model = "default", temperature = 0.3 } = {}) { const { data } = await api.post(`/ai/summarize`, { hours, limit, model, temperature }); return data; }
export async function aiInsights({ hours = 24, limit = 500 } = {}) { const { data } = await api.post(`/ai/insights`, { hours, limit }); return data; }
export async function aiChat({ messages, model = "default", temperature = 0.7, max_tokens = 500, provider = null, image_url = null, image_b64 = null, image_mime = null }) {
  const { data } = await api.post(`/ai/chat`, { messages, model, temperature, max_tokens, provider, image_url, image_b64, image_mime }); return data;
}

// Streaming via fetch (SSE over POST)
export async function* aiChatStream({ messages, model = "default", temperature = 0.7, max_tokens = 500, provider = null, image_url = null, image_b64 = null, image_mime = null }) {
  const resp = await fetch(`${BACKEND_URL}/api/ai/chat/stream`, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ messages, model, temperature, max_tokens, provider, image_url, image_b64, image_mime }) });
  if (!resp.ok || !resp.body) throw new Error(`HTTP ${resp.status}`);
  const reader = resp.body.getReader(); const decoder = new TextDecoder(); let buffer = "";
  for (;;) { const { value, done } = await reader.read(); if (done) break; buffer += decoder.decode(value, { stream: true }); const parts = buffer.split("\n\n"); buffer = parts.pop() || ""; for (const p of parts) { if (!p.startsWith("data:")) continue; const data = p.slice(5).trim(); if (data === "[DONE]") return; yield data; } }
}

// TTS
export async function ttsVoices() { const { data } = await api.get(`/tts/voices`); return data; }
export async function ttsSynthesize({ text, voice_id, model_id = "eleven_multilingual_v2" }) {
  const resp = await fetch(`${BACKEND_URL}/api/tts/synthesize`, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ text, voice_id, model_id }) });
  if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
  const blob = await resp.blob(); return URL.createObjectURL(blob);
}

// STT
export async function sttTranscribeURL({ url, options = {} }) { const { data } = await api.post(`/stt/transcribe-url`, { url, options }); return data; }
export async function sttTranscribeFile(file) {
  const form = new FormData(); form.append("file", file);
  const resp = await fetch(`${BACKEND_URL}/api/stt/transcribe-file`, { method: "POST", body: form });
  if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
  return await resp.json();
}