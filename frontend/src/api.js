import axios from "axios";

const BACKEND_URL = process.env.REACT_APP_BACKEND_URL;
if (!BACKEND_URL) {
  // eslint-disable-next-line no-console
  console.warn("REACT_APP_BACKEND_URL is not set. API calls will fail.");
}

export const api = axios.create({
  baseURL: `${BACKEND_URL}/api`,
});

// Core
export async function getHealth() {
  const { data } = await api.get(`/health`);
  return data;
}

export async function getVersion() {
  const { data } = await api.get(`/version`);
  return data;
}

export async function getConfig() {
  const { data } = await api.get(`/config`);
  return data;
}

export async function getProvidersStatus() {
  const { data } = await api.get(`/providers/status`);
  return data;
}

// Status
export async function createStatus(payload) {
  const { data } = await api.post(`/status`, payload);
  return data;
}

export async function listStatus({ limit = 10, offset = 0, q = "" } = {}) {
  const params = new URLSearchParams();
  params.set("limit", String(limit));
  params.set("offset", String(offset));
  if (q) params.set("q", q);
  const { data } = await api.get(`/status?${params.toString()}`);
  return data;
}

export async function getStatusCounts() {
  const { data } = await api.get(`/status/count`);
  return data;
}

export function exportStatusCSV(client = "") {
  const url = new URL(`${BACKEND_URL}/api/status/export`);
  if (client) url.searchParams.set("client", client);
  window.location.href = url.toString();
}

export async function purgeStatus({ client_name = "", older_than_hours = null } = {}) {
  const { data } = await api.post(`/status/purge`, { client_name, older_than_hours });
  return data;
}

// AI
export async function aiSummarize({ hours = 24, limit = 200, model = "default", temperature = 0.3 } = {}) {
  const { data } = await api.post(`/ai/summarize`, { hours, limit, model, temperature });
  return data;
}

export async function aiInsights({ hours = 24, limit = 500 } = {}) {
  const { data } = await api.post(`/ai/insights`, { hours, limit });
  return data;
}

export async function aiChat({ messages, model = "default", temperature = 0.7, max_tokens = 500, provider = null, image_url = null }) {
  const { data } = await api.post(`/ai/chat`, { messages, model, temperature, max_tokens, provider, image_url });
  return data;
}