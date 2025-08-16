import axios from "axios";

const BACKEND_URL = process.env.REACT_APP_BACKEND_URL;
if (!BACKEND_URL) {
  // eslint-disable-next-line no-console
  console.warn("REACT_APP_BACKEND_URL is not set. API calls will fail.");
}

export const api = axios.create({
  baseURL: `${BACKEND_URL}/api`,
});

export async function getHealth() {
  const { data } = await api.get(`/health`);
  return data;
}

export async function getVersion() {
  const { data } = await api.get(`/version`);
  return data;
}

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