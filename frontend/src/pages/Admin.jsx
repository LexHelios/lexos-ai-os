import { useEffect, useState } from "react";
import { Toaster, toast } from "sonner";
import { aiInsights, aiSummarize, exportStatusCSV, purgeStatus } from "../api";

export default function AdminPage() {
  const [client, setClient] = useState("");
  const [olderThan, setOlderThan] = useState("");
  const [summary, setSummary] = useState("");
  const [insights, setInsights] = useState("");
  const [loading, setLoading] = useState(false);

  const doExport = () => {
    exportStatusCSV(client.trim());
  };

  const doPurge = async () => {
    try {
      setLoading(true);
      const hours = olderThan ? Number(olderThan) : null;
      const res = await purgeStatus({ client_name: client.trim() || "", older_than_hours: hours });
      toast.success(`Deleted ${res.deleted} rows`);
    } catch (e) {
      // eslint-disable-next-line no-console
      console.error(e);
      toast.error(e?.detail || "Purge failed");
    } finally {
      setLoading(false);
    }
  };

  const doSummarize = async () => {
    try {
      setLoading(true);
      const res = await aiSummarize({ hours: 24, limit: 200 });
      setSummary(res.content);
    } catch (e) {
      // eslint-disable-next-line no-console
      console.error(e);
      toast.error(e?.detail || "Summarize failed");
    } finally {
      setLoading(false);
    }
  };

  const doInsights = async () => {
    try {
      setLoading(true);
      const res = await aiInsights({ hours: 24, limit: 500 });
      setInsights(res.content);
    } catch (e) {
      // eslint-disable-next-line no-console
      console.error(e);
      toast.error(e?.detail || "Insights failed");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-[#0f0f10] text-white">
      <div className="max-w-5xl mx-auto px-4 py-10">
        <h1 className="text-2xl font-semibold">Admin Tools</h1>
        <p className="text-gray-400 mt-1">This page is unrestricted. Use with care.</p>

        <section className="mt-8 bg-[#151517] rounded-lg p-5 border border-gray-800">
          <h2 className="text-xl font-medium">Data Operations</h2>
          <div className="mt-4 grid grid-cols-1 md:grid-cols-3 gap-4">
            <div className="md:col-span-3 flex flex-wrap gap-3 items-end">
              <div>
                <label className="block text-sm text-gray-400 mb-1">Client filter (optional)</label>
                <input
                  className="rounded border border-gray-700 bg-transparent px-3 py-2"
                  value={client}
                  onChange={(e) => setClient(e.target.value)}
                  placeholder="client name"
                />
              </div>
              <div>
                <label className="block text-sm text-gray-400 mb-1">Older than (hours, optional)</label>
                <input
                  type="number"
                  min="1"
                  className="rounded border border-gray-700 bg-transparent px-3 py-2 w-40"
                  value={olderThan}
                  onChange={(e) => setOlderThan(e.target.value)}
                  placeholder="24"
                />
              </div>
              <button onClick={doExport} className="px-4 py-2 rounded bg-emerald-600 hover:bg-emerald-500">Export CSV</button>
              <button disabled={loading} onClick={doPurge} className="px-4 py-2 rounded bg-red-600 hover:bg-red-500 disabled:opacity-50">Purge</button>
            </div>
          </div>
        </section>

        <section className="mt-8 bg-[#151517] rounded-lg p-5 border border-gray-800">
          <h2 className="text-xl font-medium">AI Utilities</h2>
          <div className="mt-4 flex gap-3">
            <button disabled={loading} onClick={doSummarize} className="px-4 py-2 rounded bg-emerald-600 hover:bg-emerald-500 disabled:opacity-50">Summarize</button>
            <button disabled={loading} onClick={doInsights} className="px-4 py-2 rounded bg-emerald-600 hover:bg-emerald-500 disabled:opacity-50">Insights</button>
          </div>
          {summary && (
            <div className="mt-4 p-3 rounded border border-gray-800 bg-black/30 whitespace-pre-wrap">{summary}</div>
          )}
          {insights && (
            <div className="mt-4 p-3 rounded border border-gray-800 bg-black/30 whitespace-pre-wrap">{insights}</div>
          )}
        </section>
      </div>
      <Toaster richColors position="top-right" />
    </div>
  );
}