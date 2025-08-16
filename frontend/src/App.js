import { useEffect, useMemo, useState } from "react";
import "./App.css";
import { BrowserRouter, Routes, Route, Link } from "react-router-dom";
import { Toaster, toast } from "sonner";
import { createStatus, getConfig, getHealth, getStatusCounts, getVersion, listStatus } from "./api";
import AdminPage from "./pages/Admin";

const Home = () => {
  const [health, setHealth] = useState(null);
  const [version, setVersion] = useState(null);
  const [counts, setCounts] = useState({ total: 0, distinct_clients: 0 });
  const [clientName, setClientName] = useState("");
  const [rows, setRows] = useState([]);
  const [q, setQ] = useState("");
  const [limit, setLimit] = useState(10);
  const [offset, setOffset] = useState(0);
  const page = useMemo(() => Math.floor(offset / limit) + 1, [offset, limit]);

  async function refresh() {
    try {
      const [h, v, c] = await Promise.all([getHealth(), getVersion(), getStatusCounts()]);
      setHealth(h);
      setVersion(v);
      setCounts(c);
    } catch (e) {
      // eslint-disable-next-line no-console
      console.error(e);
      toast.error("Backend unreachable or unhealthy");
    }
  }

  async function loadRows() {
    try {
      const data = await listStatus({ limit, offset, q });
      setRows(data);
    } catch (e) {
      // eslint-disable-next-line no-console
      console.error(e);
      toast.error("Failed loading status list");
    }
  }

  useEffect(() => {
    refresh();
  }, []);

  useEffect(() => {
    loadRows();
  }, [limit, offset, q]);

  const onSubmit = async (e) => {
    e.preventDefault();
    if (!clientName.trim()) {
      toast.error("Please enter client name");
      return;
    }
    try {
      const created = await createStatus({ client_name: clientName.trim() });
      toast.success(`Created status for ${created.client_name}`);
      setClientName("");
      await Promise.all([loadRows(), refresh()]);
    } catch (e) {
      // eslint-disable-next-line no-console
      console.error(e);
      toast.error("Create failed");
    }
  };

  const healthBadge = () => {
    if (!health) return <span className="px-2 py-1 rounded bg-gray-700">unknown</span>;
    const mongoOk = health.mongo === "ok";
    return (
      <div className="flex flex-wrap gap-2 items-center">
        <span className="px-2 py-1 rounded bg-emerald-600">API OK</span>
        <span className={`px-2 py-1 rounded ${mongoOk ? "bg-emerald-600" : "bg-red-600"}`}>
          Mongo {mongoOk ? "OK" : "DOWN"}
        </span>
        <span className="text-xs text-gray-400">{new Date(health.time).toLocaleString()}</span>
      </div>
    );
  };

  return (
    <div className="App min-h-screen bg-[#0f0f10] text-white">
      <div className="max-w-5xl mx-auto px-4 py-10">
        <header className="flex items-center justify-between">
          <a className="App-link" href="https://emergent.sh" target="_blank" rel="noopener noreferrer">
            <img
              alt="Emergent"
              className="h-14 w-14 rounded"
              src="https://avatars.githubusercontent.com/in/1201222?s=120&u=2686cf91179bbafbc7a71bfbc43004cf9ae1acea&v=4"
            />
          </a>
          <div className="text-right">
            <div className="text-sm">Version: {version ? version.version : "-"}</div>
            <div className="text-xs text-gray-400">{version ? new Date(version.time).toLocaleString() : "-"}</div>
          </div>
        </header>

        <section className="mt-8">
          {healthBadge()}
        </section>

        <section className="mt-8 grid grid-cols-1 sm:grid-cols-3 gap-4">
          <div className="bg-[#151517] rounded-lg p-4 border border-gray-800">
            <div className="text-gray-400 text-sm">Total Status Checks</div>
            <div className="text-3xl font-semibold mt-2">{counts.total}</div>
          </div>
          <div className="bg-[#151517] rounded-lg p-4 border border-gray-800">
            <div className="text-gray-400 text-sm">Distinct Clients</div>
            <div className="text-3xl font-semibold mt-2">{counts.distinct_clients}</div>
          </div>
          <div className="bg-[#151517] rounded-lg p-4 border border-gray-800">
            <div className="text-gray-400 text-sm">Page</div>
            <div className="text-3xl font-semibold mt-2">{page}</div>
          </div>
        </section>

        <section className="mt-10 bg-[#151517] rounded-lg p-5 border border-gray-800">
          <form onSubmit={onSubmit} className="flex gap-3 items-end flex-wrap">
            <div className="flex-1 min-w-60">
              <label className="block text-sm text-gray-400 mb-1">Client name</label>
              <input
                className="w-full rounded border border-gray-700 bg-transparent px-3 py-2 focus:outline-none focus:ring-2 focus:ring-emerald-600"
                placeholder="e.g. atlas-router"
                value={clientName}
                onChange={(e) => setClientName(e.target.value)}
              />
            </div>
            <button
              type="submit"
              className="px-4 py-2 rounded bg-emerald-600 hover:bg-emerald-500 transition"
            >
              Create Status
            </button>
          </form>
        </section>

        <section className="mt-10 bg-[#151517] rounded-lg p-5 border border-gray-800">
          <div className="flex items-center justify-between gap-4 flex-wrap">
            <div className="flex items-center gap-2">
              <span className="text-gray-400 text-sm">Search</span>
              <input
                className="rounded border border-gray-700 bg-transparent px-3 py-1.5 focus:outline-none focus:ring-2 focus:ring-emerald-600"
                placeholder="client name contains..."
                value={q}
                onChange={(e) => { setOffset(0); setQ(e.target.value); }}
              />
            </div>
            <div className="flex items-center gap-2">
              <span className="text-gray-400 text-sm">Per page</span>
              <select
                className="rounded border border-gray-700 bg-transparent px-2 py-1"
                value={limit}
                onChange={(e) => { setOffset(0); setLimit(Number(e.target.value)); }}
              >
                {[10, 25, 50, 100].map(n => <option key={n} value={n} className="text-black">{n}</option>)}
              </select>
            </div>
          </div>

          <div className="mt-5 overflow-x-auto">
            <table className="min-w-full text-left text-sm">
              <thead>
                <tr className="text-gray-300">
                  <th className="py-2 border-b border-gray-800">ID</th>
                  <th className="py-2 border-b border-gray-800">Client</th>
                  <th className="py-2 border-b border-gray-800">Timestamp</th>
                </tr>
              </thead>
              <tbody>
                {rows.map((r) => (
                  <tr key={r.id} className="hover:bg-black/30">
                    <td className="py-2 border-b border-gray-900 font-mono text-xs pr-4">{r.id}</td>
                    <td className="py-2 border-b border-gray-900">{r.client_name}</td>
                    <td className="py-2 border-b border-gray-900">{new Date(r.timestamp).toLocaleString()}</td>
                  </tr>
                ))}
                {rows.length === 0 && (
                  <tr>
                    <td colSpan={3} className="text-center text-gray-400 py-8">No rows</td>
                  </tr>
                )}
              </tbody>
            </table>
          </div>

          <div className="flex justify-between items-center mt-4">
            <button
              className="px-3 py-1.5 rounded bg-gray-800 disabled:opacity-50"
              disabled={offset === 0}
              onClick={() => setOffset(Math.max(0, offset - limit))}
            >
              Prev
            </button>
            <div className="text-sm text-gray-400">Page {page}</div>
            <button
              className="px-3 py-1.5 rounded bg-gray-800"
              onClick={() => setOffset(offset + limit)}
            >
              Next
            </button>
          </div>
        </section>
      </div>
      <Toaster richColors position="top-right" />
    </div>
  );
};

function Shell() {
  const [adminEnabled, setAdminEnabled] = useState(false);
  useEffect(() => {
    getConfig().then((cfg) => setAdminEnabled(!!cfg.admin_enabled)).catch(() => setAdminEnabled(false));
  }, []);

  return (
    <div className="min-h-screen">
      <nav className="bg-black/40 border-b border-gray-800">
        <div className="max-w-5xl mx-auto px-4 py-3 flex items-center gap-4">
          <Link to="/" className="text-white">Home</Link>
          {adminEnabled && <Link to="/admin" className="text-white">Admin</Link>}
        </div>
      </nav>
      <Routes>
        <Route path="/" element={<Home />} />
        {adminEnabled && <Route path="/admin" element={<AdminPage />} />}
      </Routes>
    </div>
  );
}

function App() {
  return (
    <div className="App">
      <BrowserRouter>
        <Shell />
      </BrowserRouter>
    </div>
  );
}

export default App;