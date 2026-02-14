import { BrowserRouter, Routes, Route, NavLink } from "react-router-dom";
import { useWebSocket } from "./hooks/useWebSocket";
import Dashboard from "./pages/Dashboard";
import Analytics from "./pages/Analytics";
import Config from "./pages/Config";
import System from "./pages/System";

function App() {
  const { state, connected, tradeFeed } = useWebSocket();

  return (
    <BrowserRouter>
      <div className="flex h-screen bg-gray-950 text-gray-100">
        {/* Sidebar */}
        <nav className="w-48 bg-gray-900 border-r border-gray-800 flex flex-col p-4 gap-1">
          <h1 className="text-lg font-bold mb-4 text-emerald-400">Arb Bot</h1>
          <NavLink to="/" end className={({ isActive }) =>
            `px-3 py-2 rounded text-sm ${isActive ? "bg-gray-800 text-white" : "text-gray-400 hover:text-white"}`
          }>Dashboard</NavLink>
          <NavLink to="/analytics" className={({ isActive }) =>
            `px-3 py-2 rounded text-sm ${isActive ? "bg-gray-800 text-white" : "text-gray-400 hover:text-white"}`
          }>Analytics</NavLink>
          <NavLink to="/config" className={({ isActive }) =>
            `px-3 py-2 rounded text-sm ${isActive ? "bg-gray-800 text-white" : "text-gray-400 hover:text-white"}`
          }>Config</NavLink>
          <NavLink to="/system" className={({ isActive }) =>
            `px-3 py-2 rounded text-sm ${isActive ? "bg-gray-800 text-white" : "text-gray-400 hover:text-white"}`
          }>System</NavLink>
          <div className="mt-auto">
            <div className={`text-xs px-3 py-1 rounded ${connected ? "text-emerald-400" : "text-red-400"}`}>
              {connected ? "Connected" : "Disconnected"}
            </div>
          </div>
        </nav>

        {/* Main content */}
        <main className="flex-1 overflow-auto p-6">
          <Routes>
            <Route path="/" element={<Dashboard state={state} tradeFeed={tradeFeed} />} />
            <Route path="/analytics" element={<Analytics />} />
            <Route path="/config" element={<Config state={state} />} />
            <Route path="/system" element={<System state={state} />} />
          </Routes>
        </main>
      </div>
    </BrowserRouter>
  );
}

export default App;
