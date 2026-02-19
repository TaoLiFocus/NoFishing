import { BrowserRouter, Routes, Route, Navigate } from 'react-router-dom';
import ProtectedRoute from './components/auth/ProtectedRoute';
import AppLayout from './components/common/Layout';
import Login from './pages/Login';
import Dashboard from './pages/Dashboard';
import Detection from './pages/Detection';
import History from './pages/History';
import Statistics from './pages/Statistics';
import Monitoring from './pages/Monitoring';
import Whitelist from './pages/Whitelist';
import Blacklist from './pages/Blacklist';
import UserManagement from './pages/UserManagement';
import SystemConfig from './pages/SystemConfig';
import AuditLogPage from './pages/AuditLog';
import ApiKeyManagement from './pages/ApiKeyManagement';

function App() {
  return (
    <BrowserRouter>
      <Routes>
        <Route path="/login" element={<Login />} />

        <Route element={<ProtectedRoute />}>
          <Route element={<AppLayout />}>
            <Route path="/" element={<Dashboard />} />
            <Route path="/detection" element={<Detection />} />
            <Route path="/history" element={<History />} />
            <Route path="/statistics" element={<Statistics />} />
            <Route path="/monitoring" element={<Monitoring />} />
            <Route path="/whitelist" element={<Whitelist />} />
            <Route path="/blacklist" element={<Blacklist />} />
          </Route>
        </Route>

        <Route element={<ProtectedRoute requireAdmin={true} />}>
          <Route element={<AppLayout />}>
            <Route path="/users" element={<UserManagement />} />
            <Route path="/system-config" element={<SystemConfig />} />
            <Route path="/audit-logs" element={<AuditLogPage />} />
            <Route path="/api-keys" element={<ApiKeyManagement />} />
          </Route>
        </Route>

        <Route path="*" element={<Navigate to="/" replace />} />
      </Routes>
    </BrowserRouter>
  );
}

export default App;
