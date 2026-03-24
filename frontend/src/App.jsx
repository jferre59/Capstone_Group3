import { useEffect, useState } from 'react';
import { BrowserRouter, Navigate, Route, Routes, useNavigate } from 'react-router-dom';
import { MedicalTriageDashboard } from './pages/doctor/MedicalTriageDashboard';
import { PatientInputPage } from './pages/patient/PatientInputPage';
import { PatientReportPage } from './pages/patient/PatientReportPage';

const AUTH_STORAGE_KEY = 'triage-auth';

function ProtectedRoute({ auth, allow, children }) {
  if (!auth.isAuthenticated) {
    return <Navigate to="/login" replace />;
  }

  if (!allow.includes(auth.role)) {
    return <Navigate to="/login" replace />;
  }

  return children;
}

function LoginPage({ onLogin }) {
  return (
    <div className="min-h-screen bg-slate-100 px-4 py-10">
      <div className="mx-auto w-full max-w-md rounded-xl bg-white p-8 shadow">
        <h1 className="text-2xl font-bold text-slate-900">Role Selection</h1>
        <p className="mt-2 text-sm text-slate-600">Choose your access role to continue.</p>

        <div className="mt-6 grid gap-3">
          <button
            type="button"
            onClick={() => onLogin('patient')}
            className="rounded-lg bg-blue-600 px-4 py-3 font-semibold text-white hover:bg-blue-700"
          >
            Continue as Patient
          </button>
          <button
            type="button"
            onClick={() => onLogin('doctor')}
            className="rounded-lg bg-slate-800 px-4 py-3 font-semibold text-white hover:bg-slate-900"
          >
            Continue as Doctor
          </button>
        </div>
      </div>
    </div>
  );
}

function AppRoutes() {
  const navigate = useNavigate();
  const [auth, setAuth] = useState(() => {
    const savedAuth = window.sessionStorage.getItem(AUTH_STORAGE_KEY);

    if (!savedAuth) {
      return {
        isAuthenticated: false,
        role: null,
      };
    }

    try {
      return JSON.parse(savedAuth);
    } catch {
      return {
        isAuthenticated: false,
        role: null,
      };
    }
  });

  useEffect(() => {
    window.sessionStorage.setItem(AUTH_STORAGE_KEY, JSON.stringify(auth));
  }, [auth]);

  const handleLogin = (role) => {
    setAuth({ isAuthenticated: true, role });

    if (role === 'patient') {
      navigate('/patient/input');
      return;
    }

    if (role === 'doctor') {
      navigate('/doctor/dashboard');
    }
  };

  return (
    <Routes>
      <Route path="/login" element={<LoginPage onLogin={handleLogin} />} />

      <Route
        path="/patient/input"
        element={
          <ProtectedRoute auth={auth} allow={['patient']}>
            <PatientInputPage />
          </ProtectedRoute>
        }
      />

      <Route
        path="/patient/report"
        element={
          <ProtectedRoute auth={auth} allow={['patient']}>
            <PatientReportPage />
          </ProtectedRoute>
        }
      />

      <Route
        path="/doctor/dashboard"
        element={
          <ProtectedRoute auth={auth} allow={['doctor']}>
            <MedicalTriageDashboard />
          </ProtectedRoute>
        }
      />

      <Route
        path="*"
        element={
          <Navigate
            to={
              auth.isAuthenticated
                ? auth.role === 'doctor'
                  ? '/doctor/dashboard'
                  : '/patient/input'
                : '/login'
            }
            replace
          />
        }
      />
    </Routes>
  );
}

export default function App() {
  return (
    <BrowserRouter>
      <AppRoutes />
    </BrowserRouter>
  );
}
