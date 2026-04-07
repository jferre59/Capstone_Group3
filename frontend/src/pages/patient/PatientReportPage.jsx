import { useMemo } from 'react';
import { Link, useLocation } from 'react-router-dom';
import { AlertCircle, ArrowLeft, CheckCircle2, FileWarning, ThermometerSun } from 'lucide-react';
import Navbar from '../../components/Navbar'; // ADDED: Import the shared Navbar component
const REPORT_STORAGE_KEY = 'latest-patient-report';

export function PatientReportPage() {
  const location = useLocation();
  const persistedReportState = useMemo(() => {
    const savedReport = window.sessionStorage.getItem(REPORT_STORAGE_KEY);

    if (!savedReport) {
      return null;
    }

    try {
      return JSON.parse(savedReport);
    } catch {
      return null;
    }
  }, []);
  const reportState = location.state ?? persistedReportState;
  const intake = reportState?.intake;
  const prediction = reportState?.prediction;
  const savedAt = reportState?.savedAt;
  const payload = prediction?.payload;
  const result = prediction?.result;
  const hasLiveResult = Boolean(result?.Illness || result?.Treatment);
  const intakeSymptoms = [intake?.symptom_1, intake?.symptom_2, intake?.symptom_3].filter(
    (value) => value && value !== 'NONE',
  );

  return (
     // ADDED: Wrapped in Fragment so Navbar sits above the page content
        <>
          {/* ADDED: Navbar renders at the very top of the page */}
          <Navbar />
    <div className="min-h-screen bg-gradient-to-b from-cyan-50 via-slate-50 to-white px-4 py-10 pb-24">
      <div className="mx-auto w-full max-w-5xl space-y-6">
        <div className="flex items-center justify-between gap-4">
          <div>
            <p className="text-sm font-semibold uppercase tracking-[0.2em] text-cyan-700">Patient Flow</p>
            <h1 className="mt-1 text-3xl font-bold text-slate-900">Diagnostic Report</h1>
          </div>
          <Link
            to="/patient/input"
            className="inline-flex items-center gap-2 rounded-full border border-slate-300 bg-white px-4 py-2 text-sm font-semibold text-slate-700 transition hover:border-cyan-400 hover:text-cyan-700"
          >
            <ArrowLeft className="h-4 w-4" />
            Back to Intake
          </Link>
        </div>

        <div className="rounded-2xl border border-slate-200 bg-white p-6 shadow-xl shadow-cyan-100/50">
          <div className="flex items-center gap-3">
            <div className="rounded-xl bg-cyan-100 p-2 text-cyan-700">
              <ThermometerSun className="h-5 w-5" />
            </div>
            <div>
              <h2 className="text-2xl font-bold text-slate-900">Patient Diagnostic Report</h2>
              <p className="text-sm text-slate-600">Live response from the respiratory screening backend.</p>
            </div>
          </div>

          <p className="mt-4 rounded-lg bg-slate-50 px-3 py-2 text-sm text-slate-600">
            {intake
              ? `Input received: age ${intake.age}, sex ${intake.sex}, nature ${intake.nature}, symptoms ${intakeSymptoms.join(', ') || 'none'}.`
              : 'No new intake found. Submit the intake form to generate a live prediction.'}
          </p>

          {savedAt ? (
            <p className="mt-3 text-xs font-medium uppercase tracking-[0.14em] text-slate-400">
              Last saved report: {new Date(savedAt).toLocaleString()}
            </p>
          ) : null}

          {hasLiveResult ? (
            <div className="mt-5 grid gap-4 lg:grid-cols-2">
              <div className="rounded-xl border border-cyan-100 bg-cyan-50/60 p-4">
                <p className="text-sm font-medium text-cyan-700">Predicted Condition</p>
                <p className="text-3xl font-bold text-slate-900">{result.Illness}</p>
              </div>
              <div className="rounded-xl border border-emerald-100 bg-emerald-50/70 p-4">
                <p className="text-sm font-medium text-emerald-700">Recommended Treatment</p>
                <p className="text-2xl font-bold text-slate-900">{result.Treatment}</p>
              </div>
            </div>
          ) : (
            <div className="mt-5 rounded-xl border border-amber-200 bg-amber-50 p-4 text-sm text-amber-900">
              <div className="flex items-start gap-3">
                <FileWarning className="mt-0.5 h-5 w-5 shrink-0" />
                <div>
                  <p className="font-semibold">Live prediction data is not available yet.</p>
                  <p className="mt-1">This report was not found in memory or session storage, so a fresh intake is needed.</p>
                </div>
              </div>
              <Link
                to="/patient/input"
                className="mt-4 inline-flex items-center gap-2 rounded-full bg-amber-900 px-4 py-2 font-semibold text-white transition hover:bg-amber-800"
              >
                <ArrowLeft className="h-4 w-4" />
                Return to Intake
              </Link>
            </div>
          )}
        </div>

        <section className="rounded-2xl border border-slate-200 bg-white p-6 shadow-sm">
          <h2 className="text-lg font-semibold text-slate-900">Submission Summary</h2>
          <div className="mt-4 grid gap-4 md:grid-cols-2">
            <div className="rounded-lg border border-slate-200 bg-slate-50 p-4 text-sm text-slate-700">
              <p className="font-semibold text-slate-900">Derived Payload</p>
              {payload ? (
                <ul className="mt-2 space-y-1">
                  <li>Age: {payload.age}</li>
                  <li>Age group: {payload.age_group}</li>
                  <li>Nature: {payload.nature}</li>
                  <li>Symptom count: {payload.symptom_count}</li>
                  <li>High risk: {payload.high_risk}</li>
                  <li>Symptom slots: {payload.symptom_1}, {payload.symptom_2}, {payload.symptom_3}</li>
                </ul>
              ) : (
                <p className="mt-2">No payload metadata available.</p>
              )}
            </div>

            <div className="rounded-lg border border-slate-200 bg-slate-50 p-4 text-sm text-slate-700">
              <p className="font-semibold text-slate-900">Clinical Caution</p>
              <ul className="mt-2 space-y-3">
                <li className="flex items-start gap-3">
                  <AlertCircle className="mt-0.5 h-5 w-5 shrink-0 text-amber-600" />
                  <span>This AI result is a screening aid and does not replace a licensed medical assessment.</span>
                </li>
                <li className="flex items-start gap-3">
                  <CheckCircle2 className="mt-0.5 h-5 w-5 shrink-0 text-emerald-600" />
                  <span>Use the recommended treatment output as a prompt for clinician review, not as an automatic order.</span>
                </li>
                <li className="flex items-start gap-3">
                  <AlertCircle className="mt-0.5 h-5 w-5 shrink-0 text-amber-600" />
                  <span>Escalate urgent symptoms such as severe shortness of breath or chest pain immediately.</span>
                </li>
              </ul>
            </div>
          </div>
        </section>
      </div>

      <div className="fixed bottom-0 left-0 right-0 z-50 border-t border-red-200 bg-red-50 px-4 py-3 text-center text-sm font-semibold text-red-700">
        This AI result is for reference only and does not replace professional medical advice.
      </div>
    </div>
    </> // ADDED: Closing fragment tag
  );
}
