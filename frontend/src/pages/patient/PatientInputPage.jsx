import { useMemo, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { Activity, LoaderCircle } from 'lucide-react';
import {
  buildPredictionPayload,
  natureOptions,
  predictCondition,
  sexOptions,
  symptomOptions,
} from '../../lib/prediction';
import Navbar from '../../components/Navbar'; // ADDED: Import the shared Navbar component

const REPORT_STORAGE_KEY = 'latest-patient-report';

const symptomFieldConfig = [
  { id: 'symptom_1', label: 'Primary symptom' },
];

export function PatientInputPage() {
  const navigate = useNavigate();
  const [form, setForm] = useState({
    age: '',
    sex: '',
    nature: 'unknown',
    symptom_1: 'NONE',
  });
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');

  const payloadPreview = useMemo(() => buildPredictionPayload(form), [form]);

  const canSubmit = useMemo(() => {
    const age = Number(form.age);

    return (
      form.age !== '' &&
      !Number.isNaN(age) &&
      age >= 0 &&
      form.sex !== '' &&
      form.nature !== '' &&
      payloadPreview.symptom_count > 0
    );
  }, [form, payloadPreview.symptom_count]);

  const updateField = (field, value) => {
    setForm((prev) => ({ ...prev, [field]: value }));
    if (error) {
      setError('');
    }
  };

  const handleSubmit = async (event) => {
    event.preventDefault();

    if (!canSubmit) {
      setError('Enter age, sex, nature, and choose a symptom.');
      return;
    }

    setLoading(true);
    setError('');

    try {
      const prediction = await predictCondition(form);
      const reportState = {
        intake: form,
        prediction,
        savedAt: new Date().toISOString(),
      };

      window.sessionStorage.setItem(REPORT_STORAGE_KEY, JSON.stringify(reportState));

      navigate('/patient/report', {
        state: reportState,
      });
    } catch (requestError) {
      setError(requestError.message || 'Unable to connect to the prediction service.');
    } finally {
      setLoading(false);
    }
  };

  return (
    // ADDED: Wrapped in Fragment so Navbar sits above the page content
    <>
      {/* ADDED: Navbar renders at the very top of the page */}
      <Navbar />
    <div className="min-h-screen bg-gradient-to-b from-cyan-50 via-slate-50 to-white px-4 py-10">
      <div className="mx-auto w-full max-w-3xl rounded-2xl border border-slate-200 bg-white p-8 shadow-xl shadow-cyan-100/50">
        <div className="flex items-center gap-3">
          <div className="rounded-xl bg-cyan-100 p-2 text-cyan-700">
            <Activity className="h-5 w-5" />
          </div>
          <div>
            <h1 className="text-2xl font-bold text-slate-900">Patient Symptom Intake</h1>
            <p className="text-sm text-slate-600">Submit respiratory screening data in the exact format expected by the backend API.</p>
          </div>
        </div>

        <form onSubmit={handleSubmit} className="mt-8 space-y-7">
          <div>
            <label htmlFor="age" className="mb-2 block text-sm font-semibold text-slate-700">
              Age
            </label>
            <input
              id="age"
              type="number"
              min="0"
              placeholder="e.g., 42"
              value={form.age}
              onChange={(e) => updateField('age', e.target.value)}
              className="w-full rounded-xl border border-slate-300 bg-white px-4 py-3 text-slate-900 outline-none transition focus:border-cyan-500 focus:ring-2 focus:ring-cyan-200"
            />
          </div>

          <fieldset>
            <legend className="mb-3 text-sm font-semibold text-slate-700">Sex</legend>
            <div className="grid gap-3 sm:grid-cols-2">
              {sexOptions.map((option) => (
                <label
                  key={option.value}
                  className={`flex cursor-pointer items-center gap-2 rounded-xl border px-3 py-2 text-sm transition ${
                    form.sex === option.value
                      ? 'border-cyan-500 bg-cyan-50 text-cyan-800'
                      : 'border-slate-300 bg-white text-slate-700 hover:border-cyan-300'
                  }`}
                >
                  <input
                    type="radio"
                    name="sex"
                    value={option.value}
                    checked={form.sex === option.value}
                    onChange={(e) => updateField('sex', e.target.value)}
                  />
                  {option.label}
                </label>
              ))}
            </div>
          </fieldset>

          <fieldset>
            <legend className="mb-3 text-sm font-semibold text-slate-700">Nature</legend>
            <div className="grid gap-3 sm:grid-cols-4">
              {natureOptions.map((option) => (
                <label
                  key={option.value}
                  className={`flex cursor-pointer items-center gap-2 rounded-xl border px-3 py-2 text-sm transition ${
                    form.nature === option.value
                      ? 'border-cyan-500 bg-cyan-50 text-cyan-800'
                      : 'border-slate-300 bg-white text-slate-700 hover:border-cyan-300'
                  }`}
                >
                  <input
                    type="radio"
                    name="nature"
                    value={option.value}
                    checked={form.nature === option.value}
                    onChange={(e) => updateField('nature', e.target.value)}
                  />
                  {option.label}
                </label>
              ))}
            </div>
          </fieldset>

          <section className="grid gap-4 md:grid-cols-3">
            {symptomFieldConfig.map((field) => (
              <div key={field.id}>
                <label htmlFor={field.id} className="mb-2 block text-sm font-semibold text-slate-700">
                  {field.label}
                </label>
                <select
                  id={field.id}
                  value={form[field.id]}
                  onChange={(e) => updateField(field.id, e.target.value)}
                  className="w-full rounded-xl border border-slate-300 bg-white px-4 py-3 text-sm text-slate-900 outline-none transition focus:border-cyan-500 focus:ring-2 focus:ring-cyan-200"
                >
                  {symptomOptions.map((option) => (
                    <option key={`${field.id}-${option.value}`} value={option.value}>
                      {option.label}
                    </option>
                  ))}
                </select>
              </div>
            ))}
          </section>

          <section className="rounded-xl border border-cyan-100 bg-cyan-50/70 p-4 text-sm text-slate-700">
            <p className="font-semibold text-cyan-800">Backend Payload Preview</p>
            <div className="mt-2 grid gap-2 md:grid-cols-2">
              <p>Age group: {payloadPreview.age_group || 'N/A'}</p>
              <p>Symptom count: {payloadPreview.symptom_count}</p>
              <p>High risk: {payloadPreview.high_risk}</p>
              <p>Nature: {payloadPreview.nature}</p>
              <p className="md:col-span-2">
                Symptom slots: {payloadPreview.symptom_1}, {payloadPreview.symptom_2}, {payloadPreview.symptom_3}
              </p>
            </div>
          </section>

          {error ? <p className="text-sm font-semibold text-red-600">{error}</p> : null}

          <button
            type="submit"
            disabled={loading || !canSubmit}
            className="inline-flex w-full items-center justify-center gap-2 rounded-xl bg-cyan-600 px-4 py-3 font-semibold text-white transition hover:bg-cyan-700 disabled:cursor-not-allowed disabled:bg-cyan-300"
          >
            {loading ? (
              <>
                <LoaderCircle className="h-5 w-5 animate-spin" />
                Sending prediction request...
              </>
            ) : (
              'Analyze Symptoms'
            )}
          </button>
        </form>
      </div>
    </div>
    </> // ADDED: Closing fragment tag
  );
}
