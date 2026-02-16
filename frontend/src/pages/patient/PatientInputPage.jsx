import { useMemo, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { Activity, LoaderCircle } from 'lucide-react';

const symptomOptions = [
  'Cough',
  'Fever',
  'Shortness of breath',
  'Chest pain',
  'Wheezing',
];

export function PatientInputPage() {
  const navigate = useNavigate();
  const [form, setForm] = useState({ age: '', sex: '', symptoms: [] });
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');

  const canSubmit = useMemo(() => {
    return form.age !== '' && form.sex !== '' && form.symptoms.length > 0;
  }, [form]);

  const onSymptomToggle = (symptom) => {
    setForm((prev) => {
      const exists = prev.symptoms.includes(symptom);
      return {
        ...prev,
        symptoms: exists
          ? prev.symptoms.filter((item) => item !== symptom)
          : [...prev.symptoms, symptom],
      };
    });

    if (error) {
      setError('');
    }
  };

  const handleSubmit = (event) => {
    event.preventDefault();

    if (!canSubmit) {
      setError('Please fill in all fields.');
      return;
    }

    setError('');
    setLoading(true);

    // TODO: Replace setTimeout with fetch POST to /api/predict using 'form' payload.
    setTimeout(() => {
      setLoading(false);
      navigate('/patient/report', { state: { intake: form } });
    }, 1600);
  };

  return (
    <div className="min-h-screen bg-gradient-to-b from-cyan-50 via-slate-50 to-white px-4 py-10">
      <div className="mx-auto w-full max-w-2xl rounded-2xl border border-slate-200 bg-white p-8 shadow-xl shadow-cyan-100/50">
        <div className="flex items-center gap-3">
          <div className="rounded-xl bg-cyan-100 p-2 text-cyan-700">
            <Activity className="h-5 w-5" />
          </div>
          <div>
            <h1 className="text-2xl font-bold text-slate-900">Patient Symptom Intake</h1>
            <p className="text-sm text-slate-600">Enter clinical indicators for AI-assisted respiratory screening.</p>
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
              onChange={(e) => {
                setForm((prev) => ({ ...prev, age: e.target.value }));
                if (error) {
                  setError('');
                }
              }}
              className="w-full rounded-xl border border-slate-300 bg-white px-4 py-3 text-slate-900 outline-none transition focus:border-cyan-500 focus:ring-2 focus:ring-cyan-200"
            />
          </div>

          <fieldset>
            <legend className="mb-3 text-sm font-semibold text-slate-700">Sex</legend>
            <div className="grid gap-3 sm:grid-cols-3">
              {['male', 'female', 'unknown'].map((sex) => (
                <label
                  key={sex}
                  className={`flex cursor-pointer items-center gap-2 rounded-xl border px-3 py-2 text-sm capitalize transition ${
                    form.sex === sex
                      ? 'border-cyan-500 bg-cyan-50 text-cyan-800'
                      : 'border-slate-300 bg-white text-slate-700 hover:border-cyan-300'
                  }`}
                >
                  <input
                    type="radio"
                    name="sex"
                    value={sex}
                    checked={form.sex === sex}
                    onChange={(e) => {
                      setForm((prev) => ({ ...prev, sex: e.target.value }));
                      if (error) {
                        setError('');
                      }
                    }}
                  />
                  {sex}
                </label>
              ))}
            </div>
          </fieldset>

          <fieldset>
            <legend className="mb-3 text-sm font-semibold text-slate-700">Symptoms</legend>
            <div className="grid gap-3 sm:grid-cols-2">
              {symptomOptions.map((symptom) => (
                <label
                  key={symptom}
                  className={`flex cursor-pointer items-center gap-2 rounded-xl border px-3 py-2 text-sm transition ${
                    form.symptoms.includes(symptom)
                      ? 'border-cyan-500 bg-cyan-50 text-cyan-800'
                      : 'border-slate-300 bg-white text-slate-700 hover:border-cyan-300'
                  }`}
                >
                  <input
                    type="checkbox"
                    checked={form.symptoms.includes(symptom)}
                    onChange={() => onSymptomToggle(symptom)}
                  />
                  {symptom}
                </label>
              ))}
            </div>
          </fieldset>

          {error ? <p className="text-sm font-semibold text-red-600">{error}</p> : null}

          <button
            type="submit"
            disabled={loading || !canSubmit}
            className="inline-flex w-full items-center justify-center gap-2 rounded-xl bg-cyan-600 px-4 py-3 font-semibold text-white transition hover:bg-cyan-700 disabled:cursor-not-allowed disabled:bg-cyan-300"
          >
            {loading ? (
              <>
                <LoaderCircle className="h-5 w-5 animate-spin" />
                Processing clinical data...
              </>
            ) : (
              'Analyze Symptoms'
            )}
          </button>
        </form>
      </div>
    </div>
  );
}
