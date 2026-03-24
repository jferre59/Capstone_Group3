import { useMemo, useState } from 'react';
import { AlertTriangle, ClipboardCheck, Search, ShieldAlert, Stethoscope } from 'lucide-react';
import { mockPatients } from '../../mocks/mockPatients';

const urgencyRank = {
  Critical: 0,
  'High-Risk': 1,
  High: 1,
  Medium: 2,
  Low: 3,
};

const urgencyFilters = ['All', 'Critical', 'High-Risk', 'High', 'Medium', 'Low'];

const normalizeUrgency = (urgency) => {
  if (urgency === 'High Risk') {
    return 'High-Risk';
  }
  return urgency;
};

export function MedicalTriageDashboard() {
  const [reviewedIds, setReviewedIds] = useState(new Set());
  const [searchTerm, setSearchTerm] = useState('');
  const [urgencyFilter, setUrgencyFilter] = useState('All');
  const [selectedPatientId, setSelectedPatientId] = useState(mockPatients[0]?.id ?? null);

  const filteredPatients = useMemo(() => {
    const normalizedSearch = searchTerm.trim().toLowerCase();

    return mockPatients
      .filter((patient) => {
        const urgency = normalizeUrgency(patient.urgency);
        const matchesUrgency = urgencyFilter === 'All' || urgency === urgencyFilter;
        const haystack = [patient.id, patient.name, patient.predictedCondition, ...patient.symptoms]
          .join(' ')
          .toLowerCase();
        const matchesSearch = normalizedSearch === '' || haystack.includes(normalizedSearch);

        return matchesUrgency && matchesSearch;
      })
      .sort((a, b) => {
        const aUrgency = normalizeUrgency(a.urgency);
        const bUrgency = normalizeUrgency(b.urgency);

        const aRank = urgencyRank[aUrgency] ?? 99;
        const bRank = urgencyRank[bUrgency] ?? 99;

        const rankDiff = aRank - bRank;
        if (rankDiff !== 0) {
          return rankDiff;
        }
        return b.confidence - a.confidence;
      });
  }, [searchTerm, urgencyFilter]);

  const selectedPatient =
    filteredPatients.find((patient) => patient.id === selectedPatientId) ?? filteredPatients[0] ?? null;

  const onReview = (patientId) => {
    setReviewedIds((prev) => {
      const next = new Set(prev);
      next.add(patientId);
      return next;
    });
  };

  return (
    <div className="min-h-screen bg-[radial-gradient(circle_at_top_left,_rgba(248,113,113,0.12),_transparent_35%),linear-gradient(180deg,#f8fafc_0%,#ffffff_100%)] px-4 py-10">
      <div className="mx-auto w-full max-w-7xl space-y-6">
        <div className="flex flex-col gap-4 rounded-3xl border border-slate-200 bg-white p-6 shadow-xl shadow-slate-200/60 lg:flex-row lg:items-end lg:justify-between">
          <div>
            <p className="text-sm font-semibold uppercase tracking-[0.22em] text-red-600">Clinical Dashboard</p>
            <h1 className="mt-2 text-3xl font-bold text-slate-900">Medical Triage Dashboard</h1>
            <p className="mt-2 max-w-2xl text-sm text-slate-600">
              Mock triage queue for frontend completion. Critical and high-risk patients are surfaced first while backend APIs are still pending.
            </p>
          </div>

          <div className="flex flex-wrap gap-3">
            <div className="inline-flex items-center gap-2 rounded-full border border-red-200 bg-red-50 px-4 py-2 text-sm font-medium text-red-700">
              <AlertTriangle className="h-4 w-4" />
              Critical/High-Risk on top
            </div>
            <div className="inline-flex items-center gap-2 rounded-full border border-cyan-200 bg-cyan-50 px-4 py-2 text-sm font-medium text-cyan-700">
              <Stethoscope className="h-4 w-4" />
              Mock data active
            </div>
          </div>
        </div>

        <div className="grid gap-6 lg:grid-cols-[minmax(0,1.4fr)_380px]">
          <section className="rounded-3xl border border-slate-200 bg-white p-6 shadow-lg shadow-slate-200/50">
            <div className="flex flex-col gap-4 lg:flex-row lg:items-center lg:justify-between">
              <div className="relative w-full max-w-md">
                <Search className="pointer-events-none absolute left-4 top-1/2 h-4 w-4 -translate-y-1/2 text-slate-400" />
                <input
                  type="search"
                  value={searchTerm}
                  onChange={(e) => setSearchTerm(e.target.value)}
                  placeholder="Search patient, diagnosis, symptom..."
                  className="w-full rounded-full border border-slate-300 bg-white py-3 pl-11 pr-4 text-sm text-slate-900 outline-none transition focus:border-red-400 focus:ring-2 focus:ring-red-100"
                />
              </div>

              <div className="flex flex-wrap gap-2">
                {urgencyFilters.map((filterValue) => (
                  <button
                    key={filterValue}
                    type="button"
                    onClick={() => setUrgencyFilter(filterValue)}
                    className={`rounded-full px-4 py-2 text-sm font-semibold transition ${
                      urgencyFilter === filterValue
                        ? 'bg-slate-900 text-white'
                        : 'border border-slate-300 bg-white text-slate-700 hover:border-slate-400'
                    }`}
                  >
                    {filterValue}
                  </button>
                ))}
              </div>
            </div>

            {filteredPatients.length === 0 ? (
              <div className="mt-6 rounded-2xl border border-dashed border-slate-300 bg-slate-50 p-8 text-center">
                <ShieldAlert className="mx-auto h-8 w-8 text-slate-400" />
                <p className="mt-3 text-lg font-semibold text-slate-900">No patients match the current filter.</p>
                <p className="mt-1 text-sm text-slate-600">Try clearing the search term or switching the urgency filter.</p>
              </div>
            ) : (
              <div className="mt-6 overflow-x-auto">
                <table className="min-w-full border-collapse text-left text-sm">
                  <thead>
                    <tr className="border-b border-slate-200 bg-slate-50 text-slate-700">
                      <th className="px-4 py-3 font-semibold">Patient ID</th>
                      <th className="px-4 py-3 font-semibold">Name</th>
                      <th className="px-4 py-3 font-semibold">Age</th>
                      <th className="px-4 py-3 font-semibold">Predicted Diagnosis</th>
                      <th className="px-4 py-3 font-semibold">Confidence</th>
                      <th className="px-4 py-3 font-semibold">Urgency</th>
                      <th className="px-4 py-3 font-semibold">Action</th>
                    </tr>
                  </thead>
                  <tbody>
                    {filteredPatients.map((patient) => {
                      const urgency = normalizeUrgency(patient.urgency);
                      const isUrgent = urgency === 'Critical' || urgency === 'High' || urgency === 'High-Risk';
                      const isReviewed = reviewedIds.has(patient.id);
                      const isSelected = selectedPatient?.id === patient.id;

                      return (
                        <tr
                          key={patient.id}
                          onClick={() => setSelectedPatientId(patient.id)}
                          className={`cursor-pointer border-b border-slate-100 transition ${
                            isSelected
                              ? 'bg-cyan-50/60'
                              : isUrgent
                                ? 'bg-red-50/60 hover:bg-red-50'
                                : 'bg-white hover:bg-slate-50'
                          }`}
                        >
                          <td className="px-4 py-3 font-medium text-slate-900">{patient.id}</td>
                          <td className="px-4 py-3 text-slate-700">{patient.name}</td>
                          <td className="px-4 py-3 text-slate-700">{patient.age}</td>
                          <td className="px-4 py-3 text-slate-700">{patient.predictedCondition}</td>
                          <td className="px-4 py-3 text-slate-700">{(patient.confidence * 100).toFixed(1)}%</td>
                          <td className="px-4 py-3">
                            {isUrgent ? (
                              <span className="rounded-full bg-red-600 px-3 py-1 text-xs font-semibold text-white">
                                Urgent · {urgency}
                              </span>
                            ) : (
                              <span className="rounded-full bg-slate-200 px-3 py-1 text-xs font-semibold text-slate-700">
                                {urgency}
                              </span>
                            )}
                          </td>
                          <td className="px-4 py-3">
                            <button
                              type="button"
                              onClick={(event) => {
                                event.stopPropagation();
                                onReview(patient.id);
                              }}
                              className={`inline-flex items-center gap-2 rounded-lg px-3 py-2 text-xs font-semibold transition ${
                                isReviewed
                                  ? 'bg-emerald-100 text-emerald-800 hover:bg-emerald-200'
                                  : 'bg-slate-800 text-white hover:bg-slate-900'
                              }`}
                            >
                              <ClipboardCheck className="h-4 w-4" />
                              {isReviewed ? 'Reviewed' : 'Review'}
                            </button>
                          </td>
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
              </div>
            )}
          </section>

          <aside className="rounded-3xl border border-slate-200 bg-white p-6 shadow-lg shadow-slate-200/50">
            {selectedPatient ? (
              <>
                <div className="flex items-center justify-between gap-3">
                  <div>
                    <p className="text-sm font-semibold uppercase tracking-[0.2em] text-cyan-700">Selected Case</p>
                    <h2 className="mt-2 text-2xl font-bold text-slate-900">{selectedPatient.name}</h2>
                  </div>
                  <span className="rounded-full bg-slate-100 px-3 py-1 text-xs font-semibold text-slate-700">
                    {selectedPatient.id}
                  </span>
                </div>

                <div className="mt-5 grid gap-3 sm:grid-cols-2">
                  <div className="rounded-2xl bg-slate-50 p-4">
                    <p className="text-xs font-semibold uppercase tracking-[0.15em] text-slate-500">Diagnosis</p>
                    <p className="mt-2 text-lg font-semibold text-slate-900">{selectedPatient.predictedCondition}</p>
                  </div>
                  <div className="rounded-2xl bg-slate-50 p-4">
                    <p className="text-xs font-semibold uppercase tracking-[0.15em] text-slate-500">Confidence</p>
                    <p className="mt-2 text-lg font-semibold text-slate-900">{(selectedPatient.confidence * 100).toFixed(1)}%</p>
                  </div>
                </div>

                <div className="mt-5 rounded-2xl border border-slate-200 bg-slate-50 p-4">
                  <p className="text-xs font-semibold uppercase tracking-[0.15em] text-slate-500">Urgency Assessment</p>
                  <p className="mt-2 text-lg font-semibold text-slate-900">{normalizeUrgency(selectedPatient.urgency)}</p>
                  <p className="mt-2 text-sm text-slate-600">
                    {selectedPatient.age > 65
                      ? 'Senior age bracket increases the chance of high-risk triage prioritization.'
                      : 'Current urgency is driven by the mock screening result and symptom profile.'}
                  </p>
                </div>

                <div className="mt-5">
                  <p className="text-xs font-semibold uppercase tracking-[0.15em] text-slate-500">Reported Symptoms</p>
                  <div className="mt-3 flex flex-wrap gap-2">
                    {selectedPatient.symptoms.map((symptom) => (
                      <span
                        key={symptom}
                        className="rounded-full border border-cyan-200 bg-cyan-50 px-3 py-1 text-xs font-medium text-cyan-800"
                      >
                        {symptom}
                      </span>
                    ))}
                  </div>
                </div>

                <div className="mt-6 rounded-2xl border border-dashed border-slate-300 p-4 text-sm text-slate-600">
                  <p className="font-semibold text-slate-900">Frontend note</p>
                  <p className="mt-2">
                    This panel is still powered by mock data. When backend triage APIs are ready, this view can be switched to live patient details without changing the layout.
                  </p>
                </div>
              </>
            ) : (
              <div className="rounded-2xl border border-dashed border-slate-300 bg-slate-50 p-8 text-center">
                <ShieldAlert className="mx-auto h-8 w-8 text-slate-400" />
                <p className="mt-3 text-lg font-semibold text-slate-900">No case selected</p>
                <p className="mt-1 text-sm text-slate-600">Choose a patient from the queue to inspect symptoms and urgency details.</p>
              </div>
            )}
          </aside>
        </div>
      </div>
    </div>
  );
}
