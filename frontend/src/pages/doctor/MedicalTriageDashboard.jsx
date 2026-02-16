import { useMemo, useState } from 'react';
import { AlertTriangle, ClipboardCheck } from 'lucide-react';
import { mockPatients } from '../../mocks/mockPatients';

const urgencyRank = {
  Critical: 0,
  'High-Risk': 1,
  High: 1,
  Medium: 2,
  Low: 3,
};

const normalizeUrgency = (urgency) => {
  if (urgency === 'High Risk') {
    return 'High-Risk';
  }
  return urgency;
};

export function MedicalTriageDashboard() {
  const [reviewedIds, setReviewedIds] = useState(new Set());

  // TODO: Replace mockPatients with fetch GET /api/triage/queue.
  const sortedPatients = useMemo(() => {
    return [...mockPatients].sort((a, b) => {
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
  }, []);

  const onReview = (patientId) => {
    // TODO: Call PATCH /api/triage/:patientId/review before local UI update.
    setReviewedIds((prev) => {
      const next = new Set(prev);
      next.add(patientId);
      return next;
    });
  };

  return (
    <div className="min-h-screen bg-slate-100 px-4 py-10">
      <div className="mx-auto w-full max-w-6xl rounded-xl bg-white p-6 shadow">
        <div className="mb-6 flex items-start justify-between gap-4">
          <div>
            <h1 className="text-2xl font-bold text-slate-900">Medical Triage Dashboard</h1>
            <p className="mt-1 text-sm text-slate-600">
              Patients are auto-sorted by urgency, with critical cases first.
            </p>
          </div>
          <div className="inline-flex items-center gap-2 rounded-lg border border-red-200 bg-red-50 px-3 py-2 text-sm font-medium text-red-700">
            <AlertTriangle className="h-4 w-4" />
            Critical/High-Risk on top
          </div>
        </div>

        <div className="overflow-x-auto">
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
              {sortedPatients.map((patient) => {
                const urgency = normalizeUrgency(patient.urgency);
                const isUrgent = urgency === 'Critical' || urgency === 'High' || urgency === 'High-Risk';
                const isReviewed = reviewedIds.has(patient.id);

                return (
                  <tr
                    key={patient.id}
                    className={`border-b border-slate-100 ${
                      isUrgent ? 'bg-red-50/70' : 'bg-white'
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
                        onClick={() => onReview(patient.id)}
                        className="inline-flex items-center gap-2 rounded-lg bg-slate-800 px-3 py-2 text-xs font-semibold text-white hover:bg-slate-900"
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
      </div>
    </div>
  );
}
