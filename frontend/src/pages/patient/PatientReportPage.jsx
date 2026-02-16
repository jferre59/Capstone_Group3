import { useLocation } from 'react-router-dom';
import { AlertCircle, CheckCircle2, ThermometerSun } from 'lucide-react';
import {
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  PolarAngleAxis,
  RadialBar,
  RadialBarChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts';

// TODO: Replace mockPrediction with GET /api/report/:requestId response.
const mockPrediction = {
  disease: 'Pneumonia',
  confidence: 92.5,
  featureImportance: [
    { feature: 'Shortness of breath', impact: 'High', score: 95 },
    { feature: 'Fever', impact: 'High', score: 89 },
    { feature: 'Age', impact: 'Medium', score: 63 },
    { feature: 'Chest pain', impact: 'Medium', score: 58 },
  ],
  recommendedActions: [
    { type: 'warning', text: 'Consult a medical professional within 24 hours.' },
    { type: 'check', text: 'Track body temperature and breathing pattern regularly.' },
    { type: 'warning', text: 'Seek urgent care if shortness of breath becomes severe.' },
  ],
};

const confidenceData = [
  {
    name: 'Confidence',
    value: mockPrediction.confidence,
    fill: '#0891B2',
  },
];

export function PatientReportPage() {
  const location = useLocation();
  const intake = location.state?.intake;

  return (
    <div className="min-h-screen bg-gradient-to-b from-cyan-50 via-slate-50 to-white px-4 py-10 pb-24">
      <div className="mx-auto w-full max-w-5xl space-y-6">
        <div className="rounded-2xl border border-slate-200 bg-white p-6 shadow-xl shadow-cyan-100/50">
          <div className="flex items-center gap-3">
            <div className="rounded-xl bg-cyan-100 p-2 text-cyan-700">
              <ThermometerSun className="h-5 w-5" />
            </div>
            <div>
              <h1 className="text-2xl font-bold text-slate-900">Patient Diagnostic Report</h1>
              <p className="text-sm text-slate-600">Explainable AI summary for respiratory condition screening.</p>
            </div>
          </div>

          <p className="mt-4 rounded-lg bg-slate-50 px-3 py-2 text-sm text-slate-600">
            {intake
              ? `Input received: age ${intake.age}, sex ${intake.sex}, symptoms ${intake.symptoms.join(', ')}.`
              : 'No new intake found. Showing default mock report.'}
          </p>

          <div className="mt-5 rounded-xl border border-cyan-100 bg-cyan-50/60 p-4">
            <p className="text-sm font-medium text-cyan-700">Predicted Condition</p>
            <p className="text-3xl font-bold text-slate-900">{mockPrediction.disease}</p>
          </div>
        </div>

        <div className="grid gap-6 lg:grid-cols-2">
          <section className="rounded-2xl border border-slate-200 bg-white p-6 shadow-sm">
            <h2 className="text-lg font-semibold text-slate-900">Confidence Score</h2>
            <div className="mt-4 h-72 w-full">
              <ResponsiveContainer width="100%" height="100%">
                <RadialBarChart
                  innerRadius="55%"
                  outerRadius="95%"
                  data={confidenceData}
                  startAngle={210}
                  endAngle={-30}
                  barSize={28}
                >
                  <PolarAngleAxis type="number" domain={[0, 100]} tick={false} />
                  <RadialBar background dataKey="value" cornerRadius={14} />
                  <Tooltip formatter={(value) => `${value}%`} />
                </RadialBarChart>
              </ResponsiveContainer>
            </div>
            <p className="-mt-4 text-center text-3xl font-bold text-cyan-700">{mockPrediction.confidence}%</p>
          </section>

          <section className="rounded-2xl border border-slate-200 bg-white p-6 shadow-sm">
            <h2 className="text-lg font-semibold text-slate-900">Feature Importance (XAI)</h2>
            <div className="mt-4 h-72 w-full">
              <ResponsiveContainer width="100%" height="100%">
                <BarChart
                  data={mockPrediction.featureImportance}
                  layout="vertical"
                  margin={{ top: 8, right: 16, left: 12, bottom: 8 }}
                >
                  <CartesianGrid strokeDasharray="3 3" stroke="#E2E8F0" />
                  <XAxis type="number" domain={[0, 100]} tick={{ fill: '#334155', fontSize: 12 }} />
                  <YAxis
                    type="category"
                    dataKey="feature"
                    tick={{ fill: '#334155', fontSize: 12 }}
                    width={125}
                  />
                  <Tooltip formatter={(value, _name, item) => [`${value}%`, item.payload.impact]} />
                  <Bar dataKey="score" radius={[0, 8, 8, 0]}>
                    {mockPrediction.featureImportance.map((entry) => (
                      <Cell
                        key={entry.feature}
                        fill={entry.impact === 'High' ? '#0E7490' : entry.impact === 'Medium' ? '#0EA5E9' : '#7DD3FC'}
                      />
                    ))}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            </div>
          </section>
        </div>

        <section className="rounded-2xl border border-slate-200 bg-white p-6 shadow-sm">
          <h2 className="text-lg font-semibold text-slate-900">Recommended Actions</h2>
          <ul className="mt-4 space-y-3">
            {mockPrediction.recommendedActions.map((action) => (
              <li key={action.text} className="flex items-start gap-3 rounded-lg border border-slate-200 bg-slate-50 p-3 text-sm text-slate-700">
                {action.type === 'warning' ? (
                  <AlertCircle className="mt-0.5 h-5 w-5 shrink-0 text-amber-600" />
                ) : (
                  <CheckCircle2 className="mt-0.5 h-5 w-5 shrink-0 text-emerald-600" />
                )}
                <span>{action.text}</span>
              </li>
            ))}
          </ul>
        </section>
      </div>

      <div className="fixed bottom-0 left-0 right-0 z-50 border-t border-red-200 bg-red-50 px-4 py-3 text-center text-sm font-semibold text-red-700">
        This AI result is for reference only and does not replace professional medical advice.
      </div>
    </div>
  );
}
